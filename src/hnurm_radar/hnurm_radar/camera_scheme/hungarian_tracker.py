"""
hungarian_tracker.py — 纯视觉 2D 目标关联器
==========================================================
利用 BBoxKalmanFilter 提供的高质量预测边界框，结合 YOLO 观测结果，
通过匈牙利算法实现帧间目标的 ID 关联与生命周期维护。

核心改动点：
  - 彻底解耦物理坐标变换，专职维护像素域的 RobotState。
  - 匹配逻辑由 "历史观测 vs 当前观测" 升级为 "卡尔曼先验预测 vs 当前观测"进行tracker匹配。
  - 引入 vote_pool 身份投票机制雏形。
  - 接入 shared.utils 工具箱，剔除内部冗余的 IoU 和坐标系转换代码。
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

# 导入底层数据基建与滤波引擎
from hnurm_radar.shared.type import RobotState, TrackingState, SingleDetectionResult
from hnurm_radar.filters.bbox_kalman import BBoxKalmanFilter
# 导入通用视觉计算工具箱
from hnurm_radar.shared.utils import compute_iou, xywh2xyxy

class HungarianTracker:
    """
    基于 BBoxKalmanFilter 的匈牙利关联器。
    严格只处理图像平面的 2D 追踪与关联。
    """
# // tunning: 增加 lost_thr 和 guess_thr 参数，分别控制浅层丢失和深度丢失的状态转换阈值
    def __init__(self, iou_thr=0.05, dist_thr=200, max_miss=90, lost_thr=3, guess_thr=15):
        self.tracks = []  # 存活轨迹列表 list[RobotState]
        self.kf = BBoxKalmanFilter()  # 无状态卡尔曼推演工具
        self.next_id = 1
        self.iou_thr = float(iou_thr)
        self.dist_thr = float(dist_thr)
        
        self.max_miss = int(max_miss)
        self.lost_thr = int(lost_thr)   # // tunning: 浅层丢失阈值
        self.guess_thr = int(guess_thr) # // tunning: 深度丢失盲猜阈值
        
        # 引入卡尔曼后，容忍丢失的帧数可以适当调大
        self.max_miss = int(max_miss) 

    def _cost_matrix(self, tracks, dets):
        """
        计算代价矩阵：卡尔曼预测框 vs YOLO 检测框
        tracks: list[RobotState]
        dets: list[SingleDetectionResult]
        """
        M, N = len(tracks), len(dets)
        if M == 0 or N == 0:
            return None
            
        cost = np.full((M, N), 1e6, dtype=np.float32)
        
        for i, tr in enumerate(tracks):
            # 提取卡尔曼滤波器【预测】的先验状态 [cx, cy, w, h]
            tcx, tcy, tw, th = tr.bbox_kf_state[:4]
            # 调用 utils 工具箱进行坐标系转换
            tr_xyxy = xywh2xyxy([tcx, tcy, tw, th])
            
            for j, d in enumerate(dets):
                dcx, dcy = d.xywh[0], d.xywh[1]
                # 调用 utils 工具箱计算 IoU
                iou = compute_iou(tr_xyxy, d.xyxy)
                dist = np.hypot(dcx - tcx, dcy - tcy) # 计算中心点距离
                
                # 门控逻辑：若 iou 很小且距离很大，保持高成本（拒绝匹配），防止误匹配
                if iou < self.iou_thr and dist > self.dist_thr:
                    continue
                    
                # cost: 优先 IoU（负），距离次要
                # tunning: ★ 修复分身问题。适当提高距离惩罚权重，
                # 防止由于高动态导致预测框偏离时，匈牙利算法强行开新 ID。
                cost[i, j] = -iou + 0.005 * dist
                
        return cost

    def update(self, detections):
        """
        执行一帧的追踪与状态更新。
        detections: list[SingleDetectionResult]
        返回: list[RobotState] 当前所有的存活机器人状态
        """
        # ==========================================
        # 1. 预测步 (Predict)
        # 强制所有存活轨迹利用运动学惯性向前推演一帧
        # ==========================================
        for tr in self.tracks:
            # // tunning: 只要丢失视野，立刻冻结像素层速度，防止预测框飘到别的机器人身上导致 ID 错误
            if tr.miss_cnt > 0:
                tr.bbox_kf_state[4:] = 0.0
            tr.bbox_kf_state, tr.bbox_kf_cov = self.kf.predict(tr.bbox_kf_state, tr.bbox_kf_cov)

        # ==========================================
        # 2. 匹配步 (Associate)
        # ==========================================
        cost = self._cost_matrix(self.tracks, detections)
        matches = []
        unmatched_t = list(range(len(self.tracks)))
        unmatched_d = list(range(len(detections)))
        
        if cost is not None:
            row, col = linear_sum_assignment(cost)
            matched_t = set()
            matched_d = set()
            for r, c in zip(row, col):
                if cost[r, c] >= 1e5:
                    continue
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)
            unmatched_t = [i for i in range(len(self.tracks)) if i not in matched_t]
            unmatched_d = [j for j in range(len(detections)) if j not in matched_d]

        # ==========================================
        # 3. 更新步 (Update) - 匹配成功的轨迹
        # ==========================================
        for ti, di in matches:
            tr = self.tracks[ti]
            det = detections[di]
            
            # 提取观测值 Z = [cx, cy, w, h] 并送入卡尔曼观测更新
            z = np.array(det.xywh)
            tr.bbox_kf_state, tr.bbox_kf_cov = self.kf.update(tr.bbox_kf_state, tr.bbox_kf_cov, z)
            
            # 身份投票更新 (为后续抗误检做铺垫)
            tr.vote_pool[det.label] = tr.vote_pool.get(det.label, 0) + 1
            
            # 状态机维护
            if tr.state in [TrackingState.LOST, TrackingState.GUESSING]:
                tr.state = TrackingState.RE_ACQUIRED
            else:
                tr.state = TrackingState.TRACKING
                
            tr.miss_cnt = 0
            tr.last_seen_time = time.time()

        # ==========================================
        # 4. 漏检处理 - 未匹配的轨迹
        # ==========================================
        for ti in unmatched_t:
            tr = self.tracks[ti]
            tr.miss_cnt += 1
            
            # // tunning: ★ 核心修复 - 真正的三段式状态机
            if tr.miss_cnt > self.guess_thr:
                # 漏检超过 guess_thr，进入长时遮挡，交由 guess_pts 进行赛场物理推演
                tr.state = TrackingState.GUESSING
            elif tr.miss_cnt > self.lost_thr:
                # 漏检超过 lost_thr 但未超过 guess_thr，依靠 bbox_kalman 在像素层滑行防闪烁
                tr.state = TrackingState.LOST

        # ==========================================
        # 5. 新生目标处理 - 未匹配的检测
        # ==========================================
        for dj in unmatched_d:
            det = detections[dj]
            new_id = det.track_id if det.track_id is not None else (9000 + self.next_id)
            self.next_id += 1
            
            # 实例化新的 RobotState 容器
            new_tr = RobotState(id=new_id)
            new_tr.vote_pool[det.label] = 1
            new_tr.last_seen_time = time.time()
            
            # 初始化卡尔曼状态与协方差矩阵
            z = np.array(det.xywh)
            new_tr.bbox_kf_state, new_tr.bbox_kf_cov = self.kf.initiate(z)
            
            self.tracks.append(new_tr)

        # ==========================================
        # 6. 垃圾回收 - 区分“真身”与“噪音”防卡顿
        # ==========================================
        surviving_tracks = []
        for tr in self.tracks:
            # // tunning: 统计该 ID 在整个生命周期内被 YOLO 真正“看清楚”的总次数
            total_hits = sum(tr.vote_pool.values())
            
            # // tunning: 核心策略 —— 区分对待：
            # 1. 如果是确认过的“真车”（命中>=5次），允许它在掩体后滑行 max_miss 帧 (2.8s)。
            # 2. 如果只是闪现的“噪音”（命中<5次），丢视野 3 帧立刻销毁，防止堆积导致卡顿！
            allowed_max_miss = self.max_miss if total_hits >= 5 else 3
            
            if tr.miss_cnt <= allowed_max_miss:
                surviving_tracks.append(tr)
                
        self.tracks = surviving_tracks       
        return self.tracks

    def get_active_tracks(self):
        return list(self.tracks)