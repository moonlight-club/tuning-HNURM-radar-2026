"""
hungarian_tracker.py — 纯视觉 2D 目标关联器
==========================================================
本模块在图像像素坐标系内完成目标关联与状态维护。
它结合边界框卡尔曼bbox_kalman预测结果与当前帧检测结果，通过匈牙利最优分配算法实现
稳定的跨帧 ID 关联，并输出可供上层发布逻辑直接消费的轨迹状态集合。

核心逻辑与主要功能：
    - 以 `BBoxKalmanFilter` 先验预测框作为关联基准，执行“预测框 vs 检测框”的代价匹配。
    - 使用 IoU、中心距离与身份一致性共同构建代价矩阵，并通过空间门控抑制误匹配。
    - 维护 `RobotState` 生命周期状态机（TRACKING / LOST / GUESSING / RE_ACQUIRED）。
    - 维护 `vote_pool` 身份投票池，通过时间衰减与惯性增益提升短时遮挡下的身份稳定性。
    - 对长期漏检或低可信轨迹执行回收，并进行全局同标签冲突仲裁，保证 ID 语义一致性。
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
    # [debug]适当调高vote_delay，降低遗忘速率
    def __init__(self, iou_thr=0.05, dist_thr=120, max_miss=45, lost_thr=3, guess_thr=10, 
                 label_penalty=1000.0, vote_decay=0.97):
        """初始化关联器阈值与投票参数。"""
        
        self.label_penalty = float(label_penalty) # 发生分类冲突时的巨大代价惩罚
        self.vote_decay = float(vote_decay)       # 每帧历史选票的衰减系数，用于遗忘旧状态
        self.tracks = []  # 存活轨迹列表 list[RobotState]
        self.kf = BBoxKalmanFilter()  # 无状态卡尔曼推演工具
        self.next_id = 1
        self.iou_thr = float(iou_thr)
        self.dist_thr = float(dist_thr)
        
        self.max_miss = int(max_miss)   # 引入卡尔曼后，容忍丢失的帧数可以适当调大
        self.lost_thr = int(lost_thr)   # 浅层丢失阈值
        self.guess_thr = int(guess_thr) # 深度丢失盲猜阈值
         

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
        
        # 匹配得分权重参数
        W_id = 5.0    # 身份一致性权重
        W_iou = 1.0   # 边界框交并比 (IoU) 权重
        # [debug]提高欧氏距离权重，强化空间门控的区分能力
        W_dist = 2.5  # 归一化中心距离权重
        W_bot = 1.0   # BoT-SORT bot_id 帧间关联一致性权重

        for i, tr in enumerate(tracks):
            # 提取卡尔曼滤波器【预测】的先验状态 [cx, cy, w, h]
            tcx, tcy, tw, th = tr.bbox_kf_state[:4]
            # 调用 utils 工具箱进行坐标系转换
            tr_xyxy = xywh2xyxy([tcx, tcy, tw, th])
            
            # 提取轨迹历史最高票标签作为代表身份
            best_label = max(tr.vote_pool, key=tr.vote_pool.get) if tr.vote_pool else "NULL"

            for j, d in enumerate(dets):
                dcx, dcy = d.xywh[0], d.xywh[1]
                # 调用 utils 工具箱计算 IoU
                iou = compute_iou(tr_xyxy, d.xyxy)

                # 引入归一化中心距离：远目标强调欧式距离，其余场景由 IoU + 距离联合判定
                # 消除目标尺度对距离代价的影响，提升远距离小目标的匹配鲁棒性
                dist = np.hypot(dcx - tcx, dcy - tcy)       # 绝对像素距离
                diag = np.hypot(tw, th)                     # 预测框对角线长度
                norm_dist = dist / (diag + 1e-5)            # 归一化距离 (加极小值防除零)
                
                # 空间门控过滤：距离差异过大且 IoU 低于设定阈值时，拒绝匹配
                # 使用配置参数 self.dist_thr（像素）作为硬门控，norm_dist 作为辅助
                if iou < self.iou_thr and dist > self.dist_thr:
                    continue
                    
                # 1. 计算基础几何得分
                geom_score = (iou * W_iou) + (max(0, 1.0 - norm_dist) * W_dist)

                # 2. 计算身份一致性得分与排他性惩罚
                id_score = 0.0
                if best_label != "NULL" and d.label != "NULL":
                    if best_label == d.label:
                        id_score = 1.0  # 类别标签一致
                    else:
                        # 身份互斥冲突：直接施加设定的极大惩罚代价，跳过后续得分转换
                        cost[i, j] = self.label_penalty
                        continue
                elif best_label == "NULL" or d.label == "NULL":
                    # 存在未知标签 (NULL) 时，提供基础正向得分，协助状态机平滑过渡
                    id_score = 0.5

                # 3. 计算 BoT-SORT bot_id 帧间关联一致性得分
                # bot_id 相同表示 BoT-SORT 判定为同一物理目标, 作为辅助匹配信号
                bot_score = 0.0
                det_bot_id = getattr(d, 'track_id', None)
                if tr.bot_id >= 0 and det_bot_id is not None:
                    if tr.bot_id == det_bot_id:
                        bot_score = 1.0  # bot_id 一致, 加分

                # 4. 计算最终匹配代价 (取累加得分的相反数)
                total_score = geom_score + (id_score * W_id) + (bot_score * W_bot)
                cost[i, j] = -total_score
                
                
        return cost

    def update(self, detections, dt=0.033):
        """
        执行一帧的追踪与状态更新。
        detections: list[SingleDetectionResult]
        dt: 真实时间步长
        返回: list[RobotState] 当前所有的存活机器人状态
        """
        # ==========================================
        # 0. 数据清洗 (Data Sanitization)
        # 拦截并过滤输入流中的 NULL 观测数据，确保下游张量运算合法性
        # ==========================================
        valid_detections = []
        if detections:
            for det in detections:
                if det is None:
                    continue
                # 校验核心空间属性是否完整
                if getattr(det, 'xywh', None) is None or getattr(det, 'xyxy', None) is None:
                    continue
                valid_detections.append(det)
        # 覆写原始输入，切断脏数据传播路径
        detections = valid_detections

        # ==========================================
        # 1. 预测步 (Predict)
        # 强制所有存活轨迹利用运动学惯性向前推演一帧
        # ==========================================
        for tr in self.tracks:

            # 引入身份投票池时间衰减机制
            # 每经过一帧，将所有历史选票乘以 vote_decay，实现平滑遗忘,使得错误观测的影响逐渐减弱，允许轨迹在短暂丢失后恢复正确身份。
            # 仅当目标处于视野内 (miss_cnt == 0) 时执行身份衰减。
            # 目标被遮挡期间必须冻结历史身份，防止盲猜推演因失去 ID 而中断。
            if tr.miss_cnt == 0:
                for label_key in list(tr.vote_pool.keys()):
                    tr.vote_pool[label_key] *= self.vote_decay
                    # 清理低于阈值的“死票”，防止字典无限膨胀
                    if tr.vote_pool[label_key] < 0.1:
                        del tr.vote_pool[label_key]

            # 只要丢失视野，立刻冻结像素层速度，防止预测框飘到别的机器人身上导致 ID 错误
            # if tr.miss_cnt > 0:
            #     tr.bbox_kf_state[4:] = 0.0
            tr.bbox_kf_state, tr.bbox_kf_cov = self.kf.predict(tr.bbox_kf_state, tr.bbox_kf_cov, dt)
        
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

            # 同步 BoT-SORT 的底层帧间关联 ID
            if det.track_id is not None:
                tr.bot_id = det.track_id
            
            # 身份惯性投票逻辑（EMA 优化版）
            # 核心目的：在单帧漏检数字时，利用物理框的连续性维持之前的兵种身份，且防止 NULL 稀释权重
            if det.label != "NULL":
                # 情况 A：当前帧清晰地识别到了数字，正常进行权重累加
                tr.vote_pool[det.label] = tr.vote_pool.get(det.label, 0.0) + det.conf
            else:
                # 情况 B：当前帧只看到了车身 (NULL)，启动惯性维持机制
                if len(tr.vote_pool) > 0:
                    # 获取池中除了 "NULL" 以外的所有真实兵种标签 (如 'B1', 'R3' 等)
                    real_labels = [k for k in tr.vote_pool.keys() if k != "NULL"]
                    
                    if len(real_labels) > 0:
                        # ★ 核心修正点：只把惯性权重分配给“真实身份”
                        # 这样即使当前检测是 NULL，B3 或 R1 的权重依然会小幅增长，从而压制住 NULL
                        inertia_weight = (det.conf * 0.1) / len(real_labels)
                        for k in real_labels:
                            tr.vote_pool[k] += inertia_weight
                    else:
                        # 如果池子里目前全是 NULL（说明该目标自出现起就从未被认出过数字）
                        # 则只能给 NULL 加权，作为临时的身份占位
                        tr.vote_pool["NULL"] = tr.vote_pool.get("NULL", 0.0) + det.conf * 0.1




            # 状态机维护
            if tr.state in [TrackingState.LOST, TrackingState.GUESSING]:
                tr.state = TrackingState.RE_ACQUIRED
            else:
                tr.state = TrackingState.TRACKING
            
            # 物理命中计数器递增
            tr.hit_cnt += 1
            tr.miss_cnt = 0
            tr.last_seen_time = time.time()

        # ==========================================
        # 4. 漏检处理 - 未匹配的轨迹
        # ==========================================
        for ti in unmatched_t:
            tr = self.tracks[ti]
            tr.miss_cnt += 1

      
            
            # 核心修复：真正的三段式状态机
            if tr.miss_cnt > self.guess_thr:
                # 漏检超过 guess_thr，进入长时遮挡，交由 guess_pts 进行赛场物理推演
                tr.state = TrackingState.GUESSING
            elif tr.miss_cnt > self.lost_thr:
                if tr.state != TrackingState.LOST:
                    vx, vy = tr.bbox_kf_state[4], tr.bbox_kf_state[5]
                    vw, vh = tr.bbox_kf_state[6], tr.bbox_kf_state[7]
                
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
            # 初始投票权重必须使用检测结果的真实置信度，防止 NULL 获得满额权重
            new_tr.vote_pool[det.label] = det.conf
            new_tr.last_seen_time = time.time()

            # --- ★ 修复点 3：初始化物理命中次数 ---
            new_tr.hit_cnt = 1

            # 继承 BoT-SORT 的底层帧间关联 ID
            if det.track_id is not None:
                new_tr.bot_id = det.track_id

            # 初始化卡尔曼状态与协方差矩阵
            z = np.array(det.xywh)
            new_tr.bbox_kf_state, new_tr.bbox_kf_cov = self.kf.initiate(z)

            self.tracks.append(new_tr)

        # ==========================================
        # 6. 垃圾回收 - 区分“真身”与“噪音”
        # ==========================================
        surviving_tracks = []
        for tr in self.tracks:
            # 统计该 ID 在整个生命周期内被 YOLO 真正“看清楚”的总次数
            total_hits = sum(tr.vote_pool.values())
            
            # 核心策略：区分对待
            # 1. 如果是确认过的“真车”（命中>=5次），允许它在掩体后滑行 max_miss 帧 (1.5s)。
            # 2. 如果只是闪现的“噪音”（命中<5次），丢视野 3 帧立刻销毁，防止堆积导致卡顿！
            # [debug] 终止生存策略修改，还原原始逻辑
            allowed_max_miss = self.max_miss if total_hits >= 5 else 3
            
            if tr.miss_cnt <= allowed_max_miss:
                surviving_tracks.append(tr)
                
        self.tracks = surviving_tracks 
        
        # ==========================================
        # 7. 全局 ID 唯一性抑制 (Global ID NMS)
        # 拦截并销毁抢夺真实 ID 选票的长期丢失轨迹，解决分身劫持异常
        # ==========================================
        label_to_track = {}
        for tr in self.tracks:
            best_label = max(tr.vote_pool, key=tr.vote_pool.get) if tr.vote_pool else "NULL"
            if best_label == "NULL":
                continue
                
            if best_label not in label_to_track:
                label_to_track[best_label] = tr
            else:
                # 发生身份冲突：两个轨迹均声称拥有同一高置信度标签
                existing_tr = label_to_track[best_label]
                existing_hits = sum(existing_tr.vote_pool.values())
                current_hits = sum(tr.vote_pool.values())
                
                # 仲裁逻辑 1: 优先保留当前正处于视觉锁定状态 (TRACKING) 的轨迹
                if existing_tr.state == TrackingState.TRACKING and tr.state != TrackingState.TRACKING:
                    winner, loser = existing_tr, tr
                elif tr.state == TrackingState.TRACKING and existing_tr.state != TrackingState.TRACKING:
                    winner, loser = tr, existing_tr
                else:
                    # 仲裁逻辑 2: 若状态层级一致，依据历史累计命中票数判定真伪
                    if current_hits > existing_hits:
                        winner, loser = tr, existing_tr
                    else:
                        winner, loser = existing_tr, tr
                        
                # 惩罚执行: 强行剥夺伪轨迹/失效轨迹的争议身份选票
                if best_label in loser.vote_pool:
                    loser.vote_pool[best_label] = 0.0
                
                # 更新字典映射为胜出者
                label_to_track[best_label] = winner     
                
        return self.tracks

    def get_active_tracks(self):
        return list(self.tracks)