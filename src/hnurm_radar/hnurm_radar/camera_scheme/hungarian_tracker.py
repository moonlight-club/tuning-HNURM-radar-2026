import time
import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, bbox_xyxy, label, field_xy=None):
        self.id = track_id
        self.bbox = bbox_xyxy  # [x1,y1,x2,y2] in original image coords
        self.label = label
        self.last_field = field_xy  # (fx, fy) in meters
        self.last_update = time.time()
        self.miss_cnt = 0

class HungarianTracker:
    """
    tunning: 简单匈牙利关联器（解耦于 camera_detector）
    - update(detections, to_field_fn) : detections = list of [xyxy, xywh, track_id, label]
    - get_active_tracks() -> list of Track
    """
    def __init__(self, iou_thr=0.05, dist_thr=200, max_miss=3):
        self.tracks = []  # list[Track]
        self.next_id = 1
        self.iou_thr = float(iou_thr)
        self.dist_thr = float(dist_thr)
        self.max_miss = int(max_miss)

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
        inter = interW * interH
        areaA = max(1.0, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
        areaB = max(1.0, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
        return inter / (areaA + areaB - inter)

    def _cost_matrix(self, tracks, dets):
        M, N = len(tracks), len(dets)
        if M == 0 or N == 0:
            return None
        cost = np.full((M, N), 1e6, dtype=np.float32)
        for i, tr in enumerate(tracks):
            tcx = (tr.bbox[0] + tr.bbox[2]) / 2.0
            tcy = (tr.bbox[1] + tr.bbox[3]) / 2.0
            for j, d in enumerate(dets):
                db = d[0]  # xyxy
                dcx = (db[0] + db[2]) / 2.0
                dcy = (db[1] + db[3]) / 2.0
                iou = self._iou(tr.bbox, db)
                dist = np.hypot(dcx - tcx, dcy - tcy)
                # 门控：若 iou 很小且距离很大，保持高成本（不匹配）
                if iou < self.iou_thr and dist > self.dist_thr:
                    continue
                # cost: 优先 IoU（负），距离次要
                cost[i, j] = -iou + 0.002 * dist
        return cost

    def update(self, detections, to_field_fn=None, orig_size=None, infer_size=None):
        """
        detections: list of [xyxy_box, xywh_box, track_id, label]
        to_field_fn: callable(cx, cy) -> (fx, fy) or None
        orig_size/infer_size: optional for coordinate scaling (handled by caller if needed)
        """
        det_list = []
        for det in detections:
            xyxy = det[0]
            det_list.append([xyxy, det[1], det[2], det[3]])

        # 构造代价矩阵并匈牙利匹配
        cost = self._cost_matrix(self.tracks, det_list)
        matches = []
        unmatched_t = list(range(len(self.tracks)))
        unmatched_d = list(range(len(det_list)))
        if cost is not None:
            row, col = linear_sum_assignment(cost)
            matches = []
            matched_t = set(); matched_d = set()
            for r, c in zip(row, col):
                if cost[r, c] >= 1e5:
                    continue
                matches.append((r, c))
                matched_t.add(r); matched_d.add(c)
            unmatched_t = [i for i in range(len(self.tracks)) if i not in matched_t]
            unmatched_d = [j for j in range(len(det_list)) if j not in matched_d]

        # 更新匹配到的轨迹
        for ti, di in matches:
            tr = self.tracks[ti]
            db, _, det_tid, det_label = det_list[di]
            tr.bbox = db
            tr.label = det_label
            tr.miss_cnt = 0
            tr.last_update = time.time()
            # tunning: 保存场地坐标（如果提供转换函数）
            if to_field_fn is not None:
                x1, y1, x2, y2 = db
                cx, cy = (x1 + x2) / 2.0, y2
                try:
                    tr.last_field = to_field_fn(cx, cy)
                except Exception:
                    tr.last_field = None

        # 未匹配轨迹：miss++（短期保留）
        for ti in unmatched_t:
            tr = self.tracks[ti]
            tr.miss_cnt += 1

        # 未匹配检测：新建 track
        for dj in unmatched_d:
            db, _, det_tid, det_label = det_list[dj]
            # 以检测的 track_id 做为 track id（保持与模型 track 暂时一致），否则分配内部 id
            new_id = det_tid if det_tid is not None else (9000 + self.next_id)
            self.next_id += 1
            field_xy = None
            if to_field_fn is not None:
                x1, y1, x2, y2 = db
                cx, cy = (x1 + x2) / 2.0, y2
                try:
                    field_xy = to_field_fn(cx, cy)
                except Exception:
                    field_xy = None
            self.tracks.append(Track(new_id, db, det_label, field_xy))

        # 删除超时轨迹
        self.tracks = [tr for tr in self.tracks if tr.miss_cnt <= self.max_miss]
        return self.tracks

    def get_active_tracks(self):
        return list(self.tracks)
