"""SORT (Simple Online and Realtime Tracking) 人脸跟踪器。

基于 Kalman 滤波预测 + 匈牙利算法匹配，比 IoU 贪心匹配更鲁棒。
参考: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""
import numpy as np
from typing import List
from .iou_tracker import Track, _iou


class KalmanBoxTracker:
    """单个目标的 Kalman 滤波器，状态为 [cx, cy, s, r, dx, dy, ds]。

    cx, cy: 中心坐标
    s: 面积
    r: 宽高比
    dx, dy, ds: 速度
    """

    def __init__(self, bbox):
        # 状态: [cx, cy, s, r, dx, dy, ds]
        self.dim_x = 7
        self.dim_z = 4
        self.x = np.zeros(self.dim_x)  # 状态
        self.P = np.eye(self.dim_x) * 10  # 协方差
        self.P[4:, 4:] *= 100  # 速度不确定性更大
        self.Q = np.eye(self.dim_x) * 0.01  # 过程噪声
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(self.dim_z) * 1.0  # 观测噪声
        self.F = np.eye(self.dim_x)  # 状态转移
        self.F[0, 4] = 1  # cx += dx
        self.F[1, 5] = 1  # cy += dy
        self.F[2, 6] = 1  # s += ds
        self.H = np.zeros((self.dim_z, self.dim_x))  # 观测矩阵
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1

        z = self._bbox_to_z(bbox)
        self.x[:4] = z

    @staticmethod
    def _bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(y2 - y1, 1e-6)
        return np.array([cx, cy, s, r])

    def _z_to_bbox(self, z):
        cx, cy, s, r = z
        s = max(s, 1)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    def predict(self):
        # s 不能为负
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._z_to_bbox(self.x[:4])

    def update(self, bbox):
        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x  # 残差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman 增益
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_bbox(self):
        return self._z_to_bbox(self.x[:4])


def _hungarian_match(cost_matrix, threshold):
    """简易匈牙利算法（贪心近似），返回 (matches, unmatched_rows, unmatched_cols)。"""
    rows, cols = cost_matrix.shape
    matched_rows, matched_cols = set(), set()
    matches = []

    # 按 cost 从小到大（IoU 取负后从小到大 = IoU 从大到小）
    indices = np.dstack(np.unravel_index(np.argsort(cost_matrix.ravel()), cost_matrix.shape))[0]
    for r, c in indices:
        if r in matched_rows or c in matched_cols:
            continue
        if cost_matrix[r, c] > threshold:
            break
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_rows = [i for i in range(rows) if i not in matched_rows]
    unmatched_cols = [j for j in range(cols) if j not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class SORTTracker:
    """SORT 跟踪器: Kalman 预测 + IoU 匈牙利匹配。

    Args:
        iou_threshold: IoU 匹配阈值
        max_missed: 最大连续未匹配帧数
        recognize_interval: 识别间隔帧数
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15,
                 recognize_interval: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.recognize_interval = recognize_interval
        self.tracks: List[Track] = []
        self._kalman: dict = {}  # track_id -> KalmanBoxTracker

    def update(self, detections: List[dict]) -> List[Track]:
        # Kalman 预测
        for t in self.tracks:
            kf = self._kalman.get(t.track_id)
            if kf:
                pred_bbox = kf.predict()
                t.bbox = pred_bbox

        if not detections:
            for t in self.tracks:
                t.missed += 1
            self._cleanup()
            return self.tracks

        det_boxes = [d["bbox"] for d in detections]
        det_confs = [d["confidence"] for d in detections]

        if self.tracks:
            # IoU cost matrix (取负用于最小化)
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, db in enumerate(det_boxes):
                    cost[i, j] = -_iou(t.bbox, db)

            matches, unmatched_trk, unmatched_det = _hungarian_match(
                cost, -self.iou_threshold)

            for ti, di in matches:
                self.tracks[ti].update(det_boxes[di], det_confs[di])
                self._kalman[self.tracks[ti].track_id].update(det_boxes[di])

            for i in unmatched_trk:
                self.tracks[i].missed += 1

            for j in unmatched_det:
                t = Track(det_boxes[j], det_confs[j])
                self.tracks.append(t)
                self._kalman[t.track_id] = KalmanBoxTracker(det_boxes[j])
        else:
            for j in range(len(detections)):
                t = Track(det_boxes[j], det_confs[j])
                self.tracks.append(t)
                self._kalman[t.track_id] = KalmanBoxTracker(det_boxes[j])

        self._cleanup()
        return self.tracks

    def needs_recognition(self, track: Track) -> bool:
        if not track.recognized:
            return True
        if self.recognize_interval > 0 and track.age % self.recognize_interval == 0:
            return True
        return False

    def _cleanup(self):
        removed = [t for t in self.tracks if t.missed > self.max_missed]
        for t in removed:
            self._kalman.pop(t.track_id, None)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

    def reset(self):
        self.tracks = []
        self._kalman = {}
        Track._next_id = 0
