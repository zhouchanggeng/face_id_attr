"""ByteTrack 人脸跟踪器。

核心思想: 两阶段关联。
  第一阶段: 高置信度检测与所有 track 匹配
  第二阶段: 低置信度检测与第一阶段未匹配的 track 匹配（挽救被遮挡/模糊的人脸）

参考: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
"""
import numpy as np
from typing import List
from .iou_tracker import Track, _iou
from .sort_tracker import KalmanBoxTracker, _hungarian_match


class ByteTracker:
    """ByteTrack 跟踪器: 两阶段关联 + Kalman 预测。

    Args:
        high_threshold: 高置信度阈值（第一阶段）
        low_threshold: 低置信度阈值（第二阶段下限，低于此值直接丢弃）
        iou_threshold: IoU 匹配阈值
        max_missed: 最大连续未匹配帧数
        recognize_interval: 识别间隔帧数
    """

    def __init__(self, high_threshold: float = 0.5, low_threshold: float = 0.1,
                 iou_threshold: float = 0.3, max_missed: int = 15,
                 recognize_interval: int = 30):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.recognize_interval = recognize_interval
        self.tracks: List[Track] = []
        self._kalman: dict = {}

    def update(self, detections: List[dict]) -> List[Track]:
        # Kalman 预测
        for t in self.tracks:
            kf = self._kalman.get(t.track_id)
            if kf:
                t.bbox = kf.predict()

        if not detections:
            for t in self.tracks:
                t.missed += 1
            self._cleanup()
            return self.tracks

        # 按置信度分为高/低两组
        high_dets, low_dets = [], []
        for d in detections:
            if d["confidence"] >= self.high_threshold:
                high_dets.append(d)
            elif d["confidence"] >= self.low_threshold:
                low_dets.append(d)
            # 低于 low_threshold 的直接丢弃

        matched_trk = set()

        # ---- 第一阶段: 高置信度检测 vs 所有 track ----
        if high_dets and self.tracks:
            high_boxes = [d["bbox"] for d in high_dets]
            cost = np.zeros((len(self.tracks), len(high_dets)))
            for i, t in enumerate(self.tracks):
                for j, db in enumerate(high_boxes):
                    cost[i, j] = -_iou(t.bbox, db)

            matches, unmatched_trk_idx, unmatched_high_idx = _hungarian_match(
                cost, -self.iou_threshold)

            for ti, di in matches:
                self.tracks[ti].update(high_dets[di]["bbox"], high_dets[di]["confidence"])
                self._kalman[self.tracks[ti].track_id].update(high_dets[di]["bbox"])
                matched_trk.add(ti)

            remain_trk_idx = unmatched_trk_idx
            new_det_idx = unmatched_high_idx
        else:
            remain_trk_idx = list(range(len(self.tracks)))
            new_det_idx = list(range(len(high_dets)))

        # ---- 第二阶段: 低置信度检测 vs 未匹配的 track ----
        if low_dets and remain_trk_idx:
            low_boxes = [d["bbox"] for d in low_dets]
            remain_tracks = [self.tracks[i] for i in remain_trk_idx]

            cost2 = np.zeros((len(remain_tracks), len(low_dets)))
            for i, t in enumerate(remain_tracks):
                for j, db in enumerate(low_boxes):
                    cost2[i, j] = -_iou(t.bbox, db)

            matches2, still_unmatched, _ = _hungarian_match(
                cost2, -self.iou_threshold)

            for ri, di in matches2:
                orig_idx = remain_trk_idx[ri]
                self.tracks[orig_idx].update(low_dets[di]["bbox"], low_dets[di]["confidence"])
                self._kalman[self.tracks[orig_idx].track_id].update(low_dets[di]["bbox"])
                matched_trk.add(orig_idx)

        # 未匹配的 track: missed +1
        for i in range(len(self.tracks)):
            if i not in matched_trk:
                self.tracks[i].missed += 1

        # 未匹配的高置信度检测: 创建新 track
        for j in new_det_idx:
            t = Track(high_dets[j]["bbox"], high_dets[j]["confidence"])
            self.tracks.append(t)
            self._kalman[t.track_id] = KalmanBoxTracker(high_dets[j]["bbox"])

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
