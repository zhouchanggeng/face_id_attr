"""轻量级 IoU 人脸跟踪器，无额外依赖。

通过相邻帧 bbox 的 IoU 关联同一张人脸，维护 track 生命周期。
每个 track 携带身份信息，由外部识别模块填充。
"""
import numpy as np
from typing import List, Dict, Optional


class Track:
    """单个人脸跟踪轨迹。"""
    _next_id = 0

    def __init__(self, bbox, confidence, track_id=None):
        if track_id is None:
            self.track_id = Track._next_id
            Track._next_id += 1
        else:
            self.track_id = track_id
        self.bbox = bbox              # (x1, y1, x2, y2)
        self.confidence = confidence
        self.identity = None          # 识别结果
        self.similarity = 0.0
        self.quality = None
        self.age = 0                  # 已存活帧数
        self.missed = 0              # 连续未匹配帧数
        self.recognized = False      # 是否已识别过
        self.feature = None          # 最近一次特征

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.missed = 0
        self.age += 1


def _iou(box_a, box_b):
    """计算两个 bbox 的 IoU。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """基于 IoU 匹配的多目标人脸跟踪器。

    Args:
        iou_threshold: IoU 匹配阈值，低于此值不关联
        max_missed: 连续未匹配帧数上限，超过则删除 track
        recognize_interval: 每隔多少帧重新识别一次（0=仅首次识别）
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15,
                 recognize_interval: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.recognize_interval = recognize_interval
        self.tracks: List[Track] = []

    def update(self, detections: List[dict]) -> List[Track]:
        """用当前帧的检测结果更新跟踪器。

        Args:
            detections: [{bbox, confidence, ...}, ...]
        Returns:
            当前活跃的 track 列表
        """
        if not detections:
            for t in self.tracks:
                t.missed += 1
            self._cleanup()
            return self.tracks

        det_boxes = [d["bbox"] for d in detections]
        det_confs = [d["confidence"] for d in detections]

        # 计算 IoU 矩阵
        matched_det = set()
        matched_trk = set()

        if self.tracks:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, db in enumerate(det_boxes):
                    iou_matrix[i, j] = _iou(t.bbox, db)

            # 贪心匹配（按 IoU 从高到低）
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                if max_iou < self.iou_threshold:
                    break
                ti, di = max_idx
                self.tracks[ti].update(det_boxes[di], det_confs[di])
                matched_det.add(di)
                matched_trk.add(ti)
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # 未匹配的 track: missed +1
        for i, t in enumerate(self.tracks):
            if i not in matched_trk:
                t.missed += 1

        # 未匹配的检测: 创建新 track
        for j in range(len(detections)):
            if j not in matched_det:
                self.tracks.append(Track(det_boxes[j], det_confs[j]))

        self._cleanup()
        return self.tracks

    def needs_recognition(self, track: Track) -> bool:
        """判断该 track 是否需要（重新）识别。"""
        if not track.recognized:
            return True
        if self.recognize_interval > 0 and track.age % self.recognize_interval == 0:
            return True
        return False

    def _cleanup(self):
        """移除超时的 track。"""
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

    def reset(self):
        """重置跟踪器。"""
        self.tracks = []
        Track._next_id = 0
