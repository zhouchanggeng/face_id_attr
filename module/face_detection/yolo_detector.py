import numpy as np
from typing import List
from .base import FaceDetector


class YOLOFaceDetector(FaceDetector):
    """基于 Ultralytics YOLO 的人脸检测器。

    支持 YOLOv8/v11/v12/v26 等所有 ultralytics 兼容的检测模型。
    模型权重文件为 .pt 格式（训练产出的 best.pt / last.pt）。
    """

    def __init__(
        self,
        model_path: str = "models/yolo26m_wider_face/weights/best.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        max_det: int = 300,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det

    def detect(self, image: np.ndarray) -> List[dict]:
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device or None,
            max_det=self.max_det,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "landmarks": None,
                })
        return detections
