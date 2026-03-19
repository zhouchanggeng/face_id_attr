import cv2
import numpy as np
from typing import List
from .base import FaceDetector


class DeepFaceDetector(FaceDetector):
    """基于 DeepFace 的人脸检测，支持多种 backend。

    可选 backend:
        opencv, ssd, dlib, mtcnn, fastmtcnn, retinaface,
        mediapipe, yolov8n, yolov8m, yolov8l, yolov11n,
        yolov11s, yolov11m, yolov11l, yolov12n, yolov12s,
        yolov12m, yolov12l, yunet, centerface
    """

    def __init__(self, backend: str = "mtcnn", align: bool = True,
                 conf_threshold: float = 0.5, max_size: int = 1920):
        self.backend = backend
        self.align = align
        self.conf_threshold = conf_threshold
        self.max_size = max_size

    def detect(self, image: np.ndarray) -> List[dict]:
        from deepface import DeepFace

        # 超大图缩放，防止内存溢出/段错误
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        face_objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend=self.backend,
            align=self.align,
            enforce_detection=False,
        )

        results = []
        for obj in face_objs:
            conf = obj.get("confidence", 0) or 0
            if conf < self.conf_threshold:
                continue
            region = obj["facial_area"]
            rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]

            # 坐标映射回原图尺寸
            inv = 1.0 / scale
            x1, y1 = int(rx * inv), int(ry * inv)
            x2, y2 = int((rx + rw) * inv), int((ry + rh) * inv)

            # 提取 landmarks（如果有）
            landmarks = None
            if region.get("left_eye") and region.get("right_eye"):
                pts = []
                for key in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
                    pt = region.get(key)
                    if pt:
                        pts.append((pt[0] * inv, pt[1] * inv))
                if pts:
                    landmarks = np.array(pts, dtype=np.float32)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "landmarks": landmarks,
            })
        return results
