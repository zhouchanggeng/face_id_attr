import cv2
import numpy as np
from typing import List
from .base import FaceAnalyzer


class DeepFaceAnalyzer(FaceAnalyzer):
    """基于 DeepFace 的人脸属性分析（年龄、性别、表情、种族）。

    actions 可选: ['age', 'gender', 'emotion', 'race']
    """

    def __init__(self, actions: List[str] = None,
                 detector_backend: str = "skip"):
        self.actions = actions or ["age", "gender", "emotion", "race"]
        self.detector_backend = detector_backend

    def analyze(self, image: np.ndarray, faces: List[dict]) -> List[dict]:
        from deepface import DeepFace

        results = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            crop = image[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                results.append({})
                continue
            crop = cv2.resize(crop, (224, 224))

            try:
                objs = DeepFace.analyze(
                    img_path=crop,
                    actions=self.actions,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True,
                )
                attr = objs[0] if isinstance(objs, list) else objs
                results.append({
                    "age": attr.get("age"),
                    "gender": attr.get("dominant_gender"),
                    "dominant_emotion": attr.get("dominant_emotion"),
                    "emotion": attr.get("emotion"),
                    "dominant_race": attr.get("dominant_race"),
                    "race": attr.get("race"),
                })
            except Exception:
                results.append({})
        return results
