import numpy as np
from .base import FaceRecognizer


class DeepFaceRecognizer(FaceRecognizer):
    """基于 DeepFace 的深度学习人脸识别，支持多种模型。

    可选 model_name:
        VGG-Face, Facenet, Facenet512, OpenFace, DeepFace,
        DeepID, ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L
    """

    def __init__(self, model_name: str = "Facenet512",
                 detector_backend: str = "skip",
                 normalization: str = "base"):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.normalization = normalization

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        from deepface import DeepFace

        results = DeepFace.represent(
            img_path=face_image,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            normalization=self.normalization,
            enforce_detection=False,
        )
        if not results:
            return np.zeros(512, dtype=np.float32)
        return np.array(results[0]["embedding"], dtype=np.float32)
