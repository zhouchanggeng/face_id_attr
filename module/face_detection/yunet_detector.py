import cv2
import numpy as np
from typing import List
from .base import FaceDetector


class YuNetDetector(FaceDetector):
    """基于 OpenCV FaceDetectorYN (YuNet) 的人脸检测器。

    输出包含 5 点 landmarks（右眼、左眼、鼻尖、右嘴角、左嘴角），
    可配合 SFace 的 alignCrop 实现关键点对齐。
    """

    def __init__(
        self,
        model_path: str = "models/yunet/face_detection_yunet_2023mar.onnx",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        backend_id: int = 0,
        target_id: int = 0,
    ):
        self.conf_threshold = conf_threshold
        self._model = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=backend_id,
            target_id=target_id,
        )

    def detect(self, image: np.ndarray) -> List[dict]:
        h, w = image.shape[:2]
        self._model.setInputSize((w, h))
        _, faces = self._model.detect(image)

        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            conf = float(face[14])
            # 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
            landmarks = face[4:14].reshape(5, 2).astype(np.float32)

            results.append({
                "bbox": (int(x), int(y), int(x + fw), int(y + fh)),
                "confidence": conf,
                "landmarks": landmarks,
                # 保留原始 YuNet 格式，供 SFace alignCrop 使用
                "_yunet_face": face[:-1].copy(),
            })
        return results
