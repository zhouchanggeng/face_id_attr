import cv2
import numpy as np
from .base import FaceRecognizer


class SFaceRecognizer(FaceRecognizer):
    """基于 OpenCV FaceRecognizerSF (SFace) 的人脸识别器。

    使用 SFace ONNX 模型提取 128 维特征向量，无需 deepface 依赖。
    模型文件: face_recognition_sface_2021dec.onnx
    """

    def __init__(
        self,
        model_path: str = "models/face_recognition_sface_2021dec.onnx",
        backend_id: int = 0,
        target_id: int = 0,
    ):
        self._model = cv2.FaceRecognizerSF.create(
            model=model_path,
            config="",
            backend_id=backend_id,
            target_id=target_id,
        )

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取 128 维人脸特征。输入为已对齐/裁剪的人脸图像。"""
        # SFace 期望 112x112 输入
        if face_image.shape[:2] != (112, 112):
            face_image = cv2.resize(face_image, (112, 112))
        feature = self._model.feature(face_image)
        return feature.flatten().astype(np.float32)

    def align_crop(self, image: np.ndarray, yunet_face: np.ndarray) -> np.ndarray:
        """使用 SFace 内置的 alignCrop 基于 YuNet 关键点进行仿射对齐。

        Args:
            image: 原始 BGR 图像
            yunet_face: YuNet 检测输出的单行数据 (x,y,w,h + 5 landmarks)
        Returns:
            对齐后的 112x112 人脸图像
        """
        return self._model.alignCrop(image, yunet_face)
