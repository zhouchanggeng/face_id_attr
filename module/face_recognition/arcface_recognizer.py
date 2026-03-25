import cv2
import numpy as np
import onnxruntime
from .base import FaceRecognizer


class ArcFaceRecognizer(FaceRecognizer):
    """基于 ArcFace ONNX 模型的人脸识别器。

    支持 InsightFace 导出的 R50/R100 等 ONNX 模型，提取 512 维特征向量。
    可选模型:
        - glint360k_r50.onnx  (Glint360K 数据集训练，泛化性好)
        - webface_r50.onnx    (WebFace 数据集训练)

    输入: 对齐后的 112x112 BGR 人脸图像
    输出: 512 维 L2 归一化特征向量
    """

    def __init__(self, model_path: str = "models/arcface/glint360k_r50.onnx",
                 input_size: int = 112):
        self.input_size = input_size
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=onnxruntime.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取 512 维人脸特征（L2 归一化）。"""
        if face_image.shape[:2] != (self.input_size, self.input_size):
            face_image = cv2.resize(face_image, (self.input_size, self.input_size))

        # BGR -> RGB, HWC -> CHW, normalize to [-1, 1]
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})
        feat = outputs[0].flatten().astype(np.float32)

        # L2 归一化
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat
