import cv2
import numpy as np
import onnxruntime
from .base import FaceQualityAssessor


class FQAAssessor(FaceQualityAssessor):
    """基于达摩院 FQA 模型的人脸质量评估。

    模型输入: 对齐后的 112x112 人脸图像
    模型输出: 10 个 rank 概率，通过加权求和转换为 0~1 质量分

    质量分含义:
        0.0~0.2: 极差（严重遮挡/模糊/非人脸）
        0.2~0.4: 较差（大角度侧脸/部分遮挡）
        0.4~0.6: 一般（光照不均/轻微模糊）
        0.6~0.8: 良好（正常拍摄条件）
        0.8~1.0: 优秀（正脸/清晰/光照均匀）
    """

    def __init__(self, model_path: str, input_size: int = 112):
        self.input_size = input_size
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=onnxruntime.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def assess(self, face_image: np.ndarray) -> float:
        """评估人脸质量，返回 0~1 分数。"""
        if face_image.shape[:2] != (self.input_size, self.input_size):
            face_image = cv2.resize(face_image, (self.input_size, self.input_size))

        # 预处理: BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)    # NCHW

        outputs = self.session.run(None, {self.input_name: img})
        probs = outputs[0][0]  # shape: (10,)

        # softmax
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / exp_probs.sum()

        # 加权求和: rank 0~9 映射到 0~1
        ranks = np.arange(10, dtype=np.float32) / 9.0
        score = float(np.dot(probs, ranks))
        return score
