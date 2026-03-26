import cv2
import numpy as np
import onnxruntime
from typing import List
from .base import FaceAnalyzer


class ExpressionAnalyzer(FaceAnalyzer):
    """基于 YOLO 分类模型的人脸表情识别（RAF-DB 7类）。

    输入: 对齐后的人脸图像（由 pipeline 在对齐后传入）
    输出: 7 类表情概率 + 主要表情

    RAF-DB 7 类表情 (文件夹 1-7，按名称排序后索引 0-6):
        0 -> 1 -> Surprise
        1 -> 2 -> Fear
        2 -> 3 -> Disgust
        3 -> 4 -> Happiness
        4 -> 5 -> Sadness
        5 -> 6 -> Anger
        6 -> 7 -> Neutral
    """

    EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

    def __init__(self, model_path: str, input_size: int = 224):
        self.input_size = input_size
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=onnxruntime.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """预处理: resize -> RGB -> normalize [0,1] -> CHW -> NCHW。"""
        img = cv2.resize(face_image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def analyze(self, image: np.ndarray, faces: List[dict]) -> List[dict]:
        """对每张检测到的人脸进行表情分析。

        注意: 此方法接收原图 + 检测框列表，内部裁剪人脸后分析。
        如果 pipeline 已提供对齐后的人脸，可直接调用 classify()。
        """
        results = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            crop = image[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                results.append({})
                continue
            result = self.classify(crop)
            results.append(result)
        return results

    def classify(self, face_image: np.ndarray) -> dict:
        """对单张人脸图像进行表情分类。

        Args:
            face_image: BGR 人脸图像（对齐后的 112x112 或任意尺寸）
        Returns:
            {"dominant_emotion": str, "emotion": {name: prob, ...}}
        """
        blob = self._preprocess(face_image)
        outputs = self.session.run(None, {self.input_name: blob})
        logits = outputs[0][0]

        # softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        emotion_dict = {name: float(prob) for name, prob in zip(self.EMOTIONS, probs)}
        dominant = self.EMOTIONS[int(np.argmax(probs))]

        return {
            "dominant_emotion": dominant,
            "emotion": emotion_dict,
        }
