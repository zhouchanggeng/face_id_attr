from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class FaceRecognizer(ABC):
    """人脸识别/比对基类。所有识别算法需继承此类。"""

    @abstractmethod
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取人脸特征向量。"""
        pass

    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """比对两个特征向量，返回相似度 (余弦相似度)。"""
        f1 = feat1.flatten().astype(np.float64)
        f2 = feat2.flatten().astype(np.float64)
        norm = np.linalg.norm(f1) * np.linalg.norm(f2)
        if norm == 0:
            return 0.0
        return float(np.dot(f1, f2) / norm)
