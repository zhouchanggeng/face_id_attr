from abc import ABC, abstractmethod
from typing import List
import numpy as np


class FaceAnalyzer(ABC):
    """人脸属性分析基类。"""

    @abstractmethod
    def analyze(self, image: np.ndarray, faces: List[dict]) -> List[dict]:
        """
        对检测到的人脸进行属性分析。
        Args:
            image: 原始 BGR 图像
            faces: 检测结果列表 (含 bbox 等)
        Returns:
            每张脸的属性分析结果列表，每个 dict 包含:
                - "age": int
                - "gender": str
                - "dominant_emotion": str
                - "emotion": dict  (各情绪概率)
                - "dominant_race": str
                - "race": dict  (各种族概率)
        """
        pass
