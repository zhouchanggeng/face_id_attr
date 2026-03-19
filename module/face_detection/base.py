from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class FaceDetector(ABC):
    """人脸检测基类。所有检测算法需继承此类。"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        检测人脸。
        Args:
            image: BGR格式图像 (H, W, 3)
        Returns:
            检测结果列表，每个元素为 dict:
                - "bbox": (x1, y1, x2, y2)
                - "confidence": float
                - "landmarks": np.ndarray (N, 2) 或 None
        """
        pass
