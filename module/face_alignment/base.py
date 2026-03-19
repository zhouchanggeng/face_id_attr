from abc import ABC, abstractmethod
import numpy as np


class FaceAligner(ABC):
    """人脸校正基类。所有校正算法需继承此类。"""

    @abstractmethod
    def align(self, image: np.ndarray, face: dict) -> np.ndarray:
        """
        对检测到的人脸进行校正/对齐。
        Args:
            image: 原始图像
            face: 检测结果 dict (含 bbox, landmarks 等)
        Returns:
            校正后的人脸图像
        """
        pass
