from abc import ABC, abstractmethod
import numpy as np


class FaceQualityAssessor(ABC):
    """人脸图像质量评估基类。"""

    @abstractmethod
    def assess(self, face_image: np.ndarray) -> float:
        """
        评估对齐后的人脸图像质量。
        Args:
            face_image: 对齐后的人脸图像 (BGR, 通常 112x112)
        Returns:
            质量分 0.0~1.0，分数越高质量越好
        """
        pass
