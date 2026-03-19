import cv2
import numpy as np
from .base import FaceRecognizer


class HistogramRecognizer(FaceRecognizer):
    """基于直方图的简单特征提取（演示用，无需额外模型）。"""

    def __init__(self, bins: int = 64):
        self.bins = bins

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [self.bins], [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
