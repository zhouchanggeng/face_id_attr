import cv2
import numpy as np
from .base import FaceAligner


class SimpleAligner(FaceAligner):
    """基于仿射变换的简单人脸校正（需要 landmarks）。无 landmarks 时直接裁剪。"""

    # 标准5点参考坐标 (112x112)
    REF_POINTS = np.float32([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ])

    def __init__(self, output_size: tuple = (112, 112)):
        self.output_size = output_size

    def align(self, image: np.ndarray, face: dict) -> np.ndarray:
        landmarks = face.get("landmarks")
        if landmarks is not None and len(landmarks) >= 5:
            src = np.float32(landmarks[:5])
            M = cv2.estimateAffinePartial2D(src, self.REF_POINTS)[0]
            return cv2.warpAffine(image, M, self.output_size)
        # 无 landmarks，直接裁剪 + resize
        x1, y1, x2, y2 = face["bbox"]
        crop = image[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(crop, self.output_size)
