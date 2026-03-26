"""基于 MediaPipe FaceLandmarker 的 478 点人脸关键点校正。

MediaPipe Face Mesh 提供 478 个 3D 关键点，覆盖面部轮廓、眉毛、眼睛、
鼻子、嘴唇等区域，精度高于 PFLD 98 点。

从 478 点中提取 5 个关键点用于仿射对齐:
    左眼中心: 468 (左眼虹膜中心)
    右眼中心: 473 (右眼虹膜中心)
    鼻尖: 1
    左嘴角: 61
    右嘴角: 291
"""
import cv2
import numpy as np
from typing import Optional
from .base import FaceAligner


class MediaPipeAligner(FaceAligner):
    """基于 MediaPipe FaceLandmarker 的人脸校正（478 点关键点）。

    Args:
        model_path: MediaPipe face_landmarker.task 模型路径
        output_size: 输出对齐图像尺寸 (默认 112x112)
    """

    # 标准 5 点参考坐标 (112x112)
    REF_POINTS_112 = np.float32([
        [38.2946, 51.6963],   # 左眼
        [73.5318, 51.5014],   # 右眼
        [56.0252, 71.7366],   # 鼻尖
        [41.5493, 92.3655],   # 左嘴角
        [70.7299, 92.2041],   # 右嘴角
    ])

    # 从 478 点中提取 5 关键点的索引
    FIVE_POINT_IDX = [468, 473, 1, 61, 291]

    def __init__(self, model_path: str = "models/face_landmarker.task",
                 output_size: tuple = (112, 112)):
        self.output_size = output_size
        # 延迟导入 mediapipe
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions
        )
        self._mp = mp
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def predict_478pts(self, image: np.ndarray, face: dict) -> Optional[np.ndarray]:
        """预测 478 个关键点在原图上的坐标。

        在整张图上运行 MediaPipe，选择与 bbox 最匹配的人脸。
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        x1, y1, x2, y2 = face["bbox"]
        cx_target = (x1 + x2) / 2
        cy_target = (y1 + y2) / 2

        # 如果检测到多张脸，选择中心最接近 bbox 中心的
        best_pts = None
        best_dist = float("inf")
        for lm in result.face_landmarks:
            pts = np.array([(l.x * w, l.y * h) for l in lm], dtype=np.float32)
            cx = pts[:, 0].mean()
            cy = pts[:, 1].mean()
            dist = (cx - cx_target) ** 2 + (cy - cy_target) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pts = pts

        return best_pts

    def align(self, image: np.ndarray, face: dict) -> np.ndarray:
        """预测 478 点关键点，提取 5 关键点，仿射变换对齐。"""
        pts_478 = self.predict_478pts(image, face)
        if pts_478 is not None and len(pts_478) >= 474:
            five_pts = pts_478[self.FIVE_POINT_IDX]

            sx = self.output_size[0] / 112.0
            sy = self.output_size[1] / 112.0
            ref = self.REF_POINTS_112.copy()
            ref[:, 0] *= sx
            ref[:, 1] *= sy

            M = cv2.estimateAffinePartial2D(five_pts, ref)[0]
            if M is not None:
                return cv2.warpAffine(image, M, self.output_size)

        # fallback: 直接裁剪
        x1, y1, x2, y2 = face["bbox"]
        crop = image[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(crop, self.output_size)

    def close(self):
        """释放 MediaPipe 资源。"""
        if hasattr(self, '_landmarker'):
            self._landmarker.close()

    def __del__(self):
        self.close()
