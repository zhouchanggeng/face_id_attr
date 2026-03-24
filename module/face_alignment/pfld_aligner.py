import cv2
import numpy as np
import onnxruntime
from .base import FaceAligner


class PFLDAligner(FaceAligner):
    """基于 PFLD_GhostOne 的 98 点关键点人脸校正。

    流程: 裁剪人脸区域 -> PFLD 预测 98 关键点 -> 提取 5 关键点 -> 仿射变换对齐。

    98 点关键点索引 (WFLW 标准):
        左眼中心: 96, 右眼中心: 97, 鼻尖: 54, 左嘴角: 76, 右嘴角: 82
    """

    # 标准 5 点参考坐标 (112x112)
    REF_POINTS_112 = np.float32([
        [38.2946, 51.6963],   # 左眼
        [73.5318, 51.5014],   # 右眼
        [56.0252, 71.7366],   # 鼻尖
        [41.5493, 92.3655],   # 左嘴角
        [70.7299, 92.2041],   # 右嘴角
    ])

    # 从 98 点中提取 5 关键点的索引
    FIVE_POINT_IDX = [96, 97, 54, 76, 82]

    def __init__(self, model_path: str, input_size: int = 112,
                 output_size: tuple = (112, 112)):
        self.input_size = input_size
        self.output_size = output_size
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=onnxruntime.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def _crop_face_letterbox(self, image: np.ndarray, bbox: tuple):
        """将人脸区域裁剪为正方形 letterbox 并缩放到模型输入尺寸。

        返回: (裁剪后图像, 缩放比例, x偏移, y偏移)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        max_len = max(fw, fh)

        # 扩展为正方形
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sq_x1 = int(cx - max_len / 2)
        sq_y1 = int(cy - max_len / 2)
        sq_x2 = int(sq_x1 + max_len)
        sq_y2 = int(sq_y1 + max_len)

        # 计算 padding
        pad_left = max(0, -sq_x1)
        pad_top = max(0, -sq_y1)
        pad_right = max(0, sq_x2 - w)
        pad_bottom = max(0, sq_y2 - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            sq_x1 += pad_left
            sq_y1 += pad_top
            sq_x2 += pad_left
            sq_y2 += pad_top

        crop = image[sq_y1:sq_y2, sq_x1:sq_x2]
        crop = cv2.resize(crop, (self.input_size, self.input_size),
                          interpolation=cv2.INTER_CUBIC)

        scale = max_len / self.input_size
        return crop, scale, sq_x1 - pad_left, sq_y1 - pad_top

    def _predict_landmarks(self, face_crop: np.ndarray) -> np.ndarray:
        """用 PFLD 模型预测 98 个关键点 (归一化坐标)。"""
        img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})
        preds = outputs[0][0]  # shape: (196,) -> 98 个 (x, y)
        landmarks = preds.reshape(98, 2)
        return landmarks

    def predict_98pts(self, image: np.ndarray, face: dict) -> np.ndarray:
        """预测 98 个关键点在原图上的坐标。

        Args:
            image: 原始图像
            face: 检测结果 dict (含 bbox)
        Returns:
            shape (98, 2) 的关键点坐标 (原图坐标系)
        """
        crop, scale, x_off, y_off = self._crop_face_letterbox(image, face["bbox"])
        pts_norm = self._predict_landmarks(crop)
        # 映射回原图坐标
        pts = pts_norm * self.input_size * scale
        pts[:, 0] += x_off
        pts[:, 1] += y_off
        return pts.astype(np.float32)

    def align(self, image: np.ndarray, face: dict) -> np.ndarray:
        """预测 98 点关键点，提取 5 关键点，仿射变换对齐。"""
        pts_98 = self.predict_98pts(image, face)
        five_pts = pts_98[self.FIVE_POINT_IDX]

        # 计算目标参考点（按 output_size 缩放）
        sx = self.output_size[0] / 112.0
        sy = self.output_size[1] / 112.0
        ref = self.REF_POINTS_112.copy()
        ref[:, 0] *= sx
        ref[:, 1] *= sy

        M = cv2.estimateAffinePartial2D(five_pts, ref)[0]
        if M is None:
            # fallback: 直接裁剪
            x1, y1, x2, y2 = face["bbox"]
            crop = image[max(0, y1):y2, max(0, x1):x2]
            return cv2.resize(crop, self.output_size)
        return cv2.warpAffine(image, M, self.output_size)
