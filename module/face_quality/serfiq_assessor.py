import cv2
import numpy as np
import onnxruntime
from .base import FaceQualityAssessor


class SERFIQAssessor(FaceQualityAssessor):
    """基于 SER-FIQ 思想的人脸质量评估器（复用 ArcFace 识别模型）。

    原理: 对同一张对齐人脸施加多种轻微扰动（亮度、对比度、平移、翻转），
    用识别模型提取多组特征，计算特征间的平均余弦相似度作为质量分。
    特征越稳定（相似度越高）→ 质量越好。

    优点:
        - 不需要额外的质量评估模型，直接复用识别模型
        - 质量分直接反映"该人脸对识别模型的可靠程度"
        - 与识别任务高度一致

    参数:
        model_path: ArcFace ONNX 模型路径（与识别器共用同一个模型）
        num_perturbations: 扰动次数，越多越准但越慢（默认 10）
        input_size: 模型输入尺寸（默认 112）
    """

    def __init__(self, model_path: str, num_perturbations: int = 10,
                 input_size: int = 112):
        self.input_size = input_size
        self.num_perturbations = num_perturbations
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=onnxruntime.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """ArcFace 标准预处理: BGR -> RGB, [-1, 1], CHW, NCHW。"""
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _extract(self, face_image: np.ndarray) -> np.ndarray:
        """提取 L2 归一化特征。"""
        if face_image.shape[:2] != (self.input_size, self.input_size):
            face_image = cv2.resize(face_image, (self.input_size, self.input_size))
        blob = self._preprocess(face_image)
        feat = self.session.run(None, {self.input_name: blob})[0].flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def _perturb(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """对图像施加随机轻微扰动。"""
        h, w = image.shape[:2]
        img = image.copy()

        # 1. 亮度扰动 (±15%)
        brightness = rng.uniform(0.85, 1.15)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        # 2. 对比度扰动 (±10%)
        contrast = rng.uniform(0.9, 1.1)
        mean = img.mean()
        img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # 3. 轻微平移裁剪 (±3 像素)
        dx, dy = rng.randint(-3, 4), rng.randint(-3, 4)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 4. 50% 概率水平翻转
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)

        return img

    def assess(self, face_image: np.ndarray) -> float:
        """评估人脸质量。返回 0~1 分数，越高越好。"""
        if face_image.shape[:2] != (self.input_size, self.input_size):
            face_image = cv2.resize(face_image, (self.input_size, self.input_size))

        # 提取原始特征
        feat_orig = self._extract(face_image)

        # 提取多组扰动特征
        rng = np.random.RandomState(42)
        feats = [feat_orig]
        for _ in range(self.num_perturbations):
            perturbed = self._perturb(face_image, rng)
            feats.append(self._extract(perturbed))

        feats = np.array(feats, dtype=np.float64)

        # 计算所有特征对之间的余弦相似度
        # 由于已 L2 归一化，余弦相似度 = 内积
        sim_matrix = feats @ feats.T
        n = len(feats)
        triu_idx = np.triu_indices(n, k=1)
        similarities = sim_matrix[triu_idx]

        # 平均相似度作为质量分（已在 0~1 范围内）
        score = float(np.mean(similarities))
        # clamp to [0, 1]
        return max(0.0, min(1.0, score))
