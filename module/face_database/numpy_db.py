import os
import numpy as np
from typing import List, Tuple
from .base import FaceDatabase


class NumpyFaceDatabase(FaceDatabase):
    """基于 NumPy 的简单向量数据库（适合小规模场景）。"""

    def __init__(self, dup_threshold: float = 0.9):
        """
        Args:
            dup_threshold: 去重阈值，同一身份下余弦相似度超过此值视为重复，跳过注册。
        """
        self.identities: List[str] = []
        self.features: List[np.ndarray] = []
        self.dup_threshold = dup_threshold

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def register(self, identity: str, feature: np.ndarray) -> None:
        """注册特征。若同一身份下已存在相似度 >= dup_threshold 的特征，则跳过。"""
        feat = feature.flatten().astype(np.float32)
        for i, (ident, existing) in enumerate(zip(self.identities, self.features)):
            if ident == identity and self._cosine_sim(feat, existing) >= self.dup_threshold:
                return  # 重复人脸，跳过
        self.identities.append(identity)
        self.features.append(feat)

    def search(self, feature: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        if not self.features:
            return []
        query = feature.flatten().astype(np.float64)
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        matrix = np.array(self.features, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > 0
        sims = np.zeros(len(matrix))
        sims[valid] = matrix[valid] @ query / (norms[valid] * q_norm)
        indices = np.argsort(sims)[::-1][:top_k]
        return [(self.identities[i], float(sims[i])) for i in indices]

    def list_identities(self) -> List[str]:
        return sorted(set(self.identities))

    def remove(self, identity: str) -> int:
        keep = [(ident, feat) for ident, feat in zip(self.identities, self.features)
                if ident != identity]
        removed = len(self.identities) - len(keep)
        if keep:
            self.identities, self.features = map(list, zip(*keep))
        else:
            self.identities, self.features = [], []
        return removed

    def save(self, path: str) -> None:
        feats = np.vstack(self.features) if self.features else np.array([])
        np.savez(path,
                 identities=np.array(self.identities, dtype=object),
                 features=feats)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        data = np.load(path, allow_pickle=True)
        self.identities = data["identities"].tolist()
        feats = data["features"]
        self.features = [f for f in feats] if len(feats) > 0 else []
