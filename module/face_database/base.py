from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np


class FaceDatabase(ABC):
    """人脸向量数据库基类。可替换为 FAISS、Milvus 等实现。"""

    @abstractmethod
    def register(self, identity: str, feature: np.ndarray) -> None:
        """注册一个身份的特征向量。同一 identity 可注册多条。"""
        pass

    @abstractmethod
    def search(self, feature: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        1:N 搜索，返回最相似的 top_k 个结果。
        Returns:
            [(identity, similarity), ...] 按相似度降序
        """
        pass

    @abstractmethod
    def list_identities(self) -> List[str]:
        """列出所有已注册的身份。"""
        pass

    @abstractmethod
    def remove(self, identity: str) -> int:
        """删除指定身份，返回删除的特征条数。"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """持久化到磁盘。"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载。"""
        pass
