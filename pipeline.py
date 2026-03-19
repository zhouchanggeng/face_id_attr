from typing import List, Optional
import numpy as np
import cv2

from module.face_detection.base import FaceDetector
from module.face_alignment.base import FaceAligner
from module.face_recognition.base import FaceRecognizer
from module.face_database.base import FaceDatabase
from module.face_analysis.base import FaceAnalyzer


class FaceRecogPipeline:
    """
    人脸识别流水线：检测 -> (可选)校正 -> 识别/比对/属性分析。
    支持 1:1 比对、1:N 搜索、人脸注册、属性分析。
    """

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        aligner: Optional[FaceAligner] = None,
        database: Optional[FaceDatabase] = None,
        analyzer: Optional[FaceAnalyzer] = None,
    ):
        self.detector = detector
        self.aligner = aligner
        self.recognizer = recognizer
        self.database = database
        self.analyzer = analyzer

    def _crop(self, image: np.ndarray, bbox: tuple, size: tuple = (112, 112)) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = image[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(crop, size)

    def _get_face_image(self, image: np.ndarray, face: dict) -> np.ndarray:
        if self.aligner:
            return self.aligner.align(image, face)
        return self._crop(image, face["bbox"])

    def detect(self, image: np.ndarray) -> List[dict]:
        return self.detector.detect(image)

    def extract(self, image: np.ndarray) -> List[dict]:
        """检测 + (可选校正) + 特征提取。"""
        faces = self.detector.detect(image)
        results = []
        for face in faces:
            face_img = self._get_face_image(image, face)
            feat = self.recognizer.extract(face_img)
            results.append({**face, "feature": feat})
        return results

    def compare_images(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """1:1 比对两张图片中的第一张人脸。"""
        feats1 = self.extract(image1)
        feats2 = self.extract(image2)
        if not feats1 or not feats2:
            raise ValueError("至少有一张图片未检测到人脸")
        return self.recognizer.compare(feats1[0]["feature"], feats2[0]["feature"])

    # ---- 人脸注册 & 1:N 搜索 ----

    def _require_db(self):
        if self.database is None:
            raise RuntimeError("未配置人脸数据库 (database)")

    def register(self, identity: str, image: np.ndarray) -> int:
        """注册图片中所有人脸到数据库，返回注册的人脸数。"""
        self._require_db()
        faces = self.extract(image)
        for f in faces:
            self.database.register(identity, f["feature"])
        return len(faces)

    def identify(self, image: np.ndarray, threshold: float = 0.5, top_k: int = 1) -> List[dict]:
        """
        1:N 身份识别。对图片中每张人脸在数据库中搜索。
        Returns:
            每张脸的结果: {bbox, confidence, identity, similarity, matched}
            matched=False 表示不在库中（低于阈值）。
        """
        self._require_db()
        faces = self.extract(image)
        results = []
        for face in faces:
            hits = self.database.search(face["feature"], top_k=top_k)
            if hits and hits[0][1] >= threshold:
                best_id, best_sim = hits[0]
                results.append({
                    "bbox": face["bbox"],
                    "confidence": face["confidence"],
                    "identity": best_id,
                    "similarity": best_sim,
                    "matched": True,
                    "top_k": hits,
                })
            else:
                results.append({
                    "bbox": face["bbox"],
                    "confidence": face["confidence"],
                    "identity": None,
                    "similarity": hits[0][1] if hits else 0.0,
                    "matched": False,
                    "top_k": hits,
                })
        return results

    # ---- 人脸属性分析 ----

    def analyze_faces(self, image: np.ndarray) -> List[dict]:
        """检测人脸并分析属性（年龄、性别、表情、种族）。"""
        if self.analyzer is None:
            raise RuntimeError("未配置人脸分析器 (analyzer)")
        faces = self.detector.detect(image)
        if not faces:
            return []
        attrs = self.analyzer.analyze(image, faces)
        return [{**face, "attributes": attr}
                for face, attr in zip(faces, attrs)]

    @staticmethod
    def draw_results(image: np.ndarray, results: List[dict], output_path: str) -> str:
        """在图片上绘制检测/识别/属性分析结果并保存。"""
        vis = image.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            matched = r.get("matched")
            if matched is True:
                color = (0, 255, 0)
            elif matched is False:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 构建标签
            label_parts = []
            identity = r.get("identity")
            if identity:
                label_parts.append(identity)
            similarity = r.get("similarity")
            if similarity is not None:
                label_parts.append(f"{similarity:.2f}")
            if not label_parts:
                label_parts.append(f"{r.get('confidence', 0):.2f}")

            # 属性信息
            attr = r.get("attributes", {})
            attr_lines = []
            if attr.get("age") is not None:
                attr_lines.append(f"Age:{attr['age']}")
            if attr.get("gender"):
                attr_lines.append(attr["gender"])
            if attr.get("dominant_emotion"):
                attr_lines.append(attr["dominant_emotion"])
            if attr.get("dominant_race"):
                attr_lines.append(attr["dominant_race"])

            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 在 bbox 下方逐行绘制属性
            for i, line in enumerate(attr_lines):
                ty = y2 + 18 + i * 20
                cv2.putText(vis, line, (x1, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(output_path, vis)
        return output_path
