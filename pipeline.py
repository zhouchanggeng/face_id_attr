from typing import List, Optional
import numpy as np
import cv2

from module.face_detection.base import FaceDetector
from module.face_alignment.base import FaceAligner
from module.face_recognition.base import FaceRecognizer
from module.face_database.base import FaceDatabase
from module.face_analysis.base import FaceAnalyzer
from module.face_quality.base import FaceQualityAssessor


class FaceRecogPipeline:
    """
    人脸识别流水线：检测 -> (可选)校正 -> (可选)质量评估 -> 识别/比对/属性分析。
    支持 1:1 比对、1:N 搜索、人脸注册、属性分析、质量评估。
    """

    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        aligner: Optional[FaceAligner] = None,
        database: Optional[FaceDatabase] = None,
        analyzer: Optional[FaceAnalyzer] = None,
        quality_assessor: Optional[FaceQualityAssessor] = None,
        use_align_crop: bool = True,
        max_image_size: int = 1920,
    ):
        self.detector = detector
        self.aligner = aligner
        self.recognizer = recognizer
        self.database = database
        self.analyzer = analyzer
        self.quality_assessor = quality_assessor
        self.use_align_crop = use_align_crop
        self.max_image_size = max_image_size

    @staticmethod
    def _limit_size(image: np.ndarray, max_size: int):
        """将超大图缩放到 max_size 以内，返回 (缩放后图像, 缩放比例)。"""
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image, 1.0
        scale = max_size / max(h, w)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return resized, scale

    def _crop(self, image: np.ndarray, bbox: tuple, size: tuple = (112, 112)) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = image[max(0, y1):y2, max(0, x1):x2]
        return cv2.resize(crop, size)

    def _get_face_image(self, image: np.ndarray, face: dict) -> np.ndarray:
        # 使用 SFace alignCrop（当启用且检测器提供 YuNet 原始数据时）
        yunet_face = face.get("_yunet_face")
        if self.use_align_crop and yunet_face is not None and hasattr(self.recognizer, "align_crop"):
            return self.recognizer.align_crop(image, yunet_face)
        if self.aligner:
            return self.aligner.align(image, face)
        return self._crop(image, face["bbox"])

    def detect(self, image: np.ndarray) -> List[dict]:
        resized, scale = self._limit_size(image, self.max_image_size)
        faces = self.detector.detect(resized)
        if scale < 1.0:
            inv = 1.0 / scale
            for f in faces:
                x1, y1, x2, y2 = f["bbox"]
                f["bbox"] = (int(x1 * inv), int(y1 * inv),
                             int(x2 * inv), int(y2 * inv))
                if f.get("landmarks") is not None:
                    f["landmarks"] = (f["landmarks"] * inv).astype(np.float32)
        return faces

    def extract(self, image: np.ndarray) -> List[dict]:
        """检测 + (可选校正) + (可选质量评估) + 特征提取。"""
        resized, scale = self._limit_size(image, self.max_image_size)
        faces = self.detector.detect(resized)
        results = []
        for face in faces:
            face_img = self._get_face_image(resized, face)
            # 质量评估（在对齐后、特征提取前）
            quality_score = None
            if self.quality_assessor is not None:
                quality_score = self.quality_assessor.assess(face_img)
            feat = self.recognizer.extract(face_img)
            if scale < 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = face["bbox"]
                face["bbox"] = (int(x1 * inv), int(y1 * inv),
                                int(x2 * inv), int(y2 * inv))
                if face.get("landmarks") is not None:
                    face["landmarks"] = (face["landmarks"] * inv).astype(np.float32)
            results.append({**face, "feature": feat, "quality": quality_score})
        return results

    def align_faces(self, image: np.ndarray) -> List[dict]:
        """检测人脸并返回每张脸的 5 关键点和 align 后的图像。

        Returns:
            [{bbox, confidence, five_points, aligned_face}, ...]
            five_points: shape (5, 2)，原图坐标系下的 5 个关键点
            aligned_face: 对齐后的 112x112 BGR 图像
        """
        from module.face_alignment.pfld_aligner import PFLDAligner

        resized, scale = self._limit_size(image, self.max_image_size)
        faces = self.detector.detect(resized)
        results = []
        for face in faces:
            face_img = self._get_face_image(resized, face)

            # 提取 5 关键点（缩放图坐标系）
            five_pts = None
            if isinstance(self.aligner, PFLDAligner):
                pts_98 = self.aligner.predict_98pts(resized, face)
                five_pts = pts_98[PFLDAligner.FIVE_POINT_IDX].copy()
            elif face.get("landmarks") is not None and len(face["landmarks"]) >= 5:
                five_pts = face["landmarks"][:5].copy()

            # 映射回原图坐标
            if scale < 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = face["bbox"]
                face["bbox"] = (int(x1 * inv), int(y1 * inv),
                                int(x2 * inv), int(y2 * inv))
                if five_pts is not None:
                    five_pts = (five_pts * inv).astype(np.float32)

            # 质量评估
            quality_score = None
            if self.quality_assessor is not None:
                quality_score = self.quality_assessor.assess(face_img)

            results.append({
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "five_points": five_pts,
                "aligned_face": face_img,
                "quality": quality_score,
            })
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
        """注册图片中所有人脸到数据库，返回实际新注册的人脸数（已存在的会自动跳过）。"""
        self._require_db()
        faces = self.extract(image)
        before = len(self.database.features)
        for f in faces:
            self.database.register(identity, f["feature"])
        return len(self.database.features) - before

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
        resized, scale = self._limit_size(image, self.max_image_size)
        faces = self.detector.detect(resized)
        if not faces:
            return []
        attrs = self.analyzer.analyze(resized, faces)
        if scale < 1.0:
            inv = 1.0 / scale
            for f in faces:
                x1, y1, x2, y2 = f["bbox"]
                f["bbox"] = (int(x1 * inv), int(y1 * inv),
                             int(x2 * inv), int(y2 * inv))
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
