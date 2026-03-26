import cv2
import numpy as np
import onnxruntime
from typing import List, Optional
from .base import FaceAnalyzer


# 预定义类别映射
PRESETS = {
    "rafdb_7": ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"],
    "ferplus_8": ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"],
    "affectnet_7": ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"],
    "affectnet_8": ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"],
    "smile_2": ["No_Smile", "Smile"],
}


class ExpressionAnalyzer(FaceAnalyzer):
    """通用人脸表情/属性分类器（ONNX 分类模型）。

    通过 class_names 参数配置类别，支持任意分类任务:
        - 7 类表情 (RAF-DB / AffectNet)
        - 8 类表情 (FER+ / AffectNet-8)
        - 2 类微笑检测
        - 自定义类别列表

    Args:
        model_path: ONNX 分类模型路径
        input_size: 模型输入尺寸 (默认 224)
        class_names: 类别名称列表，或预设名称 (rafdb_7/ferplus_8/affectnet_7/smile_2 等)
        threshold: 置信度阈值，低于此值输出 "Unknown" (默认 0，不过滤)
    """

    def __init__(self, model_path: str, input_size: int = 224,
                 class_names: Optional[list] = None, threshold: float = 0.0,
                 smile_mode: bool = False, smile_classes: Optional[list] = None):
        self.input_size = input_size
        self.threshold = threshold
        self.smile_mode = smile_mode
        # 哪些类别算"微笑"（默认 Happiness）
        self.smile_classes = smile_classes or ["Happiness"]
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=onnxruntime.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

        # 解析类别名称
        if class_names is None:
            # 从模型输出维度自动推断
            out_shape = self.session.get_outputs()[0].shape
            num_classes = out_shape[-1] if out_shape[-1] is not None else 7
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        elif isinstance(class_names, str) and class_names in PRESETS:
            self.class_names = PRESETS[class_names]
        elif isinstance(class_names, list):
            self.class_names = class_names
        else:
            raise ValueError(f"class_names 不支持: {class_names}，可选预设: {list(PRESETS.keys())}")

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """预处理: resize -> RGB -> normalize [0,1] -> CHW -> NCHW。"""
        img = cv2.resize(face_image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def analyze(self, image: np.ndarray, faces: List[dict]) -> List[dict]:
        """对每张检测到的人脸进行分类分析。"""
        results = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            crop = image[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                results.append({})
                continue
            results.append(self.classify(crop))
        return results

    def classify(self, face_image: np.ndarray) -> dict:
        """对单张人脸图像进行分类。

        smile_mode=True 时，将结果映射为 Smile / No_Smile。

        Returns:
            {
                "dominant_emotion": str,
                "emotion": {name: prob, ...},
                "confidence": float,
            }
        """
        blob = self._preprocess(face_image)
        outputs = self.session.run(None, {self.input_name: blob})
        logits = outputs[0][0]

        # softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        dominant = self.class_names[top_idx] if top_conf >= self.threshold else "Unknown"

        emotion_dict = {}
        for i, name in enumerate(self.class_names):
            if i < len(probs):
                emotion_dict[name] = float(probs[i])

        # 微笑检测模式：汇总 smile_classes 的概率
        if self.smile_mode:
            smile_prob = sum(emotion_dict.get(c, 0) for c in self.smile_classes)
            is_smile = dominant in self.smile_classes
            return {
                "dominant_emotion": "Smile" if is_smile else "No_Smile",
                "emotion": {"Smile": smile_prob, "No_Smile": 1 - smile_prob},
                "confidence": smile_prob if is_smile else 1 - smile_prob,
                "detail": emotion_dict,  # 保留原始 7 类详情
            }

        return {
            "dominant_emotion": dominant,
            "emotion": emotion_dict,
            "confidence": top_conf,
        }
