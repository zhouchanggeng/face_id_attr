# 人脸检测模块 (face_detection)

## 算法实现

| 类名 | 算法 | 特征维度 | 依赖 |
|------|------|----------|------|
| `YOLOFaceDetector` | YOLO 系列 (v8/v11/v12/v26) | - | ultralytics |
| `YuNetDetector` | YuNet (OpenCV FaceDetectorYN) | 5 点 landmarks | opencv-python |
| `OpenCVDetector` | OpenCV DNN SSD / Haar 级联 | - | opencv-python |

## 模型信息

### YOLOFaceDetector
- 模型格式：`.pt`（PyTorch）或 `.onnx`（ONNX Runtime）
- 默认模型：`models/yolo26m_wider_face/weights/best.pt`
- ONNX 模型：`models/yolo26m_wider_face/weights/yolo26m_facedetect_widerface.onnx`
- 训练数据：WiderFace
- 输入尺寸：640x640（自动 letterbox 缩放）
- 输出：bbox (x1, y1, x2, y2) + confidence
- ONNX 模型加载时自动设置 `task="detect"`

### YuNetDetector
- 模型格式：`.onnx`
- 默认模型：`models/yunet/face_detection_yunet_2023mar.onnx`
- 输出：bbox + confidence + 5 点 landmarks（右眼、左眼、鼻尖、右嘴角、左嘴角）
- 特点：轻量级，适合配合 SFace 的 `alignCrop` 使用
- 输出中包含 `_yunet_face` 字段，供 SFace 内置对齐使用

### OpenCVDetector
- 支持两种模式：
  1. DNN 模式：需要 Caffe SSD 模型文件（prototxt + caffemodel）
  2. Haar 级联模式：无需额外模型，使用 OpenCV 内置级联分类器
- 精度较低，仅作为 fallback 或演示用途

## 接口规范

所有检测器继承 `FaceDetector` 基类，实现 `detect(image) -> List[dict]`。

返回格式：
```python
{
    "bbox": (x1, y1, x2, y2),      # int, 像素坐标
    "confidence": float,             # 0~1
    "landmarks": np.ndarray | None,  # shape (N, 2), float32
}
```

## 注意事项

- YOLO 检测器在 GPU 不兼容时（如 sm_120 架构 + 旧版 PyTorch），需在 config 中设置 `device: "cpu"` 或升级 PyTorch
- ONNX 模型不依赖 PyTorch，适合部署场景
- 超大图片建议通过 pipeline 的 `max_image_size` 参数限制，避免内存溢出
- YuNet 的 landmarks 顺序与 PFLD 不同，不要混用
