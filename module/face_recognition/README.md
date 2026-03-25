# 人脸识别模块 (face_recognition)

## 算法实现

| 类名 | 算法 | 特征维度 | 依赖 |
|------|------|----------|------|
| `ArcFaceRecognizer` | ArcFace (InsightFace R50) | 512 | onnxruntime |
| `SFaceRecognizer` | SFace (OpenCV) | 128 | opencv-python |
| `HistogramRecognizer` | 灰度直方图 | 可配置 | opencv-python |

## 模型信息

### ArcFaceRecognizer
- 模型格式：`.onnx`
- 可选模型：
  - `models/arcface/glint360k_r50.onnx` — Glint360K 数据集训练，泛化性好（推荐）
  - `models/arcface/webface_r50.onnx` — WebFace 数据集训练
- 模型来源：[InsightFace](https://github.com/deepinsight/insightface)
- 输入：`[1, 3, 112, 112]` RGB 图像，归一化到 [-1, 1]（`pixel / 127.5 - 1.0`）
- 输出：512 维特征向量，L2 归一化
- 骨干网络：ResNet-50

### SFaceRecognizer
- 模型格式：`.onnx`
- 默认模型：`models/sface/face_recognition_sface_2021dec.onnx`
- 模型来源：[OpenCV Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- 输入：112x112 BGR 图像
- 输出：128 维特征向量
- 特点：基于 OpenCV `FaceRecognizerSF`，纯 CPU 高效运行
- 额外功能：`align_crop(image, yunet_face)` 可配合 YuNet 检测器进行内置对齐

### HistogramRecognizer
- 无需模型文件
- 基于灰度直方图提取特征，仅作为演示和基线对比
- 特征维度由 `bins` 参数控制（默认 64）

## 接口规范

所有识别器继承 `FaceRecognizer` 基类：

- `extract(face_image) -> np.ndarray`：提取特征向量
- `compare(feat1, feat2) -> float`：余弦相似度比对（基类已实现）

## 预处理差异

| 识别器 | 色彩空间 | 归一化方式 | 输入尺寸 |
|--------|----------|------------|----------|
| ArcFace | RGB | `pixel / 127.5 - 1.0` → [-1, 1] | 112x112 |
| SFace | BGR | OpenCV 内部处理 | 112x112 |
| Histogram | 灰度 | 直方图归一化 | 任意 |

## 注意事项

- ArcFace 和 SFace 的特征维度不同（512 vs 128），切换识别器后必须重新注册人脸（删除旧的 `face_db.npz`）
- ArcFace 输出已做 L2 归一化，余弦相似度等价于内积
- Glint360K 模型在跨种族、跨年龄场景下表现优于 WebFace
- SFace 的 `align_crop` 仅在使用 YuNet 检测器时可用（需要 `_yunet_face` 数据）
