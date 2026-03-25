# Face ID Attr — 模块化人脸识别流水线

一个基于 Python 的模块化人脸识别系统，支持人脸检测、关键点校正、特征提取、1:1 比对、1:N 身份识别和人脸属性分析。所有模块均可通过 YAML 配置文件灵活切换，无需修改代码。

## 特性

- **模块化架构**：检测、校正、识别、数据库、属性分析五大模块，各自独立，通过抽象基类约束接口
- **配置驱动**：通过 `config.yaml` 动态加载模块，切换算法只需改配置
- **多种检测器**：YOLO（v8/v11/v12/v26）、YuNet（OpenCV 轻量级）、OpenCV DNN/Haar
- **PFLD 关键点校正**：集成 PFLD_GhostOne 98 点关键点模型，基于 5 关键点仿射变换对齐人脸
- **SFace 特征提取**：基于 OpenCV FaceRecognizerSF，提取 128 维特征向量，纯 CPU 可运行
- **向量数据库**：内置 NumPy 余弦相似度检索，支持注册、搜索、删除，可扩展为 FAISS/Milvus
- **完整 CLI**：注册、识别、比对、检测、关键点对齐、特征可视化，支持单张和批量操作
- **注册去重**：基于特征余弦相似度自动跳过已注册的重复人脸，阈值可配置
- **特征可视化**：支持 t-SNE / PCA / UMAP 降维可视化已注册人脸特征分布，输出类内相似度统计

## 项目结构

```
face_id_attr/
├── main.py                  # CLI 入口
├── factory.py               # 根据 config.yaml 动态构建 pipeline
├── pipeline.py              # FaceRecogPipeline 流水线核心
├── config.yaml              # 模块配置文件
├── requirements.txt         # Python 依赖
├── module/
│   ├── face_detection/      # 人脸检测模块
│   │   ├── base.py          #   抽象基类 FaceDetector
│   │   ├── yolo_detector.py #   YOLO 检测器 (Ultralytics)
│   │   ├── yunet_detector.py#   YuNet 检测器 (OpenCV)
│   │   └── opencv_detector.py#  OpenCV DNN / Haar 检测器
│   ├── face_alignment/      # 人脸校正模块
│   │   ├── base.py          #   抽象基类 FaceAligner
│   │   ├── pfld_aligner.py  #   PFLD_GhostOne 98点关键点校正
│   │   └── simple_aligner.py#   简单 5 点仿射变换校正
│   ├── face_recognition/    # 人脸识别模块
│   │   ├── base.py          #   抽象基类 FaceRecognizer
│   │   ├── sface_recognizer.py # SFace 128维特征提取 (OpenCV)
│   │   └── histogram_recognizer.py # 直方图特征 (演示用)
│   ├── face_database/       # 人脸向量数据库
│   │   ├── base.py          #   抽象基类 FaceDatabase
│   │   └── numpy_db.py      #   NumPy 余弦相似度检索
│   └── face_analysis/       # 人脸属性分析模块
│       └── base.py          #   抽象基类 FaceAnalyzer
├── models/                  # 模型文件 (不纳入 git)
│   ├── yolo26m_wider_face/  #   YOLO 人脸检测模型
│   ├── sface/               #   SFace 人脸识别模型
│   ├── yunet/               #   YuNet 人脸检测模型
│   ├── webface/             #   WebFace 识别模型
│   └── PFLD_GhostOne_112_1_opt_sim.onnx  # PFLD 关键点模型
├── known_faces/             # 已知人脸图片 (按身份分子文件夹)
├── images/                  # 待识别图片
└── results/                 # 识别结果输出
```

## 安装

```bash
pip install -r requirements.txt
```

依赖：
- `opencv-python >= 4.5`
- `numpy >= 1.20`
- `pyyaml >= 6.0`
- `ultralytics >= 8.3`（YOLO 检测器）
- `onnxruntime >= 1.14`（PFLD 关键点校正）
- `scikit-learn`（特征可视化 t-SNE/PCA，可选）
- `matplotlib`（特征可视化绘图，可选）

## 模型准备

模型文件较大，不包含在仓库中，需自行下载放置到 `models/` 目录：

| 模型 | 用途 | 格式 | 路径 |
|------|------|------|------|
| YOLO WiderFace | 人脸检测 | PyTorch | `models/yolo26m_wider_face/weights/best.pt` |
| YOLO WiderFace | 人脸检测 | ONNX | `models/yolo26m_wider_face/weights/yolo26m_facedetect_widerface.onnx` |
| YuNet | 人脸检测 (轻量) | ONNX | `models/yunet/face_detection_yunet_2023mar.onnx` |
| SFace | 人脸识别 (128维) | ONNX | `models/sface/face_recognition_sface_2021dec.onnx` |
| PFLD_GhostOne | 98点关键点 | ONNX | `models/PFLD_GhostOne_112_1_opt_sim.onnx` |
| FQA | 人脸质量评估 | ONNX | `models/fqa_model.onnx` |

> YOLO 检测器同时支持 `.pt` 和 `.onnx` 格式，ONNX 格式不依赖 PyTorch，适合部署场景。
| WebFace | 人脸识别 | `models/webface/webface_r50.onnx` |

## 使用方法

### 1. 注册已知人脸

将已知人脸图片按身份分类放入 `known_faces/` 目录：

```
known_faces/
├── alice/
│   ├── alice_001.jpg
│   └── alice_002.jpg
└── bob/
    ├── bob_001.jpg
    └── bob_002.jpg
```

批量注册：

```bash
python main.py register-dir
python main.py register-dir --dir /path/to/known_faces
```

注册单张：

```bash
python main.py register --name alice --img face.jpg
```

### 2. 人脸识别 (1:N)

```bash
# 识别单张图片
python main.py identify --img query.jpg --save

# 批量识别目录下所有图片
python main.py identify-dir --save --output-dir results

# 指定阈值和 top-k
python main.py identify --img query.jpg --threshold 0.6 --top-k 3
```

### 3. 人脸比对 (1:1)

```bash
python main.py compare --img1 a.jpg --img2 b.jpg
```

### 4. 人脸检测

```bash
python main.py detect --img face.jpg --save
python main.py detect --dir images/ --save --output-dir results
```

### 5. 人脸关键点对齐

```bash
python main.py align --img face.jpg --save
python main.py align --dir images/ --save --output-dir results
```

输出：
- `results/xxx_result.jpg` — 原图标注 bbox + 5 彩色关键点
- `results/aligned/xxx_result_face0.jpg` — 对齐后的 112x112 人脸

### 6. 人脸质量评估

基于达摩院 FQA 模型，对对齐后的人脸图像打分（0~1，越高越好）。批量模式自动生成 CSV 报告。

```bash
python main.py quality --img face.jpg --save
python main.py quality --dir images/ --save --output-dir results_quality
```

输出：
- `results/xxx_result.jpg` — 原图标注 bbox + 质量分（颜色渐变：红=差，绿=好）
- `results_quality/quality_report.csv` — 批量质量评估报告

### 7. 特征可视化

将已注册的人脸特征降维到 2D 可视化，分析身份聚类质量和类内相似度。

```bash
# t-SNE 可视化（默认）
python main.py visualize

# PCA 可视化
python main.py visualize --method pca --output feature_pca.png

# UMAP 可视化（需 pip install umap-learn）
python main.py visualize --method umap
```

输出散点图（不同身份不同颜色）和类内相似度统计（mean/min/max）。

### 8. 人脸属性分析

```bash
python main.py analyze --img face.jpg --save
python main.py analyze --dir images/ --save
```

> 注：属性分析需要实现 `FaceAnalyzer` 子类并在 `config.yaml` 中配置 `analyzer`。

### 9. 数据库管理

```bash
# 列出已注册身份
python main.py list

# 删除身份
python main.py remove --name alice
```

## 配置说明

通过 `config.yaml` 配置各模块，格式为：

```yaml
detector:
  class: "module.face_detection.yolo_detector.YOLOFaceDetector"
  params:
    model_path: "models/yolo26m_wider_face/weights/best.pt"
    conf_threshold: 0.5

aligner:
  class: "module.face_alignment.pfld_aligner.PFLDAligner"
  params:
    model_path: "models/PFLD_GhostOne_112_1_opt_sim.onnx"

recognizer:
  class: "module.face_recognition.sface_recognizer.SFaceRecognizer"
  params:
    model_path: "models/sface/face_recognition_sface_2021dec.onnx"

database:
  class: "module.face_database.numpy_db.NumpyFaceDatabase"
  params:
    dup_threshold: 0.9   # 注册去重阈值，同身份下相似度超过此值视为重复
  db_path: "face_db.npz"

analyzer: null

quality_assessor:
  class: "module.face_quality.fqa_assessor.FQAAssessor"
  params:
    model_path: "models/fqa_model.onnx"
```

切换算法只需修改 `class` 和 `params`，无需改代码。

## 可选检测器/识别器组合

| 检测器 | 识别器 | 校正器 | 说明 |
|--------|--------|--------|------|
| YOLO (.pt) + PFLD | SFace | PFLDAligner | 默认配置，精度高 |
| YOLO (.onnx) + PFLD | SFace | PFLDAligner | 不依赖 PyTorch，适合部署 |
| YuNet | SFace | SFace alignCrop | 轻量级，纯 OpenCV |
| OpenCV Haar | Histogram | SimpleAligner | 零依赖演示 |

## 流水线架构

```
输入图像
  │
  ▼
┌─────────────┐
│  人脸检测    │  YOLOFaceDetector / YuNetDetector / OpenCVDetector
└──────┬──────┘
       │ faces: [{bbox, confidence, landmarks}, ...]
       ▼
┌─────────────┐
│  人脸校正    │  PFLDAligner (98点) / SimpleAligner (5点) / SFace alignCrop
└──────┬──────┘
       │ aligned_face: 112x112 BGR
       ▼
┌─────────────┐
│  质量评估    │  FQAAssessor (0~1 质量分，可选)
└──────┬──────┘
       │ quality: float
       ▼
┌─────────────┐
│  特征提取    │  SFaceRecognizer (128维) / HistogramRecognizer
└──────┬──────┘
       │ feature: np.ndarray
       ▼
┌─────────────┐
│  向量检索    │  NumpyFaceDatabase (余弦相似度)
└──────┬──────┘
       │ identity, similarity
       ▼
     输出结果
```

## 扩展开发

实现对应的抽象基类即可添加新算法：

- `FaceDetector`：实现 `detect(image) -> List[dict]`
- `FaceAligner`：实现 `align(image, face) -> np.ndarray`
- `FaceRecognizer`：实现 `extract(face_image) -> np.ndarray`
- `FaceDatabase`：实现 `register()`, `search()`, `list_identities()`, `remove()`, `save()`, `load()`
- `FaceAnalyzer`：实现 `analyze(image, faces) -> List[dict]`
- `FaceQualityAssessor`：实现 `assess(face_image) -> float`

## TODO

- [ ] 活体检测（Anti-Spoofing）— 防照片/视频/3D 面具攻击
- [ ] 视线估计（Gaze Estimation）— 眼球注视方向预测
- [ ] 人脸追踪（Face Tracking）— 视频流多目标人脸跟踪（DeepSORT / ByteTrack）
- [x] 人脸质量评估（Face Quality Assessment）— 达摩院 FQA 模型，0~1 质量打分 + CSV 报告
- [ ] 口罩检测与遮挡人脸识别 — 戴口罩/墨镜场景下的检测与识别
- [ ] 人脸属性分析 — 年龄、性别、表情、种族（轻量级 ONNX 模型替代 DeepFace）
- [ ] 头部姿态估计（Head Pose Estimation）— 偏航角/俯仰角/翻滚角
- [ ] 人脸分割（Face Parsing）— 面部区域语义分割（皮肤/眉毛/眼睛/嘴唇等）
- [ ] 视频流实时推理 — RTSP / USB 摄像头实时人脸识别
- [ ] FAISS / Milvus 向量数据库 — 大规模人脸库高效检索
- [ ] REST API 服务 — FastAPI 封装，支持 HTTP 接口调用
- [ ] 模型量化与边缘部署 — INT8 量化、TensorRT 加速、ONNX Runtime 优化

## License

MIT
