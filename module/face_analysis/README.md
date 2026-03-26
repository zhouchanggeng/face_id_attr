# 人脸属性分析模块 (face_analysis)

## 算法实现

| 类名 | 功能 | 模型 | 依赖 |
|------|------|------|------|
| `ExpressionAnalyzer` | 表情识别 / 微笑检测 | YOLO26s-cls (ONNX) | onnxruntime |

## ExpressionAnalyzer

通用人脸表情分类器，支持多种分类任务，通过 `class_names` 和 `smile_mode` 配置。

### 模型信息

- 默认模型：`models/expression/yolo26s_cls_rafdb.onnx`
- 骨干网络：YOLOv26s 分类头
- 训练数据：RAF-DB（15,339 张对齐人脸，7 类表情）
- 输入：`[1, 3, 224, 224]` RGB，归一化到 [0, 1]
- 输出：7 维 logits，经 softmax 得到概率分布

### 支持的分类模式

| 模式 | 配置 | 输出 |
|------|------|------|
| 7 类表情 | `class_names: "rafdb_7"` | Surprise / Fear / Disgust / Happiness / Sadness / Anger / Neutral |
| 8 类表情 | `class_names: "ferplus_8"` | 同上 + Contempt |
| 微笑检测 | `smile_mode: true` | Smile / No_Smile（基于 Happiness 概率） |
| 自定义 | `class_names: ["A", "B", "C"]` | 自定义类别列表 |

### 配置示例

```yaml
# 7 类表情识别
analyzer:
  class: "module.face_analysis.expression_analyzer.ExpressionAnalyzer"
  params:
    model_path: "models/expression/yolo26s_cls_rafdb.onnx"
    input_size: 224
    class_names: "rafdb_7"

# 微笑检测
analyzer:
  class: "module.face_analysis.expression_analyzer.ExpressionAnalyzer"
  params:
    model_path: "models/expression/yolo26s_cls_rafdb.onnx"
    input_size: 224
    class_names: "rafdb_7"
    smile_mode: true
    smile_classes: ["Happiness"]
```

### Pipeline 集成

表情分析在 pipeline 中的位置：**人脸校正之后**。

`analyze_faces()` 方法会先对齐人脸到 112x112，再送入表情模型（自动 resize 到 224x224），比直接裁剪原图精度更高。视频模式下每帧实时分析表情。

## 人脸表情识别（FER）技术背景

### 实现方法

人脸表情识别算法通常包含三个阶段：

1. 预处理：人脸检测 + 关键点对齐，裁剪为标准尺寸
2. 深度特征学习：CNN / ViT 提取表情特征
3. 分类输出：softmax 分类为离散表情类别

本模块采用端到端方案：对齐后的人脸图像直接送入 YOLO 分类模型，一步完成特征提取和分类。

### 公开数据集

| 数据集 | 类别数 | 样本量 | 特点 |
|--------|--------|--------|------|
| [RAF-DB](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) | 7 | ~15,000 | 已对齐，标注质量高，推荐使用 |
| [FER2013](https://hyper.ai/cn/datasets/17027) | 7 | ~35,000 | 48x48 灰度图，Disgust 类仅 600 张，类别不均衡 |
| [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions) | 5 | ~40,000 | Angry/Fear/Happy/Sad/Surprised，来源多样 |
| [AffectNet](http://mohammadmahoor.com/affectnet/) | 7/8 | ~440,000 | 最大规模，含连续 VA 标注 |
| [FER+](https://github.com/microsoft/FERPlus) | 8 | ~35,000 | FER2013 的多标注修正版，增加 Contempt 类 |
| [JAFFE](https://www.kasrl.org/jaffe.html) | 7 | 213 | 日本女性，样本少，仅适合小规模实验 |

当前使用 RAF-DB 数据集训练，因为该数据集的人脸已经过对齐校正，可直接用于分类模型训练。

### RAF-DB 类别映射

RAF-DB 数据集按文件夹 1-7 组织，Ultralytics 分类器按文件夹名排序：

| 文件夹 | 索引 | 表情 |
|--------|------|------|
| 1 | 0 | Surprise |
| 2 | 1 | Fear |
| 3 | 2 | Disgust |
| 4 | 3 | Happiness |
| 5 | 4 | Sadness |
| 6 | 5 | Anger |
| 7 | 6 | Neutral |

### 参考资源

- [FaceGg/DataSets](https://github.com/FaceGg/DataSets) — 人脸相关数据集汇总
- [UniDataPro/facial-expression-recognition-dataset](https://huggingface.co/datasets/UniDataPro/facial-expression-recognition-dataset) — HuggingFace 表情数据集
- [人脸表情识别综述 (知乎)](https://zhuanlan.zhihu.com/p/639923966)
- [FER 方法调研 (知乎)](https://zhuanlan.zhihu.com/p/1944959552505255874)

## 接口规范

所有属性分析器继承 `FaceAnalyzer` 基类：

```python
def analyze(self, image: np.ndarray, faces: List[dict]) -> List[dict]
```

`ExpressionAnalyzer` 额外提供 `classify(face_image) -> dict` 方法，接受对齐后的单张人脸图像。

### 输出字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `dominant_emotion` | str | 主要表情（或 Smile/No_Smile） |
| `emotion` | dict | 各类别概率分布 |
| `confidence` | float | 主要类别的置信度 |
| `detail` | dict | smile_mode 下保留原始 7 类概率（可选） |

## 扩展方向

- 年龄/性别分类：InsightFace `genderage.onnx` 模型
- 头部姿态估计：复用 PFLD 98 点关键点 + cv2.solvePnP
- 人脸分割：BiSeNet 等语义分割模型

## 注意事项

- 表情模型输入为 224x224，pipeline 会自动从 112x112 对齐图 resize
- `smile_mode` 不需要单独的微笑检测模型，复用 7 类表情模型即可
- 视频模式下表情每帧分析（因为表情变化快），身份识别按间隔执行
- 设置 `analyzer: null` 可禁用属性分析
