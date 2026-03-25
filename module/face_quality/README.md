# 人脸质量评估模块 (face_quality)

## 算法实现

| 类名 | 算法 | 输出 | 依赖 |
|------|------|------|------|
| `FQAAssessor` | 达摩院 FQA (Ordinal Regression) | 0~1 质量分 | onnxruntime |

## 模型信息

### FQAAssessor
- 模型格式：`.onnx`
- 默认模型：`models/fqa_model.onnx`
- 模型来源：[ModelScope - FQA](https://www.modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa)
- 输入：`[1, 3, 112, 112]` RGB 图像，归一化到 [0, 1]
- 输出：10 个 rank 概率，通过 softmax + 加权求和转换为 0~1 分数

### 算法原理

FQA 模型基于有序回归（Ordinal Regression），将人脸质量划分为 10 个等级：
1. 模型输出 10 个 logits，对应 rank 0~9
2. 经过 softmax 得到各 rank 的概率分布
3. 加权求和：`score = Σ(prob_i × i/9)`，映射到 0~1

### 质量分含义

| 分数范围 | 质量等级 | 典型场景 |
|----------|----------|----------|
| 0.0 ~ 0.2 | 极差 | 严重遮挡、模糊、非人脸 |
| 0.2 ~ 0.4 | 较差 | 大角度侧脸、部分遮挡 |
| 0.4 ~ 0.6 | 一般 | 光照不均、轻微模糊 |
| 0.6 ~ 0.8 | 良好 | 正常拍摄条件 |
| 0.8 ~ 1.0 | 优秀 | 正脸、清晰、光照均匀 |

## 接口规范

所有质量评估器继承 `FaceQualityAssessor` 基类：

- `assess(face_image) -> float`：输入对齐后的人脸图像，返回 0~1 质量分

## Pipeline 集成

质量评估在 pipeline 中的位置：**人脸校正之后、特征提取之前**。

```
检测 → 校正 → 质量评估 → 特征提取 → 检索
```

- `extract()` 返回结果包含 `quality` 字段
- `align_faces()` 返回结果包含 `quality` 字段
- `identify` 命令输出会显示质量分
- `quality` 命令专门用于批量质量评估，输出 CSV 报告

## 注意事项

- 输入必须是对齐后的 112x112 人脸图像，未对齐的图像评分不准确
- 预处理是 `pixel / 255.0`（归一化到 [0, 1]），与 ArcFace 的 [-1, 1] 不同
- 质量分可用于过滤低质量人脸，建议注册时设置质量阈值（如 > 0.3）
- 在 config.yaml 中设置 `quality_assessor: null` 可禁用质量评估
