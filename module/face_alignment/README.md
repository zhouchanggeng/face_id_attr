# 人脸校正模块 (face_alignment)

## 算法实现

| 类名 | 算法 | 关键点数 | 依赖 |
|------|------|----------|------|
| `PFLDAligner` | PFLD_GhostOne 98 点关键点 | 98 → 5 | onnxruntime |
| `SimpleAligner` | 5 点仿射变换 | 5 | opencv-python |

## 模型信息

### PFLDAligner
- 模型格式：`.onnx`
- 默认模型：`models/PFLD_GhostOne_112_1_opt_sim.onnx`
- 模型来源：[PFLD_GhostOne](https://github.com/AnthonyF333/PFLD_GhostOne)
- 输入：112x112 RGB 图像，归一化到 [-1, 1]
- 输出：98 个关键点的归一化坐标 (196 维向量)
- 关键点标准：WFLW 98 点

### 5 关键点提取

从 98 点中提取 5 个关键点用于仿射对齐：

| 关键点 | WFLW 索引 | 用途 |
|--------|-----------|------|
| 左眼中心 | 96 | 对齐参考点 |
| 右眼中心 | 97 | 对齐参考点 |
| 鼻尖 | 54 | 对齐参考点 |
| 左嘴角 | 76 | 对齐参考点 |
| 右嘴角 | 82 | 对齐参考点 |

### 标准参考坐标 (112x112)

```python
REF_POINTS_112 = [
    [38.2946, 51.6963],   # 左眼
    [73.5318, 51.5014],   # 右眼
    [56.0252, 71.7366],   # 鼻尖
    [41.5493, 92.3655],   # 左嘴角
    [70.7299, 92.2041],   # 右嘴角
]
```

### SimpleAligner
- 无需额外模型
- 需要检测器提供至少 5 个 landmarks
- 无 landmarks 时 fallback 为直接裁剪 + resize

## 接口规范

所有校正器继承 `FaceAligner` 基类，实现 `align(image, face) -> np.ndarray`。

- 输入：原始图像 + 检测结果 dict（含 bbox, landmarks）
- 输出：对齐后的人脸图像（默认 112x112 BGR）

`PFLDAligner` 额外提供 `predict_98pts(image, face) -> np.ndarray` 方法，返回原图坐标系下的 98 个关键点。

## 对齐流程

```
原始图像 + bbox
      │
      ▼
裁剪为正方形 letterbox（等比缩放 + padding）
      │
      ▼
缩放到 112x112，送入 PFLD 模型
      │
      ▼
预测 98 个关键点（归一化坐标）
      │
      ▼
映射回原图坐标，提取 5 关键点
      │
      ▼
estimateAffinePartial2D → warpAffine → 112x112 对齐人脸
```

## 注意事项

- PFLD 模型的输入预处理是 `(pixel / 255.0 - 0.5) / 0.5`，即归一化到 [-1, 1]
- 人脸区域裁剪时会扩展为正方形并添加黑色 padding，避免拉伸变形
- 仿射变换失败时（`estimateAffinePartial2D` 返回 None）会 fallback 为直接裁剪
- 输出尺寸可通过 `output_size` 参数配置，但下游模型（SFace/ArcFace）通常要求 112x112
