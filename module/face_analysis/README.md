# 人脸属性分析模块 (face_analysis)

## 当前状态

本模块目前仅提供抽象基类 `FaceAnalyzer`，尚无具体实现。原有的 DeepFace 实现已移除。

## 接口规范

所有属性分析器继承 `FaceAnalyzer` 基类：

```python
def analyze(self, image: np.ndarray, faces: List[dict]) -> List[dict]
```

- 输入：原始 BGR 图像 + 检测结果列表
- 输出：每张脸的属性字典列表

### 输出字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `age` | int | 预测年龄 |
| `gender` | str | 性别 ("Man" / "Woman") |
| `dominant_emotion` | str | 主要表情 |
| `emotion` | dict | 各表情概率分布 |
| `dominant_race` | str | 主要种族 |
| `race` | dict | 各种族概率分布 |

## 扩展方向

可基于以下方案实现 `FaceAnalyzer`：

- 轻量级 ONNX 属性分类模型（年龄/性别/表情独立模型）
- InsightFace 的属性分析模块
- ModelScope 上的人脸属性模型

实现后在 `config.yaml` 中配置 `analyzer` 即可启用。

## 注意事项

- 当前 `config.yaml` 中 `analyzer: null`，`analyze` 命令会报错提示未配置
- 属性分析的输入是原始图像 + 检测框，不是对齐后的人脸（与识别模块不同）
