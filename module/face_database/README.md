# 人脸向量数据库模块 (face_database)

## 算法实现

| 类名 | 存储方式 | 检索算法 | 依赖 |
|------|----------|----------|------|
| `NumpyFaceDatabase` | 内存 + npz 文件 | 暴力余弦相似度 | numpy |

## NumpyFaceDatabase

### 功能
- 注册：存储身份名 + 特征向量
- 检索：暴力搜索，余弦相似度排序，返回 top-k
- 去重：注册时检查同一身份下是否已有相似度 ≥ `dup_threshold` 的特征，有则跳过
- 持久化：`save/load` 使用 numpy `.npz` 格式

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dup_threshold` | 0.9 | 注册去重阈值，同身份下余弦相似度超过此值视为重复 |

### 存储格式

```
face_db.npz
├── identities: np.array of str   # 身份名列表
└── features: np.array of float32  # 特征矩阵 (N, D)
```

## 接口规范

所有数据库继承 `FaceDatabase` 基类：

- `register(identity, feature)` — 注册特征
- `search(feature, top_k) -> [(identity, similarity), ...]` — 检索
- `list_identities() -> List[str]` — 列出所有身份
- `remove(identity) -> int` — 删除身份，返回删除条数
- `save(path)` / `load(path)` — 持久化

## 注意事项

- 暴力搜索复杂度 O(N)，适合小规模场景（< 10 万条）
- 大规模场景建议实现 FAISS 或 Milvus 版本的 `FaceDatabase`
- 切换识别器（特征维度变化）后，旧数据库不兼容，需删除重建
- `dup_threshold` 设置过低会导致同一人不同角度的照片被误判为重复
