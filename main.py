"""
人脸识别流水线。

用法:
    # 从 known_faces 文件夹批量注册（子文件夹名=身份名）
    python main.py register-dir
    python main.py register-dir --dir /path/to/known_faces

    # 对 images 文件夹下所有图片进行 1:N 识别（--save 保存结果图片）
    python main.py identify-dir
    python main.py identify-dir --save --output-dir results

    # 注册单张图片
    python main.py register --name reba --img face.jpg

    # 1:N 识别单张图片
    python main.py identify --img query.jpg --save

    # 1:1 比对
    python main.py compare --img1 a.jpg --img2 b.jpg

    # 仅检测（--save 保存结果图片）
    python main.py detect --img face.jpg --save

    # 人脸属性分析（年龄、性别、表情、种族）
    python main.py analyze --img face.jpg --save
    python main.py analyze-dir --save --output-dir results

    # 列出已注册身份 / 删除身份
    python main.py list
    python main.py remove --name reba
"""
import argparse
import os
import cv2

from factory import build_pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(directory: str):
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            yield os.path.join(directory, name)


def _save_db(pipe, cfg):
    db_path = cfg.get("database", {}).get("db_path")
    if db_path:
        pipe.database.save(db_path)
        print(f"数据库已保存: {db_path}")

 
def _output_path(img_path, output_dir=None):
    """生成结果图片保存路径。"""
    out_dir = output_dir or "results"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    return os.path.join(out_dir, f"{name}_result{ext}")


def cmd_register(args, pipe, cfg):
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.img}")
    count = pipe.register(args.name, img)
    print(f"已注册 {count} 张人脸，身份: {args.name}")
    _save_db(pipe, cfg)


def cmd_register_dir(args, pipe, cfg):
    known_dir = args.dir or cfg.get("known_faces_dir", "known_faces")
    if not os.path.isdir(known_dir):
        raise FileNotFoundError(f"目录不存在: {known_dir}")

    total = 0
    for identity in sorted(os.listdir(known_dir)):
        id_dir = os.path.join(known_dir, identity)
        if not os.path.isdir(id_dir):
            continue
        count = 0
        for img_path in _iter_images(id_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [跳过] 无法读取: {img_path}")
                continue
            n = pipe.register(identity, img)
            count += n
        print(f"  身份 '{identity}': 注册 {count} 张人脸")
        total += count

    print(f"共注册 {total} 张人脸")
    _save_db(pipe, cfg)


def cmd_identify(args, pipe, cfg):
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.img}")
    threshold = args.threshold or cfg.get("identify_threshold", 0.5)
    results = pipe.identify(img, threshold=threshold, top_k=args.top_k)
    print(f"检测到 {len(results)} 张人脸:")
    for i, r in enumerate(results):
        if r["matched"]:
            print(f"  [{i}] 身份: {r['identity']}, 相似度: {r['similarity']:.4f}")
        else:
            print(f"  [{i}] 未识别 (最高相似度: {r['similarity']:.4f})")
        if args.top_k > 1 and r["top_k"]:
            for ident, sim in r["top_k"]:
                print(f"        -> {ident}: {sim:.4f}")
    if args.save and results:
        out = _output_path(args.img, args.output_dir)
        pipe.draw_results(img, results, out)
        print(f"结果已保存: {out}")


def cmd_identify_dir(args, pipe, cfg):
    images_dir = args.dir or cfg.get("images_dir", "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"目录不存在: {images_dir}")

    threshold = args.threshold or cfg.get("identify_threshold", 0.5)
    for img_path in _iter_images(images_dir):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue
        results = pipe.identify(img, threshold=threshold, top_k=args.top_k)
        filename = os.path.basename(img_path)
        if not results:
            print(f"{filename}: 未检测到人脸")
            continue
        for r in results:
            if r["matched"]:
                print(f"{filename}: {r['identity']} (相似度: {r['similarity']:.4f})")
            else:
                print(f"{filename}: 未识别 (最高相似度: {r['similarity']:.4f})")
        if args.save:
            out = _output_path(img_path, args.output_dir)
            pipe.draw_results(img, results, out)
            print(f"  -> 保存: {out}")


def cmd_compare(args, pipe, cfg):
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None:
        raise FileNotFoundError(f"无法读取图片: {args.img1}")
    if img2 is None:
        raise FileNotFoundError(f"无法读取图片: {args.img2}")
    similarity = pipe.compare_images(img1, img2)
    print(f"相似度: {similarity:.4f}")


def cmd_detect(args, pipe, cfg):
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.img}")
    faces = pipe.detect(img)
    print(f"检测到 {len(faces)} 张人脸:")
    for i, f in enumerate(faces):
        print(f"  [{i}] bbox={f['bbox']}, conf={f['confidence']:.3f}")
    if args.save and faces:
        out = _output_path(args.img, args.output_dir)
        pipe.draw_results(img, faces, out)
        print(f"结果已保存: {out}")


def cmd_detect_dir(args, pipe, cfg):
    images_dir = args.dir or cfg.get("images_dir", "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"目录不存在: {images_dir}")
    for img_path in _iter_images(images_dir):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue
        faces = pipe.detect(img)
        filename = os.path.basename(img_path)
        print(f"{filename}: 检测到 {len(faces)} 张人脸")
        if args.save:
            out = _output_path(img_path, args.output_dir)
            pipe.draw_results(img, faces, out)
            print(f"  -> 保存: {out}")


def cmd_analyze(args, pipe, cfg):
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {args.img}")
    results = pipe.analyze_faces(img)
    print(f"检测到 {len(results)} 张人脸:")
    for i, r in enumerate(results):
        attr = r.get("attributes", {})
        parts = []
        if attr.get("age") is not None:
            parts.append(f"年龄:{attr['age']}")
        if attr.get("gender"):
            parts.append(f"性别:{attr['gender']}")
        if attr.get("dominant_emotion"):
            parts.append(f"表情:{attr['dominant_emotion']}")
        if attr.get("dominant_race"):
            parts.append(f"种族:{attr['dominant_race']}")
        print(f"  [{i}] {', '.join(parts)}")
    if args.save and results:
        out = _output_path(args.img, args.output_dir)
        pipe.draw_results(img, results, out)
        print(f"结果已保存: {out}")


def cmd_analyze_dir(args, pipe, cfg):
    images_dir = args.dir or cfg.get("images_dir", "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"目录不存在: {images_dir}")
    for img_path in _iter_images(images_dir):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue
        results = pipe.analyze_faces(img)
        filename = os.path.basename(img_path)
        if not results:
            print(f"{filename}: 未检测到人脸")
            continue
        for r in results:
            attr = r.get("attributes", {})
            parts = []
            if attr.get("age") is not None:
                parts.append(f"年龄:{attr['age']}")
            if attr.get("gender"):
                parts.append(f"性别:{attr['gender']}")
            if attr.get("dominant_emotion"):
                parts.append(f"表情:{attr['dominant_emotion']}")
            if attr.get("dominant_race"):
                parts.append(f"种族:{attr['dominant_race']}")
            print(f"{filename}: {', '.join(parts)}")
        if args.save:
            out = _output_path(img_path, args.output_dir)
            pipe.draw_results(img, results, out)
            print(f"  -> 保存: {out}")


def cmd_list(args, pipe, cfg):
    pipe._require_db()
    identities = pipe.database.list_identities()
    print(f"已注册 {len(identities)} 个身份:")
    for name in identities:
        print(f"  - {name}")


def cmd_remove(args, pipe, cfg):
    pipe._require_db()
    count = pipe.database.remove(args.name)
    print(f"已删除身份 '{args.name}'，共 {count} 条特征")
    _save_db(pipe, cfg)


def cmd_visualize(args, pipe, cfg):
    """可视化已注册人脸特征的 2D 分布（t-SNE / PCA）。"""
    pipe._require_db()
    if not pipe.database.features:
        print("数据库为空，请先注册人脸")
        return

    import numpy as np
    features = np.array(pipe.database.features, dtype=np.float32)
    identities = pipe.database.identities
    unique_ids = sorted(set(identities))

    if len(features) < 2:
        print("至少需要 2 条特征才能可视化")
        return

    # 降维
    method = args.method.lower()
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(args.perplexity, len(features) - 1)
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca")
        coords = reducer.fit_transform(features)
        title = f"Face Feature t-SNE (perplexity={perp})"
    elif method == "umap":
        import umap
        n_neighbors = min(15, len(features) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        coords = reducer.fit_transform(features)
        title = "Face Feature UMAP"
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(features)
        ratio = reducer.explained_variance_ratio_
        title = f"Face Feature PCA (var: {ratio[0]:.1%} + {ratio[1]:.1%})"

    # 绘图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(unique_ids), 1))

    for idx, uid in enumerate(unique_ids):
        mask = [i for i, name in enumerate(identities) if name == uid]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(idx)], label=uid, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = args.output or "feature_visualization.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"可视化结果已保存: {out_path}")

    # 打印类间/类内相似度统计
    print(f"\n特征统计 ({len(features)} 条特征, {len(unique_ids)} 个身份):")
    for uid in unique_ids:
        mask = [i for i, name in enumerate(identities) if name == uid]
        if len(mask) < 2:
            continue
        vecs = features[mask].astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs_normed = vecs / norms
        sim_matrix = vecs_normed @ vecs_normed.T
        # 取上三角（排除对角线）
        triu_idx = np.triu_indices(len(mask), k=1)
        intra_sims = sim_matrix[triu_idx]
        print(f"  {uid}: 类内相似度 mean={intra_sims.mean():.4f}, min={intra_sims.min():.4f}, max={intra_sims.max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="人脸识别流水线")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # register
    p_reg = sub.add_parser("register", help="注册单张人脸")
    p_reg.add_argument("--name", required=True, help="身份名称")
    p_reg.add_argument("--img", required=True, help="图片路径")

    # register-dir
    p_rdir = sub.add_parser("register-dir", help="从文件夹批量注册 (子文件夹名=身份)")
    p_rdir.add_argument("--dir", default=None, help="已分类人脸文件夹路径 (默认: known_faces)")

    # identify
    p_id = sub.add_parser("identify", help="1:N 识别单张图片")
    p_id.add_argument("--img", required=True, help="图片路径")
    p_id.add_argument("--threshold", type=float, default=None, help="识别阈值")
    p_id.add_argument("--top-k", type=int, default=1, help="返回前 K 个结果")
    p_id.add_argument("--save", action="store_true", help="保存结果图片")
    p_id.add_argument("--output-dir", default=None, help="结果图片保存目录 (默认: results)")

    # identify-dir
    p_idir = sub.add_parser("identify-dir", help="对文件夹下所有图片进行 1:N 识别")
    p_idir.add_argument("--dir", default=None, help="待识别图片文件夹 (默认: images)")
    p_idir.add_argument("--threshold", type=float, default=None, help="识别阈值")
    p_idir.add_argument("--top-k", type=int, default=1, help="返回前 K 个结果")
    p_idir.add_argument("--save", action="store_true", help="保存结果图片")
    p_idir.add_argument("--output-dir", default=None, help="结果图片保存目录 (默认: results)")

    # compare
    p_cmp = sub.add_parser("compare", help="1:1 人脸比对")
    p_cmp.add_argument("--img1", required=True)
    p_cmp.add_argument("--img2", required=True)

    # detect
    p_det = sub.add_parser("detect", help="仅人脸检测")
    p_det.add_argument("--img", required=True)
    p_det.add_argument("--save", action="store_true", help="保存结果图片")
    p_det.add_argument("--output-dir", default=None, help="结果图片保存目录 (默认: results)")

    # detect-dir
    p_ddir = sub.add_parser("detect-dir", help="对文件夹下所有图片进行人脸检测")
    p_ddir.add_argument("--dir", default=None, help="图片文件夹 (默认: images)")
    p_ddir.add_argument("--save", action="store_true", help="保存结果图片")
    p_ddir.add_argument("--output-dir", default=None, help="结果图片保存目录 (默认: results)")

    # analyze
    p_ana = sub.add_parser("analyze", help="人脸属性分析 (年龄/性别/表情/种族)")
    p_ana.add_argument("--img", required=True, help="图片路径")
    p_ana.add_argument("--save", action="store_true", help="保存结果图片")
    p_ana.add_argument("--output-dir", default=None, help="结果图片保存目录 (默认: results)")

    # analyze-dir
    p_adir = sub.add_parser("analyze-dir", help="对文件夹下所有图片进行属性分析")
    p_adir.add_argument("--dir", default=None, help="图片文件夹 (默认: images)")
    p_adir.add_argument("--save", action="store_true", help="保存结果图片")
    p_adir.add_argument("--output-dir", default=None, help="结果图片保存目录 (默认: results)")

    # list / remove / visualize
    sub.add_parser("list", help="列出已注册身份")
    p_rm = sub.add_parser("remove", help="删除已注册身份")
    p_rm.add_argument("--name", required=True, help="身份名称")

    p_vis = sub.add_parser("visualize", help="可视化已注册人脸特征分布 (t-SNE/PCA/UMAP)")
    p_vis.add_argument("--method", default="tsne", choices=["tsne", "pca", "umap"], help="降维方法 (默认: tsne)")
    p_vis.add_argument("--perplexity", type=float, default=5, help="t-SNE perplexity (默认: 5)")
    p_vis.add_argument("--output", default=None, help="输出图片路径 (默认: feature_visualization.png)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    pipe, cfg = build_pipeline(args.config)

    commands = {
        "register": cmd_register,
        "register-dir": cmd_register_dir,
        "identify": cmd_identify,
        "identify-dir": cmd_identify_dir,
        "compare": cmd_compare,
        "detect": cmd_detect,
        "detect-dir": cmd_detect_dir,
        "analyze": cmd_analyze,
        "analyze-dir": cmd_analyze_dir,
        "list": cmd_list,
        "remove": cmd_remove,
        "visualize": cmd_visualize,
    }
    commands[args.command](args, pipe, cfg)


if __name__ == "__main__":
    main()
