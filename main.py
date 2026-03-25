"""
人脸识别流水线。

用法:
    # 注册人脸（子文件夹名=身份名）
    python main.py register --dir known_faces
    python main.py register --name reba --img face.jpg

    # 1:N 识别
    python main.py identify --dir images/ --save --output-dir results
    python main.py identify --img query.jpg --save

    # 1:1 比对
    python main.py compare --img1 a.jpg --img2 b.jpg

    # 人脸检测
    python main.py detect --dir images/ --save --output-dir results
    python main.py detect --img face.jpg --save

    # 关键点对齐
    python main.py align --dir images/ --save --output-dir results
    python main.py align --img face.jpg --save

    # 属性分析
    python main.py analyze --img face.jpg --save

    # 特征可视化
    python main.py visualize --method tsne

    # 数据库管理
    python main.py list
    python main.py remove --name reba
"""
import argparse
import os
import cv2

from factory import build_pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(directory: str):
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(root, name)


def _save_db(pipe, cfg):
    db_path = cfg.get("database", {}).get("db_path")
    if db_path:
        pipe.database.save(db_path)
        print(f"数据库已保存: {db_path}")


def _output_path(img_path, output_dir=None, input_dir=None):
    """生成结果图片保存路径，保留相对子目录结构。"""
    out_dir = output_dir or "results"
    if input_dir:
        rel = os.path.relpath(img_path, input_dir)
        sub_dir = os.path.dirname(rel)
        if sub_dir:
            out_dir = os.path.join(out_dir, sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    return os.path.join(out_dir, f"{name}_result{ext}")


# ---- register ----

def cmd_register(args, pipe, cfg):
    if args.img:
        # 单张注册
        if not args.name:
            print("错误: 单张注册需要指定 --name")
            return
        img = cv2.imread(args.img)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {args.img}")
        count = pipe.register(args.name, img)
        print(f"已注册 {count} 张人脸，身份: {args.name}")
    else:
        # 目录批量注册
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


# ---- identify ----

def cmd_identify(args, pipe, cfg):
    threshold = args.threshold or cfg.get("identify_threshold", 0.5)

    if args.img:
        img = cv2.imread(args.img)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {args.img}")
        results = pipe.identify(img, threshold=threshold, top_k=args.top_k)
        print(f"检测到 {len(results)} 张人脸:")
        for i, r in enumerate(results):
            q = f", 质量:{r['quality']:.2f}" if r.get("quality") is not None else ""
            if r["matched"]:
                print(f"  [{i}] 身份: {r['identity']}, 相似度: {r['similarity']:.4f}{q}")
            else:
                print(f"  [{i}] 未识别 (最高相似度: {r['similarity']:.4f}){q}")
            if args.top_k > 1 and r["top_k"]:
                for ident, sim in r["top_k"]:
                    print(f"        -> {ident}: {sim:.4f}")
        if args.save and results:
            out = _output_path(args.img, args.output_dir)
            pipe.draw_results(img, results, out)
            print(f"结果已保存: {out}")
    else:
        images_dir = args.dir or cfg.get("images_dir", "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
        for img_path in _iter_images(images_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[跳过] 无法读取: {img_path}")
                continue
            results = pipe.identify(img, threshold=threshold, top_k=args.top_k)
            filename = os.path.relpath(img_path, images_dir)
            if not results:
                print(f"{filename}: 未检测到人脸")
                continue
            for r in results:
                q = f", 质量:{r['quality']:.2f}" if r.get("quality") is not None else ""
                if r["matched"]:
                    print(f"{filename}: {r['identity']} (相似度: {r['similarity']:.4f}{q})")
                else:
                    print(f"{filename}: 未识别 (最高相似度: {r['similarity']:.4f}{q})")
            if args.save:
                out = _output_path(img_path, args.output_dir, images_dir)
                pipe.draw_results(img, results, out)
                print(f"  -> 保存: {out}")


# ---- compare ----

def cmd_compare(args, pipe, cfg):
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    if img1 is None:
        raise FileNotFoundError(f"无法读取图片: {args.img1}")
    if img2 is None:
        raise FileNotFoundError(f"无法读取图片: {args.img2}")
    similarity = pipe.compare_images(img1, img2)
    print(f"相似度: {similarity:.4f}")


# ---- detect ----

def cmd_detect(args, pipe, cfg):
    if args.img:
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
    else:
        images_dir = args.dir or cfg.get("images_dir", "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
        for img_path in _iter_images(images_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[跳过] 无法读取: {img_path}")
                continue
            faces = pipe.detect(img)
            filename = os.path.relpath(img_path, images_dir)
            print(f"{filename}: 检测到 {len(faces)} 张人脸")
            if args.save:
                out = _output_path(img_path, args.output_dir, images_dir)
                pipe.draw_results(img, faces, out)
                print(f"  -> 保存: {out}")


# ---- align ----

def _draw_align_results(image, results, output_path, align_dir=None):
    """在原图上画 5 关键点 + bbox，同时保存每张 align 后的人脸。"""
    vis = image.copy()
    POINT_NAMES = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]
    POINT_COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 165, 255)]
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    for idx, r in enumerate(results):
        x1, y1, x2, y2 = r["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        five_pts = r.get("five_points")
        if five_pts is not None:
            for j, (px, py) in enumerate(five_pts):
                color = POINT_COLORS[j % len(POINT_COLORS)]
                cv2.circle(vis, (int(px), int(py)), 4, color, -1)
                cv2.putText(vis, POINT_NAMES[j], (int(px) + 5, int(py) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        aligned = r.get("aligned_face")
        if aligned is not None and align_dir:
            os.makedirs(align_dir, exist_ok=True)
            cv2.imwrite(os.path.join(align_dir, f"{base_name}_face{idx}.jpg"), aligned)

    cv2.imwrite(output_path, vis)
    return output_path


def cmd_align(args, pipe, cfg):
    if args.img:
        img = cv2.imread(args.img)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {args.img}")
        results = pipe.align_faces(img)
        print(f"检测到 {len(results)} 张人脸:")
        for i, r in enumerate(results):
            pts_info = "有" if r["five_points"] is not None else "无"
            q = f", 质量:{r['quality']:.2f}" if r.get("quality") is not None else ""
            print(f"  [{i}] bbox={r['bbox']}, conf={r['confidence']:.3f}, 关键点={pts_info}{q}")
        if args.save and results:
            out = _output_path(args.img, args.output_dir)
            align_dir = os.path.join(args.output_dir or "results", "aligned")
            _draw_align_results(img, results, out, align_dir)
            print(f"关键点结果已保存: {out}")
            print(f"对齐人脸已保存到: {align_dir}/")
    else:
        images_dir = args.dir or cfg.get("images_dir", "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
        out_dir = args.output_dir or "results"
        align_dir = os.path.join(out_dir, "aligned")
        for img_path in _iter_images(images_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[跳过] 无法读取: {img_path}")
                continue
            results = pipe.align_faces(img)
            filename = os.path.relpath(img_path, images_dir)
            print(f"{filename}: 检测到 {len(results)} 张人脸")
            if args.save:
                out = _output_path(img_path, out_dir, images_dir)
                _draw_align_results(img, results, out, align_dir)
                print(f"  -> 保存: {out}")


# ---- quality ----

def cmd_quality(args, pipe, cfg):
    """人脸质量评估：检测 -> 对齐 -> 质量打分。"""
    if pipe.quality_assessor is None:
        print("错误: 未配置人脸质量评估器 (quality_assessor)")
        return

    if args.img:
        img = cv2.imread(args.img)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {args.img}")
        results = pipe.align_faces(img)
        print(f"检测到 {len(results)} 张人脸:")
        for i, r in enumerate(results):
            q = r.get("quality")
            q_str = f"{q:.4f}" if q is not None else "N/A"
            print(f"  [{i}] bbox={r['bbox']}, 质量={q_str}")
        if args.save and results:
            out = _output_path(args.img, args.output_dir)
            _draw_quality_results(img, results, out)
            print(f"结果已保存: {out}")
    else:
        images_dir = args.dir or cfg.get("images_dir", "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
        # 汇总 CSV
        csv_path = os.path.join(args.output_dir or "results", "quality_report.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_lines = ["file,face_idx,bbox,quality"]
        for img_path in _iter_images(images_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[跳过] 无法读取: {img_path}")
                continue
            results = pipe.align_faces(img)
            filename = os.path.relpath(img_path, images_dir)
            if not results:
                print(f"{filename}: 未检测到人脸")
                continue
            for i, r in enumerate(results):
                q = r.get("quality")
                q_str = f"{q:.4f}" if q is not None else "N/A"
                print(f"{filename}: [{i}] 质量={q_str}")
                bbox_str = f"{r['bbox'][0]}_{r['bbox'][1]}_{r['bbox'][2]}_{r['bbox'][3]}"
                csv_lines.append(f"{filename},{i},{bbox_str},{q_str}")
            if args.save:
                out = _output_path(img_path, args.output_dir, images_dir)
                _draw_quality_results(img, results, out)
                print(f"  -> 保存: {out}")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))
        print(f"\n质量报告已保存: {csv_path}")


def _draw_quality_results(image, results, output_path):
    """在图片上绘制 bbox 和质量分。"""
    vis = image.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        q = r.get("quality")
        # 颜色按质量分渐变: 红(差) -> 绿(好)
        if q is not None:
            green = int(q * 255)
            red = int((1 - q) * 255)
            color = (0, green, red)
            label = f"Q:{q:.2f}"
        else:
            color = (128, 128, 128)
            label = "Q:N/A"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(output_path, vis)


# ---- analyze ----

def cmd_analyze(args, pipe, cfg):
    def _print_attr(prefix, r):
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
        print(f"{prefix}{', '.join(parts)}")

    if args.img:
        img = cv2.imread(args.img)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {args.img}")
        results = pipe.analyze_faces(img)
        print(f"检测到 {len(results)} 张人脸:")
        for i, r in enumerate(results):
            _print_attr(f"  [{i}] ", r)
        if args.save and results:
            out = _output_path(args.img, args.output_dir)
            pipe.draw_results(img, results, out)
            print(f"结果已保存: {out}")
    else:
        images_dir = args.dir or cfg.get("images_dir", "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
        for img_path in _iter_images(images_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[跳过] 无法读取: {img_path}")
                continue
            results = pipe.analyze_faces(img)
            filename = os.path.relpath(img_path, images_dir)
            if not results:
                print(f"{filename}: 未检测到人脸")
                continue
            for r in results:
                _print_attr(f"{filename}: ", r)
            if args.save:
                out = _output_path(img_path, args.output_dir, images_dir)
                pipe.draw_results(img, results, out)
                print(f"  -> 保存: {out}")


# ---- list / remove / visualize ----

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
    """可视化已注册人脸特征的 2D 分布（t-SNE / PCA / UMAP）。"""
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
        triu_idx = np.triu_indices(len(mask), k=1)
        intra_sims = sim_matrix[triu_idx]
        print(f"  {uid}: 类内相似度 mean={intra_sims.mean():.4f}, min={intra_sims.min():.4f}, max={intra_sims.max():.4f}")


# ---- CLI ----

def _add_img_dir_args(parser, img_help="图片路径", dir_help="图片文件夹"):
    """为子命令添加互斥的 --img / --dir 参数。"""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--img", default=None, help=img_help)
    group.add_argument("--dir", default=None, help=dir_help)


def main():
    parser = argparse.ArgumentParser(description="人脸识别流水线")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", help="子命令")

    # register
    p_reg = sub.add_parser("register", help="注册人脸 (--img 单张 / --dir 批量)")
    p_reg.add_argument("--name", default=None, help="身份名称 (单张注册时必填)")
    _add_img_dir_args(p_reg, "单张图片路径", "已分类人脸文件夹 (子文件夹名=身份，默认: known_faces)")

    # identify
    p_id = sub.add_parser("identify", help="1:N 识别 (--img 单张 / --dir 批量)")
    _add_img_dir_args(p_id, "图片路径", "图片文件夹 (默认: images)")
    p_id.add_argument("--threshold", type=float, default=None, help="识别阈值")
    p_id.add_argument("--top-k", type=int, default=1, help="返回前 K 个结果")
    p_id.add_argument("--save", action="store_true", help="保存结果图片")
    p_id.add_argument("--output-dir", default=None, help="结果保存目录 (默认: results)")

    # compare
    p_cmp = sub.add_parser("compare", help="1:1 人脸比对")
    p_cmp.add_argument("--img1", required=True)
    p_cmp.add_argument("--img2", required=True)

    # detect
    p_det = sub.add_parser("detect", help="人脸检测 (--img 单张 / --dir 批量)")
    _add_img_dir_args(p_det)
    p_det.add_argument("--save", action="store_true", help="保存结果图片")
    p_det.add_argument("--output-dir", default=None, help="结果保存目录 (默认: results)")

    # align
    p_align = sub.add_parser("align", help="关键点对齐 (--img 单张 / --dir 批量)")
    _add_img_dir_args(p_align)
    p_align.add_argument("--save", action="store_true", help="保存结果图片")
    p_align.add_argument("--output-dir", default=None, help="结果保存目录 (默认: results)")

    # analyze
    p_ana = sub.add_parser("analyze", help="属性分析 (--img 单张 / --dir 批量)")
    _add_img_dir_args(p_ana)
    p_ana.add_argument("--save", action="store_true", help="保存结果图片")
    p_ana.add_argument("--output-dir", default=None, help="结果保存目录 (默认: results)")

    # quality
    p_qa = sub.add_parser("quality", help="人脸质量评估 (--img 单张 / --dir 批量，批量输出 CSV 报告)")
    _add_img_dir_args(p_qa)
    p_qa.add_argument("--save", action="store_true", help="保存结果图片")
    p_qa.add_argument("--output-dir", default=None, help="结果保存目录 (默认: results)")

    # list / remove / visualize
    sub.add_parser("list", help="列出已注册身份")
    p_rm = sub.add_parser("remove", help="删除已注册身份")
    p_rm.add_argument("--name", required=True, help="身份名称")

    p_vis = sub.add_parser("visualize", help="可视化人脸特征分布 (t-SNE/PCA/UMAP)")
    p_vis.add_argument("--method", default="tsne", choices=["tsne", "pca", "umap"], help="降维方法")
    p_vis.add_argument("--perplexity", type=float, default=5, help="t-SNE perplexity")
    p_vis.add_argument("--output", default=None, help="输出图片路径")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    pipe, cfg = build_pipeline(args.config)

    commands = {
        "register": cmd_register,
        "identify": cmd_identify,
        "compare": cmd_compare,
        "detect": cmd_detect,
        "align": cmd_align,
        "analyze": cmd_analyze,
        "quality": cmd_quality,
        "list": cmd_list,
        "remove": cmd_remove,
        "visualize": cmd_visualize,
    }
    commands[args.command](args, pipe, cfg)


if __name__ == "__main__":
    main()
