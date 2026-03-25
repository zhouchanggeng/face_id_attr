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


def _algo_tag(cfg):
    """从配置中提取算法短名，用于自动命名输出目录。

    返回格式: "检测器_对齐器_识别器"，如 "yolo26m_pfld_arcface_glint360k"
    """
    def _short_name(module_cfg):
        if not module_cfg:
            return "none"
        model_path = (module_cfg.get("params") or {}).get("model_path", "")
        if model_path:
            # 取模型文件名（不含扩展名），去掉常见后缀
            name = os.path.splitext(os.path.basename(model_path))[0]
            # 简化: 去掉常见冗余后缀
            for suffix in ["_opt_sim", "_facedetect_widerface", "_2021dec",
                           "_2023mar", "_int8", "_int8bq", "_112_1",
                           "face_detection_", "face_recognition_"]:
                name = name.replace(suffix, "")
            return name
        # fallback: 取类名
        cls_name = module_cfg.get("class", "")
        return cls_name.rsplit(".", 1)[-1] if cls_name else "unknown"

    det = _short_name(cfg.get("detector"))
    ali = _short_name(cfg.get("aligner"))
    rec = _short_name(cfg.get("recognizer"))
    return f"{det}_{ali}_{rec}"


def _default_output_dir(input_dir, cfg, task=""):
    """生成默认输出目录名: result_{输入目录名}_{算法标签}_{任务名}"""
    dir_name = os.path.basename(os.path.normpath(input_dir))
    tag = _algo_tag(cfg)
    name = f"result_{dir_name}_{tag}"
    if task:
        name += f"_{task}"
    return name


def _nms_detections(detections, nms_threshold=0.4):
    """对检测结果做 NMS 去重，按 confidence 降序保留。"""
    if len(detections) <= 1:
        return detections
    boxes = [d["bbox"] for d in detections]
    confs = [d["confidence"] for d in detections]
    indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(i)
        remaining = []
        for j in indices:
            bx1 = max(boxes[i][0], boxes[j][0])
            by1 = max(boxes[i][1], boxes[j][1])
            bx2 = min(boxes[i][2], boxes[j][2])
            by2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, bx2 - bx1) * max(0, by2 - by1)
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            iou = inter / max(area_i + area_j - inter, 1e-6)
            if iou < nms_threshold:
                remaining.append(j)
        indices = remaining
    return [detections[i] for i in keep]


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
        out_dir = args.output_dir or _default_output_dir(images_dir, cfg, "identify")
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
                out = _output_path(img_path, out_dir, images_dir)
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
        out_dir = args.output_dir or _default_output_dir(images_dir, cfg, "detect")
        for img_path in _iter_images(images_dir):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[跳过] 无法读取: {img_path}")
                continue
            faces = pipe.detect(img)
            filename = os.path.relpath(img_path, images_dir)
            print(f"{filename}: 检测到 {len(faces)} 张人脸")
            if args.save:
                out = _output_path(img_path, out_dir, images_dir)
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
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
        out_dir = args.output_dir or _default_output_dir(images_dir, cfg, "align")
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
        out_dir = args.output_dir or _default_output_dir(images_dir, cfg, "quality")
        csv_path = os.path.join(out_dir, "quality_report.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
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
                out = _output_path(img_path, out_dir, images_dir)
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
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (255, 255, 255), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
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
        out_dir = args.output_dir or _default_output_dir(images_dir, cfg, "analyze")
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
                out = _output_path(img_path, out_dir, images_dir)
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


# ---- video ----

def cmd_video(args, pipe, cfg):
    """视频人脸识别：检测 + 跟踪 + 识别，输出标注视频。"""
    from module.face_tracking.iou_tracker import IoUTracker
    from module.face_tracking.sort_tracker import SORTTracker
    from module.face_tracking.byte_tracker import ByteTracker

    TRACKERS = {"iou": IoUTracker, "sort": SORTTracker, "byte": ByteTracker}

    pipe._require_db()
    video_path = args.input
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    threshold = args.threshold or cfg.get("identify_threshold", 0.5)

    # 输出路径
    tag = _algo_tag(cfg)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = args.output_dir or f"result_{base}_{tag}_video"
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"{base}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    tracker_cls = TRACKERS.get(args.tracker, IoUTracker)
    tracker_kwargs = {
        "iou_threshold": args.iou_threshold,
        "max_missed": int(fps * 0.5),
        "recognize_interval": int(fps * args.recog_interval),
    }
    if args.tracker == "byte":
        tracker_kwargs["high_threshold"] = 0.5
        tracker_kwargs["low_threshold"] = 0.1
    tracker = tracker_cls(**tracker_kwargs)

    print(f"视频: {video_path} ({w}x{h}, {fps:.1f}fps, {total_frames} 帧)")
    print(f"识别阈值: {threshold}, 识别间隔: {args.recog_interval}s")
    print(f"输出: {out_video}\n")

    frame_idx = 0
    id_log = {}  # track_id -> {identity, first_frame, last_frame, appearances}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测（在缩放图上）
        resized, scale = pipe._limit_size(frame, pipe.max_image_size)
        raw_dets = pipe.detector.detect(resized)

        # NMS 去重（防止重叠框）
        raw_dets = _nms_detections(raw_dets, nms_threshold=0.4)

        # 映射回原图坐标用于跟踪和绘制
        detections = []
        for d in raw_dets:
            x1, y1, x2, y2 = d["bbox"]
            if scale < 1.0:
                inv = 1.0 / scale
                d["bbox"] = (int(x1 * inv), int(y1 * inv),
                             int(x2 * inv), int(y2 * inv))
            # 保留缩放图上的原始 bbox 用于对齐
            d["_resized_bbox"] = (x1, y1, x2, y2)
            detections.append(d)

        # 跟踪
        tracks = tracker.update(detections)

        # 识别：仅对需要识别的 track 做推理（首次 + 定期刷新）
        for track in tracks:
            if track.missed > 0:
                continue
            if not tracker.needs_recognition(track):
                continue
            face_dict = {"bbox": track.bbox, "landmarks": None}
            if scale < 1.0:
                ox1, oy1, ox2, oy2 = track.bbox
                face_dict["bbox"] = (int(ox1 * scale), int(oy1 * scale),
                                     int(ox2 * scale), int(oy2 * scale))
            face_img = pipe._get_face_image(resized, face_dict)
            feat = pipe.recognizer.extract(face_img)
            track.feature = feat
            hits = pipe.database.search(feat, top_k=1)
            if hits:
                pred_id, sim = hits[0]
                # 保留历史最高相似度（质量好的帧自然分高）
                if sim > track.similarity:
                    track.similarity = sim
                    if sim >= threshold:
                        track.identity = pred_id
            track.recognized = True

        # 绘制
        vis = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            if track.identity:
                color = (0, 255, 0)
                label = f"#{track.track_id} {track.identity} {track.similarity:.2f}"
            else:
                color = (0, 0, 255)
                label = f"#{track.track_id} unknown"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th_ - 6), (x1 + tw + 4, y1), (255, 255, 255), -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 记录日志
            tid = track.track_id
            if tid not in id_log:
                id_log[tid] = {"identity": track.identity, "first_frame": frame_idx,
                               "last_frame": frame_idx, "max_sim": track.similarity}
            else:
                id_log[tid]["last_frame"] = frame_idx
                if track.identity:
                    id_log[tid]["identity"] = track.identity
                    id_log[tid]["max_sim"] = max(id_log[tid]["max_sim"], track.similarity)

        writer.write(vis)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  处理中: {frame_idx}/{total_frames} ({frame_idx/max(total_frames,1)*100:.0f}%)")

    cap.release()
    writer.release()
    print(f"\n处理完成: {frame_idx} 帧")
    print(f"输出视频: {out_video}")

    # 保存跟踪日志
    log_path = os.path.join(out_dir, "track_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("track_id,identity,first_frame,last_frame,duration_frames,max_similarity\n")
        for tid in sorted(id_log.keys()):
            info = id_log[tid]
            duration = info["last_frame"] - info["first_frame"] + 1
            ident = info["identity"] or "unknown"
            f.write(f"{tid},{ident},{info['first_frame']},{info['last_frame']},{duration},{info['max_sim']:.4f}\n")
    print(f"跟踪日志: {log_path}")

    # 打印摘要
    identities = set(v["identity"] for v in id_log.values() if v["identity"])
    print(f"\n识别到 {len(identities)} 个身份, {len(id_log)} 条轨迹:")
    for ident in sorted(identities):
        tracks_of = [v for v in id_log.values() if v["identity"] == ident]
        total_dur = sum(v["last_frame"] - v["first_frame"] + 1 for v in tracks_of)
        max_sim = max(v["max_sim"] for v in tracks_of)
        print(f"  {ident}: {len(tracks_of)} 条轨迹, {total_dur} 帧, 最高相似度 {max_sim:.4f}")


# ---- evaluate ----

def cmd_evaluate(args, pipe, cfg):
    """评测人脸识别性能。测试目录结构: test_dir/{identity}/xxx.jpg"""
    import numpy as np

    pipe._require_db()
    test_dir = args.dir
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"目录不存在: {test_dir}")

    threshold = args.threshold or cfg.get("identify_threshold", 0.5)
    out_dir = args.output_dir or _default_output_dir(test_dir, cfg, "evaluate")
    os.makedirs(out_dir, exist_ok=True)

    # 收集测试数据: ground truth identity -> image paths
    gt_data = []
    for identity in sorted(os.listdir(test_dir)):
        id_dir = os.path.join(test_dir, identity)
        if not os.path.isdir(id_dir):
            continue
        for img_path in _iter_images(id_dir):
            gt_data.append((identity, img_path))

    if not gt_data:
        print("测试目录为空")
        return

    print(f"评测数据: {len(gt_data)} 张图片, {len(set(g[0] for g in gt_data))} 个身份")
    print(f"识别阈值: {threshold}\n")

    # 逐张识别
    results_log = []  # (gt_id, pred_id, similarity, matched, quality)
    all_similarities = []  # (similarity, is_same_person)

    for gt_id, img_path in gt_data:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue
        faces = pipe.extract(img)
        if not faces:
            results_log.append((gt_id, None, 0.0, False, None))
            continue
        # 取最大人脸
        face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        hits = pipe.database.search(face["feature"], top_k=1)
        if hits:
            pred_id, sim = hits[0]
            matched = sim >= threshold
            quality = face.get("quality")
            results_log.append((gt_id, pred_id if matched else None, sim, matched, quality))
            # 记录相似度用于 ROC
            is_same = (pred_id == gt_id)
            all_similarities.append((sim, is_same))
        else:
            results_log.append((gt_id, None, 0.0, False, face.get("quality")))

    # ---- 1:N Rank-1 指标 ----
    total = len(results_log)
    no_face = sum(1 for r in results_log if r[1] is None and r[2] == 0.0 and not r[3])
    detected = total - no_face  # 排除未检测到人脸的

    # Rank-1: 预测身份 == GT 身份（不考虑阈值，只看 top-1 是否正确）
    rank1_correct = sum(1 for gt, pred_id, sim, matched, _ in results_log
                        if sim > 0 and _get_top1_id(pipe, gt, results_log, all_similarities, gt_id=gt) == gt)
    # 简化: 直接用 all_similarities 中 is_same=True 的
    rank1_correct = sum(1 for sim, is_same in all_similarities if is_same)
    rank1_acc = rank1_correct / max(detected, 1)

    # 在当前阈值下的指标
    tp = sum(1 for gt, pred, sim, matched, _ in results_log if matched and pred == gt)
    fp = sum(1 for gt, pred, sim, matched, _ in results_log if matched and pred != gt)
    fn = sum(1 for gt, pred, sim, matched, _ in results_log if not matched or pred != gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print("=" * 60)
    print(f"{'评测结果':^56}")
    print("=" * 60)
    print(f"  总图片数:       {total}")
    print(f"  检测到人脸:     {detected} ({detected/max(total,1)*100:.1f}%)")
    print(f"  识别阈值:       {threshold}")
    print(f"  Rank-1 准确率:  {rank1_acc:.4f} ({rank1_correct}/{detected})")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print("=" * 60)

    # ---- 每个身份的详细指标 ----
    gt_ids = sorted(set(r[0] for r in results_log))
    print(f"\n{'身份':<20} {'总数':>4} {'正确':>4} {'错误':>4} {'未识别':>6} {'准确率':>8}")
    print("-" * 56)
    for gid in gt_ids:
        id_results = [r for r in results_log if r[0] == gid]
        id_total = len(id_results)
        id_correct = sum(1 for r in id_results if r[3] and r[1] == gid)
        id_wrong = sum(1 for r in id_results if r[3] and r[1] != gid)
        id_miss = id_total - id_correct - id_wrong
        id_acc = id_correct / max(id_total, 1)
        print(f"  {gid:<18} {id_total:>4} {id_correct:>4} {id_wrong:>4} {id_miss:>6} {id_acc:>8.2%}")

    # ---- ROC 曲线 & AUC ----
    if all_similarities:
        _plot_roc(all_similarities, out_dir, threshold)

    # ---- 保存详细 CSV ----
    csv_path = os.path.join(out_dir, "evaluate_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,gt_identity,pred_identity,similarity,matched,quality\n")
        for (gt_id, img_path), (_, pred, sim, matched, quality) in zip(gt_data, results_log):
            fname = os.path.relpath(img_path, test_dir)
            q_str = f"{quality:.4f}" if quality is not None else ""
            pred_str = pred or ""
            f.write(f"{fname},{gt_id},{pred_str},{sim:.4f},{matched},{q_str}\n")
    print(f"\n详细报告已保存: {csv_path}")

    # ---- 保存汇总 ----
    summary_path = os.path.join(out_dir, "evaluate_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        tag = _algo_tag(cfg)
        f.write(f"算法配置: {tag}\n")
        f.write(f"测试集: {test_dir} ({total} 张图片, {len(gt_ids)} 个身份)\n")
        f.write(f"识别阈值: {threshold}\n\n")
        f.write(f"Rank-1 准确率: {rank1_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"TP={tp}, FP={fp}, FN={fn}\n")
    print(f"汇总已保存: {summary_path}")


def _get_top1_id(pipe, gt, results_log, all_similarities, gt_id):
    """辅助函数，不实际使用，仅占位。"""
    return gt_id


def _plot_roc(all_similarities, out_dir, current_threshold):
    """绘制 ROC 曲线，计算 AUC 和 EER。"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims = np.array([s for s, _ in all_similarities])
    labels = np.array([1 if is_same else 0 for _, is_same in all_similarities])

    if len(set(labels)) < 2:
        print("  (正负样本不足，跳过 ROC 曲线)")
        return

    # 计算不同阈值下的 TAR 和 FAR
    thresholds = np.linspace(0, 1, 500)
    tars, fars = [], []
    for t in thresholds:
        preds = sims >= t
        tp = np.sum(preds & (labels == 1))
        fp = np.sum(preds & (labels == 0))
        fn = np.sum(~preds & (labels == 1))
        tn = np.sum(~preds & (labels == 0))
        tar = tp / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)
        tars.append(tar)
        fars.append(far)

    tars = np.array(tars)
    fars = np.array(fars)
    frrs = 1 - tars

    # AUC
    sorted_idx = np.argsort(fars)
    auc = np.trapz(tars[sorted_idx], fars[sorted_idx])

    # EER: FAR ≈ FRR
    eer_idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    # TAR@FAR=1e-3
    far_target = 1e-3
    tar_at_far = tars[np.argmin(np.abs(fars - far_target))]

    print(f"\n  AUC:            {auc:.4f}")
    print(f"  EER:            {eer:.4f} (阈值={eer_threshold:.3f})")
    print(f"  TAR@FAR=0.001:  {tar_at_far:.4f}")

    # 绘制 ROC
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC 曲线
    ax = axes[0]
    ax.plot(fars, tars, "b-", linewidth=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.scatter([fars[eer_idx]], [tars[eer_idx]], c="red", s=80, zorder=5,
               label=f"EER={eer:.4f} (t={eer_threshold:.3f})")
    ax.set_xlabel("FAR (False Accept Rate)")
    ax.set_ylabel("TAR (True Accept Rate)")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TAR/FAR vs Threshold
    ax = axes[1]
    ax.plot(thresholds, tars, "g-", linewidth=2, label="TAR")
    ax.plot(thresholds, fars, "r-", linewidth=2, label="FAR")
    ax.plot(thresholds, frrs, "b--", linewidth=1, label="FRR")
    ax.axvline(x=current_threshold, color="orange", linestyle=":", linewidth=2,
               label=f"当前阈值={current_threshold:.2f}")
    ax.axvline(x=eer_threshold, color="red", linestyle=":", linewidth=1,
               label=f"EER阈值={eer_threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("TAR / FAR / FRR vs Threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC 曲线已保存: {roc_path}")


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

    # evaluate
    p_eval = sub.add_parser("evaluate", help="评测识别性能 (测试目录: {identity}/xxx.jpg)")
    p_eval.add_argument("--dir", required=True, help="测试图片目录 (子文件夹名=GT身份)")
    p_eval.add_argument("--threshold", type=float, default=None, help="识别阈值")
    p_eval.add_argument("--output-dir", default=None, help="评测报告保存目录")

    # video
    p_vid = sub.add_parser("video", help="视频人脸识别 (检测+跟踪+识别)")
    p_vid.add_argument("--input", required=True, help="输入视频路径")
    p_vid.add_argument("--tracker", default="sort", choices=["iou", "sort", "byte"],
                       help="跟踪算法: iou(贪心IoU), sort(Kalman+匈牙利), byte(ByteTrack两阶段)")
    p_vid.add_argument("--threshold", type=float, default=None, help="识别阈值")
    p_vid.add_argument("--iou-threshold", type=float, default=0.3, help="跟踪 IoU 阈值")
    p_vid.add_argument("--recog-interval", type=float, default=1.0, help="重新识别间隔 (秒，默认1.0)")
    p_vid.add_argument("--output-dir", default=None, help="输出目录")

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
        "evaluate": cmd_evaluate,
        "video": cmd_video,
    }
    commands[args.command](args, pipe, cfg)


if __name__ == "__main__":
    main()
