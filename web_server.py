"""
人脸识别 Web 服务 - Flask API + 前端页面

启动方式:
    python web_server.py
    python web_server.py --config config.yaml --port 5000

API:
    POST /api/register   - 注册人脸 (name + image)
    POST /api/identify    - 识别人脸 (image)
    POST /api/detect      - 检测人脸 (image)
    GET  /api/identities  - 列出已注册身份
    DELETE /api/identity/<name> - 删除身份
"""
import argparse
import base64
import io
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from factory import build_pipeline
except ImportError:
    from .factory import build_pipeline

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

pipe = None
cfg = None


def _decode_image(file_or_b64):
    """从上传文件或 base64 解码图片。"""
    if hasattr(file_or_b64, "read"):
        data = file_or_b64.read()
    else:
        # base64 string
        if "," in file_or_b64:
            file_or_b64 = file_or_b64.split(",", 1)[1]
        data = base64.b64decode(file_or_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片")
    return img


def _encode_result_image(image, results):
    """在图片上绘制结果并返回 base64 JPEG。"""
    vis = image.copy()
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        matched = r.get("matched")
        if matched is True:
            color = (0, 255, 0)
        elif matched is False:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label_parts = []
        if r.get("identity"):
            label_parts.append(r["identity"])
        if r.get("similarity") is not None:
            label_parts.append(f"{r['similarity']:.2f}")
        if r.get("quality") is not None:
            label_parts.append(f"Q:{r['quality']:.2f}")
        if not label_parts:
            label_parts.append(f"{r.get('confidence', 0):.2f}")

        attr = r.get("attributes", {})
        if attr.get("dominant_emotion"):
            label_parts.append(attr["dominant_emotion"])

        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (255, 255, 255), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _save_db():
    db_path = cfg.get("database", {}).get("db_path")
    if db_path:
        pipe.database.save(db_path)


# ---- Routes ----

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/register", methods=["POST"])
def api_register():
    """注册人脸。表单字段: name (身份名), image (图片文件)。"""
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "请提供身份名称 (name)"}), 400
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "请上传图片 (image)"}), 400
    try:
        img = _decode_image(file)
        count = pipe.register(name, img)
        _save_db()
        # 返回检测到的人脸框用于前端展示
        faces = pipe.detect(img)
        result_img = _encode_result_image(img, faces)
        return jsonify({
            "success": True,
            "registered": count,
            "name": name,
            "faces_detected": len(faces),
            "result_image": result_img,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/identify", methods=["POST"])
def api_identify():
    """识别人脸。表单字段: image (图片文件), threshold (可选)。"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "请上传图片 (image)"}), 400
    try:
        threshold = float(request.form.get("threshold", cfg.get("identify_threshold", 0.5)))
        img = _decode_image(file)
        results = pipe.identify(img, threshold=threshold, top_k=3)
        result_img = _encode_result_image(img, results)
        # 序列化结果
        faces = []
        for r in results:
            faces.append({
                "bbox": list(r["bbox"]),
                "confidence": round(r["confidence"], 4),
                "identity": r["identity"],
                "similarity": round(r["similarity"], 4),
                "matched": r["matched"],
                "top_k": [[name, round(sim, 4)] for name, sim in (r.get("top_k") or [])],
            })
        return jsonify({
            "success": True,
            "faces": faces,
            "result_image": result_img,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """检测人脸。表单字段: image (图片文件)。"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "请上传图片 (image)"}), 400
    try:
        img = _decode_image(file)
        faces = pipe.detect(img)
        result_img = _encode_result_image(img, faces)
        face_list = []
        for f in faces:
            face_list.append({
                "bbox": list(f["bbox"]),
                "confidence": round(f["confidence"], 4),
            })
        return jsonify({
            "success": True,
            "faces": face_list,
            "result_image": result_img,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """属性分析（微笑检测/表情识别）。表单字段: image (图片文件)。"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "请上传图片 (image)"}), 400
    try:
        img = _decode_image(file)
        results = pipe.analyze_faces(img)
        result_img = _encode_result_image(img, results)
        face_list = []
        for r in results:
            attr = r.get("attributes", {})
            face_list.append({
                "bbox": list(r["bbox"]),
                "confidence": round(r.get("confidence", 0), 4),
                "dominant_emotion": attr.get("dominant_emotion", ""),
                "emotion_confidence": round(attr.get("confidence", 0), 4),
                "emotion": {k: round(v, 4) for k, v in attr.get("emotion", {}).items()},
                "detail": {k: round(v, 4) for k, v in attr.get("detail", {}).items()} if attr.get("detail") else None,
                "quality": round(r["quality"], 4) if r.get("quality") is not None else None,
            })
        return jsonify({
            "success": True,
            "faces": face_list,
            "result_image": result_img,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/identities", methods=["GET"])
def api_identities():
    """列出已注册身份。"""
    try:
        identities = pipe.database.list_identities()
        # 统计每个身份的特征数
        counts = {}
        for name in pipe.database.identities:
            counts[name] = counts.get(name, 0) + 1
        return jsonify({
            "success": True,
            "identities": [{"name": n, "count": counts.get(n, 0)} for n in identities],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/identity/<name>", methods=["DELETE"])
def api_delete_identity(name):
    """删除指定身份。"""
    try:
        count = pipe.database.remove(name)
        _save_db()
        return jsonify({"success": True, "removed": count, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人脸识别 Web 服务")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    args = parser.parse_args()

    print(f"加载配置: {args.config}")
    pipe, cfg = build_pipeline(args.config)
    print(f"流水线就绪，启动 Web 服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
