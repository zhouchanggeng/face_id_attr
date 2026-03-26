"""根据配置文件动态构建流水线各模块。"""
import importlib
import yaml

from .pipeline import FaceRecogPipeline


def _create_instance(cfg: dict):
    """根据 {"class": "module.ClassName", "params": {...}} 动态实例化。

    config 中的 class 路径支持两种格式:
        - "module.face_detection.yolo_detector.YOLOFaceDetector" (短路径，自动加 face_id_attr 前缀)
        - "face_id_attr.module.face_detection.yolo_detector.YOLOFaceDetector" (完整路径)
    """
    module_path, class_name = cfg["class"].rsplit(".", 1)
    # 尝试直接导入，失败则加 face_id_attr 前缀
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        module = importlib.import_module(f"face_id_attr.{module_path}")
    cls = getattr(module, class_name)
    params = cfg.get("params") or {}
    return cls(**params)


def build_pipeline(config_path: str) -> tuple:
    """从 YAML 配置文件构建 FaceRecogPipeline，返回 (pipeline, config_dict)。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detector = _create_instance(cfg["detector"])

    aligner = None
    if cfg.get("aligner"):
        aligner = _create_instance(cfg["aligner"])

    recognizer = _create_instance(cfg["recognizer"])

    database = None
    if cfg.get("database"):
        database = _create_instance(cfg["database"])
        db_path = cfg["database"].get("db_path")
        if db_path:
            database.load(db_path)

    analyzer = None
    if cfg.get("analyzer"):
        analyzer = _create_instance(cfg["analyzer"])

    quality_assessor = None
    if cfg.get("quality_assessor"):
        quality_assessor = _create_instance(cfg["quality_assessor"])

    pipe = FaceRecogPipeline(
        detector, recognizer, aligner, database, analyzer,
        quality_assessor=quality_assessor,
        use_align_crop=cfg.get("use_align_crop", True),
        max_image_size=cfg.get("max_image_size", 1920),
    )
    return pipe, cfg
