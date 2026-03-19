"""根据配置文件动态构建流水线各模块。"""
import importlib
import yaml

from pipeline import FaceRecogPipeline


def _create_instance(cfg: dict):
    """根据 {"class": "module.ClassName", "params": {...}} 动态实例化。"""
    module_path, class_name = cfg["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
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

    pipe = FaceRecogPipeline(detector, recognizer, aligner, database, analyzer)
    return pipe, cfg
