"""face_id_attr — 模块化人脸识别流水线。

提供人脸检测、关键点校正、特征提取、1:N 识别、质量评估、视频跟踪等基础能力。
可作为 Python 包安装后供其他项目依赖使用。

用法（作为库）:
    from face_id_attr.pipeline import FaceRecogPipeline
    from face_id_attr.factory import build_pipeline
    from face_id_attr.module.face_detection import YOLOFaceDetector
    from face_id_attr.module.face_alignment import PFLDAligner
    from face_id_attr.module.face_recognition import ArcFaceRecognizer
"""

__version__ = "0.1.0"
