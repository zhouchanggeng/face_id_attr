import cv2
import numpy as np
from typing import List
from .base import FaceDetector


class OpenCVDetector(FaceDetector):
    """基于 OpenCV DNN 的人脸检测 (Caffe SSD)。"""

    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

    def __init__(self, model_path: str = None, config_path: str = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        if model_path and config_path:
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            # 回退到 Haar 级联
            self.net = None
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, image: np.ndarray) -> List[dict]:
        h, w = image.shape[:2]
        if self.net:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            detections = self.net.forward()
            results = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf >= self.conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    results.append({"bbox": (x1, y1, x2, y2), "confidence": conf, "landmarks": None})
            return results
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            return [
                {"bbox": (x, y, x + w_, y + h_), "confidence": 1.0, "landmarks": None}
                for (x, y, w_, h_) in faces
            ]
