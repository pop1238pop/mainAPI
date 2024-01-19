# For machine learning
import torch
# For array computations
import numpy as np
# For image decoding / editing
import cv2
# For environment variables
import os
# For detecting which ML Devices we can use
import platform
# For actually using the YOLO models
from ultralytics import YOLO

class YoloV8ImageObjectDetection:
    PATH        = PATH = os.environ.get(r"D:\Senoir Proj\hosting-yolo-fastapi-master\best.pt", "best.pt")
    # Path to a model. yolov8n.pt means download from PyTorch Hub
    CONF_THRESH = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.70")) # Confidence threshold

    def __init__(self, chunked: bytes = None):

        self._bytes = chunked
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _get_device(self):

        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self):

        model = YOLO(YoloV8ImageObjectDetection.PATH)
        return model

    async def __call__(self):

        frame = self._get_image_from_chunked()
        results = self.score_frame(frame)
        frame, labels = self.plot_boxes(results, frame)
        return frame, set(labels)
    
    def _get_image_from_chunked(self):

        arr = np.asarray(bytearray(self._bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        return img
    
    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(
            frame, 
            conf=YoloV8ImageObjectDetection.CONF_THRESH, 
            save_conf=True
        )
        return results

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        for r in results:
            boxes = r.boxes
            labels = []
            for box in boxes:
                c = box.cls
                l = self.model.names[int(c)]
                labels.append(l)
        frame = results[0].plot()
        return frame, labels
