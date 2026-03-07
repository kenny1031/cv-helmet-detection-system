from ultralytics import YOLO
from typing import List, Dict

class HelmetPredictor:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float=0.4
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold


    def predict(
        self, image_path: str
    ) -> List[Dict[str, str | float | list]]:
        results = self.model(image_path)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])

            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2]
            })

        return detections