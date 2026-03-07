import os
import tempfile

from inference.predictor import HelmetPredictor

MODEL_PATH = "runs/detect/train2/weights/best.pt"

predictor = HelmetPredictor(MODEL_PATH)

def detect_objects(image_bytes):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        image_path = tmp.name

    detections = predictor.predict(image_path)
    os.remove(image_path)

    return detections