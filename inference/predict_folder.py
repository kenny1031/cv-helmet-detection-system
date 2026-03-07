import os
import json
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train2/weights/best.pt"
IMAGE_DIR = "examples"
OUTPUT_FILE = "detections.json"

model = YOLO(MODEL_PATH)

results_json = []

for filename in os.listdir(IMAGE_DIR):
    if not filename.endswith(".jpg"):
        continue

    path = os.path.join(IMAGE_DIR, filename)
    results = model(path)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "label": label,
            "confidence": float(box.conf[0]),
            "bbox": [x1, y1, x2, y2],
        })

    results_json.append({
        "image": filename,
        "detections": detections
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(results_json, f, indent=2)

print("Results saved to", OUTPUT_FILE)