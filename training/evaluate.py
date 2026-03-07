from ultralytics import YOLO

MODEL_PATH = "runs/detect/train2/weights/best.pt"

model = YOLO(MODEL_PATH)
metrics = model.val()

print(metrics)