import time
import glob
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train2/weights/best.pt"
IMAGE_DIR = "examples"

model = YOLO(MODEL_PATH)
image_paths = glob.glob(f"{IMAGE_DIR}/*.jpg")

print(f"Testing {len(image_paths)} images...")

start = time.time()

for img in image_paths:
    model(img, verbose=False)

end = time.time()

total_time = end - start
fps = len(image_paths) / total_time

print("Total images:", len(image_paths))
print("Total time:", round(total_time, 2), "seconds")
print("FPS:", round(fps, 2))