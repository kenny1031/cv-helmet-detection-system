from fastapi import FastAPI, UploadFile, File
from api.service import detect_objects
from api.schemas import DetectionResponse

app = FastAPI(
    title="Helmet Detection API",
    description="Detects safety helmet usage using YOLOv8",
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detections = detect_objects(image_bytes)
    return {"detections": detections}