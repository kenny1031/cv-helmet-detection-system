from pydantic import BaseModel
from typing import List

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]

class DetectionResponse(BaseModel):
    detections: List[Detection]