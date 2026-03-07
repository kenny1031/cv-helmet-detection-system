# test_inference.py

import sys
import os

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predictor import HelmetPredictor

MODEL_PATH = "runs/detect/train2/weights/best.pt"

def test_model_load():
    predictor = HelmetPredictor(MODEL_PATH)
    assert predictor.model is not None

def test_prediction_output():
    predictor = HelmetPredictor(MODEL_PATH)
    results = predictor.predict("examples/helmet1.jpg")
    assert isinstance(results, list)