from ultralytics import YOLO
import yaml

CONFIG_PATH = "configs/train_config.yml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    model = YOLO(cfg["model"])
    model.train(
        data=cfg["data"],
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        device=cfg["device"]
    )

if __name__ == "__main__":
    train()