import fiftyone as fo
from ultralytics import YOLO
from tqdm import tqdm
import sys

version = sys.argv[1]
DATASET_NAME = "helmet_hardhats_csv"
MODEL_PATH = f"runs/detect/train{version}/weights/best.pt"

def main():
    # Load data
    dataset = fo.load_dataset(DATASET_NAME)

    # Only take validation
    view = dataset.match({"split": "valid"})

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    print("Running inference on validation set...")

    # Batch prediction
    for sample in tqdm(view):
        results = model(sample.filepath)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]
            w_img, h_img = results.orig_shape[1], results.orig_shape[0]

            # Normalise xywh
            x = float(x1 / w_img)
            y = float(y1 / h_img)
            w = float((x2 - x1) / w_img)
            h = float((y2 - y1) / h_img)

            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=[x, y, w, h],
                    confidence=float(box.conf[0]),
                )
            )

        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    print("Inference complete.")

    # Evaluate
    results = view.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
    )

    results.print_report()

    # Open fiftyone
    session = fo.launch_app(view)
    session.wait()


if __name__ == "__main__":
    main()