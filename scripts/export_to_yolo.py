import os
import shutil
import fiftyone as fo
from tqdm import tqdm

DATASET_NAME = "helmet_hardhats_csv"
EXPORT_ROOT = "exports/yolo"

CLASS_MAP = {
    "Hardhat": 0,
    "NO-Hardhat": 1,
}

def ensure_dirs() -> None:
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{EXPORT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{EXPORT_ROOT}/labels/{split}", exist_ok=True)

def export_split(dataset, split_name: str) -> None:
    print(f"\nExporting {split_name}...")

    view = dataset.match({"split": split_name})

    for sample in tqdm(view):
        filename = os.path.basename(sample.filepath)

        # Copy image
        dest_img_path = f"{EXPORT_ROOT}/images/{split_name}/{filename}"
        shutil.copy(sample.filepath, dest_img_path)

        # Write label
        label_path = f"{EXPORT_ROOT}/labels/{split_name}/{filename.replace('.jpg', '.txt')}"
        detections = sample["ground_truth"].detections

        with open(label_path, "w") as f:
            for det in detections:
                cls_id = CLASS_MAP[det.label]
                x, y, w, h = det.bounding_box

                cx = x + w / 2
                cy = y + h / 2

                f.write(f"{cls_id} {cx} {cy} {w} {h}\n")

    print(f"{split_name} done.")

def write_data_yaml() -> None:
    yaml_content = f"""path: {os.path.abspath(EXPORT_ROOT)}
train: images/train
val: images/valid
test: images/test

names:
  0: Hardhat
  1: NO-Hardhat
"""
    with open(f"{EXPORT_ROOT}/data.yaml", "w") as f:
        f.write(yaml_content)

def main():
    ensure_dirs()

    dataset = fo.load_dataset(DATASET_NAME)

    for split in ["train", "valid", "test"]:
        export_split(dataset, split)

    write_data_yaml()

    print("\nExport complete.")
    print("YOLO dataset ready.")

if __name__ == "__main__":
    main()