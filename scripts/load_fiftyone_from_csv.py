import os
import pandas as pd
import fiftyone as fo
from collections import Counter

DATA_ROOT = "data"
DATA_NAME = "helmet_hardhats_csv"

CLASSES = ["Hardhat", "NO-Hardhat"]

def load_split(dataset, split_name):
    print(f"\nLoading {split_name}...")

    split_dir = os.path.join(DATA_ROOT, split_name)
    csv_path = os.path.join(split_dir, "_annotations.csv")

    df = pd.read_csv(csv_path)
    images_dir = os.path.join(split_dir, "images")

    class_counter = Counter()
    samples = []

    for filename, group in df.groupby("filename"):
        filepath = os.path.join(images_dir, filename)
        if not os.path.exists(filepath):
            continue

        detections = []

        for _, row in group.iterrows():
            cls = str(row["class"])
            if cls not in CLASSES:
                continue

            class_counter[cls] += 1

            w = float(row["width"])
            h = float(row["height"])

            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            x = xmin / w
            y = ymin / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h

            detections.append(
                fo.Detection(
                    label=cls,
                    bounding_box=[x, y, box_w, box_h],
                )
            )

        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Detections(detections=detections)
        sample["split"] = split_name

        samples.append(sample)

    dataset.add_samples(samples)

    print(f"{split_name} loaded.")
    print("Class distribution:", dict(class_counter))

def main():
    if DATA_NAME in fo.list_datasets():
        fo.delete_dataset(DATA_NAME)

    dataset = fo.Dataset(DATA_NAME)

    for split in ["train", "valid", "test"]:
        load_split(dataset, split)

    print("\nTotal samples:", len(dataset))

    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()

