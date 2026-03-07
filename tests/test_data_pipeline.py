import os
import pandas as pd

def test_dataset_structure():

    assert os.path.exists("data/train/_annotations.csv")
    assert os.path.exists("data/valid/_annotations.csv")
    assert os.path.exists("data/test/_annotations.csv")


def test_annotations_columns():

    df = pd.read_csv("data/train/_annotations.csv")

    required_columns = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    for col in required_columns:
        assert col in df.columns