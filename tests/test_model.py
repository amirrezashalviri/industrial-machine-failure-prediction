import pandas as pd


def test_dataset_shape():
    data = pd.read_csv("data/raw/ai4i2020.csv")

    assert data.shape == (10000, 14)


def test_target_exists():
    data = pd.read_csv("data/raw/ai4i2020.csv")

    assert "Machine failure" in data.columns