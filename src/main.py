import pandas as pd


DATA_PATH = "data/raw/ai4i2020.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load the predictive maintenance dataset."""
    return pd.read_csv(path)


def main():
    data = load_data(DATA_PATH)

    print("Dataset shape:")
    print(data.shape)

    print("\nDataset columns:")
    print(data.columns.tolist())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nMachine failure distribution:")
    print(data["Machine failure"].value_counts())

    print("\nNumerical feature statistics:")
    print(data.describe())


if __name__ == "__main__":
    main()