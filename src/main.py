import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/raw/ai4i2020.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load the predictive maintenance dataset."""
    return pd.read_csv(path)


def main():
    data = load_data(DATA_PATH)

    print("\n****************************************************************************************************")
    print("\nDataset Info:")
    print(data.info())

    print("\n****************************************************************************************************")
    print("\nGeneral Machine failure Info:")
    print(data["Machine failure"].value_counts())

    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    print("\n****************************************************************************************************")
    print("\nMachine failure Info By Selected Cols:")
    print(data.groupby("Machine failure")[features].mean())

    print("\n****************************************************************************************************")
    print("\nTool Wear Distribution by Machine Failure:")

    #failure_0 = data[data["Machine failure"] == 0]["Tool wear [min]"]
    #failure_1 = data[data["Machine failure"] == 1]["Tool wear [min]"]

    failure_0 = data[data["Machine failure"] == 0]["Torque [Nm]"]
    failure_1 = data[data["Machine failure"] == 1]["Torque [Nm]"]

    plt.hist(failure_0, bins=30, alpha=0.6, label="No Failure", density=True)
    plt.hist(failure_1, bins=30, alpha=0.6, label="Failure", density=True)

    plt.xlabel("Tool wear [min]")
    plt.ylabel("Number of machines")
    plt.title("Tool Wear Distribution by Machine Failure")
    plt.legend()

    #plt.show()
    plt.savefig("data/tool_wear_distribution.png")


if __name__ == "__main__":
    main()