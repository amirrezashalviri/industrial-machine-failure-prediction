import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/raw/ai4i2020.csv"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_model(model, X_test, y_test, name: str):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("F1:", f1_score(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))
    print("PR-AUC:", average_precision_score(y_test, probabilities))

    return predictions, probabilities


def main():
    data = load_data(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    print("Dataset shape:", data.shape)
    print("\nFailure distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression",
    )

    evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest",
    )

    feature_importance = pd.Series(
        random_forest_model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\n===== Feature Importance =====")
    print(feature_importance)

    feature_importance.plot(kind="bar")
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png")
    plt.close()

    predictions = random_forest_model.predict(X_test)

    errors = X_test.copy()
    errors["actual"] = y_test
    errors["predicted"] = predictions

    print("\n===== False Negatives =====")
    print(errors[(errors["actual"] == 1) & (errors["predicted"] == 0)])

    cm = confusion_matrix(y_test, predictions)

    plt.imshow(cm)
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    joblib.dump(
        random_forest_model,
        "models/random_forest.pkl",
    )

    print("\nModel saved to models/random_forest.pkl")


if __name__ == "__main__":
    main()