"""Evaluate a trained fraud detection model on the held-out test set."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import (
    EXPERIMENTS_DIR,
    FEATURE_COLUMNS,
    FRAUD_THRESHOLD,
    MODEL_PATH,
    PROCESSED_DIR,
    TARGET_COLUMN,
)


def load_test_set(path=None):
    path = path or PROCESSED_DIR / "test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python train.py` first to generate the "
            "processed test split."
        )
    df = pd.read_csv(path)
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def evaluate(model_path=MODEL_PATH, save_plots: bool = True):
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python train.py` first.")

    model = joblib.load(model_path)
    X_test, y_test = load_test_set()

    y_proba = model.predict_proba(X_test)[:, 1]
    # Threshold explicitly rather than calling model.predict(), which is hard-wired
    # to 0.5 and would report numbers the API and consumer never actually produce.
    y_pred = (y_proba >= FRAUD_THRESHOLD).astype(int)

    print(f"Decision threshold: {FRAUD_THRESHOLD}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)

    if save_plots:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

        ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig(EXPERIMENTS_DIR / "confusion_matrix.png", bbox_inches="tight")
        plt.close()

        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve")
        plt.savefig(EXPERIMENTS_DIR / "roc_curve.png", bbox_inches="tight")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title("Precision-Recall Curve")
        plt.savefig(EXPERIMENTS_DIR / "precision_recall_curve.png", bbox_inches="tight")
        plt.close()

        print(f"\nPlots saved to {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    evaluate()
