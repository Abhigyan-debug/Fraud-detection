"""Train the fraud detection model and persist it (and the scaler) to disk."""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from config import MODEL_PATH, MODELS_DIR, RANDOM_STATE
from preprocessing import prepare_dataset


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )


def train(save: bool = True):
    X_train, X_test, y_train, y_test, scaler = prepare_dataset()

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")

    return model, scaler


if __name__ == "__main__":
    train()
