"""Load, clean and preprocess the raw credit-card transaction data."""

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCALED_COLUMNS,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw dataset and drop exact duplicate rows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download the Kaggle "
            "'Credit Card Fraud Detection' dataset and place creditcard.csv "
            "there, or set the FRAUD_RAW_DATA environment variable to its path."
        )
    df = pd.read_csv(path)
    return df.drop_duplicates()


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """Scale Time/Amount; the V1-V28 columns are already PCA output.

    Pass ``fit=False`` with an existing ``scaler`` for any data that must not
    influence the fitted statistics (i.e. the test fold and live inference).
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[SCALED_COLUMNS] = scaler.fit_transform(df[SCALED_COLUMNS])
    else:
        df[SCALED_COLUMNS] = scaler.transform(df[SCALED_COLUMNS])

    return df, scaler


def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def balance_classes(X_train: pd.DataFrame, y_train: pd.Series):
    """Oversample the minority (fraud) class with SMOTE — training data only."""
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)


def prepare_dataset(path=RAW_DATA_PATH, save_artifacts: bool = True, balance: bool = True):
    """Run the full pipeline: load -> split -> scale -> balance.

    The split happens before scaling on purpose: fitting the scaler on the full
    frame would let the test fold's Time/Amount statistics inform the training
    data. SMOTE likewise only ever sees the training fold.

    Returns (X_train, X_test, y_train, y_test, scaler).
    """
    df = load_raw_data(path)

    X_train, X_test, y_train, y_test = split_data(df)

    X_train, scaler = scale_features(X_train, fit=True)
    X_test, _ = scale_features(X_test, scaler=scaler, fit=False)

    if balance:
        X_train, y_train = balance_classes(X_train, y_train)

    if save_artifacts:
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(
            PROCESSED_DIR / "test.csv", index=False
        )

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = prepare_dataset()

    print(f"Training set (after balancing): {X_train.shape}")
    print(y_train.value_counts())
    print(f"\nTest set: {X_test.shape}")
    print(y_test.value_counts())
