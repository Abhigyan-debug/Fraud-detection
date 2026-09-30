"""Central configuration for paths and constants used across the pipeline."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def _resolve_raw_data_path() -> Path:
    env_path = os.environ.get("FRAUD_RAW_DATA")
    if env_path:
        return Path(env_path)

    default_path = DATA_DIR / "raw" / "creditcard.csv"
    if default_path.exists():
        return default_path

    # Falls back to this machine's known dataset location so the app works
    # without requiring FRAUD_RAW_DATA to be set for local development.
    fallback_path = Path("D:/archive (6)/creditcard.csv")
    if fallback_path.exists():
        return fallback_path

    return default_path


RAW_DATA_PATH = _resolve_raw_data_path()

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

EXPERIMENTS_DIR = BASE_DIR / "experiments"

# V1-V28 are already PCA components from the source dataset; only Time and
# Amount need scaling before they're fed to the model.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
SCALED_COLUMNS = ["Time", "Amount"]
TARGET_COLUMN = "Class"

TEST_SIZE = 0.2
RANDOM_STATE = 42
# Tuned for F1 on a validation slice of the training fold by
# train.select_threshold() -- never on the test fold, which would make the
# reported precision/recall optimistic. Lower it to trade precision for recall.
FRAUD_THRESHOLD = 0.455

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TRANSACTIONS_TOPIC = os.environ.get("KAFKA_TRANSACTIONS_TOPIC", "transactions")
KAFKA_ALERTS_TOPIC = os.environ.get("KAFKA_ALERTS_TOPIC", "fraud_alerts")
