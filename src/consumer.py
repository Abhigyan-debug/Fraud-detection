"""Consume transactions from Kafka, score them with the trained model, and
republish anything that looks fraudulent to an alerts topic.
"""

import json

import joblib
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from config import (
    FEATURE_COLUMNS,
    FRAUD_THRESHOLD,
    KAFKA_ALERTS_TOPIC,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TRANSACTIONS_TOPIC,
    MODEL_PATH,
    SCALED_COLUMNS,
    SCALER_PATH,
)


def score_transaction(record: dict, model, scaler) -> float:
    row = pd.DataFrame([record])[FEATURE_COLUMNS]
    row[SCALED_COLUMNS] = scaler.transform(row[SCALED_COLUMNS])
    return float(model.predict_proba(row)[0, 1])


def run():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
    # One message at a time, so the forest's thread pool is pure overhead here
    # (see the note in api.py).
    model.n_jobs = 1
    scaler = joblib.load(SCALER_PATH)

    consumer = KafkaConsumer(
        KAFKA_TRANSACTIONS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="fraud-detection-consumer",
    )
    alert_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Listening on '{KAFKA_TRANSACTIONS_TOPIC}'...")
    try:
        for message in consumer:
            record = message.value
            probability = score_transaction(record, model, scaler)
            amount = record.get("Amount", 0.0)

            if probability >= FRAUD_THRESHOLD:
                alert_producer.send(KAFKA_ALERTS_TOPIC, {**record, "fraud_probability": probability})
                print(f"[ALERT] amount={amount:.2f} probability={probability:.4f}")
            else:
                print(f"[OK]    amount={amount:.2f} probability={probability:.4f}")
    finally:
        consumer.close()
        alert_producer.close()


if __name__ == "__main__":
    run()
