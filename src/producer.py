"""Simulate a live transaction stream by publishing rows of the raw
credit-card dataset to a Kafka topic, one transaction at a time.
"""

import json
import os
import time
from typing import Optional

import pandas as pd
from kafka import KafkaProducer

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TRANSACTIONS_TOPIC, RAW_DATA_PATH


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def stream_transactions(path=RAW_DATA_PATH, delay: float = 0.1, limit: Optional[int] = None):
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}.")

    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    producer = build_producer()
    try:
        for _, row in df.iterrows():
            producer.send(KAFKA_TRANSACTIONS_TOPIC, row.to_dict())
            time.sleep(delay)
        producer.flush()
        print(f"Streamed {len(df)} transactions to '{KAFKA_TRANSACTIONS_TOPIC}'")
    finally:
        producer.close()


if __name__ == "__main__":
    env_limit = os.environ.get("FRAUD_STREAM_LIMIT")
    env_delay = os.environ.get("FRAUD_STREAM_DELAY")
    stream_transactions(
        delay=float(env_delay) if env_delay else 0.1,
        limit=int(env_limit) if env_limit else None,
    )
