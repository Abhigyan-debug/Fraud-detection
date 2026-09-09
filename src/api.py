"""FastAPI service that scores individual transactions for fraud risk.

Run with: uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from config import FEATURE_COLUMNS, FRAUD_THRESHOLD, MODEL_PATH, SCALED_COLUMNS, SCALER_PATH

_state: dict = {}


def _build_transaction_model():
    fields = {
        col: (float, Field(...))
        for col in FEATURE_COLUMNS
        if col not in SCALED_COLUMNS
    }
    fields["Time"] = (float, Field(..., ge=0, description="Seconds since the first transaction"))
    fields["Amount"] = (float, Field(..., ge=0, description="Transaction amount"))
    return create_model("Transaction", **fields)


Transaction = _build_transaction_model()


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError(
            f"Model or scaler not found ({MODEL_PATH}, {SCALER_PATH}). "
            "Run `python train.py` first."
        )
    _state["model"] = joblib.load(MODEL_PATH)
    _state["scaler"] = joblib.load(SCALER_PATH)
    yield
    _state.clear()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model = _state.get("model")
    scaler = _state.get("scaler")
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([transaction.model_dump()])[FEATURE_COLUMNS]
    row[SCALED_COLUMNS] = scaler.transform(row[SCALED_COLUMNS])

    probability = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(is_fraud=probability >= FRAUD_THRESHOLD, fraud_probability=probability)
