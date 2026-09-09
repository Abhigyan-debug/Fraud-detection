"""Streamlit dashboard for exploring the dataset, visualizing fraud patterns,
scoring transactions, reviewing model performance, and watching live Kafka
fraud alerts.

Run with: streamlit run src/dashboard.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from kafka import KafkaConsumer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from config import (
    FEATURE_COLUMNS,
    FRAUD_THRESHOLD,
    KAFKA_ALERTS_TOPIC,
    KAFKA_BOOTSTRAP_SERVERS,
    MODEL_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    SCALED_COLUMNS,
    SCALER_PATH,
    TARGET_COLUMN,
)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


@st.cache_data
def load_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


@st.cache_data
def get_test_predictions(_model, _scaler):
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    y_pred = _model.predict(X_test)
    y_proba = _model.predict_proba(X_test)[:, 1]
    # st.cache_data may hand back read-only numpy arrays; sklearn's metric
    # functions try to flip a writeable flag internally, so force real copies.
    return np.array(y_test, copy=True), np.array(y_pred, copy=True), np.array(y_proba, copy=True)


def score(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    scored = df.copy()
    scored[SCALED_COLUMNS] = scaler.transform(scored[SCALED_COLUMNS])
    scored["fraud_probability"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    return scored


def show_verdict(probability: float):
    if probability >= FRAUD_THRESHOLD:
        st.error(f"FRAUD suspected — probability {probability:.4f}")
    else:
        st.success(f"Looks legitimate — probability {probability:.4f}")


def poll_fraud_alerts(timeout_ms: int = 3000):
    consumer = KafkaConsumer(
        KAFKA_ALERTS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="dashboard-viewer",
        consumer_timeout_ms=timeout_ms,
    )
    try:
        return [message.value for message in consumer]
    finally:
        consumer.close()


st.title("Credit Card Fraud Detection Dashboard")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error(f"No trained model found at {MODEL_PATH}. Run `python train.py` first.")
    st.stop()

model, scaler = load_artifacts()

tab_overview, tab_visualize, tab_explore, tab_predict, tab_performance, tab_live = st.tabs(
    [
        "Overview",
        "Visualize",
        "Explore Data",
        "Score Transactions",
        "Model Performance",
        "Live Alerts",
    ]
)

with tab_overview:
    st.subheader("Dataset overview")
    if RAW_DATA_PATH.exists():
        df = load_data()
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", f"{len(df):,}")
        col2.metric("Fraud cases", f"{int(df['Class'].sum()):,}")
        col3.metric("Fraud rate", f"{df['Class'].mean() * 100:.3f}%")
        st.bar_chart(df["Class"].value_counts())
    else:
        st.warning(f"Raw dataset not found at {RAW_DATA_PATH}. Set FRAUD_RAW_DATA to override the path.")

with tab_visualize:
    st.subheader("Visual exploration")
    if not RAW_DATA_PATH.exists():
        st.warning(f"Raw dataset not found at {RAW_DATA_PATH}.")
    else:
        df = load_data()

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df["Amount"], bins=50, kde=True, ax=ax)
            ax.set_title("Transaction Amount Distribution")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Class", y="Amount", ax=ax)
            ax.set_title("Amount by Class")
            st.pyplot(fig)
            plt.close(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots()
            sns.histplot(df["Time"], bins=50, ax=ax)
            ax.set_title("Transaction Time Distribution")
            st.pyplot(fig)
            plt.close(fig)
        with col4:
            fig, ax = plt.subplots()
            sample = df.sample(min(5000, len(df)), random_state=42)
            sns.scatterplot(data=sample, x="Amount", y="Time", hue="Class", alpha=0.5, ax=ax)
            ax.set_title("Amount vs. Time (sampled)")
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Feature correlation heatmap")
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

with tab_explore:
    st.subheader("Sample transactions")
    if RAW_DATA_PATH.exists():
        df = load_data()
        n = st.slider("Rows to display", 10, 500, 50)
        st.dataframe(df.sample(n, random_state=42))
    else:
        st.warning(f"Raw dataset not found at {RAW_DATA_PATH}.")

with tab_predict:
    st.subheader("Try a single transaction")
    source = st.radio("Pick a transaction", ["Random from dataset", "Manual entry"], horizontal=True)

    if source == "Random from dataset":
        if not RAW_DATA_PATH.exists():
            st.warning(f"Raw dataset not found at {RAW_DATA_PATH}.")
        else:
            if st.button("Pick random transaction"):
                st.session_state.sample_row = load_data().sample(1).iloc[0]

            if "sample_row" in st.session_state:
                row = st.session_state.sample_row
                st.dataframe(row.to_frame().T)
                result = score(row.to_frame().T, model, scaler)
                show_verdict(float(result["fraud_probability"].iloc[0]))
    else:
        st.caption(
            "V1-V28 are anonymized PCA features from the original dataset and default to 0 "
            "(an 'average' transaction). Adjust Amount/Time to see their effect on the score."
        )
        amount = st.number_input("Amount", min_value=0.0, value=100.0)
        time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)

        if st.button("Score this transaction"):
            manual_row = {col: 0.0 for col in FEATURE_COLUMNS}
            manual_row["Amount"] = amount
            manual_row["Time"] = time_val
            result = score(pd.DataFrame([manual_row]), model, scaler)
            show_verdict(float(result["fraud_probability"].iloc[0]))

    st.divider()
    st.subheader("Batch scoring")
    st.caption("Upload a CSV containing the Time, V1-V28 and Amount columns.")

    uploaded = st.file_uploader("Upload a CSV of transactions to score", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = set(FEATURE_COLUMNS) - set(batch.columns)
        if missing:
            st.error(f"Uploaded file is missing columns: {sorted(missing)}")
        else:
            results = score(batch, model, scaler)
            results["is_fraud"] = results["fraud_probability"] >= FRAUD_THRESHOLD
            st.dataframe(results[[*FEATURE_COLUMNS, "fraud_probability", "is_fraud"]])
            st.download_button(
                "Download scored results",
                results.to_csv(index=False).encode("utf-8"),
                "scored_transactions.csv",
                "text/csv",
            )

with tab_performance:
    st.subheader("Model performance (held-out test set)")
    test_path = PROCESSED_DIR / "test.csv"
    if not test_path.exists():
        st.warning(f"{test_path} not found. Run `python train.py` first to generate it.")
    else:
        y_test, y_pred, y_proba = get_test_predictions(model, scaler)

        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
        col2.metric("PR-AUC", f"{average_precision_score(y_test, y_proba):.4f}")
        col3.metric("Fraud recall", f"{recall_score(y_test, y_pred):.4f}")

        report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        col4, col5, col6 = st.columns(3)
        with col4:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Legit", "Fraud"]).plot(
                ax=ax, cmap="Blues", colorbar=False
            )
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close(fig)
        with col5:
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            ax.set_title("ROC Curve")
            st.pyplot(fig)
            plt.close(fig)
        with col6:
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
            ax.set_title("Precision-Recall Curve")
            st.pyplot(fig)
            plt.close(fig)

with tab_live:
    st.subheader("Live fraud alerts")
    st.caption(
        f"Reads from the '{KAFKA_ALERTS_TOPIC}' Kafka topic at {KAFKA_BOOTSTRAP_SERVERS}. "
        "Start the producer/consumer (see README) to generate live alerts."
    )

    if "fraud_alerts" not in st.session_state:
        st.session_state.fraud_alerts = []

    if st.button("Check for new alerts"):
        try:
            new_alerts = poll_fraud_alerts()
            st.session_state.fraud_alerts = new_alerts + st.session_state.fraud_alerts
            if new_alerts:
                st.success(f"Found {len(new_alerts)} new alert(s)")
            else:
                st.info("No new alerts.")
        except Exception as exc:
            st.error(f"Could not reach Kafka at {KAFKA_BOOTSTRAP_SERVERS}: {exc}")

    if st.session_state.fraud_alerts:
        st.dataframe(pd.DataFrame(st.session_state.fraud_alerts))
    else:
        st.info("No alerts collected yet. Click 'Check for new alerts' once the consumer has flagged something.")
