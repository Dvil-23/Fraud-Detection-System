import os
import secrets
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.feature_engineering import build_feature_vector, classify_risk  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: #0f172a;
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: #111827;
}
h1, h2, h3 {
    color: #60a5fa;
}
div[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 12px;
    border-radius: 12px;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
}
.stButton > button:hover {
    background: #1d4ed8;
}
.block-container {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load the trained model and scaler (paths are relative to this file, not the CWD)."""
    model_candidates = [
        "random_forest_model.pkl",
        "logistic_regression_model.pkl",
    ]
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    model = scaler = used_model_path = None
    errors = []

    for name in model_candidates:
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            model = joblib.load(path)
            used_model_path = path
            break
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            errors.append(f"scaler.pkl: {type(exc).__name__}: {exc}")

    if model is None or scaler is None:
        st.error("❌ Model files could not be loaded.")
        for err in errors:
            st.warning(err)
        st.info("Train the models from the project root, then reload this page:")
        st.code("python backend/train_model.py")
        st.stop()

    return model, scaler, used_model_path, scaler_path


def generate_transactions(n=500):
    """Build SIMULATED demo transactions for the dashboard pages.

    These rows are random, and ``Is_Fraud`` comes from simple rules plus noise -
    it is not produced by the trained model. Use "Manual Prediction" to score a
    transaction with the real model.
    """
    rng = np.random.default_rng(42)
    now = datetime.now()
    transactions = []

    for idx in range(n):
        tx_time = idx * 120 + int(rng.integers(0, 61))
        amount = round(float(rng.lognormal(mean=4.0, sigma=1.0)), 2)
        v1 = float(rng.normal(0, 1.5))
        v2 = float(rng.normal(0, 1.2))
        extra_features = rng.normal(0, 1, size=26)

        risk_seed = (
            (amount > 800)
            or (v1 < -3.0)
            or (abs(v2) > 2.5)
            or (rng.random() < 0.04)
        )

        transactions.append({
            "Transaction_ID": f"TXN{idx + 1}",
            "Time": float(tx_time),
            "Amount": float(amount),
            "V1": v1,
            "V2": v2,
            **{f"V{i}": float(v) for i, v in enumerate(extra_features, 3)},
            "Is_Fraud": int(risk_seed),
            "Timestamp": now - timedelta(seconds=tx_time),
        })

    return pd.DataFrame(transactions)


def predict_fraud(model, scaler, transaction):
    """Score one transaction dict with the trained model; returns (probability, label)."""
    row = build_feature_vector(transaction, scaler)
    prob = float(model.predict_proba(row)[0][1])
    return prob, int(prob >= 0.5)


def new_captcha_code():
    return str(secrets.randbelow(9000) + 1000)


model, scaler, model_path, scaler_path = load_models()

if "transactions" not in st.session_state:
    st.session_state.transactions = generate_transactions()

if "captcha_code" not in st.session_state:
    st.session_state.captcha_code = new_captcha_code()

if "verified" not in st.session_state:
    st.session_state.verified = False


st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Transactions", "Alerts", "Manual Prediction", "Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {os.path.relpath(model_path, BASE_DIR)}")
st.sidebar.caption(f"Scaler: {os.path.relpath(scaler_path, BASE_DIR)}")


if page == "Dashboard":
    st.title("🛡️ Fraud Detection Dashboard")
    st.caption("Demo dashboard over simulated transactions (not live data and not scored by the model). "
               "Use Manual Prediction to run the trained model.")

    df = st.session_state.transactions
    total = len(df)
    frauds = int(df["Is_Fraud"].sum())
    legit = total - frauds
    fraud_rate = (frauds / total) * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total)
    c2.metric("Fraud Cases", frauds)
    c3.metric("Legitimate", legit)
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        pie_df = pd.DataFrame({
            "Type": ["Legitimate", "Fraud"],
            "Count": [legit, frauds]
        })
        fig = px.pie(
            pie_df,
            names="Type",
            values="Count",
            color="Type",
            color_discrete_map={"Legitimate": "#22c55e", "Fraud": "#ef4444"},
            title="Transaction Split"
        )
        st.plotly_chart(fig)

    with col2:
        chart_df = df.copy()
        chart_df["HourBucket"] = (chart_df["Time"] // 3600).astype(int)
        grp = chart_df.groupby("HourBucket")["Is_Fraud"].mean().reset_index()
        fig = px.line(
            grp,
            x="HourBucket",
            y="Is_Fraud",
            markers=True,
            title="Fraud Rate by Hour Bucket"
        )
        st.plotly_chart(fig)


elif page == "Transactions":
    st.title("📄 Transactions")

    df = st.session_state.transactions.copy()

    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Filter", ["All", "Fraud", "Legitimate"])
    with col2:
        rows_to_show = st.selectbox("Rows", [20, 50, 100, 200], index=1)

    if filter_type == "Fraud":
        df = df[df["Is_Fraud"] == 1]
    elif filter_type == "Legitimate":
        df = df[df["Is_Fraud"] == 0]

    display_cols = ["Transaction_ID", "Timestamp", "Time", "Amount", "V1", "V2", "Is_Fraud"]
    st.dataframe(
        df[display_cols].sort_values("Timestamp", ascending=False).head(rows_to_show)
    )


elif page == "Alerts":
    st.title("🚨 Fraud Alerts")

    df = st.session_state.transactions.copy()
    alerts = df[df["Is_Fraud"] == 1].sort_values("Timestamp", ascending=False).head(20)

    if alerts.empty:
        st.success("No fraud alerts found.")
    else:
        for _, row in alerts.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.error(
                    f"{row['Transaction_ID']} | Amount: ₹{row['Amount']:.2f} | "
                    f"Time: {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                c2.button("Review", key=f"review_{row['Transaction_ID']}")
                c3.button("Block", key=f"block_{row['Transaction_ID']}")


elif page == "Manual Prediction":
    st.title("🔍 Manual Prediction")
    st.caption("The model expects features in the order Time, V1-V28, Amount; only Time and Amount are scaled.")

    if not st.session_state.verified:
        st.warning("Verification required before prediction.")

        st.markdown(
            f"""
            <div style="
                font-size:42px;
                font-weight:800;
                letter-spacing:8px;
                color:#f8fafc;
                background:#1e293b;
                border:2px dashed #60a5fa;
                border-radius:12px;
                padding:18px;
                text-align:center;
                width:240px;">
                {st.session_state.captcha_code}
            </div>
            """,
            unsafe_allow_html=True
        )

        entered_code = st.text_input("Enter verification code")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Verify Code"):
                if entered_code.strip() == st.session_state.captcha_code:
                    st.session_state.verified = True
                    st.success("Verification successful.")
                    st.rerun()
                else:
                    st.error("Wrong verification code.")

        with col2:
            if st.button("Refresh Code"):
                st.session_state.captcha_code = new_captcha_code()
                st.rerun()

    else:
        st.success("Verified successfully.")

        with st.form("manual_prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                time_val = st.number_input("Time", value=406.0)
                amount_val = st.number_input("Amount", value=149.62)

            feature_inputs = {}

            with col2:
                for i in range(1, 15):
                    feature_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"input_v{i}")

            with col3:
                for i in range(15, 29):
                    feature_inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"input_v{i}")

            submitted = st.form_submit_button("Predict Fraud")

            if submitted:
                transaction = {
                    "Time": float(time_val),
                    "Amount": float(amount_val)
                }

                for i in range(1, 29):
                    transaction[f"V{i}"] = float(feature_inputs[f"V{i}"])

                try:
                    prob, pred = predict_fraud(model, scaler, transaction)

                    if pred == 1:
                        st.error(f"🚨 Fraud Detected | Probability: {prob:.4f}")
                    else:
                        st.success(f"✅ Legitimate Transaction | Probability: {prob:.4f}")

                    st.info(f"Risk Level: {classify_risk(prob)}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


elif page == "Analytics":
    st.title("📊 Analytics")

    df = st.session_state.transactions.copy()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df,
            x="Amount",
            y="V1",
            color=df["Is_Fraud"].astype(str),
            title="Amount vs V1",
            color_discrete_map={"0": "#22c55e", "1": "#ef4444"}
        )
        st.plotly_chart(fig)

    with col2:
        bins = pd.cut(df["Amount"], bins=10)
        grp = df.groupby(bins, observed=True)["Is_Fraud"].mean().reset_index()
        grp["AmountRange"] = grp["Amount"].astype(str)
        fig = px.bar(
            grp,
            x="AmountRange",
            y="Is_Fraud",
            title="Fraud Rate by Amount Range"
        )
        st.plotly_chart(fig)


st.markdown("---")
st.caption(
    "Predictions use the Random Forest model with feature order Time, V1-V28, Amount "
    "and a StandardScaler applied to Time and Amount. Dashboard, Transactions, Alerts and "
    "Analytics pages show simulated demo data."
)