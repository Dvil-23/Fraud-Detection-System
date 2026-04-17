import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import random
from datetime import datetime, timedelta

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
    model_paths = [
        "models/random_forest_model.pkl",
        "models/randomforestmodel.pkl",
        "random_forest_model.pkl",
        "randomforestmodel.pkl",
        "models/logistic_regression_model.pkl",
        "models/logisticregressionmodel.pkl"
    ]

    scaler_paths = [
        "models/scaler.pkl",
        "scaler.pkl"
    ]

    model = None
    scaler = None
    used_model_path = None
    used_scaler_path = None

    for path in model_paths:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            used_model_path = path
            break
        except:
            pass

    for path in scaler_paths:
        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            used_scaler_path = path
            break
        except:
            pass

    if model is None or scaler is None:
        st.error("❌ Model files not found.")
        st.info("Keep files like this:")
        st.code(
"""fraud_detection_system/
├── streamlit_app.py
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl""")
        st.info("Or run this from project root:")
        st.code("python src/model_training.py")
        st.stop()

    return model, scaler, used_model_path, used_scaler_path


def generate_transactions(n=500):
    np.random.seed(42)
    random.seed(42)
    now = datetime.now()
    transactions = []

    for idx in range(n):
        tx_time = idx * 120 + random.randint(0, 60)
        amount = round(np.random.lognormal(mean=4.0, sigma=1.0), 2)
        v1 = np.random.normal(0, 1.5)
        v2 = np.random.normal(0, 1.2)

        extra_features = [float(np.random.normal(0, 1)) for _ in range(26)]

        risk_seed = (
            (amount > 800) or
            (v1 < -3.0) or
            (abs(v2) > 2.5) or
            (random.random() < 0.04)
        )
        is_fraud = 1 if risk_seed else 0

        transactions.append({
            'Transaction_ID': f'TXN{idx + 1}',
            'Time': float(tx_time),
            'Amount': float(amount),
            'V1': float(v1),
            'V2': float(v2),
            **{f'V{i}': v for i, v in enumerate(extra_features, 3)},
            'Is_Fraud': int(is_fraud),
            'Timestamp': now - timedelta(seconds=tx_time)
        })

    return pd.DataFrame(transactions)


def predict_fraud(model, scaler, transaction):
    feature_row = [transaction["Time"]]

    for i in range(1, 29):
        feature_row.append(transaction[f"V{i}"])

    feature_row.append(transaction["Amount"])

    row = pd.DataFrame([feature_row], columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0][1])
    else:
        pred = int(model.predict(row)[0])
        prob = 1.0 if pred == 1 else 0.0

    pred = 1 if prob >= 0.5 else 0
    return prob, pred


model, scaler, model_path, scaler_path = load_models()

if "transactions" not in st.session_state:
    st.session_state.transactions = generate_transactions()

if "captcha_code" not in st.session_state:
    st.session_state.captcha_code = str(random.randint(1000, 9999))

if "verified" not in st.session_state:
    st.session_state.verified = False


st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Transactions", "Alerts", "Manual Prediction", "Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {model_path}")  # local path shown to help debugging
st.sidebar.caption(f"Scaler: {scaler_path}")  # local path shown to help debugging


if page == "Dashboard":
    st.title("🛡️ Fraud Detection Dashboard")
    st.caption("Real-time fraud monitoring view based on your trained model setup.[file:1][file:2]")

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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)


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
        df[display_cols].sort_values("Timestamp", ascending=False).head(rows_to_show),
        use_container_width=True
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
    st.caption("The model expects features in the order Time, V1-V28, Amount, and only Time and Amount are scaled.[file:203]")

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
                st.session_state.captcha_code = str(random.randint(1000, 9999))
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

                    risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"
                    st.info(f"Risk Level: {risk}")

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
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bins = pd.cut(df["Amount"], bins=10)
        grp = df.groupby(bins)["Is_Fraud"].mean().reset_index()
        grp["AmountRange"] = grp["Amount"].astype(str)
        fig = px.bar(
            grp,
            x="AmountRange",
            y="Is_Fraud",
            title="Fraud Rate by Amount Range"
        )
        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.caption(
    "Project uses Random Forest as the best model in the provided project documents, "
    "with feature order Time, V1-V28, Amount and StandardScaler applied to Time and Amount.[file:1][file:203]"
)