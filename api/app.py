import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "backend"))

from flask import Flask, request, jsonify
import sqlite3
import joblib
from feature_engineering import build_feature_vector

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DB_PATH = os.path.join(BASE_DIR, "database", "fraud_system.db")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    amount = float(data["amount"])
    time = int(data["time"])
    customer_id = int(data["customer_id"])
    location = data.get("location", "India")
    device = data.get("device", "mobile")
    is_international = int(data.get("is_international", 0))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # store transaction
    cursor.execute("""
        INSERT INTO transactions (customer_id, amount, transaction_time, location, device, is_international)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (customer_id, amount, time, location, device, is_international))

    txn_id = cursor.lastrowid

    # ML prediction
    X = build_feature_vector(amount, time)
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0][1])
    label = int(prob >= 0.5)

    # store prediction
    cursor.execute("""
        INSERT INTO predictions (transaction_id, fraud_probability, predicted_label, model_version)
        VALUES (?, ?, ?, ?)
    """, (txn_id, prob, label, "RF_v1"))

    # alert if fraud
    if label == 1:
        cursor.execute("""
            INSERT INTO alerts (transaction_id, risk_level, status)
            VALUES (?, ?, ?)
        """, (txn_id, "HIGH", "OPEN"))

    conn.commit()
    conn.close()

    return jsonify({
        "transaction_id": txn_id,
        "fraud_probability": prob,
        "fraud": bool(label)
    })
if __name__ == "__main__":
    app.run(debug=True)
