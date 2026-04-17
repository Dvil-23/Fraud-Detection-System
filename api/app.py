<<<<<<< HEAD
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
=======
import os
import sqlite3

import joblib  # FIX: use joblib consistently (train_model.py uses joblib to save)
from flask import Flask, jsonify, request
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    build_feature_vector,
    validate_transaction_features,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DB_PATH = os.path.join(DATABASE_DIR, "fraud_system.db")
MODEL_VERSION = "random_forest_model.pkl"

app = Flask(__name__)


# ── FIX: Wrap model loading in try/except so the Flask server starts
# even when .pkl files are missing, and /health reports the real status.
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODEL_LOAD_ERROR = None
except FileNotFoundError as e:
    model = None
    scaler = None
    MODEL_LOAD_ERROR = str(e)


def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def classify_risk(probability):
    if probability < 0.3:
        return "LOW"
    if probability < 0.7:
        return "MEDIUM"
    return "HIGH"


def insert_transaction(conn, validated_payload, device="API_CLIENT"):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO transactions (
            customer_id, amount, transaction_time, location, device, is_international
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            None,
            float(validated_payload["Amount"]),
            int(validated_payload["Time"]),
            "UNKNOWN",
            device,
            0,
        ),
    )
    return cursor.lastrowid


def insert_prediction(conn, transaction_id, fraud_probability, predicted_label):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            transaction_id, fraud_probability, predicted_label, model_version
        ) VALUES (?, ?, ?, ?)
        """,
        (transaction_id, float(fraud_probability), int(predicted_label), MODEL_VERSION),
    )


def insert_alert(conn, transaction_id, risk_level):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO alerts (transaction_id, risk_level, status)
        VALUES (?, ?, ?)
        """,
        (transaction_id, risk_level, "OPEN"),
    )


def generate_prediction(validated_payload):
    feature_vector = build_feature_vector(validated_payload, scaler)
    probability = float(model.predict_proba(feature_vector)[0][1])
    predicted_label = int(model.predict(feature_vector)[0])
    is_fraud = bool(predicted_label)
    risk_level = classify_risk(probability)

    return {
        "fraud_probability": round(probability, 6),
        "predicted_label": predicted_label,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }


def process_prediction(validated_payload, device="API_CLIENT"):
    prediction = generate_prediction(validated_payload)

    conn = get_db_connection()
    try:
        transaction_id = insert_transaction(conn, validated_payload, device=device)
        insert_prediction(
            conn, transaction_id,
            prediction["fraud_probability"], prediction["predicted_label"]
        )
        if prediction["is_fraud"]:
            insert_alert(conn, transaction_id, prediction["risk_level"])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "transaction_id": transaction_id,
        "fraud_probability": prediction["fraud_probability"],
        "is_fraud": prediction["is_fraud"],
        "risk_level": prediction["risk_level"],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    # Build a human-readable field reference
    field_reference = {col: FEATURE_DISPLAY_NAMES.get(col, col) for col in FEATURE_COLUMNS}
    return jsonify({
        "message": "Fraud Detection API is running.",
        "available_routes": {
            "GET /health": "Check API health and model status",
            "GET /stats":  "Fraud statistics from the database",
            "POST /predict": "Predict fraud from JSON input",
        },
        "required_fields": field_reference,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "database_configured": os.path.exists(DB_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
        "expected_features": FEATURE_COLUMNS,
    }), 200


@app.route("/stats", methods=["GET"])
def stats():
    """Return fraud statistics aggregated from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM transactions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 1")
        fraud_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT risk_level, COUNT(*) as cnt FROM alerts GROUP BY risk_level"
        )
        risk_breakdown = {row["risk_level"]: row["cnt"] for row in cursor.fetchall()}
        conn.close()

        fraud_rate = round(fraud_count / total * 100, 2) if total > 0 else 0.0
        return jsonify({
            "total_transactions": total,
            "fraud_detected": fraud_count,
            "legitimate": total - fraud_count,
            "fraud_rate_percent": fraud_rate,
            "alert_breakdown": risk_breakdown,
        }), 200
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except sqlite3.Error as exc:
        return jsonify({"error": f"Database error: {str(exc)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first.",
            "detail": MODEL_LOAD_ERROR,
        }), 503

    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        validated_payload = validate_transaction_features(payload)
        result = process_prediction(validated_payload, device="API_CLIENT")
        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except sqlite3.Error as exc:
        return jsonify({"error": f"Database error: {str(exc)}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Internal server error: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
>>>>>>> 29a82d4 (Initial commit)
