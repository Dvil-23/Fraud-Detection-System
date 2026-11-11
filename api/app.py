"""Flask REST API for fraud prediction.

Run from the project root::

    python api/app.py              # http://127.0.0.1:5000
    API_HOST=0.0.0.0 API_PORT=8000 python api/app.py

Environment variables: API_HOST (default 127.0.0.1), API_PORT (default 5000),
FLASK_DEBUG=1 to enable Flask's debug mode (never use it on a public interface).
"""
import logging
import os
import sqlite3
import sys

import joblib
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_DISPLAY_NAMES,
    build_feature_vector,
    classify_risk,
    validate_transaction_features,
)
from init_db import init_db  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DB_PATH = os.path.join(BASE_DIR, "database", "fraud_system.db")
MODEL_VERSION = "random_forest_model.pkl"

logger = logging.getLogger(__name__)
app = Flask(__name__)


def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load (model, scaler, error). On any failure returns (None, None, message)."""
    try:
        return joblib.load(model_path), joblib.load(scaler_path), None
    except Exception as exc:  # missing file, corrupt pickle, version mismatch ...
        logger.error("Could not load model artifacts: %s", exc)
        return None, None, f"{type(exc).__name__}: {exc}"


# The server starts even if the models are missing; /health and /predict report it.
model, scaler, MODEL_LOAD_ERROR = load_artifacts()


def get_db_connection():
    """Open the database, creating it from schema.sql first if it does not exist."""
    if not os.path.exists(DB_PATH):
        init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
    conn.execute(
        """
        INSERT INTO predictions (
            transaction_id, fraud_probability, predicted_label, model_version
        ) VALUES (?, ?, ?, ?)
        """,
        (transaction_id, float(fraud_probability), int(predicted_label), MODEL_VERSION),
    )


def insert_alert(conn, transaction_id, risk_level):
    conn.execute(
        "INSERT INTO alerts (transaction_id, risk_level, status) VALUES (?, ?, ?)",
        (transaction_id, risk_level, "OPEN"),
    )


def generate_prediction(validated_payload):
    feature_vector = build_feature_vector(validated_payload, scaler)
    probability = float(model.predict_proba(feature_vector)[0][1])
    predicted_label = int(model.predict(feature_vector)[0])
    return {
        "fraud_probability": round(probability, 6),
        "predicted_label": predicted_label,
        "is_fraud": bool(predicted_label),
        "risk_level": classify_risk(probability),
    }


def process_prediction(validated_payload, device="API_CLIENT"):
    prediction = generate_prediction(validated_payload)

    conn = get_db_connection()
    try:
        transaction_id = insert_transaction(conn, validated_payload, device=device)
        insert_prediction(
            conn, transaction_id, prediction["fraud_probability"], prediction["predicted_label"]
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


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fraud Detection API is running.",
        "available_routes": {
            "GET /health": "Check API health and model status",
            "GET /stats": "Prediction statistics from the database",
            "POST /predict": "Predict fraud from a JSON body with all 30 features",
        },
        "required_fields": {col: FEATURE_DISPLAY_NAMES.get(col, col) for col in FEATURE_COLUMNS},
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model is not None and scaler is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "database_configured": os.path.exists(DB_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
        "expected_features": FEATURE_COLUMNS,
    }), 200


@app.route("/stats", methods=["GET"])
def stats():
    """Counts from the database. Fraud figures are based on scored transactions only."""
    try:
        conn = get_db_connection()
        try:
            total_transactions = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            scored = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            fraud_count = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE predicted_label = 1"
            ).fetchone()[0]
            risk_breakdown = {
                row["risk_level"]: row["cnt"]
                for row in conn.execute(
                    "SELECT risk_level, COUNT(*) AS cnt FROM alerts GROUP BY risk_level"
                )
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.exception("Database error in /stats")
        return jsonify({"error": f"Database error: {exc}"}), 500

    return jsonify({
        "total_transactions": total_transactions,
        "predictions_made": scored,
        "fraud_detected": fraud_count,
        "legitimate": scored - fraud_count,
        "fraud_rate_percent": round(fraud_count / scored * 100, 2) if scored else 0.0,
        "alert_breakdown": risk_breakdown,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({
            "error": "Model not loaded. Run `python backend/train_model.py` first.",
            "detail": MODEL_LOAD_ERROR,
        }), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        validated_payload = validate_transaction_features(payload)
        return jsonify(process_prediction(validated_payload, device="API_CLIENT")), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except sqlite3.Error as exc:
        logger.exception("Database error in /predict")
        return jsonify({"error": f"Database error: {exc}"}), 500
    except Exception:
        logger.exception("Unexpected error in /predict")
        return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
