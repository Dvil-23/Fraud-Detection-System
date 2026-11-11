"""Shared feature handling for the fraud-detection model.

The model expects 30 numeric features in this exact order:
``Time, V1 ... V28, Amount``. Only ``Time`` and ``Amount`` are scaled (with the
``StandardScaler`` saved by ``backend/train_model.py``); V1-V28 are left as-is
because the dataset already provides them as PCA components.

Both the Flask API and the Streamlit app use this module so that they build
features, scale them and classify risk in exactly the same way.
"""
import math

import pandas as pd

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_FIELDS = FEATURE_COLUMNS.copy()
SCALED_COLUMNS = ["Time", "Amount"]

# V1-V28 are anonymised PCA components supplied by the dataset; they have no
# published real-world meaning, so they are described as such.
FEATURE_DISPLAY_NAMES = {
    "Time": "Seconds elapsed since the first transaction",
    "Amount": "Transaction amount",
    **{f"V{i}": f"Anonymised PCA component {i}" for i in range(1, 29)},
}

# Probability thresholds used to turn a fraud probability into a risk label.
MEDIUM_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.7


def get_display_name(field: str) -> str:
    """Return a short description for a feature field."""
    return FEATURE_DISPLAY_NAMES.get(field, field)


def classify_risk(probability: float) -> str:
    """Map a fraud probability to LOW / MEDIUM / HIGH."""
    if probability < MEDIUM_RISK_THRESHOLD:
        return "LOW"
    if probability < HIGH_RISK_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def validate_transaction_features(payload):
    """Validate a raw payload and return ``{field: float}`` for all 30 features.

    Raises ``ValueError`` with a user-readable message when the payload is not a
    JSON object, has missing fields, or contains non-numeric / non-finite values.
    """
    if not isinstance(payload, dict):
        raise ValueError("Input payload must be a JSON object.")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing_fields:
        raise ValueError("Missing required fields: " + ", ".join(missing_fields))

    validated = {}
    for field in REQUIRED_FIELDS:
        value = payload[field]
        if isinstance(value, bool):
            raise ValueError(f"Field '{field}' must be a numeric value.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{field}' must be a numeric value.") from None

        if not math.isfinite(number):
            raise ValueError(f"Field '{field}' must be a finite number.")
        if field == "Time" and number < 0:
            raise ValueError("'Time' must be >= 0 (seconds since first transaction).")
        if field == "Amount" and number < 0:
            raise ValueError("'Amount' must be >= 0.")

        validated[field] = number

    return validated


def build_feature_vector(payload, scaler):
    """Return a one-row DataFrame, ordered and scaled the way the model was trained."""
    validated = validate_transaction_features(payload)
    row = pd.DataFrame([[validated[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    row[SCALED_COLUMNS] = scaler.transform(row[SCALED_COLUMNS])
    return row
