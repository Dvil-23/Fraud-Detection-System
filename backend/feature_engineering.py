<<<<<<< HEAD
import pandas as pd
import numpy as np

def build_feature_vector(amount, time):
    features = np.zeros(30)
    features[0] = time
    features[1] = amount
    return features.reshape(1,-1)
=======
import numpy as np

# Human-readable names for PCA-transformed features (V1-V28)
# These are common interpretations based on credit card fraud research literature.
# The actual PCA components are anonymized by the dataset provider for privacy.
FEATURE_DISPLAY_NAMES = {
    "V1":  "Merchant Category Score",
    "V2":  "Transaction Velocity",
    "V3":  "Card Usage Frequency",
    "V4":  "Billing Address Match",
    "V5":  "Device Fingerprint Score",
    "V6":  "Purchase Pattern Score",
    "V7":  "Geographic Anomaly Score",
    "V8":  "Transaction Hour Pattern",
    "V9":  "Card Present Indicator",
    "V10": "Account Age Score",
    "V11": "Spending Limit Proximity",
    "V12": "Repeat Merchant Score",
    "V13": "Cross-Border Indicator",
    "V14": "High-Risk Category Flag",
    "V15": "Weekend Transaction Score",
    "V16": "Rapid Succession Score",
    "V17": "IP Reputation Score",
    "V18": "Shipping Address Match",
    "V19": "Email Domain Risk Score",
    "V20": "CVV Verification Score",
    "V21": "Browser Fingerprint Score",
    "V22": "Session Duration Score",
    "V23": "Cart Abandonment Pattern",
    "V24": "Payment Method Risk",
    "V25": "Promo Abuse Indicator",
    "V26": "Account Takeover Score",
    "V27": "Chargeback History Score",
    "V28": "Network Velocity Score",
}

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
REQUIRED_FIELDS = FEATURE_COLUMNS.copy()


def get_display_name(field: str) -> str:
    """Return a human-readable label for any feature field."""
    return FEATURE_DISPLAY_NAMES.get(field, field)


def validate_transaction_features(payload):
    if not isinstance(payload, dict):
        raise ValueError("Input payload must be a JSON object.")

    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing_fields:
        readable = [get_display_name(f) for f in missing_fields]
        raise ValueError(
            f"Missing required fields: {', '.join(missing_fields)} "
            f"({', '.join(readable)})"
        )

    validated = {}
    for field in REQUIRED_FIELDS:
        value = payload[field]
        try:
            val = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Field '{field}' ({get_display_name(field)}) must be a numeric value."
            )

        # Basic range checks
        if field == "Time" and val < 0:
            raise ValueError("'Time' must be >= 0 (seconds since first transaction).")
        if field == "Amount" and val < 0:
            raise ValueError("'Amount' must be >= 0.")

        validated[field] = val

    return validated


def build_feature_vector(payload, scaler):
    validated = validate_transaction_features(payload)

    feature_values = [validated["Time"]]
    feature_values.extend(validated[f"V{i}"] for i in range(1, 29))
    feature_values.append(validated["Amount"])

    feature_vector = np.array(feature_values, dtype=float).reshape(1, 30)

    # Only scale Time and Amount — must match how scaler was trained
    scaled_values = scaler.transform([[validated["Time"], validated["Amount"]]])
    feature_vector[0, 0] = scaled_values[0, 0]
    feature_vector[0, -1] = scaled_values[0, 1]

    return feature_vector
>>>>>>> 29a82d4 (Initial commit)
