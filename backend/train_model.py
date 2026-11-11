"""Train, evaluate and save the fraud-detection models.

Usage (from the project root)::

    python backend/train_model.py
    python backend/train_model.py --data path/to/creditcard.csv --model-dir models

Outputs (in ``models/`` by default):
    random_forest_model.pkl, logistic_regression_model.pkl, scaler.pkl

Only ``Time`` and ``Amount`` are scaled; the scaler is fitted on the training
split only. SMOTE oversampling is applied to the training split only, and only
when the classes are clearly imbalanced and ``imbalanced-learn`` is installed
(the bundled dataset is ~50/50, so it is normally skipped).
"""
import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.feature_engineering import FEATURE_COLUMNS, SCALED_COLUMNS  # noqa: E402

DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")
IMBALANCE_THRESHOLD = 0.4  # apply SMOTE when the minority class is below 40%


def _maybe_oversample(X_train, y_train, random_state):
    minority_share = y_train.value_counts(normalize=True).min()
    if minority_share >= IMBALANCE_THRESHOLD:
        print(f"Classes are balanced enough (minority share {minority_share:.1%}); skipping SMOTE.")
        return X_train, y_train
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("WARNING: data is imbalanced but imbalanced-learn is not installed; "
              "training without SMOTE (pip install imbalanced-learn).")
        return X_train, y_train
    print(f"Minority share {minority_share:.1%}; applying SMOTE to the training split.")
    return SMOTE(random_state=random_state).fit_resample(X_train, y_train)


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key:<10}: {value:.4f}")
    return metrics


def train(data_path=DEFAULT_DATA_PATH, model_dir=DEFAULT_MODEL_DIR,
          n_estimators=120, random_state=42):
    """Train both models, print test-set metrics, save artifacts, return metrics."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLUMNS + ["Class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["Class"].astype(int)
    print(f"Rows: {len(df)}  |  fraud share: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[SCALED_COLUMNS] = scaler.fit_transform(X_train[SCALED_COLUMNS])
    X_test[SCALED_COLUMNS] = scaler.transform(X_test[SCALED_COLUMNS])

    X_fit, y_fit = _maybe_oversample(X_train, y_train, random_state)

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=12, random_state=random_state, n_jobs=-1
    )
    rf_model.fit(X_fit, y_fit)

    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_fit, y_fit)

    print(f"\nTest-set results ({len(X_test)} rows, stratified 80/20 split, seed {random_state}):")
    metrics = {
        "random_forest": evaluate("Random Forest", rf_model, X_test, y_test),
        "logistic_regression": evaluate("Logistic Regression", lr_model, X_test, y_test),
    }

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(model_dir, "random_forest_model.pkl"))
    joblib.dump(lr_model, os.path.join(model_dir, "logistic_regression_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"\nSaved random_forest_model.pkl, logistic_regression_model.pkl and scaler.pkl to {model_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="path to creditcard-style CSV")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="where to write .pkl files")
    args = parser.parse_args()
    try:
        train(args.data, args.model_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
