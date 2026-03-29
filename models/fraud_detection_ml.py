# =============================================================================
# Intelligent Fraud Detection System — Member 2: ML & Model Evaluation
# =============================================================================
# Dataset  : creditcard.csv  (10,000 transactions, 31 features, 50/50 balanced)
# Models   : Logistic Regression, Random Forest Classifier
# Output   : Trained models saved as .pkl files
# =============================================================================

import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score
)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  FRAUD DETECTION — ML & MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv("creditcard.csv")

fraud_count  = df["Class"].sum()
normal_count = (df["Class"] == 0).sum()
fraud_pct    = df["Class"].mean() * 100

print(f"\n[1] Dataset loaded : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"    Fraud cases    : {fraud_count} ({fraud_pct:.2f}%)")
print(f"    Normal cases   : {normal_count}")

# ── 2. FEATURE / TARGET SPLIT ─────────────────────────────────────────────────

X = df.drop("Class", axis=1)
y = df["Class"]

# ── 3. SCALE Time & Amount (V1-V28 are already PCA-scaled) ───────────────────

scaler = StandardScaler()
X = X.copy()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])
print("\n[2] Scaled 'Time' and 'Amount' with StandardScaler")

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Train/Test split (80/20, stratified):")
print(f"    Training : {X_train.shape[0]}  (Fraud: {y_train.sum()}, Normal: {(y_train==0).sum()})")
print(f"    Testing  : {X_test.shape[0]}   (Fraud: {y_test.sum()}, Normal: {(y_test==0).sum()})")

# ── 5. TRAIN MODELS ───────────────────────────────────────────────────────────

print("\n[4] Training models...")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("    Logistic Regression trained")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("    Random Forest trained")

# ── 6. PREDICTIONS ────────────────────────────────────────────────────────────

lr_pred       = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

rf_pred       = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# ── 7. EVALUATION ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)

def print_metrics(name, y_true, y_pred, y_proba):
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))
    roc = roc_auc_score(y_true, y_proba)
    f1  = f1_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred)
    re  = recall_score(y_true, y_pred)
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  Precision : {pr:.4f}")
    print(f"  Recall    : {re:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return roc, f1, pr, re

lr_roc, lr_f1, lr_pr, lr_re = print_metrics("Logistic Regression",     y_test, lr_pred, lr_pred_proba)
rf_roc, rf_f1, rf_pr, rf_re = print_metrics("Random Forest Classifier", y_test, rf_pred, rf_pred_proba)

# ── 8. COMPARISON SUMMARY ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Metric':<20} {'Logistic Reg':>15} {'Random Forest':>15}")
print(f"  {'-'*50}")
for label, lv, rv in [("ROC-AUC",   lr_roc, rf_roc),
                       ("Precision", lr_pr,  rf_pr),
                       ("Recall",    lr_re,  rf_re),
                       ("F1-Score",  lr_f1,  rf_f1)]:
    print(f"  {label:<20} {lv:>15.4f} {rv:>15.4f}")

best     = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
best_key = "random_forest" if rf_f1 >= lr_f1 else "logistic_regression"
print(f"\n  Best model by F1-Score: {best}")

# ── 9. SAVE MODELS ────────────────────────────────────────────────────────────

with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n[5] Models saved:")
print("    -> logistic_regression_model.pkl")
print("    -> random_forest_model.pkl")
print("    -> scaler.pkl  (required for Member 3's API)")

# ── 10. HANDOFF NOTES FOR MEMBER 3 ────────────────────────────────────────────

print("\n" + "=" * 60)
print("  HANDOFF NOTES FOR MEMBER 3")
print("=" * 60)
print(f"""
  Best model : {best}

  Load in your Flask backend:

      import pickle, pandas as pd

      with open("{best_key}_model.pkl", "rb") as f:
          model = pickle.load(f)
      with open("scaler.pkl", "rb") as f:
          scaler = pickle.load(f)

      def predict_fraud(transaction: dict) -> dict:
          row = pd.DataFrame([transaction])
          row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])
          prob  = model.predict_proba(row)[0][1]
          label = int(model.predict(row)[0])
          return {{"fraud_probability": round(prob, 4), "is_fraud": bool(label)}}
""")

print("=" * 60)
print("  Done. All outputs ready for submission.")
print("=" * 60)
