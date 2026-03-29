# =============================================================================
# Intelligent Fraud Detection System — Member 2: ML & Model Evaluation
# =============================================================================
# Dataset  : creditcard.csv  (10,000 transactions, 31 features)
# Models   : Logistic Regression, Random Forest Classifier
# Note     : Dataset is ~50/50 balanced — SMOTE not required
# Output   : Trained models saved as .pkl files + full evaluation dashboard
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

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

# Auto-detect imbalance — use SMOTE only if minority class < 20%
USE_SMOTE = fraud_pct < 20
print(f"    Class balance  : {'IMBALANCED — SMOTE will be applied' if USE_SMOTE else 'BALANCED — no oversampling needed'}")

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
print(f"    Training samples : {X_train.shape[0]}  (Fraud: {y_train.sum()}, Normal: {(y_train==0).sum()})")
print(f"    Testing  samples : {X_test.shape[0]}   (Fraud: {y_test.sum()}, Normal: {(y_test==0).sum()})")

# ── 5. OPTIONAL SMOTE ─────────────────────────────────────────────────────────

if USE_SMOTE:
    print("\n[4] Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"    After SMOTE — Fraud: {y_train.sum()}, Normal: {(y_train==0).sum()}")
else:
    print("\n[4] SMOTE skipped — dataset is already balanced")

# ── 6. TRAIN MODELS ───────────────────────────────────────────────────────────

print("\n[5] Training models...")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("    Logistic Regression trained")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("    Random Forest trained")

# ── 7. PREDICTIONS ────────────────────────────────────────────────────────────

lr_pred       = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

rf_pred       = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# ── 8. EVALUATION ─────────────────────────────────────────────────────────────

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

# ── 9. COMPARISON TABLE ───────────────────────────────────────────────────────

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

best      = "Random Forest" if rf_f1 >= lr_f1 else "Logistic Regression"
best_key  = "random_forest" if rf_f1 >= lr_f1 else "logistic_regression"
print(f"\n  Best model by F1-Score: {best}")

# ── 10. VISUALIZATIONS ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Fraud Detection — Model Evaluation Dashboard", fontsize=15, fontweight="bold")

# Confusion Matrix — LR
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"], ax=axes[0][0])
axes[0][0].set_title("Confusion Matrix\nLogistic Regression")
axes[0][0].set_ylabel("Actual"); axes[0][0].set_xlabel("Predicted")

# Confusion Matrix — RF
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"], ax=axes[0][1])
axes[0][1].set_title("Confusion Matrix\nRandom Forest")
axes[0][1].set_ylabel("Actual"); axes[0][1].set_xlabel("Predicted")

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
axes[0][2].plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC={lr_roc:.3f})", color="steelblue")
axes[0][2].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_roc:.3f})", color="seagreen")
axes[0][2].plot([0,1],[0,1],"k--",alpha=0.4)
axes[0][2].set_title("ROC Curves"); axes[0][2].legend()
axes[0][2].set_xlabel("False Positive Rate"); axes[0][2].set_ylabel("True Positive Rate")

# Precision-Recall Curves
pre_lr, rec_lr, _ = precision_recall_curve(y_test, lr_pred_proba)
pre_rf, rec_rf, _ = precision_recall_curve(y_test, rf_pred_proba)
axes[1][0].plot(rec_lr, pre_lr, label="Logistic Reg",  color="steelblue")
axes[1][0].plot(rec_rf, pre_rf, label="Random Forest", color="seagreen")
axes[1][0].set_title("Precision-Recall Curves"); axes[1][0].legend()
axes[1][0].set_xlabel("Recall"); axes[1][0].set_ylabel("Precision")

# Feature Importances — RF top 15
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp.nlargest(15).sort_values().plot(kind="barh", ax=axes[1][1], color="seagreen")
axes[1][1].set_title("Top 15 Feature Importances\n(Random Forest)")
axes[1][1].set_xlabel("Importance Score")

# Metric Bar Comparison
metrics = ["ROC-AUC", "Precision", "Recall", "F1-Score"]
lr_vals = [lr_roc, lr_pr, lr_re, lr_f1]
rf_vals = [rf_roc, rf_pr, rf_re, rf_f1]
x = np.arange(len(metrics)); w = 0.35
axes[1][2].bar(x - w/2, lr_vals, w, label="Logistic Reg",  color="steelblue")
axes[1][2].bar(x + w/2, rf_vals, w, label="Random Forest", color="seagreen")
axes[1][2].set_xticks(x); axes[1][2].set_xticklabels(metrics)
axes[1][2].set_ylim(0, 1.15); axes[1][2].set_title("Metric Comparison")
axes[1][2].set_ylabel("Score"); axes[1][2].legend()
for i, (lv, rv) in enumerate(zip(lr_vals, rf_vals)):
    axes[1][2].text(i - w/2, lv + 0.02, f"{lv:.2f}", ha="center", fontsize=8)
    axes[1][2].text(i + w/2, rv + 0.02, f"{rv:.2f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("evaluation_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[6] Evaluation dashboard saved -> evaluation_dashboard.png")

# ── 11. SAVE MODELS ───────────────────────────────────────────────────────────

with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("[7] Models saved:")
print("    -> logistic_regression_model.pkl")
print("    -> random_forest_model.pkl")
print("    -> scaler.pkl  (required for Member 3's API)")

# ── 12. HANDOFF NOTES FOR MEMBER 3 ───────────────────────────────────────────

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
