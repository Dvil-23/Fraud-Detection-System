<<<<<<< HEAD
=======
import os
>>>>>>> 29a82d4 (Initial commit)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
from imblearn.over_sampling import SMOTE

print("Loading Kaggle dataset...")
import os
=======
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

print("Loading dataset...")
>>>>>>> 29a82d4 (Initial commit)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1)
y = df["Class"]

<<<<<<< HEAD
# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# handle imbalance
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# model
model = RandomForestClassifier(n_estimators=120, max_depth=12)
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

# save
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))


print("Model saved")
=======
# ── FIX: Only scale Time and Amount (columns 0 and 29).
# feature_engineering.py only applies the scaler to these two columns,
# so training must match exactly — otherwise predictions will be wrong.
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ── Model 1: Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
rf_preds = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=["Legitimate", "Fraud"]))

# ── Model 2: Logistic Regression (promised in README/API docs)
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_res, y_train_res)
lr_preds = lr_model.predict(X_test)
print(f"Logistic Regression Accuracy: {lr_model.score(X_test, y_test):.4f}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_preds, target_names=["Legitimate", "Fraud"]))

# ── Save models
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# FIX: Save as "random_forest_model.pkl" — this is what app.py loads.
joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("\n✅ All models and scaler saved to /models/")
print("   - random_forest_model.pkl")
print("   - logistic_regression_model.pkl")
print("   - scaler.pkl")
>>>>>>> 29a82d4 (Initial commit)
