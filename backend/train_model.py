import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("Loading Kaggle dataset...")
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1)
y = df["Class"]

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
