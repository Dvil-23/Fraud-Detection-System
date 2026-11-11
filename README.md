# Intelligent Fraud Detection System

A small end-to-end fraud-detection project: it trains machine-learning models on credit-card-style
transaction data, serves predictions through a **Flask REST API** that logs every request to a
**SQLite** database, and includes a **Streamlit** app for trying the model interactively.

## What it does

- Trains a **Random Forest** and a **Logistic Regression** classifier on `data/creditcard.csv`
  and evaluates them on a held-out test set (`backend/train_model.py`).
- `POST /predict` takes the 30 transaction features, returns a fraud probability, a fraud/not-fraud
  decision and a `LOW` / `MEDIUM` / `HIGH` risk level, and stores the transaction, prediction and
  (for predicted fraud) an alert in SQLite.
- The Streamlit app (`streamlit_app.py`) has a **Manual Prediction** page that runs the real trained
  model, plus Dashboard / Transactions / Alerts / Analytics pages that show **simulated demo data**
  (random transactions, not scored by the model and not read from the database).

### Dataset

`data/creditcard.csv` has 10,000 rows: `Time`, `V1`–`V28` (anonymised PCA components), `Amount` and
the label `Class` (1 = fraud). It is roughly 50/50 (5,002 fraud / 4,998 legitimate), unlike the
well-known full Kaggle dataset (≈284k rows, <1 % fraud). Because the data is balanced, the trainer
skips SMOTE; SMOTE is applied automatically only if you train on clearly imbalanced data and
`imbalanced-learn` is installed.

### Model performance (measured)

Stratified 80/20 split, seed 42, 2,000 test rows, produced by `python backend/train_model.py`
with scikit-learn 1.9.1 (numbers can shift slightly with other library versions):

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Logistic Regression | 85.8 %   | 87.9 %    | 82.9 % | 85.3 %| 0.9348  |
| Random Forest       | 94.1 %   | 94.9 %    | 93.2 % | 94.1 %| 0.9802  |

The API and the Streamlit app use the Random Forest.

## Project layout

```
.
├── api/app.py                      Flask API (/, /health, /stats, /predict)
├── backend/
│   ├── feature_engineering.py      Feature order, validation, scaling, risk levels (shared)
│   └── train_model.py              Train + evaluate + save models
├── data/
│   ├── creditcard.csv              Dataset
│   └── import_kaggle_data.py       Load the CSV into the SQLite `transactions` table
├── models/                         *.pkl files written by train_model.py
├── database/fraud_system.db        SQLite file (created by init_db.py)
├── schema.sql                      Database schema (single source of truth)
├── init_db.py                      Create the database from schema.sql
├── streamlit_app.py                Streamlit UI
├── tests/                          pytest suite
├── requirements.txt / requirements-dev.txt / pytest.ini
```

## Prerequisites

- Python 3.10+ (developed and tested on **Python 3.12**; other versions are untested)
- `pip`
- About 10 MB of free disk space for the models and database

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # use requirements-dev.txt instead if you want to run the tests
```

## Setup and running

Run everything from the project root.

```bash
# 1. Train the models (writes models/*.pkl). Takes a few seconds.
python backend/train_model.py

# 2. Create the database (safe to re-run; add --reset to wipe and recreate it)
python init_db.py

# 3. Optional: load the 10,000 dataset rows into the transactions table
python data/import_kaggle_data.py          # add --replace to clear and re-import

# 4a. Start the API  ->  http://127.0.0.1:5000
python api/app.py

# 4b. Start the Streamlit app  ->  http://localhost:8501
streamlit run streamlit_app.py
```

Steps 2 and 3 are optional for the API: it creates an empty database automatically on first use.
Without step 3, `/stats` simply shows zero historical transactions. The trained `.pkl` files must
exist (step 1), otherwise `/predict` returns `503` and the Streamlit app shows a "train the models"
message. Re-train after upgrading scikit-learn, because pickled models are tied to the version that
created them.

## Configuration

| Setting | How | Default |
|---|---|---|
| API host | env `API_HOST` | `127.0.0.1` (local only) |
| API port | env `API_PORT` | `5000` |
| Flask debug mode | env `FLASK_DEBUG=1` | off (never enable on a public interface) |
| Training data / output folder | `python backend/train_model.py --data FILE --model-dir DIR` | `data/creditcard.csv`, `models/` |
| Risk thresholds | `MEDIUM_RISK_THRESHOLD`, `HIGH_RISK_THRESHOLD` in `backend/feature_engineering.py` | 0.3 / 0.7 |

Example: `API_HOST=0.0.0.0 API_PORT=8000 python api/app.py` (this exposes the API to your network;
it has no authentication).

## API usage

All 30 features are required: `Time`, `V1`…`V28`, `Amount` (numbers, finite, `Time` and `Amount` ≥ 0).

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "Time": 406, "Amount": 149.62,
  "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0,
  "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0,
  "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0
}'
```

Response (`200`) for the request above (probability varies with the input and the trained model;
`transaction_id` is 10001 if you imported the dataset first):

```json
{"transaction_id": 10001, "fraud_probability": 0.366595, "is_fraud": false, "risk_level": "MEDIUM"}
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Route list and the required fields |
| `GET /health` | `ok` / `degraded`, whether the model, scaler and database are available |
| `GET /stats` | Totals from the database. Fraud figures (`predictions_made`, `fraud_detected`, `fraud_rate_percent`) count only transactions scored through the API |
| `POST /predict` | Score one transaction. `400` for invalid input, `503` if the model is not loaded |

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite (39 tests) covers feature validation and scaling, risk levels, training, database
creation and import, every API endpoint (using a temporary database, so your real one is untouched),
and a headless smoke test of all Streamlit pages. The Streamlit tests are skipped if `models/*.pkl`
do not exist yet, so run the training step first.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtualenv and run `pip install -r requirements.txt` |
| `/predict` returns `503` / Streamlit says "Model files could not be loaded" | Run `python backend/train_model.py`. `/health` shows the underlying error in `model_load_error` |
| `InconsistentVersionWarning` or odd predictions after upgrading packages | Models were saved by a different scikit-learn version: re-run `python backend/train_model.py` |
| `Address already in use` | Pick another port: `API_PORT=5001 python api/app.py` or `streamlit run streamlit_app.py --server.port 8502` |
| `The transactions table already has N rows` | The import already ran. Use `python data/import_kaggle_data.py --replace` to start over |
| `no such column` / `no such table` errors from an old `fraud_system.db` | The file was created with an older schema: `python init_db.py --reset` (deletes its data), then re-import |
| API unreachable from another machine | It binds to `127.0.0.1` by default; set `API_HOST=0.0.0.0` knowing it has no authentication |

## Known limitations

- The Streamlit Dashboard, Transactions, Alerts and Analytics pages use simulated data; the "Review"
  and "Block" buttons on the Alerts page are placeholders and do nothing.
- The API has no authentication or rate limiting and uses Flask's development server; it is not a
  production deployment.
- Only Python 3.12 on Linux was tested. Windows, macOS and other Python versions are untested.
- Models were trained and evaluated only on the bundled 10,000-row balanced sample. Performance on
  real, heavily imbalanced data has not been measured, and the SMOTE path has not been exercised.

## Team

Dhruvil Pandya (data & database), Dhyey Joshi (machine learning & evaluation), Kavy Modi (API &
integration).

## License

MIT, see [LICENSE](LICENSE).
