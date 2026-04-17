# 🔐 Intelligent Fraud Detection System for Financial Transactions

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📌 Project Overview

With the rapid growth of digital payments and online banking, financial institutions face increasing risks of fraudulent transactions. Manual rule-based systems are no longer sufficient to detect complex and evolving fraud patterns. This project focuses on building an AI-driven fraud detection system that analyzes historical transaction data to identify suspicious activities in real time.

The system leverages machine learning models and database integration to assist financial institutions in minimizing losses while ensuring secure and seamless transactions for users.

## 💡 Core Idea

The core idea is to automatically detect high-risk financial transactions by learning patterns from past transaction data. The system assigns a fraud risk score to each transaction based on features such as transaction amount, time, and behavioral patterns. Transactions with higher risk scores are flagged for further verification, enabling faster and more accurate fraud prevention.

### Key Objectives
- ✅ Detect fraudulent transactions with high accuracy
- ✅ Minimize false alarms while maximizing fraud capture
- ✅ Provide a scalable and real-world applicable solution for FinTech systems
- ✅ Real-time fraud prediction via REST API
- ✅ Persistent database storage for transactions and predictions

## 🛠️ Tech Stack

- **Programming Language:** Python 3.8+
- **Data Handling & Analysis:** Pandas, NumPy
- **Database:** SQLite
- **Machine Learning:**
  - Logistic Regression
  - Random Forest Classifier
- **Model Evaluation:** Scikit-learn (Precision, Recall, F1-Score, ROC-AUC)
- **Backend:** Flask REST API
- **Version Control:** Git & GitHub

## 📂 Project Structure

```
fraud_detection_system/
├── data/
│   └── creditcard.csv             # Transaction dataset (10,000 rows)
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── src/
│   ├── data_preprocessing.py      # Data cleaning and feature engineering
│   └── model_training.py          # ML model training and evaluation
├── database/
│   ├── create_db.py               # Database initialization
│   └── queries.py                 # SQL query helpers
├── api/
│   ├── app.py                     # Flask API application
│   └── routes.py                  # API endpoints
├── tests/
│   └── test_api.py                # API test suite
├── docs/
│   └── API_DOCUMENTATION.md       # API reference
├── requirements.txt
├── config.py
├── run_all.py                     # Full pipeline runner
└── README.md
```

## 👥 Team Responsibilities

<<<<<<< HEAD
### 👤 Team Member 1 : Dhruvil Pandya — Data & Database Management
=======
### 👤 Team Member 1 — Data & Database Management
>>>>>>> 29a82d4 (Initial commit)
- ✅ Collection and understanding of transaction dataset
- ✅ Data cleaning, preprocessing, and feature scaling
- ✅ Designing and managing SQL database for transactions
- ✅ Exploratory Data Analysis (EDA) and insights generation
- **Deliverables:** `data_preprocessing.py`, `create_db.py`, `queries.py`

<<<<<<< HEAD
### 👤 Team Member 2 : Dhyey Joshi — Machine Learning & Model Evaluation
=======
### 👤 Team Member 2 — Machine Learning & Model Evaluation
>>>>>>> 29a82d4 (Initial commit)
- ✅ Selection and implementation of ML models
- ✅ Training and tuning fraud detection models
- ✅ Model evaluation using Precision, Recall, F1-Score, ROC-AUC
- ✅ Saving trained models for API integration
- **Deliverables:** `model_training.py`, trained `.pkl` model files

<<<<<<< HEAD
### 👤 Team Member 3 : Kavy Modi — System Integration & Application Layer
=======
### 👤 Team Member 3 — System Integration & Application Layer
>>>>>>> 29a82d4 (Initial commit)
- ✅ Developing backend logic to load ML models
- ✅ Integrating database with prediction system
- ✅ Building REST API for fraud risk prediction
- ✅ End-to-end system testing and deployment setup
- **Deliverables:** `app.py`, `routes.py`, `test_api.py`, `API_DOCUMENTATION.md`

## 🚀 Quick Start

### Prerequisites
```
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the complete pipeline**
```bash
python run_all.py
```

The API will be available at `http://localhost:5000`

## 📊 Dataset

The project uses a credit card transaction dataset with the following features:

- **Time:** Seconds elapsed between this transaction and the first transaction
- **V1–V28:** Principal components from PCA transformation (anonymized features)
- **Amount:** Transaction amount in currency units
- **Class:** Target label — 0 (legitimate) or 1 (fraudulent)

The dataset contains **10,000 transactions** with a balanced class distribution, enabling reliable model training and evaluation.

## 🎯 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 86.0%    | 87.9%     | 82.9%  | 85.3%    | 0.9347  |
| Random Forest       | **94.0%**| **94.9%** | **94.0%** | **94.4%** | **0.9816** |

*Evaluated on a stratified 80/20 train-test split with random seed 42.*

## 🔌 API Usage

### Predict Fraud for a Single Transaction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406,
    "Amount": 149.62,
    "V1": -1.358354,
    "V2": -0.072781,
    "V3": 2.536347,
    ...
  }'
```

### Response
```json
{
  "fraud_probability": 0.87,
  "is_fraud": true,
  "risk_level": "HIGH",
  "timestamp": "2024-01-27T10:30:00"
}
```

See [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) for the full API reference.

## 📈 Key Features

1. **Real-time Fraud Detection** — Instant prediction API for live transactions
2. **High Accuracy** — 94%+ accuracy with strong recall for fraud detection
3. **Scalable Architecture** — Modular design for easy extension
4. **Database Integration** — Persistent storage for transactions and predictions
5. **REST API** — Clean endpoints for single and batch predictions
6. **Fully Tested** — Automated test suite for all API endpoints

## 🔄 Future Enhancements

- [ ] Deep learning models (LSTM, Autoencoders)
- [ ] Web dashboard for real-time monitoring
- [ ] PostgreSQL integration for production scale
- [ ] Docker containerization
- [ ] Real-time streaming with Apache Kafka
- [ ] Model retraining pipeline

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Authors

- **Team Member 1** — Data & Database Management
- **Team Member 2** — Machine Learning & Model Evaluation
- **Team Member 3** — System Integration & Application Layer

## 🙏 Acknowledgments

- Dataset: Credit Card Fraud Detection Dataset
- Scikit-learn Documentation
- Flask Documentation
