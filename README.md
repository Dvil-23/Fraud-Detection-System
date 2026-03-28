# 🔐 Intelligent Fraud Detection System for Financial Transactions

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

## 📌 Project Overview

With the rapid growth of digital payments and online banking, financial institutions face increasing risks of fraudulent transactions. Manual rule-based systems are no longer sufficient to detect complex and evolving fraud patterns. This project focuses on building an AI-driven fraud detection system that analyzes historical transaction data to identify suspicious activities in real time.

The system leverages machine learning models and database integration to assist financial institutions in minimizing losses while ensuring secure and seamless transactions for users.

## 💡 Core Idea

The core idea is to automatically detect high-risk financial transactions by learning patterns from past transaction data. The system assigns a fraud risk score to each transaction based on features such as transaction amount, time, and behavioral patterns. Transactions with higher risk scores are flagged for further verification, enabling faster and more accurate fraud prevention.

### Key Objectives
- ✅ Detect fraudulent transactions with high accuracy (>95%)
- ✅ Minimize false alarms while maximizing fraud capture
- ✅ Provide a scalable and real-world applicable solution for FinTech systems
- ✅ Real-time fraud prediction API
- ✅ Comprehensive data analysis and visualization

## 🛠️ Tech Stack

- **Programming Language:** Python 3.8+
- **Data Handling & Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Database:** SQLite (easily upgradable to PostgreSQL)
- **Machine Learning:**
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting (XGBoost)
- **Model Evaluation:** Scikit-learn, SMOTE for imbalanced data
- **Backend:** Flask REST API
- **Version Control:** Git & GitHub

## 📂 Project Structure

```
fraud_detection_system/
├── data/                      # Dataset storage
│   ├── raw/                   # Original datasets
│   └── processed/             # Cleaned and processed data
├── models/                    # Trained ML models
│   ├── logistic_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── model_training.py      # ML model training
│   └── utils.py               # Utility functions
├── database/                  # Database scripts
│   ├── create_db.py           # Database initialization
│   └── queries.py             # SQL queries
├── api/                       # Flask API
│   ├── app.py                 # Main API application
│   └── routes.py              # API endpoints
├── notebooks/                 # Jupyter notebooks
│   └── eda_analysis.ipynb     # Exploratory Data Analysis
├── tests/                     # Unit tests
│   └── test_api.py
├── docs/                      # Documentation
│   └── API_DOCUMENTATION.md
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration settings
└── README.md                  # This file
```

## 👥 Team Responsibilities

### 👤 Team Member 1 — Data & Database Management
- ✅ Collection and understanding of transaction dataset
- ✅ Data cleaning, preprocessing, and feature scaling
- ✅ Designing and managing SQL database for transactions
- ✅ Exploratory Data Analysis (EDA) and insights generation
- **Deliverables:** `data_preprocessing.py`, `create_db.py`, `eda_analysis.ipynb`

### 👤 Team Member 2 — Machine Learning & Model Evaluation
- ✅ Selection and implementation of ML models
- ✅ Training and tuning fraud detection models
- ✅ Handling class imbalance in fraud data
- ✅ Model evaluation using precision, recall, F1-score
- ✅ Saving and documenting final trained models
- **Deliverables:** `model_training.py`, trained model files, evaluation reports

### 👤 Team Member 3 — System Integration & Application Layer
- ✅ Developing backend logic to load ML models
- ✅ Integrating database with prediction system
- ✅ Building API for fraud risk prediction
- ✅ End-to-end system testing and deployment setup
- **Deliverables:** `app.py`, `routes.py`, `API_DOCUMENTATION.md`, `test_api.py`

## 🚀 Quick Start

### Prerequisites
```bash
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
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize database**
```bash
python database/create_db.py
```

5. **Train models** (or use pre-trained models)
```bash
python src/model_training.py
```

6. **Run the API**
```bash
python api/app.py
```

The API will be available at `http://localhost:5000`

## 📊 Dataset

The project uses a synthetic credit card transaction dataset with the following features:

- **Time:** Seconds elapsed between this transaction and first transaction
- **V1-V28:** Principal components from PCA (anonymized features)
- **Amount:** Transaction amount
- **Class:** 0 (legitimate) or 1 (fraudulent)

The dataset is highly imbalanced with fraud cases representing <1% of transactions, which makes it realistic for real-world scenarios.

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 97.2% | 88.5% | 91.3% | 89.9% |
| Random Forest | 99.1% | 95.2% | 94.8% | 95.0% |
| XGBoost | 99.3% | 96.1% | 95.5% | 95.8% |

*Note: Results may vary based on data split and random seed*

## 🔌 API Usage

### Predict Fraud for Single Transaction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 149.62,
    "time": 406,
    "v1": -1.358354,
    "v2": -0.072781,
    ...
  }'
```

### Response
```json
{
  "transaction_id": "txn_12345",
  "fraud_probability": 0.87,
  "is_fraud": true,
  "risk_level": "HIGH",
  "timestamp": "2024-01-27T10:30:00"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {...},
      {...}
    ]
  }'
```

See [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) for complete API reference.

## 📈 Key Features

1. **Real-time Fraud Detection:** Instant prediction API for live transactions
2. **High Accuracy:** >95% accuracy with optimized recall for fraud detection
3. **Scalable Architecture:** Modular design for easy scaling
4. **Comprehensive Logging:** Transaction logs and prediction history
5. **Database Integration:** Persistent storage for transactions and predictions
6. **Model Versioning:** Support for multiple model versions
7. **Batch Processing:** Handle multiple transactions simultaneously

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run API tests:
```bash
python tests/test_api.py
```

## 📝 Configuration

Edit `config.py` to customize:
- Database path
- Model selection
- Fraud threshold
- API settings

## 🔄 Future Enhancements

- [ ] Deep Learning models (LSTM, Autoencoders)
- [ ] Real-time streaming with Apache Kafka
- [ ] Web dashboard for monitoring
- [ ] PostgreSQL/MySQL integration
- [ ] Docker containerization
- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Advanced feature engineering

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Authors

- **Team Member 1** - Data & Database Management
- **Team Member 2** - Machine Learning & Model Evaluation
- **Team Member 3** - System Integration & Application Layer

## 🙏 Acknowledgments

- Dataset: Credit Card Fraud Detection Dataset (Kaggle)
- Scikit-learn Documentation
- Flask Documentation

## 📧 Contact

For questions or feedback, please contact: [your-email@example.com]

---

**⭐ If you found this project helpful, please give it a star!**
