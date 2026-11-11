import sqlite3

from api import app as api_app


def test_home_and_health(client):
    assert client.get("/").status_code == 200
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["model_loaded"] and body["scaler_loaded"]
    assert len(body["expected_features"]) == 30


def test_predict_logs_prediction_and_alert(client, sample_rows):
    fraud, _ = sample_rows
    resp = client.post("/predict", json=fraud)
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"transaction_id", "fraud_probability", "is_fraud", "risk_level"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    conn = sqlite3.connect(api_app.DB_PATH)
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
    alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    conn.close()
    assert alerts == (1 if body["is_fraud"] else 0)


def test_fraud_scores_higher_than_legit(client, sample_rows):
    fraud, legit = sample_rows
    p_fraud = client.post("/predict", json=fraud).get_json()["fraud_probability"]
    p_legit = client.post("/predict", json=legit).get_json()["fraud_probability"]
    assert p_fraud > p_legit


def test_invalid_requests_return_400_and_write_nothing(client, sample_rows):
    _, legit = sample_rows
    missing = {k: v for k, v in legit.items() if k != "V7"}
    assert client.post("/predict", json=missing).status_code == 400
    assert client.post("/predict", json={**legit, "Amount": "abc"}).status_code == 400
    assert client.post("/predict", json={**legit, "Amount": -1}).status_code == 400
    assert client.post("/predict", json=[1, 2]).status_code == 400
    assert client.post("/predict", data="hello", content_type="text/plain").status_code == 400
    assert client.get("/stats").get_json()["total_transactions"] == 0


def test_stats_counts_scored_transactions(client, sample_rows):
    fraud, legit = sample_rows
    client.post("/predict", json=fraud)
    client.post("/predict", json=legit)
    stats = client.get("/stats").get_json()
    assert stats["predictions_made"] == 2
    assert stats["fraud_detected"] + stats["legitimate"] == 2
    assert 0 <= stats["fraud_rate_percent"] <= 100


def test_predict_without_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(api_app, "model", None)
    assert client.post("/predict", json={}).status_code == 503
    assert client.get("/health").get_json()["status"] == "degraded"


def test_load_artifacts_reports_errors_instead_of_crashing(tmp_path):
    model, scaler, err = api_app.load_artifacts(str(tmp_path / "x.pkl"), str(tmp_path / "y.pkl"))
    assert model is None and scaler is None and err
