import os

import pandas as pd
import pytest

from backend.train_model import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "creditcard.csv")


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    """A 1,500-row sample of the bundled dataset, so training in tests is fast."""
    df = pd.read_csv(DATA_PATH).sample(1500, random_state=0)
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def trained_dir(small_dataset, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    metrics = train(small_dataset, str(model_dir), n_estimators=20)
    return str(model_dir), metrics


@pytest.fixture()
def sample_rows():
    df = pd.read_csv(DATA_PATH)
    fraud = df[df.Class == 1].iloc[0].drop("Class").to_dict()
    legit = df[df.Class == 0].iloc[0].drop("Class").to_dict()
    return fraud, legit


@pytest.fixture()
def client(trained_dir, tmp_path, monkeypatch):
    """Flask test client wired to a temporary DB and a freshly trained model."""
    from api import app as api_app

    model_dir, _ = trained_dir
    model, scaler, err = api_app.load_artifacts(
        os.path.join(model_dir, "random_forest_model.pkl"),
        os.path.join(model_dir, "scaler.pkl"),
    )
    assert err is None
    monkeypatch.setattr(api_app, "model", model)
    monkeypatch.setattr(api_app, "scaler", scaler)
    monkeypatch.setattr(api_app, "MODEL_LOAD_ERROR", None)
    monkeypatch.setattr(api_app, "DB_PATH", str(tmp_path / "test.db"))
    api_app.app.config["TESTING"] = True
    return api_app.app.test_client()
