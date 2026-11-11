import os

import joblib
import pytest

from backend.train_model import train


def test_training_saves_loadable_artifacts_with_sane_metrics(trained_dir):
    model_dir, metrics = trained_dir
    for name in ("random_forest_model.pkl", "logistic_regression_model.pkl", "scaler.pkl"):
        assert os.path.exists(os.path.join(model_dir, name))
    rf = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
    assert rf.n_features_in_ == 30
    assert metrics["random_forest"]["roc_auc"] > 0.85
    assert metrics["random_forest"]["accuracy"] > 0.8


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(str(tmp_path / "nope.csv"), str(tmp_path))


def test_dataset_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        train(str(bad), str(tmp_path))
