import os

import joblib
import pandas as pd
import pytest

from backend.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_vector,
    classify_risk,
    validate_transaction_features,
)


def full_payload(**overrides):
    payload = {c: 0.0 for c in FEATURE_COLUMNS}
    payload.update(overrides)
    return payload


def test_feature_order_is_time_v1_v28_amount():
    assert len(FEATURE_COLUMNS) == 30
    assert FEATURE_COLUMNS[0] == "Time" and FEATURE_COLUMNS[-1] == "Amount"
    assert FEATURE_COLUMNS[1:-1] == [f"V{i}" for i in range(1, 29)]


def test_valid_payload_is_converted_to_floats():
    out = validate_transaction_features(full_payload(Amount="12.5", Time=3))
    assert out["Amount"] == 12.5 and out["Time"] == 3.0


@pytest.mark.parametrize("payload", [None, [], "x", 5])
def test_non_object_payload_rejected(payload):
    with pytest.raises(ValueError, match="JSON object"):
        validate_transaction_features(payload)


def test_missing_fields_reported():
    payload = full_payload()
    del payload["V3"]
    with pytest.raises(ValueError, match="V3"):
        validate_transaction_features(payload)


@pytest.mark.parametrize("bad", ["abc", None, [1], True, float("nan"), float("inf")])
def test_non_numeric_or_non_finite_rejected(bad):
    with pytest.raises(ValueError):
        validate_transaction_features(full_payload(V1=bad))


def test_negative_time_and_amount_rejected():
    with pytest.raises(ValueError):
        validate_transaction_features(full_payload(Time=-1))
    with pytest.raises(ValueError):
        validate_transaction_features(full_payload(Amount=-0.01))


@pytest.mark.parametrize("p,level", [(0.0, "LOW"), (0.29, "LOW"), (0.3, "MEDIUM"),
                                     (0.69, "MEDIUM"), (0.7, "HIGH"), (1.0, "HIGH")])
def test_classify_risk(p, level):
    assert classify_risk(p) == level


def test_build_feature_vector_scales_only_time_and_amount(trained_dir):
    model_dir, _ = trained_dir
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    row = build_feature_vector(full_payload(Time=100.0, Amount=50.0, V5=1.25), scaler)
    assert list(row.columns) == FEATURE_COLUMNS and row.shape == (1, 30)
    assert row.loc[0, "V5"] == 1.25  # untouched
    expected = scaler.transform(pd.DataFrame([[100.0, 50.0]], columns=["Time", "Amount"]))[0]
    assert row.loc[0, "Time"] == pytest.approx(expected[0])
    assert row.loc[0, "Amount"] == pytest.approx(expected[1])
