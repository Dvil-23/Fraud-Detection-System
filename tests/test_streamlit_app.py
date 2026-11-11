"""Headless smoke test of every page of the Streamlit app (uses the models in models/)."""
import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "streamlit_app.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "models", "random_forest_model.pkl")),
    reason="run `python backend/train_model.py` first",
)

FAILURE_MARKERS = ("could not be loaded", "Prediction failed")


def failure_messages(at):
    """Error boxes that signal a real failure (the Alerts page uses st.error for fraud alerts)."""
    return [e.value for e in at.error if any(m in e.value for m in FAILURE_MARKERS)]


PAGES = ["Dashboard", "Transactions", "Alerts", "Manual Prediction", "Analytics"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not failure_messages(at), failure_messages(at)


def test_manual_prediction_after_verification():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Manual Prediction").run()
    at.text_input[0].set_value(at.session_state.captcha_code)
    next(b for b in at.button if b.label == "Verify Code").click().run()
    assert at.session_state.verified is True
    next(b for b in at.button if b.label == "Predict Fraud").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not failure_messages(at), failure_messages(at)
    assert any("Risk Level" in i.value for i in at.info)
