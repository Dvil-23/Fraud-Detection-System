# Start here

See [README.md](README.md) for full instructions. Quick start, from the project root:

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python backend/train_model.py       # train and save the models
python init_db.py                   # create the SQLite database
python api/app.py                   # API at http://127.0.0.1:5000
streamlit run streamlit_app.py      # UI at http://localhost:8501 (separate terminal)
```

Tests: `pip install -r requirements-dev.txt && python -m pytest`
