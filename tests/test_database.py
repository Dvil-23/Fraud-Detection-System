import os
import sqlite3
import sys

import pytest

from init_db import init_db

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import import_kaggle_data as importer  # noqa: E402

TABLES = {"customers", "transactions", "features", "predictions", "alerts"}


def count(path, table="transactions"):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_init_db_creates_tables_is_rerunnable_and_resettable(tmp_path):
    path = str(tmp_path / "sub" / "db.sqlite")
    init_db(path)
    conn = sqlite3.connect(path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.execute("INSERT INTO transactions (amount, transaction_time) VALUES (1, 1)")
    conn.commit()
    conn.close()
    assert TABLES <= names

    init_db(path)  # must not wipe data
    assert count(path) == 1
    init_db(path, reset=True)  # reset wipes it
    assert count(path) == 0


def test_import_is_guarded_against_duplicates(tmp_path, small_dataset):
    db = str(tmp_path / "db.sqlite")
    assert importer.import_transactions(small_dataset, db) == 1500
    with pytest.raises(RuntimeError, match="--replace"):
        importer.import_transactions(small_dataset, db)
    assert count(db) == 1500
    assert importer.import_transactions(small_dataset, db, replace=True) == 1500
    assert count(db) == 1500
