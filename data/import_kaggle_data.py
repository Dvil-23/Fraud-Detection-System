"""Load data/creditcard.csv into the ``transactions`` table of the SQLite database.

Usage (from anywhere)::

    python data/import_kaggle_data.py             # refuses to run if transactions already exist
    python data/import_kaggle_data.py --replace   # clear transactions/predictions/alerts/features first

Each CSV row becomes one transaction. ``customer_id`` (1-200), ``location``,
``device`` and ``is_international`` are placeholder values because the dataset
does not contain them. The ``Class`` label is not stored in the database.
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from init_db import DB_PATH, init_db  # noqa: E402

CSV_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")


def import_transactions(csv_path=CSV_PATH, db_path=DB_PATH, replace=False):
    """Import the CSV and return the number of rows inserted."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    init_db(db_path)  # creates tables if needed (no-op for existing ones)
    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    try:
        existing = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        if existing and not replace:
            raise RuntimeError(
                f"The transactions table already has {existing} rows. "
                "Re-run with --replace to clear it first."
            )
        if replace:
            for table in ("alerts", "predictions", "features", "transactions"):
                conn.execute(f"DELETE FROM {table}")

        rows = [
            ((i % 200) + 1, float(amount), int(t), "India", "mobile", 0)
            for i, (t, amount) in enumerate(zip(df["Time"], df["Amount"]))
        ]
        conn.executemany(
            "INSERT INTO transactions "
            "(customer_id, amount, transaction_time, location, device, is_international) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import creditcard.csv into the database.")
    parser.add_argument("--replace", action="store_true",
                        help="delete existing transactions (and dependent rows) before importing")
    args = parser.parse_args()
    try:
        count = import_transactions(replace=args.replace)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Transactions inserted: {count}")
