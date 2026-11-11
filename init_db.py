"""Create the SQLite database from schema.sql.

Usage (from anywhere)::

    python init_db.py            # create tables that don't exist yet (safe to re-run)
    python init_db.py --reset    # delete the database file and recreate it (DESTROYS DATA)

``schema.sql`` is the single source of truth for the database layout.
"""
import argparse
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "fraud_system.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")


def init_db(db_path=DB_PATH, schema_path=SCHEMA_PATH, reset=False):
    """Create the database (and parent folder) from the schema; return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if reset and os.path.exists(db_path):
        os.remove(db_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        conn.commit()
    finally:
        conn.close()
    return db_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the fraud-detection SQLite database.")
    parser.add_argument("--reset", action="store_true",
                        help="delete the existing database file first (destroys all data)")
    args = parser.parse_args()
    path = init_db(reset=args.reset)
    print("Database ready at:", path)
