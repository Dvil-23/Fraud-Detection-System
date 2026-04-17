<<<<<<< HEAD
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "fraud_system.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "schema.sql")

# make sure database folder exists
os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

conn = sqlite3.connect(DB_PATH)

with open(SCHEMA_PATH, "r") as f:
    conn.executescript(f.read())

conn.close()

print("Database created successfully at:", DB_PATH)
=======
"""
init_db.py
Run this ONCE before starting the Flask app to create the database and tables.

Usage:
    python init_db.py
"""
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "fraud_system.db")

os.makedirs(DATABASE_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ── Table 1: transactions ────────────────────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id      TEXT,
    amount           REAL    NOT NULL,
    transaction_time INTEGER NOT NULL,
    location         TEXT    DEFAULT 'UNKNOWN',
    device           TEXT    DEFAULT 'API_CLIENT',
    is_international INTEGER DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ── Table 2: predictions ─────────────────────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id   INTEGER NOT NULL REFERENCES transactions(id),
    fraud_probability REAL   NOT NULL,
    predicted_label  INTEGER NOT NULL,   -- 0 = legitimate, 1 = fraud
    model_version    TEXT    NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ── Table 3: alerts ──────────────────────────────────────────────────────────
cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL REFERENCES transactions(id),
    risk_level     TEXT    NOT NULL,   -- LOW / MEDIUM / HIGH
    status         TEXT    NOT NULL DEFAULT 'OPEN',  -- OPEN / RESOLVED
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("✅ Database initialised at:", DB_PATH)
print("   Tables created: transactions, predictions, alerts")
>>>>>>> 29a82d4 (Initial commit)
