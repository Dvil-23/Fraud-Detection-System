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
