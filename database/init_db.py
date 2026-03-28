import sqlite3

db_path = "database/fraud_system.db"

conn = sqlite3.connect(db_path)

with open("schema.sql", "r") as f:
    conn.executescript(f.read())

conn.close()

print("Database created at:", db_path)
