import sqlite3
import pandas as pd

DB = "../database/fraud_system.db"
CSV = "../data/creditcard.csv"

conn = sqlite3.connect(DB)
cursor = conn.cursor()

print("Loading dataset...")
df = pd.read_csv(CSV)

# rename to system format
df = df.rename(columns={
    "Time":"transaction_time",
    "Amount":"amount",
    "Class":"fraud_label"
})

# create customers automatically
df["customer_id"] = (df.index % 200) + 1
df["location"] = "India"
df["device"] = "mobile"
df["is_international"] = 0

transactions = df[[
    "customer_id","amount","transaction_time",
    "location","device","is_international"
]]

transactions.to_sql("transactions", conn, if_exists="append", index=False)

print("Transactions inserted:", len(transactions))
conn.close()
