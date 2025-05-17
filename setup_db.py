import sqlite3
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Load CSV data
stocks_df = pd.read_csv('data/stocks.csv')
mutual_funds_df = pd.read_csv('data/mutual_funds.csv')

# Connect to SQLite
conn = sqlite3.connect('data/finance.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS stocks (
    name TEXT,
    volatility REAL,
    expected_return REAL,
    previous_year_return REAL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS mutual_funds (
    name TEXT,
    risk INTEGER,
    return REAL,
    previous_year_return REAL
)
''')

# Insert data
stocks_df.to_sql('stocks', conn, if_exists='replace', index=False)
mutual_funds_df.to_sql('mutual_funds', conn, if_exists='replace', index=False)

conn.commit()
conn.close()
print("Database setup complete.")
