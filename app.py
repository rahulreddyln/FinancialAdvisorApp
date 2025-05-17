from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline

app = Flask(__name__)
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_investment_goal(prompt):
    labels = ["growth", "balanced", "conservative"]
    try:
        result = classifier(prompt, candidate_labels=labels)
        return result["labels"][0]
    except Exception as e:
        print("LLM fallback:", e)
        return "balanced"

def get_db_connection():
    conn = sqlite3.connect(os.path.join(data_dir, 'finance.db'))
    conn.row_factory = sqlite3.Row
    return conn

def load_data_from_db():
    conn = get_db_connection()
    stocks_df = pd.read_sql_query("SELECT * FROM stocks", conn)
    mutual_funds_df = pd.read_sql_query("SELECT * FROM mutual_funds", conn)
    conn.close()
    return stocks_df, mutual_funds_df

# Load data and model
stocks_df, mutual_funds_df = load_data_from_db()
model_path = os.path.join(data_dir, 'stock_model.pkl')

if not os.path.exists(model_path):
    X = stocks_df[['volatility']]
    y = stocks_df['expected_return']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
else:
    model = joblib.load(model_path)

def recommend_stocks(df, model, top_n=3):
    df = df.copy()
    df['predicted_return'] = model.predict(df[['volatility']])
    df = df.sort_values(by='predicted_return', ascending=False)
    return df[['name', 'predicted_return', 'volatility', 'previous_year_return']].head(top_n)

def recommend_mutual_funds(df, risk_level='low', age=None, income=None, savings=None):
    df = df.copy()
    if age is not None:
        if age < 35:
            risk_level = 'high'
        elif age < 50:
            risk_level = 'medium'
        else:
            risk_level = 'low'

    if risk_level == 'low':
        df = df[df['risk'] <= 3]
    elif risk_level == 'medium':
        df = df[(df['risk'] > 3) & (df['risk'] <= 6)]
    else:
        df = df[df['risk'] > 6]

    df = df.sort_values(by='return', ascending=False)
    return df[['name', 'risk', 'return', 'previous_year_return']].head(3)

def retirement_plan(age, income, budget, risk_level):
    retirement_age = 60
    years_left = retirement_age - age
    target = income * 0.7 * 20
    monthly_savings = target / (years_left * 12)
    inflation_factor = 1.03 if risk_level == 'low' else 1.05 if risk_level == 'medium' else 1.07
    inflation_adjusted = monthly_savings * inflation_factor
    return round(inflation_adjusted, 2)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = int(request.form['age'])
        income = float(request.form['income'])
        budget = float(request.form['budget'])
        risk = request.form['risk']
        goal_text = request.form['goal']

        goal_type = classify_investment_goal(goal_text)

        stock_recs = recommend_stocks(stocks_df.copy(), model)
        fund_recs = recommend_mutual_funds(mutual_funds_df.copy(), risk_level=risk, age=age, income=income, savings=budget)
        retirement = retirement_plan(age, income, budget, risk)

        return render_template('result.html',
                               stocks=stock_recs.to_dict('records'),
                               funds=fund_recs.to_dict('records'),
                               retirement=retirement,
                               goal=goal_type)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
