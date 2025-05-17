from flask import Flask, render_template, request
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
import openai

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

app = Flask(__name__)

# Load datasets from CSV
data_dir = os.path.join(os.path.dirname(__file__), 'data')
stocks_df = pd.read_csv(os.path.join(data_dir, 'stocks.csv'))
mutual_funds_df = pd.read_csv(os.path.join(data_dir, 'mutual_funds.csv'))

# Train and save stock model if not already trained
stock_model_path = os.path.join(data_dir, 'stock_model.pkl')
if not os.path.exists(stock_model_path):
    X = stocks_df[['volatility']]
    y = stocks_df['expected_return']
    stock_model = RandomForestRegressor(n_estimators=100, random_state=42)
    stock_model.fit(X, y)
    joblib.dump(stock_model, stock_model_path)
else:
    stock_model = joblib.load(stock_model_path)

def recommend_stocks(df, model, top_n=3):
    df = df.copy()
    df['predicted_return'] = model.predict(df[['volatility']])
    df = df.sort_values(by='predicted_return', ascending=False)
    return df[['name', 'predicted_return', 'volatility', 'previous_year_return']].head(top_n)

def recommend_mutual_funds(df, risk_level='low', age=None, income=None, savings=None):
    df = df.copy()

    if age is not None:
        if age < 35:
            adjusted_risk = 'high'
        elif age < 50:
            adjusted_risk = 'medium'
        else:
            adjusted_risk = 'low'
    else:
        adjusted_risk = risk_level

    if adjusted_risk == 'low':
        df = df[df['risk'] <= 3]
    elif adjusted_risk == 'medium':
        df = df[(df['risk'] > 3) & (df['risk'] <= 6)]
    else:
        df = df[df['risk'] > 6]

    df = df.sort_values(by='return', ascending=False)
    return df[['name', 'risk', 'return', 'previous_year_return']].head(3)

def classify_investment_goal(prompt):
    system_prompt = (
        "You are a financial assistant. Classify the user's goal into one of: 'growth', 'balanced', or 'conservative'."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip().lower()
    except Exception as e:
        print("LLM fallback due to:", e)
        return "balanced"

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

        stock_recs = recommend_stocks(stocks_df.copy(), stock_model)
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
