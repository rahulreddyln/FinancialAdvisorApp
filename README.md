# Financial Advisor App

This is a Flask-based financial advisor system that recommends stocks, mutual funds, and retirement plans based on user input, using machine learning and OpenAI GPT-4.

## Setup Instructions

1. Clone or download the repository.

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows



pip install -r requirements.txt
python setup_db.py
pip install -r requirements.txt
python app.py
Go to: http://127.0.0.1:5000/

pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

FinancialAdvisorApp/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── stocks.csv
│   └── mutual_funds.csv
└── templates/
    ├── form.html
    └── result.html
