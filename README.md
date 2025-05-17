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
Add your OpenAI API key:

Open app.py.

Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key.

python app.py

http://127.0.0.1:5000/



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
