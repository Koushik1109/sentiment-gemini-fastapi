# Sentiment Analysis API (Gemini + FastAPI)

## How to run locally
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Put your Gemini key in .env
uvicorn main:app --reload --port 8000
