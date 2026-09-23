# Cane Advisory

A modern, premium AI-Driven Irrigation Advisory & Live News web app built for sugarcane farming.

## How to run locally

```bash
# 1. Install dependencies
pip install fastapi "uvicorn[standard]" pydantic python-dotenv
pip install "google-genai"   # optional, only needed for real Gemini calls

# 2. Setup your local .env file
cp .env.example .env
# Edit .env and paste your API key (if available):
# GEMINI_API_KEY=your-api-key

# 3. Run the backend server
uvicorn backend:app --reload --port 8000
```

Then visit **http://localhost:8000** in your browser.
