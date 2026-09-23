"""
Cane Advisory — Premium AI-Driven Irrigation & News System for Sugarcane
Backend: FastAPI server + Gemini API integration

Run locally:
    pip install fastapi uvicorn "google-genai" pydantic python-dotenv
    cp .env.example .env
    uvicorn backend:app --reload --port 8000

Then open http://localhost:8000/ in a browser. 
"""

import os
import random
from datetime import datetime, timedelta
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load variables from a local .env file
load_dotenv()

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(
    title="Cane Advisory API",
    description="Backend for Cane Advisory: Premium AI-Driven Irrigation & News System.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

_gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _gemini_client = None

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

Language = Literal["English", "Marathi", "Kannada", "Hindi"]

class AdvisoryRequest(BaseModel):
    farm_id: str = Field(..., examples=["CANE-12345"])
    soil_moisture: float = Field(..., description="Soil moisture in % (0-100)")
    soil_temp: float = Field(..., description="Soil temperature in Celsius")
    crop_stage: str = Field(..., examples=["Tillering", "Grand Growth", "Maturity"])
    language: Language = "English"

class AdvisoryResponse(BaseModel):
    farm_id: str
    language: Language
    advisory_text: str
    recommended_action: str
    source: Literal["gemini", "rule-based-fallback"]

# --------------------------------------------------------------------------
# Advisory generation
# --------------------------------------------------------------------------

def _call_gemini(req: AdvisoryRequest) -> Optional[str]:
    if not _gemini_client:
        return None
    try:
        prompt = (
            f"Write a short, 2-sentence agricultural advisory for a sugarcane farmer. "
            f"Plot ID: {req.farm_id}, Soil Moisture: {req.soil_moisture}%, Soil Temp: {req.soil_temp}C, "
            f"Crop Stage: {req.crop_stage}, Language: {req.language}. "
            f"Do not use markdown, bullet points, or introductory text. Just output the two sentences."
        )
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        text = text.replace("**", "").replace("*", "").strip()
        return text or None
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

def _rule_based_advisory(req: AdvisoryRequest) -> str:
    low_moisture = req.soil_moisture < 35
    high_moisture = req.soil_moisture > 65
    hot_soil = req.soil_temp > 32

    templates = {
        "English": {
            "irrigate": "Irrigate today for about 2 hours, preferably in the early morning. Soil moisture at your {stage} stage plot is low ({moisture}%), and warm soil means the crop will use water faster.",
            "delay": "You can delay irrigation for 2-3 days. Soil moisture is still comfortable ({moisture}%) for the {stage} stage, so the roots have enough water for now.",
            "skip": "No irrigation is needed right now. Soil moisture is high ({moisture}%), and irrigating now could waterlog the root zone and reduce yield.",
        },
        "Marathi": {
            "irrigate": "आज सकाळी सुमारे 2 तास पाणी द्या. तुमच्या {stage} टप्प्यातील प्लॉटमध्ये मातीतील ओलावा कमी आहे ({moisture}%), आणि जमीन उष्ण असल्याने पिकाला अधिक पाणी लागेल.",
            "delay": "पुढील 2-3 दिवस पाणी देण्याची गरज नाही. सध्या मातीतील ओलावा ({moisture}%) {stage} टप्प्यासाठी पुरेसा आहे.",
            "skip": "सध्या पाणी देण्याची गरज नाही. मातीतील ओलावा जास्त आहे ({moisture}%), आत्ता पाणी दिल्यास मुळांभोवती पाणी साचून उत्पादनावर परिणाम होऊ शकतो.",
        },
        "Kannada": {
            "irrigate": "ಇಂದು ಬೆಳಿಗ್ಗೆ ಸುಮಾರು 2 ಗಂಟೆಗಳ ಕಾಲ ನೀರು ಕೊಡಿ. ನಿಮ್ಮ {stage} ಹಂತದ ಪ್ಲಾಟ್‌ನಲ್ಲಿ ಮಣ್ಣಿನ ತೇವಾಂಶ ಕಡಿಮೆ ಇದೆ ({moisture}%), ಮತ್ತು ಮಣ್ಣು ಬಿಸಿಯಾಗಿರುವುದರಿಂದ ಬೆಳೆಗೆ ಹೆಚ್ಚು ನೀರು ಬೇಕಾಗುತ್ತದೆ.",
            "delay": "ಮುಂದಿನ 2-3 ದಿನ ನೀರು ಕೊಡುವ ಅಗತ್ಯವಿಲ್ಲ. ಪ್ರಸ್ತುತ ಮಣ್ಣಿನ ತೇವಾಂಶ ({moisture}%) {stage} ಹಂತಕ್ಕೆ ಸಾಕಾಗುತ್ತದೆ.",
            "skip": "ಸದ್ಯಕ್ಕೆ ನೀರಿನ ಅಗತ್ಯವಿಲ್ಲ. ಮಣ್ಣಿನ ತೇವಾಂಶ ಹೆಚ್ಚಿದೆ ({moisture}%), ಈಗ ನೀರು ಕೊಟ್ಟರೆ ಬೇರುಗಳ ಬಳಿ ನೀರು ನಿಂತು ಇಳುವರಿ ಕಡಿಮೆಯಾಗಬಹುದು.",
        },
        "Hindi": {
            "irrigate": "आज सुबह लगभग 2 घंटे सिंचाई करें। आपके {stage} चरण वाले प्लॉट में मिट्टी की नमी कम है ({moisture}%), और मिट्टी गर्म होने से फसल को अधिक पानी की जरूरत होगी।",
            "delay": "अगले 2-3 दिन सिंचाई की जरूरत नहीं है। मिट्टी की नमी ({moisture}%) {stage} चरण के लिए अभी पर्याप्त है।",
            "skip": "अभी सिंचाई की जरूरत नहीं है। मिट्टी में नमी अधिक है ({moisture}%), अभी पानी देने से जड़ों के पास पानी भर सकता है और उत्पादन घट सकता है।",
        },
    }

    if low_moisture or hot_soil:
        key = "irrigate"
    elif high_moisture:
        key = "skip"
    else:
        key = "delay"

    text = templates[req.language][key].format(
        stage=req.crop_stage, moisture=round(req.soil_moisture, 1)
    )
    return text

@app.post("/api/v1/generate-advisory", response_model=AdvisoryResponse)
def generate_advisory(req: AdvisoryRequest):
    gemini_text = _call_gemini(req)
    if gemini_text:
        advisory_text = gemini_text
        source = "gemini"
    else:
        advisory_text = _rule_based_advisory(req)
        source = "rule-based-fallback"

    if req.soil_moisture < 35 or req.soil_temp > 32:
        action = "Irrigate soon"
    elif req.soil_moisture > 65:
        action = "Skip irrigation"
    else:
        action = "Delay irrigation"

    return AdvisoryResponse(
        farm_id=req.farm_id,
        language=req.language,
        advisory_text=advisory_text,
        recommended_action=action,
        source=source,
    )

# --------------------------------------------------------------------------
# Mock telemetry data (dashboard.html)
# --------------------------------------------------------------------------

VILLAGES = ["Sector A", "Sector B", "Sector C", "Sector D", "Sector E"]
CROP_STAGES = ["Germination", "Tillering", "Grand Growth", "Maturity"]

def _make_plot(i: int):
    random.seed(i * 17 + 3)
    moisture = round(random.uniform(18, 78), 1)
    temp = round(random.uniform(24, 36), 1)
    humidity = round(random.uniform(40, 85), 1)
    status = "Low" if moisture < 35 else ("High" if moisture > 65 else "Normal")
    return {
        "plot_id": f"CANE-PLT-{1000 + i}",
        "village": VILLAGES[i % len(VILLAGES)],
        "crop_stage": CROP_STAGES[i % len(CROP_STAGES)],
        "area_acres": round(random.uniform(1.5, 6.0), 1),
        "soil_moisture": moisture,
        "soil_moisture_status": status,
        "soil_temp": temp,
        "humidity": humidity,
        "water_stress_index": round(random.uniform(0.1, 0.9), 2),
        "next_irrigation_date": (datetime.now() + timedelta(days=random.randint(0, 5))).strftime("%d %b %Y"),
        "last_irrigation_date": (datetime.now() - timedelta(days=random.randint(1, 10))).strftime("%d %b %Y"),
    }

@app.get("/api/v1/telemetry")
def get_telemetry():
    plots = [_make_plot(i) for i in range(6)]
    forecast = []
    base = datetime.now()
    conditions = ["Clear", "Partly Cloudy", "Overcast", "Light Showers", "Clear"]
    for i, cond in enumerate(conditions):
        forecast.append({
            "date": (base + timedelta(days=i)).strftime("%a, %d %b"),
            "condition": cond,
            "temp_high": round(random.uniform(30, 38), 1),
            "temp_low": round(random.uniform(20, 26), 1),
            "rain_chance_pct": random.choice([0, 10, 20, 60, 5]),
        })
    alerts = [
        {"level": "warning", "message": "Soil moisture dropping in CANE-PLT-1002"},
        {"level": "info", "message": "Favorable weather for fertilization expected tomorrow"},
        {"level": "success", "message": "All irrigation systems operating normally in Sector A"},
    ]
    return {"plots": plots, "forecast": forecast, "alerts": alerts}

# --------------------------------------------------------------------------
# Mock analytics data (analytics.html)
# --------------------------------------------------------------------------

@app.get("/api/v1/analytics")
def get_analytics():
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    random.seed(99)
    water_before = [round(random.uniform(1500, 2000), 0) for _ in months]
    water_after = [round(w * random.uniform(0.60, 0.75), 0) for w in water_before]
    yield_before = [round(random.uniform(80, 95), 1) for _ in months]
    yield_after = [round(y * random.uniform(1.05, 1.20), 1) for y in yield_before]
    electricity_savings_pct = [round(random.uniform(15, 35), 1) for _ in months]

    return {
        "months": months,
        "water_usage_m3": {"before_ai": water_before, "after_ai": water_after},
        "yield_tonnes_per_acre": {"before_ai": yield_before, "after_ai": yield_after},
        "electricity_savings_pct": electricity_savings_pct,
        "summary": {
            "farms_onboarded": 5201,
            "total_water_saved_million_litres": 950.2,
            "avg_yield_increase_pct": 14.2,
            "avg_electricity_savings_pct": 28.1,
        },
    }

# --------------------------------------------------------------------------
# Mock live local news (news.html)
# --------------------------------------------------------------------------

@app.get("/api/v1/news")
def get_news():
    now = datetime.now()
    news_items = [
        {
            "id": "news-1",
            "title": "Government Announces New Subsidies for Drip Irrigation",
            "snippet": "Farmers adopting advanced drip irrigation technologies can now claim up to 40% subsidy under the revised agricultural scheme.",
            "source": "Agri Ministry Update",
            "timestamp": (now - timedelta(minutes=45)).strftime("%I:%M %p, %d %b"),
            "category": "Policy"
        },
        {
            "id": "news-2",
            "title": "Unseasonal Rains Expected in Northern Districts",
            "snippet": "Meteorological department warns of unexpected light to moderate rainfall over the next 48 hours. Farmers advised to delay harvesting.",
            "source": "Local Met Dept",
            "timestamp": (now - timedelta(hours=2, minutes=15)).strftime("%I:%M %p, %d %b"),
            "category": "Weather"
        },
        {
            "id": "news-3",
            "title": "Breakthrough in Sugarcane Pest Control",
            "snippet": "A new bio-friendly pesticide has shown promising results in early trials against the common cane borer, reducing crop damage by 30%.",
            "source": "AgriTech Daily",
            "timestamp": (now - timedelta(hours=5)).strftime("%I:%M %p, %d %b"),
            "category": "Research"
        },
        {
            "id": "news-4",
            "title": "Local Sugar Mill Increases Procurement Price",
            "snippet": "In response to increased demand, the regional sugar cooperative has announced a 5% hike in the procurement price per tonne of sugarcane.",
            "source": "Market Watch",
            "timestamp": (now - timedelta(days=1)).strftime("%I:%M %p, %d %b"),
            "category": "Market"
        },
        {
            "id": "news-5",
            "title": "Workshop: Optimizing Water Usage in Grand Growth Stage",
            "snippet": "Join our free online seminar this weekend featuring agricultural experts discussing water conservation strategies during the critical growth phase.",
            "source": "Cane Advisory Extension",
            "timestamp": (now - timedelta(days=2)).strftime("%I:%M %p, %d %b"),
            "category": "Events"
        }
    ]
    return {"news": news_items}

# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__)) or ".", html=True), name="frontend")
