"""
FloodWatch NLP API Service
FastAPI endpoint for flood advisory classification
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
from typing import List

# ---------------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------------
app = FastAPI(title="FloodWatch NLP API", version="1.2")

# Enable CORS for the frontend (port 5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Model Loading (Correct Paths)
# ---------------------------------------------------------
MODEL_PATH = Path("models_nlp/nb_model.pkl")
VECTORIZER_PATH = Path("models_nlp/tfidf_vectorizer.pkl")

model = None
vectorizer = None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ NLP Model and Vectorizer loaded successfully")
except Exception as e:
    print("\n⚠ WARNING: Could not load model or vectorizer.")
    print(f"Model path:      {MODEL_PATH}")
    print(f"Vectorizer path: {VECTORIZER_PATH}")
    print(f"Error: {e}")
    print("Run: python train_nlp.py to regenerate the model.\n")

# ---------------------------------------------------------
# Static Advisory Messages (Used for Risk-Based NLP)
# ---------------------------------------------------------
ADVISORY_MESSAGES = {
    0: {
        "level_name": "Low Chance (Yellow Warning)",
        "advisory": (
            "Light to moderate rains expected. Flooding possible in low-lying areas. "
            "Monitor weather updates and avoid flood-prone areas."
        ),
        "color": "#FFC107"
    },
    1: {
        "level_name": "Moderate Chance (Orange Warning)",
        "advisory": (
            "Heavy rainfall expected. Road and street flooding likely in multiple areas. "
            "Prepare emergency supplies and avoid unnecessary travel."
        ),
        "color": "#FF9800"
    },
    2: {
        "level_name": "High Chance (Red Warning)",
        "advisory": (
            "Intense to torrential rains forecast. Serious flooding expected. "
            "Evacuate if in high-risk areas and monitor official advisories."
        ),
        "color": "#F44336"
    }
}

# ---------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------
class AdvisoryRequest(BaseModel):
    text: str

class AdvisoryResponse(BaseModel):
    level: int
    level_name: str
    advisory: str
    color: str
    confidence: dict

class RiskRequest(BaseModel):
    risk: str


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "FloodWatch NLP API",
        "model_loaded": model is not None and vectorizer is not None,
        "endpoints": ["/predict", "/batch-predict", "/generate_from_risk"]
    }

# ---------------------------------------------------------
# Predict Advisory from Full Text
# ---------------------------------------------------------
@app.post("/predict", response_model=AdvisoryResponse)
def predict_flood_level(request: AdvisoryRequest):

    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_nlp.py first."
        )

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        text_vector = vectorizer.transform([request.text])
        prediction = int(model.predict(text_vector)[0])
        probabilities = model.predict_proba(text_vector)[0]
        confidence = {f"level_{i}": float(prob) for i, prob in enumerate(probabilities)}

        info = ADVISORY_MESSAGES[prediction]

        return AdvisoryResponse(
            level=prediction,
            level_name=info["level_name"],
            advisory=info["advisory"],
            color=info["color"],
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------
# Batch Prediction (For Testing)
# ---------------------------------------------------------
@app.post("/batch-predict")
def batch_predict(texts: List[str]):

    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        results = []
        for text in texts:
            v = vectorizer.transform([text])
            p = int(model.predict(v)[0])
            probs = model.predict_proba(v)[0]

            results.append({
                "text": text,
                "level": p,
                "level_name": ADVISORY_MESSAGES[p]["level_name"],
                "confidence": {f"level_{i}": float(prob) for i, prob in enumerate(probs)}
            })

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ---------------------------------------------------------
# Risk Level → Advisory Message
# (Used by main FloodWatch frontend)
# ---------------------------------------------------------
@app.post("/generate_from_risk")
def generate_from_risk(req: RiskRequest):

    risk_map = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    if req.risk not in risk_map:
        raise HTTPException(status_code=400, detail="Risk must be: Low, Moderate, or High")

    level = risk_map[req.risk]
    info = ADVISORY_MESSAGES[level]

    return {
        "level": level,
        "level_name": info["level_name"],
        "advisory": info["advisory"],
        "color": info["color"]
    }

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
# In app.py:
# ...
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
