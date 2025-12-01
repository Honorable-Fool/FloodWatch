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

app = FastAPI(title="FloodWatch NLP API", version="1.1")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and vectorizer
# Load trained model and vectorizer
# We correct the path to look inside the 'models' folder
# and correct the filenames to match the files in your directory ('nb_model.pkl' and 'tfidf.pkl')

# Assuming 'serve_api_nlp.py' is next to the 'models' folder
MODEL_PATH = Path("models") / "nb_model.pkl"
VECTORIZER_PATH = Path("models") / "tfidf.pkl"

try:
    # Use 'joblib.load' as you have it
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Model and vectorizer loaded successfully")
except Exception as e:
    # Print the specific paths being used to help debug future issues
    print(f"⚠ Warning: Could not load model or vectorizer.")
    print(f"Attempted Model Path: {MODEL_PATH}")
    print(f"Attempted Vectorizer Path: {VECTORIZER_PATH}")
    print(f"Error details: {e}")
    model = None
    vectorizer = None

# ... rest of your code ...

# Advisory messages for each level
ADVISORY_MESSAGES = {
    0: {
        "level_name": "Low Chance (Yellow Warning)",
        "advisory": "Light to moderate rains expected. Flooding possible in low-lying areas. Monitor weather updates and avoid flood-prone areas.",
        "color": "#FFC107"
    },
    1: {
        "level_name": "Moderate Chance (Orange Warning)",
        "advisory": "Heavy rainfall expected. Road/street flooding likely in multiple areas. Prepare emergency supplies and avoid unnecessary travel.",
        "color": "#FF9800"
    },
    2: {
        "level_name": "High Chance (Red Warning)",
        "advisory": "Intense to torrential rains forecast. Serious flooding expected in Dasmariñas. Evacuate if in high-risk areas. Stay indoors and monitor official advisories.",
        "color": "#F44336"
    }
}

class AdvisoryRequest(BaseModel):
    text: str

class AdvisoryResponse(BaseModel):
    level: int
    level_name: str
    advisory: str
    color: str
    confidence: dict

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FloodWatch NLP API",
        "model_loaded": model is not None,
        "levels": {
            "0": "Low Chance (Yellow Warning)",
            "1": "Moderate Chance (Orange Warning)",
            "2": "High Chance (Red Warning)"
        }
    }

@app.post("/predict", response_model=AdvisoryResponse)
def predict_flood_level(request: AdvisoryRequest):
    """
    Predict flood advisory level from text
    """
    
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Advisory text cannot be empty"
        )
    
    try:
        # Transform input text using TF-IDF
        text_tfidf = vectorizer.transform([request.text])
        
        # Predict flood level
        prediction = int(model.predict(text_tfidf)[0])
        
        # Get probability scores
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = {
            f"level_{i}": float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        # Advisory details
        advisory_info = ADVISORY_MESSAGES.get(prediction, ADVISORY_MESSAGES[0])
        
        return AdvisoryResponse(
            level=prediction,
            level_name=advisory_info["level_name"],
            advisory=advisory_info["advisory"],
            color=advisory_info["color"],
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
def batch_predict(texts: list[str]):
    """Predict multiple advisory texts"""
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for text in texts:
            text_tfidf = vectorizer.transform([text])
            prediction = int(model.predict(text_tfidf)[0])
            probabilities = model.predict_proba(text_tfidf)[0]
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "level": prediction,
                "level_name": ADVISORY_MESSAGES[prediction]["level_name"],
                "confidence": {f"level_{i}": float(prob) for i, prob in enumerate(probabilities)}
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """Get information about the loaded model"""
    
    if model is None or vectorizer is None:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "model_type": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF",
        "num_features": len(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else "Unknown",
        "num_classes": len(model.classes_),
        "classes": model.classes_.tolist()
    }


# ============================================================
# ✅ NEW ENDPOINT: Generate advisory based on ML prediction
# ============================================================

class RiskRequest(BaseModel):
    risk: str  # "Low", "Moderate", "High"

@app.post("/generate_from_risk")
def generate_from_risk(req: RiskRequest):
    """
    Generate advisory directly using ML risk classification.
    No NLP input needed.
    """

    risk_map = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    if req.risk not in risk_map:
        raise HTTPException(status_code=400, detail="Risk must be Low, Moderate, or High")

    level = risk_map[req.risk]
    advisory_info = ADVISORY_MESSAGES[level]

    return {
        "level": level,
        "level_name": advisory_info["level_name"],
        "advisory": advisory_info["advisory"],
        "color": advisory_info["color"]
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
