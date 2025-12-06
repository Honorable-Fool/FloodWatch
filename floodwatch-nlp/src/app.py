""" FloodWatch NLP API Service FastAPI endpoint for flood advisory classification """
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import random
import os

# ---------------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------------
app = FastAPI(title="FloodWatch NLP API", version="1.7")

# Enable CORS for the frontend (port 5500)
# Allows the browser to fetch data from this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------
# Get the directory where app.py is located to define relative paths
BASE_DIR = Path(__file__).resolve().parent.parent 

MODEL_PATH = BASE_DIR / "models_nlp" / "nb_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models_nlp" / "tfidf.pkl"

DATA_DIR = BASE_DIR / "data"
EXCEL_FILE = DATA_DIR / "Nlp dataset.xlsx"
CSV_SOURCE = DATA_DIR / "Nlp dataset.xlsx - Source.csv"
CSV_PARA = DATA_DIR / "Nlp dataset.xlsx - Paraphrased.csv"

model = None
vectorizer = None
# Database to store advisory text for each risk level
ADVISORY_DATABASE: Dict[int, List[str]] = {0: [], 1: [], 2: []}

# ---------------------------------------------------------
# Load Resources
# ---------------------------------------------------------
def load_resources():
    """Loads the trained model, vectorizer, and advisory dataset into memory."""
    global model, vectorizer, ADVISORY_DATABASE
    
    # 1. Load Model & Vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✓ NLP Model and Vectorizer loaded successfully")
    except Exception as e:
        print(f"⚠ WARNING: Could not load model. Error: {e}")

    # 2. Load Advisory Dataset
    print("Loading advisory dataset...")
    df = pd.DataFrame()
    
    try:
        # Priority: Try loading Excel first
        if EXCEL_FILE.exists():
            df1 = pd.read_excel(EXCEL_FILE, sheet_name='Source', usecols=['Advisory Text', 'Level'])
            df2 = pd.read_excel(EXCEL_FILE, sheet_name='Paraphrased', usecols=['Advisory Text', 'Level'])
            df = pd.concat([df1, df2], ignore_index=True)
            print("✓ Loaded data from Excel")
        # Fallback: Try loading CSVs
        elif CSV_SOURCE.exists() and CSV_PARA.exists():
            df1 = pd.read_csv(CSV_SOURCE)
            df2 = pd.read_csv(CSV_PARA)
            if 'Advisory Text' in df1.columns:
                df = pd.concat([df1, df2], ignore_index=True)
            print("✓ Loaded data from CSVs")
        else:
            print("⚠ Data files not found.")
            
        # 3. Process Data
        if not df.empty:
            # Clean up types and remove empty rows
            df = df.dropna(subset=['Level', 'Advisory Text'])
            df['Level'] = pd.to_numeric(df['Level'], errors='coerce').astype(int)
            df['Advisory Text'] = df['Advisory Text'].astype(str)
            
            # Organize text by risk level (0, 1, 2)
            for level in [0, 1, 2]:
                texts = df[df['Level'] == level]['Advisory Text'].tolist()
                # Remove duplicates and very short strings
                texts = list(set([t for t in texts if len(t) > 10]))
                if texts:
                    ADVISORY_DATABASE[level] = texts
            
            print(f"✓ Dataset Ready: {len(ADVISORY_DATABASE[0])} Low, {len(ADVISORY_DATABASE[1])} Mod, {len(ADVISORY_DATABASE[2])} High advisories.")
            
    except Exception as e:
        print(f"⚠ Error loading dataset: {e}")

load_resources()

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
FALLBACK_MESSAGES = {
    0: "Low Risk: Light rains expected. Monitor local news for updates.",
    1: "Moderate Risk: Moderate rains detected. Flooding is possible in low-lying areas.",
    2: "High Risk: Heavy rainfall warning. Severe flooding expected. Evacuate if advised."
}

COLORS = {0: "#FFC107", 1: "#FF9800", 2: "#F44336"}
LEVEL_NAMES = {
    0: "Low Chance (Yellow Warning)", 
    1: "Moderate Chance (Orange Warning)", 
    2: "High Chance (Red Warning)"
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def format_advisory_paragraph(texts: List[str]) -> str:
    """
    Joins a list of sentences into a coherent paragraph.
    Ensures each sentence ends with a period before joining.
    """
    cleaned = []
    for t in texts:
        t = t.strip()
        if t:
            # Remove trailing punctuation then add a period
            t = t.rstrip('.!,;') + "."
            cleaned.append(t)
            
    return " ".join(cleaned)

# ---------------------------------------------------------
# Schemas
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
# API Routes
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "service": "FloodWatch NLP API"}

@app.post("/predict", response_model=AdvisoryResponse)
def predict_flood_level(request: AdvisoryRequest):
    """
    Classifies input text into a risk level using the NLP model.
    Then selects a matching advisory from the dataset.
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Convert text to numbers (TF-IDF) and predict
        text_vector = vectorizer.transform([request.text])
        prediction = int(model.predict(text_vector)[0])
        probabilities = model.predict_proba(text_vector)[0]
        confidence = {f"level_{i}": float(prob) for i, prob in enumerate(probabilities)}
        
        # --- COMBINE SENTENCES ---
        # Pick random sentences from the predicted level to form a new advisory
        candidates = ADVISORY_DATABASE.get(prediction, [])
        if len(candidates) >= 3:
            selected_texts = random.sample(candidates, 3)
            selected_advisory = format_advisory_paragraph(selected_texts)
        elif candidates:
            selected_advisory = format_advisory_paragraph([candidates[0]])
        else:
            selected_advisory = FALLBACK_MESSAGES[prediction]

        return AdvisoryResponse(
            level=prediction,
            level_name=LEVEL_NAMES[prediction],
            advisory=selected_advisory,
            color=COLORS[prediction],
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/generate_from_risk")
def generate_from_risk(req: RiskRequest):
    """
    Generates a detailed advisory based strictly on the provided risk level (Low/Mod/High).
    Used by the frontend after ML prediction.
    """
    risk_map = {"Low": 0, "Moderate": 1, "High": 2}
    
    if req.risk not in risk_map:
        level = 0 
    else:
        level = risk_map[req.risk]
    
    candidates = ADVISORY_DATABASE.get(level, [])
    
    # --- COMBINE SENTENCES FOR DETAIL ---
    # Pick 3-4 random sentences to ensure the advisory is detailed and long
    if len(candidates) >= 4:
        selected_texts = random.sample(candidates, 3)
        advisory_text = format_advisory_paragraph(selected_texts)
    elif candidates:
        advisory_text = format_advisory_paragraph([candidates[0]])
    else:
        advisory_text = FALLBACK_MESSAGES[level]

    return {
        "level": level,
        "level_name": LEVEL_NAMES[level],
        "advisory": advisory_text,
        "color": COLORS[level]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)