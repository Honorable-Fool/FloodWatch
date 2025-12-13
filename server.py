import os
import joblib
import pandas as pd
import numpy as np
import random
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize App
app = FastAPI()

# Enable CORS (Important for preventing fetch errors)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATH CONFIGURATION ----------------
ROOT = Path(__file__).parent
ML_DIR = ROOT / "floodwatch-ml"
NLP_DIR = ROOT / "floodwatch-nlp"

# ---------------- GLOBAL STATE ----------------
rf_model = None
scaler = None
advisory_database = {0: [], 1: [], 2: []}
barangay_data = {} # Cache for barangay elevations

# ---------------- LOAD RESOURCES ON STARTUP ----------------
print("--- STARTING SERVER INITIALIZATION ---")

# 1. Load ML Model & Scaler
try:
    model_path = ML_DIR / "models/random_forest_floodwatch.joblib"
    scaler_path = ML_DIR / "models/scaler_floodwatch.joblib"
    
    if model_path.exists():
        rf_model = joblib.load(model_path)
        print("✓ ML Model Loaded")
    else:
        print(f"⚠ ML Model not found at: {model_path}")

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("✓ Scaler Loaded")
    else:
        print(f"⚠ Scaler not found at: {scaler_path}")
except Exception as e:
    print(f"⚠ Critical ML Load Error: {e}")

# 2. Load and Cache CSV Data (OPTIMIZATION)
try:
    csv_path = ML_DIR / "data/floodwatch_MLdataset.csv"
    if not csv_path.exists():
        # Fallback to static if data folder is missing
        csv_path = ML_DIR / "static/data/floodwatch_MLdataset.csv"

    if csv_path.exists():
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        # Create a dictionary for fast lookups: {'BarangayName': [elev1, elev2...]}
        # Normalize keys to lowercase for easier matching
        temp_grouped = df.groupby("Barangay")["Elevation_m"].apply(list).to_dict()
        barangay_data = {k.lower().strip(): v for k, v in temp_grouped.items()}
        # Store original case keys for the API list
        barangay_list_data = [{"barangay": k, "elevations": v} for k, v in temp_grouped.items()]
        print("✓ Dataset Loaded & Cached")
    else:
        print("⚠ Dataset CSV not found!")
        barangay_list_data = []
except Exception as e:
    print(f"⚠ CSV Load Error: {e}")
    barangay_list_data = []

# 3. Load NLP Models
try:
    nlp_model_path = NLP_DIR / "models_nlp/nb_model.pkl"
    if nlp_model_path.exists():
        nlp_model = joblib.load(nlp_model_path)
        vectorizer = joblib.load(NLP_DIR / "models_nlp/tfidf.pkl")
        
        # Load NLP Excel/CSV
        nlp_df = pd.DataFrame()
        nlp_excel = NLP_DIR / "data/Nlp dataset.xlsx"
        
        if nlp_excel.exists():
            df1 = pd.read_excel(nlp_excel, sheet_name='Source', usecols=['Advisory Text', 'Level'])
            df2 = pd.read_excel(nlp_excel, sheet_name='Paraphrased', usecols=['Advisory Text', 'Level'])
            nlp_df = pd.concat([df1, df2], ignore_index=True)
        
        if not nlp_df.empty:
            nlp_df = nlp_df.dropna(subset=['Level', 'Advisory Text'])
            nlp_df['Level'] = pd.to_numeric(nlp_df['Level'], errors='coerce').astype(int)
            for level in [0, 1, 2]:
                texts = nlp_df[nlp_df['Level'] == level]['Advisory Text'].astype(str).tolist()
                advisory_database[level] = list(set([t for t in texts if len(t) > 10]))
        print("✓ NLP Resources Loaded")
    else:
        print("⚠ NLP Model not found")
except Exception as e:
    print(f"⚠ NLP Load Error: {e}")

# ---------------- SCHEMAS ----------------
class PredictRequest(BaseModel):
    Barangay: str
    Duration_hr: float
    Rainfall_mm: float
    Elevation_m: float = None 

class NlpRequest(BaseModel):
    risk: str

# ---------------- API ROUTES ----------------

@app.get("/api/barangays")
def get_barangays():
    """Returns the cached list of barangays instantly."""
    return barangay_list_data

@app.post("/api/ml/predict")
def predict_flood(req: PredictRequest):
    try:
        # 1. Determine Elevation (Use cached data)
        elevation = req.Elevation_m
        
        if elevation is None:
            # Look up in our pre-loaded dictionary
            b_key = req.Barangay.lower().strip()
            if b_key in barangay_data:
                # Pick a random elevation from the list for that barangay
                elevation = float(random.choice(barangay_data[b_key]))
            else:
                # Fallback: Random elevation from ALL data (if available)
                all_elevs = [e for sublist in barangay_data.values() for e in sublist]
                if all_elevs:
                    elevation = float(random.choice(all_elevs))
                else:
                    elevation = 10.0 # Ultimate fallback

        # 2. Prepare Features
        # Create DataFrame to match model expected format
        X = pd.DataFrame([[elevation, req.Rainfall_mm, req.Duration_hr]], 
                        columns=["Elevation_m", "Rainfall_mm", "Duration_hr"])
        
        if scaler:
            X = scaler.transform(X)
        
        # 3. Predict
        numeric_label = 0
        if rf_model:
            numeric_label = int(rf_model.predict(X)[0])
        else:
            # Fallback logic if model failed to load
            print("Using fallback logic (No Model Loaded)")
            if req.Rainfall_mm >= 28: numeric_label = 2
            elif req.Rainfall_mm >= 11: numeric_label = 1
            else: numeric_label = 0
        
        return {
            "numeric_label": numeric_label,
            "chosen_elevation": elevation,
            "risk_label": ["Low", "Moderate", "High"][numeric_label]
        }

    except Exception as e:
        print(f"Server Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/nlp/generate")
def generate_advisory(req: NlpRequest):
    risk_map = {"Low": 0, "Moderate": 1, "High": 2}
    level = risk_map.get(req.risk, 0)
    
    candidates = advisory_database.get(level, [])
    
    if len(candidates) >= 3:
        advisory = " ".join(random.sample(candidates, 3))
    elif candidates:
        advisory = candidates[0]
    else:
        advisory = "Advisory not available at this time."
        
    return {"advisory": advisory}

# ---------------- SERVE STATIC FILES ----------------
# This must be last
app.mount("/", StaticFiles(directory=".", html=True), name="static")