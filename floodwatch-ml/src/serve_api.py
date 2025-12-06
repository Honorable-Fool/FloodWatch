# src/serve_api.py
import csv
import random
import os
from pathlib import Path
from typing import Optional, List, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Import Compatibility Fix ---
# Handles importing 'utils' whether run via 'run.py' (as module) or directly (as script)
try:
    from src.utils import models_path
except ImportError:
    from utils import models_path

# ------------------------
# App & static mounting
# ------------------------
app = FastAPI(title="FloodWatch Machine Learning Model (Random Forest) API")

# Define root directory relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Mount 'static' folder to serve dataset/images if needed via HTTP
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

# Enable CORS (Cross-Origin Resource Sharing)
# This allows the frontend (port 5500) to communicate with this backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Data / model file paths
# ------------------------
# List of potential locations for the CSV dataset, checked in order
DATA_PATHS = [
    PROJECT_ROOT / "static" / "data" / "FloodWatch_MLDataset.csv",
    Path(r"static/data/floodwatch_MLdataset.csv"),
    Path("/mnt/data/floodwatch_MLdataset.csv")    
      
]

# Define paths for the trained model and scaler
DEFAULT_MODEL = models_path("random_forest_floodwatch.joblib")
DEFAULT_SCALER = models_path("scaler_floodwatch.joblib")
# Allow environment variables to override paths (useful for deployment)
MODEL_PATH = Path(os.getenv("FLOODWATCH_MODEL_PATH", DEFAULT_MODEL))
SCALER_PATH = Path(os.getenv("FLOODWATCH_SCALER_PATH", DEFAULT_SCALER))

LABEL_MAP = {0: "Yellow", 1: "Orange", 2: "Red"}


# ------------------------
# Utilities
# ------------------------
def find_dataset_path() -> Optional[Path]:
    """Scans DATA_PATHS and returns the first one that actually exists."""
    for p in DATA_PATHS:
        if p and p.exists():
            return p
    return None


def load_model_and_scaler():
    """Loads the trained ML model and data scaler from disk."""
    model = None
    scaler = None
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
    except Exception as e:
        print("Warning: failed to load model:", e)
    try:
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print("Warning: failed to load scaler:", e)
    return model, scaler


def sample_elevation_for_barangay(barangay: str) -> Optional[float]:
    """
    Look up the elevation for a specific barangay in the dataset.
    If multiple entries exist, pick one at random (weighted sampling).
    """
    path = find_dataset_path()
    if not path:
        return None
    vals = []
    # Read CSV manually to find matching rows
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("Barangay") or "").strip()
            if not name:
                continue
            # Case-insensitive comparison
            if name.lower() == barangay.strip().lower():
                v = r.get("Elevation_m", "")
                try:
                    v = float(v)
                    vals.append(v)
                except:
                    continue
    if not vals:
        return None
    return float(random.choice(vals))


def sample_elevation_from_all() -> Optional[float]:
    """Fallback: if barangay not found, pick a random elevation from ANYWHERE in the dataset."""
    path = find_dataset_path()
    if not path:
        return None
    vals = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                v = float(r.get("Elevation_m", ""))
                vals.append(v)
            except:
                continue
    if not vals:
        return None
    return float(random.choice(vals))


# ------------------------
# Load model & scaler at startup
# ------------------------
MODEL, SCALER = load_model_and_scaler()
if MODEL is None:
    print("Warning: No model loaded. /predict will use a simple heuristic fallback.")
if SCALER is None:
    print("Info: No scaler loaded. Features will not be scaled before prediction.")


# ------------------------
# Request / Response schemas (Pydantic)
# ------------------------
class PredictRequest(BaseModel):
    # Input schema for the /predict endpoint
    Barangay: Optional[str] = Field(None, description="Name of barangay")
    Elevation_m: Optional[float] = Field(
        None, description="Elevation in meters; if omitted, backend samples from dataset"
    )
    Duration_hr: float = Field(..., gt=0, description="Rainfall duration in hours")
    Rainfall_mm: float = Field(..., ge=0, description="Rainfall intensity (mm/hr)")


class PredictResponse(BaseModel):
    # Output schema for the /predict endpoint
    predicted_label: str
    numeric_label: int
    chosen_elevation: float
    class_probabilities: Dict[str, float]


# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    # Simple endpoint to check if API is running and paths are correct
    return {
        "status": "ok",
        "model_file": str(MODEL_PATH.name) if MODEL_PATH else None,
        "scaler_file": str(SCALER_PATH.name) if SCALER_PATH else None,
        "dataset": str(find_dataset_path()) if find_dataset_path() else None,
    }


@app.get("/api/barangays")
def api_barangays():
    """Returns a list of all barangays and their recorded elevations from the dataset."""
    path = find_dataset_path()
    if not path:
        raise HTTPException(status_code=500, detail="Dataset not found on server")

    grouped = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("Barangay") or "").strip()
            if not name:
                continue
            try:
                elev = float(r.get("Elevation_m", ""))
            except:
                continue
            grouped.setdefault(name, []).append(elev)

    # Format list for frontend dropdowns
    out = [{"barangay": k, "elevations": v} for k, v in grouped.items()]
    out.sort(key=lambda x: x["barangay"])
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Main prediction logic:
    1. Determine Elevation (User provided -> Barangay lookup -> Random fallback)
    2. Prepare Data (Create DataFrame, Scale)
    3. Predict (Using Random Forest or Fallback Heuristic)
    """

    # Basic input validation
    if req.Duration_hr is None or req.Duration_hr <= 0:
        raise HTTPException(status_code=400, detail="Duration_hr must be > 0")
    if req.Rainfall_mm is None or req.Rainfall_mm < 0:
        raise HTTPException(status_code=400, detail="Rainfall_mm must be >= 0")

    barangay = (req.Barangay or "").strip()

    # 1. Determine Elevation
    if req.Elevation_m is not None:
        chosen_elevation = float(req.Elevation_m)
    else:
        # Try to find elevation for this specific barangay
        chosen_elevation = sample_elevation_for_barangay(barangay) if barangay else None
        # If not found, just pick a random valid elevation from the whole dataset
        if chosen_elevation is None:
            chosen_elevation = sample_elevation_from_all()

    if chosen_elevation is None:
        raise HTTPException(status_code=500, detail="No elevation values available to sample.")

    # 2. Prepare Data
    # Create a DataFrame with specific column names to match what the Scaler expects
    # This prevents the "X does not have valid feature names" warning
    X = pd.DataFrame([[
        float(chosen_elevation),
        float(req.Rainfall_mm),
        float(req.Duration_hr)
    ]], columns=["Elevation_m", "Rainfall_mm", "Duration_hr"])

    # Scale the features (e.g., normalize elevation and rainfall to similar ranges)
    X_proc = SCALER.transform(X) if SCALER is not None else X

    # 3. Predict
    if MODEL is not None:
        # Use Random Forest Model
        probs_arr = MODEL.predict_proba(X_proc)[0]
        pred_numeric = int(MODEL.predict(X_proc)[0])

        class_probs = {
            LABEL_MAP.get(int(cls), str(cls)): float(round(float(probs_arr[i]), 4))
            for i, cls in enumerate(MODEL.classes_)
        }

    else:
        # Fallback Heuristic (Simple rules if AI model is missing)
        r = float(req.Rainfall_mm)
        if r >= 27.6:
            pred_numeric = 2 # Red
            class_probs = {"Yellow": 0.05, "Orange": 0.25, "Red": 0.70}
        elif r >= 10.6:
            pred_numeric = 1 # Orange
            class_probs = {"Yellow": 0.10, "Orange": 0.80, "Red": 0.10}
        else:
            pred_numeric = 0 # Yellow
            class_probs = {"Yellow": 0.85, "Orange": 0.10, "Red": 0.05}

    predicted_label = LABEL_MAP.get(pred_numeric, str(pred_numeric))

    return PredictResponse(
        predicted_label=predicted_label,
        numeric_label=pred_numeric,
        chosen_elevation=float(chosen_elevation),
        class_probabilities=class_probs,
    )