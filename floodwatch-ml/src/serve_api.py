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
from src.utils import models_path

# ------------------------
# App & static mounting
# ------------------------
app = FastAPI(title="FloodWatch Machine Learning Model (Random Forest) API")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

# Enable CORS for local dev; restrict origins in production
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
# Candidate dataset paths (tries these in order)
DATA_PATHS = [
    PROJECT_ROOT / "static" / "data" / "FloodWatch_MLDataset.csv",
    Path(r"static/data/floodwatch_MLdataset.csv"),
    Path(r"static/data/FloodWatch_MLDataset.csv"),
    Path("/mnt/data/floodwatch_MLdataset.csv"),    # uploaded file fallback
    Path("/mnt/data/FloodWatch_MLDataset.csv"),   # alternative uploaded name
]

# Model & scaler (use models_path helper to resolve inside repo)
DEFAULT_MODEL = models_path("random_forest_floodwatch.joblib")
DEFAULT_SCALER = models_path("scaler_floodwatch.joblib")
MODEL_PATH = Path(os.getenv("FLOODWATCH_MODEL_PATH", DEFAULT_MODEL))
SCALER_PATH = Path(os.getenv("FLOODWATCH_SCALER_PATH", DEFAULT_SCALER))

LABEL_MAP = {0: "Yellow", 1: "Orange", 2: "Red"}


# ------------------------
# Utilities
# ------------------------
def find_dataset_path() -> Optional[Path]:
    """Return the first existing dataset path from DATA_PATHS or None."""
    for p in DATA_PATHS:
        if p and p.exists():
            return p
    return None


def load_model_and_scaler():
    """Try to load model and scaler. Return (model, scaler). Missing artifacts -> None."""
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
    Sample a random elevation value from rows in dataset that match the given barangay.
    Keeps duplicates so repeated values are weighted by frequency.
    Returns None if no matching rows found or if dataset missing.
    """
    path = find_dataset_path()
    if not path:
        return None
    vals = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("Barangay") or "").strip()
            if not name:
                continue
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
    """Fallback: sample a random elevation from all rows in the dataset."""
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
# Load model & scaler at startup (if available)
# ------------------------
MODEL, SCALER = load_model_and_scaler()
if MODEL is None:
    print("Warning: No model loaded. /predict will use a simple heuristic fallback.")
if SCALER is None:
    print("Info: No scaler loaded. Features will not be scaled before prediction.")


# ------------------------
# Request / Response schemas
# ------------------------
class PredictRequest(BaseModel):
    Barangay: Optional[str] = Field(None, description="Name of barangay (optional but recommended)")
    Elevation_m: Optional[float] = Field(
        None, description="Elevation in meters; if omitted, backend samples from dataset"
    )
    Duration_hr: float = Field(..., gt=0, description="Rainfall duration in hours")
    Rainfall_mm: float = Field(..., ge=0, description="Rainfall intensity (mm/hr)")


class PredictResponse(BaseModel):
    predicted_label: str
    numeric_label: int
    chosen_elevation: float
    class_probabilities: Dict[str, float]


# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": str(MODEL_PATH.name) if MODEL_PATH else None,
        "scaler_file": str(SCALER_PATH.name) if SCALER_PATH else None,
        "dataset": str(find_dataset_path()) if find_dataset_path() else None,
    }


@app.get("/api/barangays")
def api_barangays():
    """
    Return a list of barangays, each with the array of elevations observed in the dataset.
    Example:
      [{"barangay": "Burol I", "elevations": [53.0, 55.1, 56.2]}, ...]
    """
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
                # skip rows with invalid elevation
                continue
            grouped.setdefault(name, []).append(elev)

    # Return elevations as-is (keeps duplicates -> weighted sampling)
    out = [{"barangay": k, "elevations": v} for k, v in grouped.items()]
    out.sort(key=lambda x: x["barangay"])
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict route:
    - If Elevation_m provided, use it.
    - Else sample a random elevation for the Barangay.
    - Match model feature order EXACTLY:
          [Elevation_m, Rainfall_mm, Duration_hr]
    """

    if req.Duration_hr is None or req.Duration_hr <= 0:
        raise HTTPException(status_code=400, detail="Duration_hr must be > 0")
    if req.Rainfall_mm is None or req.Rainfall_mm < 0:
        raise HTTPException(status_code=400, detail="Rainfall_mm must be >= 0")

    barangay = (req.Barangay or "").strip()

    # Determine chosen elevation
    if req.Elevation_m is not None:
        chosen_elevation = float(req.Elevation_m)
    else:
        chosen_elevation = sample_elevation_for_barangay(barangay) if barangay else None
        if chosen_elevation is None:
            chosen_elevation = sample_elevation_from_all()

    if chosen_elevation is None:
        raise HTTPException(status_code=500, detail="No elevation values available to sample.")

    # Build feature vector in correct order
    X = np.array([[
        float(chosen_elevation),
        float(req.Rainfall_mm),
        float(req.Duration_hr)
    ]], dtype=float)

    # Apply scaler
    X_proc = SCALER.transform(X) if SCALER is not None else X

    # Perform prediction
    if MODEL is not None:
        probs_arr = MODEL.predict_proba(X_proc)[0]
        pred_numeric = int(MODEL.predict(X_proc)[0])

        class_probs = {
            LABEL_MAP.get(int(cls), str(cls)): float(round(float(probs_arr[i]), 4))
            for i, cls in enumerate(MODEL.classes_)
        }

    else:
        # fallback heuristic
        r = float(req.Rainfall_mm)
        if r >= 27.6:
            pred_numeric = 2
            class_probs = {"Yellow": 0.05, "Orange": 0.25, "Red": 0.70}
        elif r >= 10.6:
            pred_numeric = 1
            class_probs = {"Yellow": 0.10, "Orange": 0.80, "Red": 0.10}
        else:
            pred_numeric = 0
            class_probs = {"Yellow": 0.85, "Orange": 0.10, "Red": 0.05}

    predicted_label = LABEL_MAP.get(pred_numeric, str(pred_numeric))

    return PredictResponse(
        predicted_label=predicted_label,
        numeric_label=pred_numeric,
        chosen_elevation=float(chosen_elevation),
        class_probabilities=class_probs,
    )

