# src/utils.py
from pathlib import Path
import pandas as pd
import joblib

# Determine project root by going up two levels from this script (src -> floodwatch-ml -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def data_path(*parts):
    # Helper to construct paths to the 'data' directory
    return PROJECT_ROOT.joinpath("data", *parts)

def models_path(*parts):
    # Helper to construct paths to the 'models' directory
    # Automatically creates the directory if it doesn't exist
    p = PROJECT_ROOT.joinpath("models", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_dataset(csv_path=None):
    # Default to the standard dataset location if no path is provided
    if csv_path is None:
        csv_path = data_path("floodwatch_MLdataset.csv")
    
    csv_path = Path(csv_path)
    
    # Fallback logic: if not in data folder, check static folder (common in web deployments)
    if not csv_path.exists():
        fallback = PROJECT_ROOT / "static" / "data" / "floodwatch_MLdataset.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
            
    # Load with specific encoding to handle special characters in Filipino names
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    
    # Validation: Ensure critical columns exist before proceeding
    expected = {"Barangay", "Elevation_m", "Duration_hr", "Rainfall_mm", "Numeric_Label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

def get_feature_matrix(df, features=None):
    # Prepare X (features) and y (target) for model training
    df = df.copy()
    
    # Feature Engineering: Calculate intensity (mm per hour)
    df["Rainfall_intensity_mm_per_hr"] = df["Rainfall_mm"].astype(float) / df["Duration_hr"].astype(float)
    
    if features is None:
        features = ["Elevation_m", "Rainfall_mm", "Duration_hr", "Rainfall_intensity_mm_per_hr"]
        
    X = df[features].astype(float)
    y = df["Numeric_Label"].astype(int)
    return X, y

def save_pickle(obj, path):
    # Save a Python object (model/scaler) to disk using joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path):
    # Load a Python object from disk
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    return joblib.load(path)