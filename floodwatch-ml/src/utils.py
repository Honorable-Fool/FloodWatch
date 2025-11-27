# src/utils.py
from pathlib import Path
import pandas as pd
import joblib

# Project root: one directory above src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def data_path(*parts):
    return PROJECT_ROOT.joinpath("data", *parts)

def models_path(*parts):
    p = PROJECT_ROOT.joinpath("models", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_dataset(csv_path=None):
    if csv_path is None:
        csv_path = data_path(r"floodwatch_MLdataset.csv", encoding='ISO-8859-1')
    csv_path = Path(r"data\floodwatch_MLdataset.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {r"data\floodwatch_MLdataset.csv"}")
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    expected = {"Barangay", "Elevation_m", "Duration_hr", "Rainfall_mm", "Numeric_Label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df

def get_feature_matrix(df, features=None):
    df = df.copy()
    df["Rainfall_intensity_mm_per_hr"] = df["Rainfall_mm"].astype(float) / df["Duration_hr"].astype(float)
    if features is None:
        features = ["Elevation_m", "Rainfall_mm", "Duration_hr", "Rainfall_intensity_mm_per_hr"]
    X = df[features].astype(float)
    y = df["Numeric_Label"].astype(int)
    return X, y

def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")
    return joblib.load(path)
