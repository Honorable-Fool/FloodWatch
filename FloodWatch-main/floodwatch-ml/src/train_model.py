# src/train_model.py

from pathlib import Path
import argparse, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from utils import data_path, models_path, load_dataset, save_pickle

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="Optional CSV override")
    p.add_argument("--model-out", type=str, default=None, help="Optional model path")
    p.add_argument("--scaler-out", type=str, default=None, help="Optional scaler path")
    return p.parse_args()

def get_feature_matrix(df):
    """
    Extract numeric features used by the ML model.
    Only uses:
        Elevation_m
        Rainfall_mm
        Duration_hr
    """
    X = df[["Elevation_m", "Rainfall_mm", "Duration_hr"]].astype(float)
    y = df["Numeric_Label"].astype(int)
    return X, y

def main():
    args = parse_args()

    csv_path = Path(args.csv) if args.csv else data_path("FloodWatch_MLDataset.csv")
    model_path = Path(args.model_out) if args.model_out else models_path("random_forest_floodwatch.joblib")
    scaler_path = Path(args.scaler_out) if args.scaler_out else models_path("scaler_floodwatch.joblib")

    fi_png = models_path("feature_importances.png")
    cm_png = models_path("confusion_matrix.png")
    cv_png = models_path("kfold_cv_scores.png")

    print("CSV:", csv_path)

    try:
        df = load_dataset(csv_path)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)

    print("\nLoaded df shape:", df.shape)
    print("Label counts:\n", df["Numeric_Label"].value_counts())

    # Extract features
    X, y = get_feature_matrix(df)
    print("Using features:", list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2022
    )

    print("\nTrain/test shapes:", X_train.shape, X_test.shape)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_pickle(scaler, scaler_path)
    print("Saved scaler to", scaler_path)

    # Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=2022,
        n_jobs=-1
    )

    # Cross-validation (K-Fold)
    cv = KFold(n_splits=5, shuffle=True, random_state=2022)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores, ddof=1)

    print("\nCV F1 macro (per fold):", cv_scores)
    print("CV F1 macro mean: {:.4f}, std: {:.4f}".format(cv_mean, cv_std))

    # Plot K-Fold results: bar plot per fold with mean line and ±std band
    try:
        folds = np.arange(1, len(cv_scores) + 1)
        plt.figure(figsize=(8,4))
        sns.barplot(x=folds, y=cv_scores, palette="Blues_d")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Fold")
        plt.ylabel("F1 macro score")
        plt.title("K-Fold Cross-Validation F1 (macro) per fold")
        # mean line and shaded std
        plt.axhline(cv_mean, color="red", linestyle="--", label=f"Mean = {cv_mean:.4f}")
        plt.fill_between([0.5, len(folds)+0.5], cv_mean - cv_std, cv_mean + cv_std, color="red", alpha=0.1, label=f"±1 std = {cv_std:.4f}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(cv_png)
        plt.close()
        print("Saved K-Fold CV plot to", cv_png)
    except Exception as e:
        print("Warning: failed to plot CV scores:", e)

    # Train final model on full training set
    rf.fit(X_train_scaled, y_train)
    save_pickle(rf, model_path)
    print("Saved model to", model_path)

    # Evaluate on held-out test set
    y_pred = rf.predict(X_test_scaled)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Yellow(0)", "Orange(1)", "Red(2)"],
        yticklabels=["Yellow(0)", "Orange(1)", "Red(2)"]
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()
    print("Saved confusion matrix to", cm_png)

    # Feature importance plot
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:\n", importances)

    plt.figure(figsize=(6,4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(fi_png)
    plt.close()
    print("Saved feature importances to", fi_png)

    print("\nTest F1 (macro):", f1_score(y_test, y_pred, average="macro"))
    print("\nTraining complete.\n")

if __name__ == "__main__":
    main()
