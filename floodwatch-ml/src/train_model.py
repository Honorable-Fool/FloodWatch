# src/train_model.py
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz, export_text
import joblib
import os
try:
    from utils import data_path, models_path, load_dataset, save_pickle
except ImportError:
    from src.utils import data_path, models_path, load_dataset, save_pickle


# CLI / args: Allows running script with custom file paths
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="Optional CSV override")
    p.add_argument("--model-out", type=str, default=None, help="Optional model path")
    p.add_argument("--scaler-out", type=str, default=None, help="Optional scaler path")
    return p.parse_args()


# Feature extraction: Selects relevant columns for the ML model
def get_feature_matrix(df):
    """
        Features: Elevation_m, Rainfall_mm, Duration_hr
        Target: Numeric_Label (0, 1, 2)
    """
    X = df[["Elevation_m", "Rainfall_mm", "Duration_hr"]].astype(float)
    y = df["Numeric_Label"].astype(int)
    return X, y


# Main training pipeline
def main():
    args = parse_args()

    # Define paths for input data and output artifacts (models, plots)
    csv_path = Path(args.csv) if args.csv else data_path("floodWatch_MLDataset.csv")
    model_path = Path(args.model_out) if args.model_out else models_path("random_forest_floodwatch.joblib")
    scaler_path = Path(args.scaler_out) if args.scaler_out else models_path("scaler_floodwatch.joblib")

    # Paths for analysis charts
    fi_png = models_path("feature_importances.png")
    cm_png = models_path("confusion_matrix.png")
    cv_png = models_path("kfold_cv_scores.png")
    gini_png = models_path("gini_impurity_distribution.png")
    tree_dot = models_path("decision_tree.dot")
    tree_png = models_path("decision_tree.png")
    tree_rules = models_path("decision_tree_rules.txt")

    print("CSV:", csv_path)

    # 1. Load Data
    try:
        df = load_dataset(csv_path)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)

    print("\nLoaded df shape:", df.shape)
    print("Label counts:\n", df["Numeric_Label"].value_counts())

    # 2. Prepare Features
    X, y = get_feature_matrix(df)
    print("Using features:", list(X.columns))

    # 3. Train/Test Split (80% training, 20% testing)
    # Stratify ensures the class balance is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2022
    )
    print("\nTrain/test shapes:", X_train.shape, X_test.shape)

    # 4. Scaling
    # Standardize features (mean=0, variance=1) for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_pickle(scaler, scaler_path)
    print("Saved scaler to", scaler_path)

    # 5. Initialize Random Forest Model
    rf = RandomForestClassifier(
        n_estimators=300,           # Number of trees
        min_samples_leaf=3,         # Prevents overfitting
        class_weight="balanced",    # Handles imbalanced classes (if any)
        random_state=2022,
        n_jobs=-1
    )

    # 6. Cross-Validation (K-Fold)
    # Checks model stability by splitting training data into 5 parts
    cv = KFold(n_splits=5, shuffle=True, random_state=2022)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores, ddof=1)

    print("\nCV F1 macro (per fold):", cv_scores)
    print("CV F1 macro mean: {:.4f}, std: {:.4f}".format(cv_mean, cv_std))

    # Plot CV results (Bar chart of scores)
    try:
        folds = np.arange(1, len(cv_scores) + 1)
        plt.figure(figsize=(8,4))
        
        # --- FIX: Updated syntax to satisfy Future Warning ---
        sns.barplot(x=folds, y=cv_scores, hue=folds, legend=False, palette="Blues_d")
        
        plt.ylim(0.0, 1.0)
        plt.xlabel("Fold")
        plt.ylabel("F1 macro score")
        plt.title("K-Fold Cross-Validation F1 (macro) per fold")
        plt.axhline(cv_mean, color="red", linestyle="--", label=f"Mean = {cv_mean:.4f}")
        plt.fill_between([0.5, len(folds)+0.5], cv_mean - cv_std, cv_mean + cv_std, color="red", alpha=0.1,
                         label=f"±1 std = {cv_std:.4f}")
        for i, v in enumerate(cv_scores):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(cv_png)
        plt.close()
        print("Saved K-Fold CV plot to", cv_png)
    except Exception as e:
        print("Warning: failed to plot CV scores:", e)

    # 7. Final Training
    rf.fit(X_train_scaled, y_train)
    save_pickle(rf, model_path)
    print("Saved model to", model_path)

    # 8. Evaluation on Test Set
    y_pred = rf.predict(X_test_scaled)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix (Heatmap)
    try:
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
    except Exception as e:
        print("Warning: failed to create confusion matrix plot:", e)

    # Feature Importance Plot
    # Shows which inputs (Rainfall, Duration, Elevation) mattered most
    try:
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nFeature Importances:\n", importances)

        plt.figure(figsize=(6,4))
        
        # --- FIX: Updated syntax here too just in case ---
        sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False)
        
        plt.title("Feature Importances (MDI)")
        plt.tight_layout()
        plt.savefig(fi_png)
        plt.close()
        print("Saved feature importances to", fi_png)
    except Exception as e:
        print("Warning: failed to compute / plot feature importances:", e)


    # Gini Impurity Distribution 
    try:
        all_impurities = np.concatenate([est.tree_.impurity for est in rf.estimators_])
        plt.figure(figsize=(8,4))
        sns.histplot(all_impurities, bins=50, kde=True)
        plt.xlabel("Node Gini impurity")
        plt.ylabel("Count")
        plt.title("Distribution of Node Gini Impurity (all trees)")
        mean_imp = float(np.mean(all_impurities))
        std_imp = float(np.std(all_impurities, ddof=1))
        plt.axvline(mean_imp, color="red", linestyle="--", label=f"Mean={mean_imp:.4f}")
        plt.fill_betweenx([0, plt.gca().get_ylim()[1]], mean_imp - std_imp, mean_imp + std_imp, color="red", alpha=0.08,
                          label=f"±1 std = {std_imp:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(gini_png)
        plt.close()
        print("Saved Gini impurity distribution to", gini_png)
    except Exception as e:
        print("Warning: failed to plot gini impurity distribution:", e)


    # Export Decision Tree Visualization (Graphviz)
    try:
        estimator = rf.estimators_[0]
        # Text rules
        try:
            rules = export_text(estimator, feature_names=list(X.columns))
            with open(tree_rules, "w", encoding="utf-8") as f:
                f.write(rules)
            print("Saved decision tree rules to", tree_rules)
        except Exception as e:
            print("Warning: failed to export textual rules:", e)

        # Visual Dot file
        try:
            export_graphviz(
                estimator,
                out_file=str(tree_dot),
                feature_names=list(X.columns),
                class_names=["Yellow (0)", "Orange (1)", "Red (2)"],
                rounded=True,
                precision=2,
                filled=True
            )
            print("Saved decision tree DOT to", tree_dot)

            # Convert to PNG
            try:
                import graphviz
                with open(tree_dot, "r", encoding="utf-8") as f:
                    dot_graph = f.read()
                graph = graphviz.Source(dot_graph)
                graph.format = "png"
                graph.render(tree_png.with_suffix("").as_posix(), cleanup=True)
                print("Rendered decision tree PNG to", tree_png)
            except Exception as e:
                print("Graphviz rendering to PNG failed (system graphviz may be missing).", e)
                print("The DOT file remains at", tree_dot, "— you can convert it to PNG using Graphviz manually.")
        except Exception as e:
            print("Warning: failed to export decision tree DOT:", e)
    except Exception as e:
        print("Warning: could not access individual estimator to export tree:", e)

    # Final F1 Score
    print("\nTest F1 (macro):", f1_score(y_test, y_pred, average="macro"))
    print("\nTraining complete.\n")

if __name__ == "__main__":
    main()