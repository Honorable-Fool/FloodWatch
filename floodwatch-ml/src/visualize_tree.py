import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
from pathlib import Path

# Import path helpers
try:
    from utils import models_path
except ImportError:
    from src.utils import models_path

def visualize_sample_tree():
    # 1. Load trained Random Forest model
    model_path = models_path("random_forest_floodwatch.joblib")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please run train_model.py first.")
        return

    print(f"Loading model from {model_path}...")
    rf_model = joblib.load(model_path)

    # 2. Extract ONE tree (e.g., the first one)
    # Estimator [0], but could pick [1], [5], etc.
    single_tree = rf_model.estimators_[0]

    # 3. Setup the plot
    # 'figsize' controls the resolution. (20, 10) is usually big enough for a paper figure.
    plt.figure(figsize=(30, 10), dpi=300)

    # 4. Plot the tree with specific limits for readability
    plot_tree(single_tree, 
              max_depth=3,                    # <--- KEY CHANGE: Limits depth so it fits on a page
              feature_names=["Elevation_m", "Rainfall_mm", "Duration_hr"],
              class_names=["Yellow", "Orange", "Red"],
              filled=True,                    # Adds color
              rounded=True,                   # Rounded corners for nodes
              fontsize=12,                    # Readable font size
              precision=2)                   

    plt.title("Simplified Decision Logic (Sample Tree from Random Forest)", fontsize=15)

    # 5. Save the output
    output_path = models_path("paper_sample_tree.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"Success! Readable tree saved to: {output_path}")

if __name__ == "__main__":
    visualize_sample_tree()