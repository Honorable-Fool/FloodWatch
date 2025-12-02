"""
train_nlp.py
Trains TF-IDF + Naive Bayes model using combined Source and Paraphrased sheets
from a single Excel file.
Saves trained model to models/ directory.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# NOTE: The Excel file MUST be in the same folder as this script.
DATA_FILE_XLSX = "Nlp dataset.xlsx" # <--- ASSUMED NAME OF YOUR EXCEL FILE
SOURCE_SHEET = 'Source' # <--- Check your actual sheet name for Source data
PARAPHRASED_SHEET = 'Paraphrased' # <--- Check your actual sheet name for Paraphrased data

MODEL_DIR = "models_nlp"
MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf.pkl")

def ensure_directories():
    """Create necessary directories"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"✓ Models directory: {MODEL_DIR}")

def load_and_clean_data():
    """Load, combine, and preprocess the flood advisory datasets from Excel sheets"""
    print("\n" + "="*60)
    print("LOADING AND COMBINING DATASETS FROM EXCEL SHEETS")
    print("="*60)

    try:
        # Load data from the first sheet (Source)
        df_source = pd.read_excel(
            DATA_FILE_XLSX, 
            sheet_name=SOURCE_SHEET, 
            usecols=['Advisory Text', 'Level']
        )
        print(f"✓ Loaded Sheet '{SOURCE_SHEET}': {len(df_source)} rows")
        
        # Load data from the second sheet (Paraphrased)
        df_paraphrased = pd.read_excel(
            DATA_FILE_XLSX, 
            sheet_name=PARAPHRASED_SHEET, 
            usecols=['Advisory Text', 'Level']
        )
        print(f"✓ Loaded Sheet '{PARAPHRASED_SHEET}': {len(df_paraphrased)} rows")
        
        # Combine the datasets
        df = pd.concat([df_source, df_paraphrased], ignore_index=True)
        print(f"Initial combined shape: {df.shape[0]} rows")
        
    except Exception as e:
        print(f"✗ Failed to load or combine datasets. Check file/sheet names and openpyxl installation.")
        print(f"Error details: {e}")
        return None

    # --- Cleaning Steps ---
    
    # Drop rows where 'Level' is missing
    df = df.dropna(subset=['Level'])
    
    # Convert 'Level' to int, ensuring no non-numeric values remain
    df = df[pd.to_numeric(df['Level'], errors='coerce').notna()]
    df['Level'] = df['Level'].astype(int)

    # Ensure 'Advisory Text' is a string and remove duplicates
    df['Advisory Text'] = df['Advisory Text'].astype(str)
    df = df.drop_duplicates(subset=['Advisory Text'])
    
    # Final cleanup: ensure no empty strings or whitespace-only texts
    df = df[df['Advisory Text'].str.strip().astype(bool)]
    
    print(f"After cleaning and deduplication: {df.shape[0]} rows")

    print("\nCombined Class Distribution:")
    class_counts = df['Level'].value_counts().sort_index()
    for level, count in class_counts.items():
        level_names = {0: "Yellow", 1: "Orange", 2: "Red"}
        print(f"  Level {level} ({level_names.get(level, 'Unknown')}): {count} samples")

    # Rename columns to match what the rest of the script expects
    df = df.rename(columns={'Advisory Text': "Advisory", 'Level': "Level"})
    
    return df

def train_model(df):
    """Train TF-IDF + Naive Bayes classifier"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Split features and labels
    X = df['Advisory']
    y = df['Level']
    
    # Train-test split (80-20), stratified for class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create TF-IDF Vectorizer
    print("\nCreating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1, 2), 
        min_df=2, 
        lowercase=True,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"  Features: {X_train_tfidf.shape[1]}")
    
    # Train Naive Bayes
    print("\nTraining Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    print("✓ Model trained")
    
    # Predictions and Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    target_names = ['Level 0 (Yellow)', 'Level 1 (Orange)', 'Level 2 (Red)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Yellow (0)', 'Orange (1)', 'Red (2)'],
                yticklabels=['Yellow (0)', 'Orange (1)', 'Red (2)'])
    plt.title('Confusion Matrix - Flood Advisory Classification (Combined Data)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # Top predictive features (kept for diagnostics)
    print("\n" + "="*60)
    print("TOP PREDICTIVE WORDS")
    print("="*60)
    
    feature_names = vectorizer.get_feature_names_out()
    level_names = ['Yellow (Low)', 'Orange (Moderate)', 'Red (High)']
    
    for i, level_name in enumerate(level_names):
        if i < len(model.feature_log_prob_):
            top_indices = model.feature_log_prob_[i].argsort()[-10:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            print(f"\n{level_name}:")
            print("  " + ", ".join(top_features))
    
    return model, vectorizer, accuracy

def save_model(model, vectorizer):
    """Save trained model and vectorizer"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    
    print(f"✓ Model saved: {MODEL_FILE}")
    print(f"✓ Vectorizer saved: {VECTORIZER_FILE}")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("FLOODWATCH NLP MODEL TRAINING (COMBINED DATA)")
    print("TF-IDF + Naive Bayes Classifier")
    print("="*60)
    
    ensure_directories()
    
    df = load_and_clean_data()
    if df is None or len(df) == 0:
        print("\n✗ Training failed. Check if your Excel file is present and sheet names are correct.")
        return
    
    model, vectorizer, accuracy = train_model(df)
    
    save_model(model, vectorizer)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Final Accuracy: {accuracy*100:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()