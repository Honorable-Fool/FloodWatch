"""
train_nlp.py
Trains TF-IDF + Naive Bayes model using combined Source and Paraphrased sheets.
Generates comprehensive analysis reports (Confusion Matrix, Keywords, Predictions).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent # src/
ROOT_DIR = SCRIPT_DIR.parent                 # floodwatch-nlp/

# Data Path
DATA_FILE_XLSX = ROOT_DIR / "data" / "Nlp dataset.xlsx"

# Output Paths (Saved in the parent folder models_nlp for access by app.py)
MODEL_DIR = ROOT_DIR / "models_nlp"
MODEL_FILE = MODEL_DIR / "nb_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "tfidf.pkl"

# Report Files
PREDICTIONS_CSV = MODEL_DIR / "nlp_predictions.csv"       
VOCAB_CSV = MODEL_DIR / "tfidf_vocabulary.csv"            
FULL_DATASET_CSV = MODEL_DIR / "nlp_dataset_tfidf_analysis.csv" 
SAMPLE_VECTORS_CSV = MODEL_DIR / "tfidf_sample_vectors.csv"

SOURCE_SHEET = 'Source'
PARAPHRASED_SHEET = 'Paraphrased'

def ensure_directories():
    """Create necessary directories for models and reports"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"✓ Models directory: {MODEL_DIR}")

def load_and_clean_data():
    """Load, combine, and preprocess the flood advisory datasets from Excel"""
    print("\n" + "="*60)
    print("LOADING AND COMBINING DATASETS")
    print(f"File: {DATA_FILE_XLSX}")
    print("="*60)

    try:
        if not DATA_FILE_XLSX.exists():
            print(f"✗ File not found: {DATA_FILE_XLSX}")
            return None

        # Load both sheets
        df_source = pd.read_excel(DATA_FILE_XLSX, sheet_name=SOURCE_SHEET, usecols=['Advisory Text', 'Level'])
        df_para = pd.read_excel(DATA_FILE_XLSX, sheet_name=PARAPHRASED_SHEET, usecols=['Advisory Text', 'Level'])
        
        # Combine
        df = pd.concat([df_source, df_para], ignore_index=True)
        print(f"✓ Loaded {len(df)} total rows")
        
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return None

    # Cleaning: Remove rows with missing levels or empty text
    df = df.dropna(subset=['Level'])
    df = df[pd.to_numeric(df['Level'], errors='coerce').notna()]
    df['Level'] = df['Level'].astype(int)
    df['Advisory Text'] = df['Advisory Text'].astype(str)
    df = df.drop_duplicates(subset=['Advisory Text'])
    df = df[df['Advisory Text'].str.strip().astype(bool)]
    
    df = df.rename(columns={'Advisory Text': "Advisory", 'Level': "Level"})
    return df

def analyze_full_dataset(vectorizer, df):
    """
    Runs the entire NLP Dataset through the vectorizer to analyze keywords.
    Exports 'nlp_dataset_tfidf_analysis.csv' showing keywords for every row.
    """
    print("\nGENERATING FULL DATASET ANALYSIS...")
    
    # 1. Transform the WHOLE dataset
    tfidf_matrix = vectorizer.transform(df['Advisory'])
    feature_names = vectorizer.get_feature_names_out()
    
    analysis_data = []
    
    # 2. Iterate through every row to extract top keywords
    for i in range(len(df)):
        row = tfidf_matrix[i]
        indices = row.indices
        values = row.data
        
        # Sort words by importance (highest TF-IDF score first)
        sorted_items = sorted(zip(indices, values), key=lambda x: x[1], reverse=True)
        
        # Format: "word(score)"
        keywords = [f"{feature_names[idx]}({val:.2f})" for idx, val in sorted_items]
        
        analysis_data.append({
            'Row ID': i + 1,
            'Original Advisory': df.iloc[i]['Advisory'],
            'Level': df.iloc[i]['Level'],
            'Keywords Extracted': ", ".join(keywords) 
        })
        
    # 3. Save to CSV
    pd.DataFrame(analysis_data).to_csv(FULL_DATASET_CSV, index=False)
    print(f"✓ Full dataset analysis saved to: {FULL_DATASET_CSV}")

    # 4. Save Vocabulary List (Global Word Importance)
    idf_scores = vectorizer.idf_
    vocab_df = pd.DataFrame({
        'Word': feature_names,
        'Global Importance (IDF)': idf_scores
    }).sort_values(by='Global Importance (IDF)', ascending=False)
    vocab_df.to_csv(VOCAB_CSV, index=False)
    print(f"✓ Global vocabulary saved to: {VOCAB_CSV}")

def train_model(df):
    """Train TF-IDF + Naive Bayes classifier pipeline"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    X = df['Advisory']
    y = df['Level']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create Vectorizer (Converts text to numbers)
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1, 2), # Use unigrams ("rain") and bigrams ("heavy rain")
        min_df=2, 
        lowercase=True,
        stop_words='english'
    )
    
    # Fit on TRAINING data only
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Naive Bayes Model
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    print("✓ Model trained")
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Accuracy: {accuracy*100:.2f}%")

    # --- INSERTED CLASSIFICATION REPORT HERE ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- ANALYSIS PHASE ---
    analyze_full_dataset(vectorizer, df)

    # Save Test Predictions for manual inspection
    results_df = pd.DataFrame({
        'Advisory Text': X_test.values,
        'Actual Level': y_test.values,
        'Predicted Level': y_pred,
        'Correct': y_test.values == y_pred
    })
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    
    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'confusion_matrix.png')
    plt.close()
    
    return model, vectorizer, accuracy

def save_model(model, vectorizer):
    """Save trained model and vectorizer to disk"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    
    print(f"✓ Model saved: {MODEL_FILE}")
    print(f"✓ Vectorizer saved: {VECTORIZER_FILE}")

def main():
    print("\n" + "="*60)
    print("FLOODWATCH NLP MODEL TRAINING")
    print("="*60)
    
    ensure_directories()
    
    df = load_and_clean_data()
    if df is None or len(df) == 0:
        print("\n✗ Training failed. Check input data.")
        return
    
    model, vectorizer, accuracy = train_model(df)
    
    save_model(model, vectorizer)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()