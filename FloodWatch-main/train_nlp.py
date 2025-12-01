"""
train_nlp.py
Trains TF-IDF + Naive Bayes model for flood advisory classification
Saves trained model to models/ directory
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

# Configuration
DATA_FILE = "Nlp dataset.xlsx"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf.pkl")

def ensure_directories():
    """Create necessary directories"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"✓ Models directory: {MODEL_DIR}")

def load_and_clean_data():
    """Load and preprocess the flood advisory dataset"""
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)

    try:
        # Load the data assuming NO header (header=None) to use reliable numerical indices.
        df = pd.read_excel(DATA_FILE, header=None)
        print(f"✓ Loaded: {DATA_FILE}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

    if df.shape[1] < 2:
        print("✗ Dataset has fewer than 2 columns. Check file format.")
        return None

    # Select the first two columns (index 0 and 1) and rename them
    df = df[[0, 1]]
    df = df.rename(columns={0: "Advisory", 1: "Level"})

    print(f"Original shape ( with potential header row): {df.shape[0]} rows")

    # Skip the first row if it contains the header string 'Level', which causes the ValueError.
    if df.iloc[0]['Level'] == 'Level':
        df = df.iloc[1:] 

    # Drop missing values
    df = df.dropna()
    print(f"After removing header/nulls: {df.shape[0]} rows")

    # Convert Level to int (This will now succeed)
    df = df[pd.to_numeric(df['Level'], errors='coerce').notna()]
    df['Level'] = df['Level'].astype(int)

    # Remove duplicates... (The rest of the function remains the same)
    df = df.drop_duplicates(subset=['Advisory'])
    print(f"After cleaning and deduplication: {df.shape[0]} rows")

    print("\nClass Distribution:")
    class_counts = df['Level'].value_counts().sort_index()
    for level, count in class_counts.items():
         level_names = {0: "Yellow", 1: "Orange", 2: "Red"}
         print(f"  Level {level} ({level_names.get(level, 'Unknown')}): {count} samples")

    return df

def train_model(df):
    """Train TF-IDF + Naive Bayes classifier"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Split features and labels
    X = df['Advisory']
    y = df['Level']
    
    # Train-test split (80-20)
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
        max_features=1000,      # Top 1000 features
        ngram_range=(1, 2),     # Unigrams and bigrams
        min_df=2,               # Ignore rare terms
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
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluation
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    target_names = ['Level 0 (Yellow)', 'Level 1 (Orange)', 'Level 2 (Red)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Yellow (0)', 'Orange (1)', 'Red (2)'],
                yticklabels=['Yellow (0)', 'Orange (1)', 'Red (2)'])
    plt.title('Confusion Matrix - Flood Advisory Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # Top predictive features
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

def test_predictions(model, vectorizer):
    """Test model with sample predictions"""
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    test_cases = [
        "Light to moderate rains are expected over Dasmariñas, Cavite",
        "Heavy to intense rainfall is forecast over Dasmariñas area",
        "Flooding is possible in low-lying areas",
        "Road/street flooding is possible due to impounded water and high tide",
        "Torrential rains expected. Serious flooding anticipated. Evacuate immediately",
        "Scattered rains are likely but will not cause adverse impact"
    ]
    
    level_names = {
        0: "Yellow Warning (Low)",
        1: "Orange Warning (Moderate)",
        2: "Red Warning (High)"
    }
    
    for text in test_cases:
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        print(f" Text: {text[:65]}...")
        print(f"Prediction: Level {prediction} - {level_names[prediction]}")
        
        # Show probabilities
        prob_str = " | ".join([
            f"L{i}={prob:.1%}" for i, prob in enumerate(probabilities)
        ])
        print(f"Confidence: {prob_str}")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("FLOODWATCH NLP MODEL TRAINING")
    print("TF-IDF + Naive Bayes Classifier")
    print("="*60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Load data
    df = load_and_clean_data()
    if df is None or len(df) == 0:
        print("\n✗ No data to train on. Please check your dataset.")
        return
    
    # Train model
    model, vectorizer, accuracy = train_model(df)
    
    # Save model
    save_model(model, vectorizer)
    
    # Test predictions
    test_predictions(model, vectorizer)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Final Accuracy: {accuracy*100:.2f}%")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python app.py")
    print("  2. Test API at: http://127.0.0.1:5000")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
