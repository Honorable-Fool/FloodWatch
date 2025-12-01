"""
FloodWatch Backend API
Flask server with TF-IDF + Naive Bayes NLP model
Receives user inputs and returns flood predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Safe Path Handling
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "nb_model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "tfidf.pkl")

# Load trained model and vectorizer
try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    print("✓ Model and vectorizer loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    print("Run: python train_model.py")
    model = None
    vectorizer = None
    MODEL_LOADED = False

# Advisory Messages
ADVISORY_MESSAGES = {
    0: {
        "level_name": "Low Chance (Yellow Warning)",
        "advisory": "Light to moderate rains expected over {location}. Flooding possible in low-lying areas.",
        "color": "#FFC107",
        "icon": "⚠️"
    },
    1: {
        "level_name": "Moderate Chance (Orange Warning)",
        "advisory": "Heavy rainfall forecast in {location}. Road flooding likely. Prepare supplies and avoid travel.",
        "color": "#FF9800",
        "icon": "🔶"
    },
    2: {
        "level_name": "High Chance (Red Warning)",
        "advisory": "Intense to torrential rains expected in {location}. Serious flooding anticipated. Evacuate immediately.",
        "color": "#F44336",
        "icon": "🔴"
    }
}

@app.route("/api/barangays")
def barangays():
    return jsonify({
        "barangays": [
            "Baan", "San Vicente", "Pagatpatan", "Ambago",
            "Datu Silongan", "Dagohoy", "Tandang Sora"
        ]
    })

def generate_advisory_text(location, duration, intensity):
    if intensity < 2.5:
        advisory = f"Light rains are expected over {location}"
    elif intensity < 7.5:
        advisory = f"Light to moderate rains are expected over {location}"
    elif intensity < 15:
        advisory = f"Moderate to heavy rains are expected over {location}"
    elif intensity < 30:
        advisory = f"Heavy to intense rainfall is forecast over {location}"
    else:
        advisory = f"Torrential rains are expected over {location}"

    # Duration
    if duration <= 1:
        advisory += " within the next hour"
    elif duration <= 3:
        advisory += f" within the next {duration} hours"
    elif duration <= 6:
        advisory += f" over the next {duration} hours"
    else:
        advisory += " for an extended period"

    advisory += ". "

    # Flood risk
    if intensity < 7.5:
        advisory += "Flooding is possible in low-lying areas."
    elif intensity < 15:
        advisory += "Road flooding is possible in several areas."
    else:
        advisory += "Serious flooding expected. Prepare for evacuation."

    return advisory


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "FloodWatch NLP Backend",
        "model_loaded": MODEL_LOADED,
        "version": "1.0"
    })


# ---------------------------------------------------------
# FINAL Predict Route (fixed — no duplicates)
# ---------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        location = data.get('location', '').strip()
        duration = float(data.get('duration', 0))
        intensity = float(data.get('intensity', 0))

        if not location:
            return jsonify({"error": "Location cannot be empty"}), 400
        if duration <= 0 or intensity <= 0:
            return jsonify({"error": "Duration and intensity must be positive"}), 400

    except Exception:
        return jsonify({"error": "Invalid input format"}), 400

    try:
        advisory_text = generate_advisory_text(location, duration, intensity)
        text_tfidf = vectorizer.transform([advisory_text])

        prediction = int(model.predict(text_tfidf)[0])
        probabilities = model.predict_proba(text_tfidf)[0]

        confidence = {
            f"level_{i}": round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        level_info = ADVISORY_MESSAGES[prediction]

        return jsonify({
            "level": prediction,
            "level_name": level_info["level_name"],
            "advisory": level_info["advisory"].format(location=location),
            "color": level_info["color"],
            "icon": level_info["icon"],
            "confidence": confidence,
            "generated_text": advisory_text
        })

    except Exception as e:
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@app.route("/model-info")
def model_info():
    if not MODEL_LOADED:
        return jsonify({"loaded": False})

    return jsonify({
        "loaded": True,
        "model_type": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF",
        "num_features": len(vectorizer.get_feature_names_out()),
        "classes": model.classes_.tolist()
    })


# Run Flask server
if __name__ == "__main__":
    print("Starting FloodWatch API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
