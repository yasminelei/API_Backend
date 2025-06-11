from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and features
try:
    model = joblib.load('trained_data/student_classifier_model.pkl')
    features = joblib.load('trained_data/model_features.pkl')  # list of feature column names
    logger.info("Model and features loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or features: {e}")
    model = None
    features = []

@app.route('/')
def home():
    return "🎓 Welcome to the Student Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not features:
        return jsonify({'error': 'Model or features not loaded properly'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Extract and validate input values
    try:
        input_data = {
            'Study_Hours_per_Week': float(data.get('study_hours_per_week', 0)),
            'Sleep_Hours_per_Night': float(data.get('sleep_hours_per_night', 0)),
            'Attendance (%)': float(data.get('attendance', 0)),
            'Participation_Score': float(data.get('participation_score', 0))
        }
    except ValueError:
        return jsonify({'error': 'All inputs must be numeric values'}), 400

    logger.info(f"Received input: {input_data}")

    # Create DataFrame and align with feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Predict
    try:
        prob = model.predict_proba(input_df)[0]
        prediction = int(prob[1] >= 0.5)
        label = "Pass" if prediction == 1 else "Fail"
        confidence = round(prob[1] * 100, 2)

        return jsonify({
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
