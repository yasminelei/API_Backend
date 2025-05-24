# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and feature list
cls_model = joblib.load('trained_data/student_classifier_model.pkl')
features = joblib.load('trained_data/model_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Prepare the input dictionary with default fallback values
    input_data = {
        'Study_Hours_per_Week': data.get('study_hours_per_week', 0),
        'Sleep_Hours_per_Night': data.get('sleep_hours_per_night', 0),
        'Attendance (%)': data.get('attendance', 0),
        'Participation_Score': data.get('participation_score', 0)
    }

    # Convert input to DataFrame and align feature columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Predict probability and result
    prob = cls_model.predict_proba(input_df)[0]
    prediction = int(prob[1] >= 0.5)
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "label": label,               # "Pass" or "Fail"
        "confidence": round(prob[1] * 100, 2)  # in %
    })

if __name__ == '__main__':
    app.run(debug=True)
