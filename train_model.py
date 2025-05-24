# train_model.py

# 📦 Import libraries for data handling, ML, and saving models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import joblib
import os

# 📁 Ensure trained_data directory exists
os.makedirs("trained_data", exist_ok=True)

# 📥 Load dataset
df = pd.read_csv("csv/Students_Grading_Dataset.csv")

# 🎯 Select input features and target label
X = df[['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Participation_Score']]
y = df['Pass_or_Fail']  # 1 = Pass, 0 = Fail (Make sure this column exists)

# 🧪 Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🛠 Define pipeline: scaling + MLP classifier
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',          # ReLU works well with MLPClassifier
        max_iter=2000,
        early_stopping=True,
        random_state=42
    ))
])

# 🧠 Train the model
clf_pipeline.fit(X_train, y_train)

# 💾 Save trained model and feature list
joblib.dump(clf_pipeline, 'trained_data/student_classifier_model.pkl')
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# ✅ Success message
print("✅ Student classification model trained and saved to 'trained_data/'")
