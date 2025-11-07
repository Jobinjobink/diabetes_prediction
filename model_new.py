import numpy as np
import joblib
import pandas as pd

# Load model, scaler, and feature list
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load('features.pkl')

# User input (must follow feature order)
data = [[10, 162, 84, 0, 0, 27.7, 0.182, 54]]

# Convert to DataFrame with column names
data = pd.DataFrame(data, columns=feature_names)

# ✅ Scale input
data_scaled = scaler.transform(data)

# Predict
prediction = model.predict(data_scaled)

print("Prediction:", prediction)
if prediction[0] == 1:
    print("Result: Diabetic")
else:
    print("Result: Not Diabetic")
