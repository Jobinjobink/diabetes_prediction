import streamlit as st
import pandas as pd
import joblib

# Load Model, Scaler, and Feature Names
model = joblib.load("src/train_data/diabetes_model.pkl")
scaler = joblib.load("src/train_data/scaler.pkl")
feature_names = joblib.load("src/train_data/features.pkl")


st.title("🩺 Diabetes Prediction Web App")
st.write("Enter the patient details below:")

# Create input fields
user_data = {}
for feature in feature_names:
    user_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_data = pd.DataFrame([user_data])

if st.button("Predict"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Patient is likely **Diabetic**")
    else:
        st.success("✅ Patient is **Not Diabetic**")
