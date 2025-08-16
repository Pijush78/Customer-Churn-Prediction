import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and preprocessors
model = load_model("churn_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
gender_encoder = pickle.load(open("gender_encoder.pkl", "rb"))

st.title("💡 Customer Churn Prediction App")
st.write("Enter customer details below to predict if the customer will churn.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=1000.0, step=100.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
country = st.selectbox("Country", ["France", "Germany", "Spain"])

# Encode categorical features
gender_val = gender_encoder.transform([gender])[0]

# One-hot encode country (France = dropped column)
country_encoded = [0, 0]
if country == "Germany":
    country_encoded = [1, 0]
elif country == "Spain":
    country_encoded = [0, 1]

# Final features in same order as training
features = np.array([[credit_score, gender_val, age, tenure, balance, num_products,
                      has_cr_card, is_active_member, estimated_salary,
                      country_encoded[0], country_encoded[1]]])

# Scale features
features_scaled = scaler.transform(features)

# Predict
if st.button("🔮 Predict"):
    prediction = model.predict(features_scaled)[0][0]
    if prediction > 0.5:
        st.error("⚠️ This customer is likely to CHURN!")
    else:
        st.success("✅ This customer is NOT likely to churn.")
