import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib # Used to load your trained model pipeline

# ==========================================
# 1. LOAD YOUR TRAINED PIPELINE
# ==========================================
# Make sure to save your model first using: joblib.dump(best_model, 'churn_model.pkl')
@st.cache_resource
def load_pipeline():
    try:
        path = Path('models/telco_customer_churn_rf_v1.joblib')
        return joblib.load(path)
    except FileNotFoundError:
        st.error("⚠️ 'churn_model.pkl' not found. Please save your best_model pipeline first using joblib.")
        return None

pipeline = load_pipeline()

# ==========================================
# 2. DASHBOARD HEADER
# ==========================================
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Dashboard")
st.markdown("Enter a customer's details below to calculate their risk of leaving.")

# ==========================================
# 3. INTERACTIVE INPUT FORMS
# ==========================================
st.subheader("👤 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=5)

with col2:
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=85.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=425.0, step=10.0)

# Collapsible section for additional technical services
with st.expander("🛠️ Additional Add-on Services"):
    col3, col4 = st.columns(2)
    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with col4:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Convert "Yes"/"No" mapping back for SeniorCitizen to match your schema
senior_numeric = 1 if senior_citizen == "Yes" else 0

# ==========================================
# 4. CONSTRUCT THE NEW CUSTOMER DATAFRAME
# ==========================================
new_customer = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_numeric],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# ==========================================
# 5. ML PREDICTION AND RISK DISPLAY
# ==========================================
st.markdown("---")
if st.button("🚀 Calculate Churn Risk", type="primary"):
    if pipeline is not None:
        # Get raw probability scores from your Logistic Regression
        probabilities = pipeline.predict_proba(new_customer)[0]
        churn_prob = probabilities[1] # Probability of Churn (Class 1)
        
        # Display the result with intuitive color coding
        st.subheader("🔮 Prediction Analysis")
        
        if churn_prob > 0.5:
            st.error(f"🚨 **High Churn Risk!** This customer has a **{churn_prob:.1%}** chance of leaving.")
        elif churn_prob > 0.3:
            st.warning(f"⚠️ **Moderate Churn Risk.** This customer has a **{churn_prob:.1%}** chance of leaving.")
        else:
            st.success(f"✅ **Low Churn Risk.** This customer has a **{churn_prob:.1%}** chance of leaving.")
            
        # Optional: Display raw data payload for verification
        with st.sidebar:
            st.write("📋 **Processed Payload:**")
            st.dataframe(new_customer.T.rename(columns={0: "Value"}))
