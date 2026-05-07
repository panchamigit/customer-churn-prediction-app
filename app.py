import streamlit as st
import numpy as np
import pickle

# IMPORTANT IMPORT
from sklearn.ensemble import RandomForestClassifier


model = pickle.load(open("telecom_churn_model.pkl", "rb"))


st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    layout="centered"
)

st.title(" Telecom Customer Churn Prediction")

st.write("Enter customer details below")

# =========================================
# USER INPUTS
# =========================================

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges ($)",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges ($)",
    min_value=0.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL (Digital Subscriber Line)",
        "Fiber Optic",
        "No Internet"
    ]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# =========================================
# ENCODING
# =========================================

gender = 1 if gender == "Male" else 0

senior = 1 if senior == "Yes" else 0

# Contract Encoding
if contract == "Month-to-month":
    contract = 0
elif contract == "One year":
    contract = 1
else:
    contract = 2

# Internet Service Encoding
if internet_service == "DSL (Digital Subscriber Line)":
    internet_service = 0
elif internet_service == "Fiber Optic":
    internet_service = 1
else:
    internet_service = 2

# Payment Method Encoding
payment_dict = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

payment_method = payment_dict[payment_method]

input_data = np.array([[
    gender,
    senior,
    tenure,
    monthly_charges,
    total_charges,
    contract,
    internet_service,
    payment_method
]])

if st.button("Predict Churn"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")