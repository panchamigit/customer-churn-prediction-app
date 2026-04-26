import streamlit as st
import numpy as np
import pickle

# --- Load model ---
model = pickle.load(open("gb_churn_model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

# --- INPUTS (match your dataset columns) ---

credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
age = st.number_input("Age", min_value=18, max_value=100)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0.0)
products = st.number_input("Number of Products", min_value=1, max_value=4)
salary = st.number_input("Estimated Salary", min_value=0.0)

# --- Categorical inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])

# --- Convert categorical → numeric (same as training) ---
gender = 1 if gender == "Male" else 0

geo_map = {"France": 0, "Spain": 1, "Germany": 2}
geography = geo_map[geography]

# --- Create input array (ORDER MUST MATCH TRAINING) ---
input_data = np.array([[credit_score, geography, gender, age,
                        tenure, balance, products, salary]])

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")