
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime



# Load model and preprocessors
model = joblib.load("Documents/customer_churn_project/models/01_churn_model.pkl")
scaler = joblib.load("Documents/customer_churn_project/models/02_scaler.pkl")
encoder = joblib.load("Documents/customer_churn_project/models/03_encoder.pkl")


 # Optional: not used directly here, but mentioned in case recruiters ask

# Define mappings for categorical values (to improve UX)
gender_map = {"Female": 0, "Male": 1}
yes_no_map = {"Yes": 1, "No": 0}
internet_service_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_method_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

# Reverse mappings if needed later
inv_gender_map = {v: k for k, v in gender_map.items()}

# UI Header
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔍 Customer Churn Prediction App")
st.markdown("Use this tool to predict if a customer is likely to **churn** based on their service details.")

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", list(internet_service_map.keys()))
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
contract = st.sidebar.selectbox("Contract Type", list(contract_map.keys()))
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", list(payment_method_map.keys()))
monthly_charges = st.sidebar.slider("Monthly Charges", 10, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 10, 10000, 2000)

# Convert to encoded input
input_data = pd.DataFrame([[
    gender_map[gender],
    1 if senior == "Yes" else 0,
    yes_no_map[partner],
    yes_no_map[dependents],
    tenure,
    yes_no_map[phone_service],
    yes_no_map[multiple_lines],
    internet_service_map[internet_service],
    yes_no_map[online_security],
    yes_no_map[online_backup],
    yes_no_map[device_protection],
    yes_no_map[tech_support],
    yes_no_map[streaming_tv],
    yes_no_map[streaming_movies],
    contract_map[contract],
    yes_no_map[paperless_billing],
    payment_method_map[payment_method],
    monthly_charges,
    total_charges
]], columns=[
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
])

# Scale numerical features
scaled_input = scaler.transform(input_data)

# Predict churn
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Show result
st.subheader("🔮 Prediction Result:")
if prediction[0] == 1:
    st.error("⚠️ The customer is likely to **churn**.")
else:
    st.success("✅ The customer is likely to **stay**.")

st.write(f"**Churn Probability:** {prediction_proba[0][1]*100:.2f}%")

# Save prediction logs
log = input_data.copy()
log["prediction"] = prediction
log["prediction_proba"] = prediction_proba[0][1]
log["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
try:
    pd.read_csv("prediction_logs.csv")  # check if exists
    log.to_csv("prediction_logs.csv", mode='a', header=False, index=False)
except:
    log.to_csv("prediction_logs.csv", index=False)

# Optional: Show prediction log (recruiter-friendly UI)
with st.expander("📄 Show past prediction logs"):
    try:
        logs_df = pd.read_csv("prediction_logs.csv")
        st.dataframe(logs_df.tail(10))  # Show last 10 predictions
    except FileNotFoundError:
        st.info("No prediction logs available yet.")
