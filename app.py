import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# Load the saved model bundle
# ------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("readmission_model_bundle.joblib")
    return bundle["model"], bundle["feature_names"]

model, feature_names = load_model()

st.title("🏥 Hospital Readmission Predictor (30-Day) – XGBoost")
st.write("Enter encounter details below to predict the probability of readmission within 30 days.")

# --------------------------------------
# Helper: Create an empty input template
# --------------------------------------
def create_input_template():
    return pd.DataFrame(columns(feature_names)).iloc[:1].fillna(0)

# --------------------------------------
# Sidebar Inputs (example key features)
# --------------------------------------
st.sidebar.header("Patient Encounter Details")

age = st.sidebar.selectbox("Age Group", [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
])

time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 20, 5)
num_medications = st.sidebar.slider("Number of Medications", 1, 50, 10)
num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 100, 40)

a1c_result = st.sidebar.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
glu_result = st.sidebar.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])

diabetes_med = st.sidebar.selectbox("Diabetes Medication?", ["Yes", "No"])
change_med = st.sidebar.selectbox("Medication Changed?", ["Yes", "No"])

# --------------------------------------
# Convert to numeric (same mapping as training)
# --------------------------------------
a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
binary_map = {"Yes": 1, "No": 0}

# --------------------------------------
# Build a single-row dataframe for prediction
# --------------------------------------
data = pd.DataFrame([{
    "time_in_hospital": time_in_hospital,
    "num_medications": num_medications,
    "num_lab_procedures": num_lab_procedures,
    "A1Cresult": a1c_map[a1c_result],
    "max_glu_serum": glu_map[glu_result],
    "diabetesMed": binary_map[diabetes_med],
    "change": binary_map[change_med],
    # Age one-hot
    f"age_{age}": 1,
}])

# Add missing columns (Streamlit input may not include every one-hot)
for col in feature_names:
    if col not in data.columns:
        data[col] = 0

# Reorder columns to match training data
data = data[feature_names]

# --------------------------------------
# PREDICT
# --------------------------------------
if st.button("Predict Readmission Risk"):
    pred = model.predict_proba(data)[0][1]
    #st.subheader(f"🔮 Predicted Probability of 30-Day Readmission: **{pred:.2%}**")
    st.subheader(f"🔮 Predicted Probability of 30-Day Readmission: **{pred}**")

    if pred < 0.04:
        st.success("Low risk patient. 👍")
    elif pred < 0.45:
        st.warning("Moderate risk patient. ⚠️")
    else:
        st.error("High risk of readmission. 🚨")
