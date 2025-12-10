import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------------------
# LOAD TRAINED MODEL + SCALER
# -------------------------------------------------------------
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🩺 Breast Cancer Detection using KNN")
st.write("This app predicts whether a breast tumor is **Malignant (0)** or **Benign (1)** using a trained KNN model.")

st.markdown("---")
st.header("🔢 Enter Patient Feature Values")

# -------------------------------------------------------------
# FEATURE NAMES (EXACT ORDER FROM DATASET)
# -------------------------------------------------------------
feature_names = [
    'radius_mean','texture_mean','perimeter_mean','area_mean',
    'smoothness_mean','compactness_mean','concavity_mean',
    'concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave points_se','symmetry_se',
    'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst',
    'area_worst','smoothness_worst','compactness_worst','concavity_worst',
    'concave points_worst','symmetry_worst','fractal_dimension_worst'
]

inputs = []

# -------------------------------------------------------------
# TAKE INPUT FROM USER
# -------------------------------------------------------------
cols = st.columns(3)

i = 0
for feature in feature_names:
    with cols[i % 3]:
        value = st.number_input(f"{feature}", min_value=0.0, format="%.5f")
        inputs.append(value)
    i += 1

st.markdown("---")

# -------------------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------------------
if st.button("🔍 Predict"):
    data = np.array([inputs])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success("🟢 **Benign Tumor (No Cancer Detected)**")
    else:
        st.error("🔴 **Malignant Tumor (Cancer Detected)**")

    st.info("Prediction complete using KNN model.")

st.markdown("---")
st.caption("Developed for Lab 10 - KNN Classification Project")
