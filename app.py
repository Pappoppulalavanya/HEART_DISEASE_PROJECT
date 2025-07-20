import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction App")

# Input fields
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.selectbox("Rest ECG Results (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("No. of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                       'restecg', 'thalach', 'exang', 'oldpeak',
                                       'slope', 'ca', 'thal'])

    prediction = model.predict(input_data)[0]
    st.success("Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected ✅")
