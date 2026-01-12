import streamlit as st
from PIL import Image  
import joblib
from data import UserCreate
import pandas as pd
model=joblib.load('model.joblib')
img = Image.open("heart-img.png") 
st.image(img, width=200) 
st.title("heart-disease-predictor")
st.header("enter the values ")


age = st.number_input("Age", min_value=1, max_value=120, value=45)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=240)
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

cp_type = st.selectbox(
    "Chest Pain Type (string)",
    options=["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
)

sex_str = st.selectbox("Sex (string)", options=["male", "female"])
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "cp_type": cp_type
    }])
    

    pred = model.predict(input_df)[0]          
   
    if pred=="disease":
        st.error("heart disease = yes")
    else:
        st.success("heart disease = NO")
