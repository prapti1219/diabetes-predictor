import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
model = RandomForestClassifier()
model.fit(X, y)
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health parameters to check diabetes risk")

preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("🔴 You may be diabetic!")
    else:
        st.success("🟢 You are not diabetic.")