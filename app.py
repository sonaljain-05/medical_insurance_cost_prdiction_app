import streamlit as st
from PIL import Image
import joblib
import numpy as np

st.set_page_config(layout="wide")

model = joblib.load("best_model.pkl")

st.markdown("""
<style>
div[data-testid="stRadio"] label {
    color: #2e7d32 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

img = Image.open("picture/medical.png")
img = img.resize((500, 900))

col1, col2 = st.columns(2)

with col1:
    st.image(img)

with col2:
    st.markdown("<h1 style='color:#0ba8b0;'>Medical Prediction App</h1>", unsafe_allow_html=True)

    st.image("picture/globe.png", width=50)

    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    st.image("picture/age.png", width=50)
    age = st.selectbox("Age", list(range(1, 101)), index=26)

    st.image("picture/bmi.png", width=50)
    bmi = st.number_input("BMI", 10.0, 60.0, 24.0)

    st.image("picture/children.png", width=50)
    children = st.number_input("Children", 0, 10, 0)

    st.image("picture/sex.png", width=50)
    sex = st.radio("Sex", ["male", "female"])

    st.image("picture/smoker.png", width=50)
    smoker = st.radio("Smoker", ["yes", "no"])

    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0

    region_northeast = 1 if region == "northeast" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    if st.button("Predict"):

        prediction = model.predict([[
            age,
            bmi,
            children,
            sex,
            smoker,
            region_northeast,
            region_northwest,
            region_southeast,
            region_southwest
        ]])

        actual_cost = np.exp(prediction[0])

        st.success(f"Predicted Medical Cost: ₹ {actual_cost:,.2f}")
