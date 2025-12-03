import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import requests
from streamlit_lottie import st_lottie

# ---------------------------------------------------
# Load Lottie Animations
# ---------------------------------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

heart_animation = load_lottie("https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json")
doctor_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
)

# ---------------------------------------------------
# Background Styling
# ---------------------------------------------------
background_style = """
<style>
    .main {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    .stApp {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
        color: #F8FAFC;
        text-align: center;
    }
    .sub-font {
        font-size: 20px !important;
        color: #E2E8F0;
        text-align: center;
    }
</style>
"""
st.markdown(background_style, unsafe_allow_html=True)

# ---------------------------------------------------
# Title + Animation
# ---------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(heart_animation, height=170)

st.markdown("<h1 class='big-font'>Heart Disease Prediction (Stacking Model)</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-font'>A modern interactive medical ML app.</p>", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model with Pickle
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/stack_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------------------------------
# Load Feature Names
# ---------------------------------------------------
df = pd.read_csv("heart.csv")
feature_names = df.drop("target", axis=1).columns.tolist()

# ---------------------------------------------------
# Streamlit-Only Gauge Component
# ---------------------------------------------------
def streamlit_gauge(label, value):
    """
    Custom circular gauge made using Streamlit HTML + CSS.
    value: 0–100
    """
    st.markdown(f"""
        <style>
        .gauge-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
        }}
        .gauge {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: conic-gradient(
                #ff4b4b {value * 3.6}deg,
                #e6e6e6 {value * 3.6}deg
            );
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 30px;
            font-weight: bold;
            color: #ffffff;
        }}
        </style>
        <div class="gauge-wrapper">
            <div class="gauge">{value:.0f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='text-align:center;font-weight:600;color:white'>{label}</p>", unsafe_allow_html=True)


# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
st.markdown("### 🧪 Enter Patient Information")

user_input = {}

colA, colB = st.columns(2)
for i, col in enumerate(feature_names):
    if i % 2 == 0:
        with colA:
            user_input[col] = st.number_input(
                col, value=float(df[col].median())
            )
    else:
        with colB:
            user_input[col] = st.number_input(
                col, value=float(df[col].median())
            )

input_df = pd.DataFrame([user_input])

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
if st.button("🔍 Predict", use_container_width=True):

    with st.spinner("Analyzing patient data..."):
        time.sleep(1)
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1] * 100

    st.success("Prediction completed!")

    # Gauge Meter
    streamlit_gauge("Heart Disease Risk", prediction_prob)

    # Result Box
    if prediction == 1:
        st.error("### 🚨 High chance of Heart Disease")
    else:
        st.success("### 🟢 Low chance of Heart Disease")

# ---------------------------------------------------
# Footer Animation
# ---------------------------------------------------
st_lottie(doctor_animation, height=150)
st.markdown("<p class='sub-font'>Built with ❤️ by Saidul</p>", unsafe_allow_html=True)
