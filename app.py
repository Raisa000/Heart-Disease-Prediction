import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# -----------------------------
# Load Lottie Animation
# -----------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

heart_animation = load_lottie("https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json")
doctor_animation = load_lottie("https://assets2.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
)

# -----------------------------
# Title & Animation
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(heart_animation, height=170)

st.markdown("<h1 style='text-align:center;color:#F8FAFC;font-size:40px;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#E2E8F0;font-size:20px;'>Interactive ML-based heart disease predictor</p>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return pickle.load("models/stack_model.pkl")

model = load_model()

# -----------------------------
# Load CSV to get features
# -----------------------------
df = pd.read_csv("heart.csv")
feature_names = df.drop("target", axis=1).columns.tolist()

# -----------------------------
# Input Form
# -----------------------------
st.markdown("### 🧪 Enter Patient Details")
colA, colB = st.columns(2)
user_input = {}
for i, col in enumerate(feature_names):
    with (colA if i % 2 == 0 else colB):
        user_input[col] = st.number_input(f"{col}", value=float(df[col].median()))
input_df = pd.DataFrame([user_input])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):

    with st.spinner("Running Model..."):
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1] * 100

    st.success("Prediction Complete!")

    # -----------------------------
    # Gauge Meter
    # -----------------------------
    st.markdown("### ❤️‍🔥 Risk Meter")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_prob,
        title={'text': "Heart Disease Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 35], 'color': "#10B981"},
                {'range': [35, 70], 'color': "#FACC15"},
                {'range': [70, 100], 'color': "#EF4444"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Prediction Result
    # -----------------------------
    if prediction == 1:
        st.error("### 🚨 High chance of Heart Disease")
    else:
        st.success("### 🟢 Low chance of Heart Disease")

# Footer
st_lottie(doctor_animation, height=150)
st.markdown("<p style='text-align:center;color:#E2E8F0;'>Built with ❤️ by Saidul</p>", unsafe_allow_html=True)

