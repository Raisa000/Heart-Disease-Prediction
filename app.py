import streamlit as st
import numpy as np
import pickle
import time

# ---------------------------
# 🎨 Custom CSS for Aesthetic UI
# ---------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #f7f9fc;}
    .title {text-align: center; font-size: 40px; font-weight: 900; color: #D64550;}
    .subtitle {text-align: center; font-size: 20px; color: #555;}
    .footer {text-align:center; padding-top:20px; color:#888;}
    .stButton>button {
        background-color: #D64550;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #b73a42;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------
# ▒ Streamlit Gauge Component
# ---------------------------
def streamlit_gauge(label, value):
    """
    A Streamlit-only circular gauge meter (0-100%).
    """
    st.markdown(f"""
        <style>
        .gauge-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 15px;
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
            color: #333;
        }}
        </style>
        <div class="gauge-wrapper">
            <div class="gauge">{value:.0f}%</div>
        </div>
        <p style="text-align:center; font-weight:600">{label}</p>
    """, unsafe_allow_html=True)


# ---------------------------
# 🧠 Load Stacking Model
# ---------------------------
model = pickle.load("models/stack_model.pkl")


# ---------------------------
# 🎯 Title & Subtitle
# ---------------------------
st.markdown("<p class='title'>❤️ Heart Disease Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered Stacking Model for accurate risk analysis</p>", unsafe_allow_html=True)


# ---------------------------
# 📥 Input Fields (Neat Layout)
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("CA", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])


# ---------------------------
# 🔘 Prediction Button
# ---------------------------
if st.button("Predict"):
    with st.spinner("Analyzing your health data..."):
        time.sleep(1)

        # Prepare input
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

        # Predict
        prediction = model.predict(data)[0]
        score = model.predict_proba(data)[0][1]

        # Output message
        if prediction == 1:
            st.error(f"⚠️ High chance of Heart Disease (Confidence: {score:.2f})")
        else:
            st.success(f"✔️ No Heart Disease Detected (Confidence: {score:.2f})")

        # Gauge Meter
        risk_percent = score * 100
        streamlit_gauge("Heart Disease Risk Level", risk_percent)


# ---------------------------
# Footer
# ---------------------------
st.markdown("<p class='footer'>Developed with ❤️ by Saidul</p>", unsafe_allow_html=True)
