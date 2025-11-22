import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import time

# ------------------------------------------------------
# Load Model Bundle
# ------------------------------------------------------
MODEL_PATH = "loan_prediction_model.pkl"

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["best_model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_column"]
except Exception as e:
    st.error(f"❌ Model Loading Error: {e}")
    st.stop()

# ------------------------------------------------------
# Page Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="centered",
)

# ------------------------------------------------------
# Custom CSS Styling (Glass UI + Animations)
# ------------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #e2ebf0 0%, #ffffff 100%);
}

.title {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
    color: #243447;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
    color: #566573;
}

.glass-box {
    background: rgba(255, 255, 255, 0.75);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    backdrop-filter: blur(12px);
    animation: fadeIn 1.3s ease-in-out;
}

label {
    font-weight: 600 !important;
}

.stButton>button {
    background: #3498db;
    color: white;
    padding: 14px 28px;
    border-radius: 12px;
    border: none;
    font-size: 20px;
    transition: 0.3s;
}

.stButton>button:hover {
    background: #2980b9;
    transform: translateY(-3px);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0px); }
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Title Section
# ------------------------------------------------------
st.markdown("<div class='title'>💳 Loan Approval Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill out the applicant information below</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# Glass UI Container
# ------------------------------------------------------
st.markdown("<div class='glass-box'>", unsafe_allow_html=True)

# Input fields dictionary
inputs = {}

col1, col2 = st.columns(2)

# Proper Labels for Every Feature
label_map = {
    "Gender": "Gender (0 = Male, 1 = Female)",
    "Married": "Married (0 = No, 1 = Yes)",
    "Education": "Education (0 = Not Graduate, 1 = Graduate)",
    "Self_Employed": "Self Employed (0 = No, 1 = Yes)",
    "ApplicantIncome": "Applicant Income (Monthly)",
    "CoapplicantIncome": "Co-applicant Income (Monthly)",
    "LoanAmount": "Loan Amount (in Thousands)",
    "Loan_Amount_Term": "Loan Term (Days)",
    "Credit_History": "Credit History (0 = Bad, 1 = Good)",
    "Property_Area": "Property Area (0 = Rural, 1 = Semi-Urban, 2 = Urban)",
}

for feature in feature_columns:

    # pick a readable label
    label = label_map.get(feature, feature)

    # Numeric input boxes
    if any(key in feature.lower() for key in ["income", "amount", "term"]):
        with col1:
            inputs[feature] = st.number_input(f"📌 {label}", min_value=0, step=1)

    # Binary fields (0/1)
    elif feature in ["Gender", "Married", "Education", "Self_Employed", "Credit_History"]:
        with col2:
            inputs[feature] = st.selectbox(f"🔘 {label}", [0, 1])

    # Categorical (0/1/2)
    elif "property" in feature.lower():
        with col2:
            inputs[feature] = st.selectbox(f"🏡 {label}", [0, 1, 2])

    # Default fallback
    else:
        with col1:
            inputs[feature] = st.number_input(f"📌 {label}", min_value=0, step=1)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------
# Predict Button
# ------------------------------------------------------
st.write("")
predict_btn = st.button("🚀 Predict Loan Approval")

if predict_btn:
    try:
        # Create DataFrame
        df = pd.DataFrame([[inputs[f] for f in feature_columns]], columns=feature_columns)

        with st.spinner("⏳ Evaluating loan application..."):
            time.sleep(1.5)  # Animation delay
            scaled = scaler.transform(df)
            pred = model.predict(scaled)[0]

        if pred == 1:
            st.success("🎉 **Loan Approved! Congratulations!**")
            st.balloons()
        else:
            st.error("❌ **Loan Rejected. Criteria not met.**")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
