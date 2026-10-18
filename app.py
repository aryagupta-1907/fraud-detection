import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from geolocation import detect_impossible_travel, CITY_COORDINATES

# Page Configuration
st.set_page_config(page_title="Fraud Risk Analyzer", page_icon="🛡️", layout="wide")
st.title("🛡️ Financial Transaction Fraud Risk Analyzer")

# Load Trained Model
MODEL_PATH = "models/xgboost_fraud_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# Create Tabs for Interface
tab1, tab2 = st.tabs(["💳 Transaction Risk Scoring & SHAP", "🌍 Impossible Travel Detector"])

with tab1:
    st.header("Real-Time Transaction Risk Assessment")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Parameters")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.0)
        hour = st.slider("Hour of Day", 0, 23, 15)

    with col2:
        st.subheader("Engineered Features")
        log_amount = np.log1p(amount)
        is_night = 1 if (hour < 6 or hour > 22) else 0
        st.metric("Log Amount", f"{log_amount:.2f}")
        st.metric("Nighttime Status", "Yes" if is_night else "No")

    if st.button("Score Transaction"):
        if model is None:
            st.warning("Model not found! Run `python src/train.py` first.")
        else:
            # Create input row matching model feature columns
            input_df = pd.DataFrame(columns=model.feature_names_in_)
            input_df.loc[0] = 0 # Default initialize
            input_df['Amount'] = amount
            input_df['Time'] = hour * 3600
            input_df['Log_Amount'] = log_amount
            input_df['Hour'] = hour
            input_df['Is_Night'] = is_night
            
            risk_score = float(model.predict_proba(input_df)[:, 1][0])
            
            st.progress(risk_score)
            st.write(f"**Risk Score:** `{risk_score * 100:.2f}%`")
            
            if risk_score > 0.5:
                st.error("🚨 High Risk: Transaction flagged as potential fraud!")
            else:
                st.success("✅ Low Risk: Transaction appears normal.")
                
            # --- SHAP Explainability Section ---
            st.markdown("---")
            st.subheader("🔍 Model Decision Breakdown (SHAP)")
            st.write("Visualizing how each feature contributed to this specific risk score:")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], max_display=5, show=False)
            st.pyplot(fig)

with tab2:
    st.header("Geolocation Velocity & Anomaly Analyzer")
    st.markdown("Check if a sequence of transactions involves physically impossible travel times between locations.")

    col_geo1, col_geo2, col_geo3 = st.columns(3)

    with col_geo1:
        last_city = st.selectbox("Previous Transaction City", list(CITY_COORDINATES.keys()), index=0)

    with col_geo2:
        current_city = st.selectbox("Current Transaction City", list(CITY_COORDINATES.keys()), index=1)

    with col_geo3:
        time_diff = st.number_input("Hours Since Last Transaction", min_value=0.1, max_value=48.0, value=1.0, step=0.5)

    if st.button("Check Velocity & Distance"):
        geo_result = detect_impossible_travel(last_city, current_city, time_diff)
        
        st.write(f"**Distance Between Cities:** `{geo_result['distance_miles']} miles`")
        st.write(f"**Required Speed:** `{geo_result['required_speed_mph']} mph`")
        
        if geo_result['is_impossible']:
            st.error(f"🚨 IMPOSSIBLE TRAVEL DETECTED! Moving {geo_result['distance_miles']} miles in {time_diff} hour(s) requires traveling at {geo_result['required_speed_mph']} mph (faster than a commercial flight).")
        else:
            st.success("✅ Normal Travel Velocity: Physical movement between locations is feasible.")