import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import time
from datetime import datetime

# ✅ Configure API Key securely
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Go to Streamlit Cloud → Settings → Secrets and add your API key.")
    st.stop()

# 🎮 Gamified AI Challenge: Dynamic AI Adjustments for Peak Hours
st.title("🏆 Dynamic AI Peak Hour Challenge")
st.write("**Objective:** Optimize restaurant efficiency using AI predictions to reduce wait times, enhance table turnover, and improve customer satisfaction.")

# Simulated AI Predictions
@st.cache_data
def predict_peak_traffic(date):
    np.random.seed(hash(date) % 1000)
    return np.random.randint(50, 200)

# Historical Data
previous_month_data = {
    "avg_wait_time": 12,
    "table_turnover_rate": 1.8,
    "customer_satisfaction": 4.3,
    "labor_cost_percentage": 32,
}

def adjust_staffing(predicted_traffic):
    base_staff = 5
    return base_staff + (predicted_traffic // 40)

def dynamic_menu_adjustment(predicted_traffic):
    return ["Fast-prep meals", "Combo offers"] if predicted_traffic > 150 else ["Regular menu", "Limited specials"]

def evaluate_performance(current_data):
    score = sum([
        current_data["avg_wait_time"] <= previous_month_data["avg_wait_time"] * 0.85,
        current_data["table_turnover_rate"] >= previous_month_data["table_turnover_rate"] * 1.1,
        current_data["customer_satisfaction"] >= 4.5,
        current_data["labor_cost_percentage"] <= 30,
        "Fast-prep meals" in current_data["menu_adjustments"]
    ])
    return score

# AI Dashboard Execution
today = datetime.today().strftime('%Y-%m-%d')
predicted_traffic = predict_peak_traffic(today)
recommended_staffing = adjust_staffing(predicted_traffic)
menu_suggestions = dynamic_menu_adjustment(predicted_traffic)

# Simulated Current Performance Data
current_data = {
    "avg_wait_time": np.random.randint(8, 15),
    "table_turnover_rate": round(np.random.uniform(1.7, 2.2), 2),
    "customer_satisfaction": round(np.random.uniform(4.2, 4.7), 1),
    "labor_cost_percentage": np.random.randint(28, 34),
    "menu_adjustments": menu_suggestions
}

# Gamification Score & Reward Tiers
score = evaluate_performance(current_data)
reward_tier = "🎖 Tier 3 (Good Effort)"
if score >= 5:
    reward_tier = "🏆 Tier 1 (Champion!)"
elif score >= 4:
    reward_tier = "🥈 Tier 2 (Great Job!)"

# Display Results with Progress Bar
st.progress(score / 5)
st.write(f"📊 **Predicted Traffic:** {predicted_traffic} customers")
st.write(f"👨‍🍳 **Recommended Staff:** {recommended_staffing} members")
st.write(f"🍽 **Menu Adjustments:** {menu_suggestions}")
st.write(f"🎯 **Performance Score:** {score}/5")
st.subheader(f"🏅 **Achieved Reward Tier:** {reward_tier}")
