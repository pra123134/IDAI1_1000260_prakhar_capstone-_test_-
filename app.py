import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
from datetime import datetime
import plotly.express as px
import google.generativeai as genai

# ✅ Configure API Key for Gemini 1.5 Pro
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ✅ Configure Streamlit
st.set_page_config(page_title="AI Restaurant Challenges", layout="wide")
st.title("🏆 AI-Powered Restaurant Management Challenges")
st.write("**Objective:** Solve real-time restaurant challenges using AI insights to optimize operations.")

# Multiplayer Progress Tracking
st.sidebar.title("🏅 Progress Tracking")
user_id = st.sidebar.text_input("Enter Manager ID:")

# Challenge Selection with Dynamic Challenges
challenges = {
    "Dynamic AI Adjustments for Peak Hours": 5,
    "Inventory Optimization": 4,
    "Customer Personalization & Upselling": 3,
    "Energy Efficiency Management": 4,
    "Seasonal Menu Adaptation": 3,
    "AI-Driven Special Promotions": 4,
    "Real-Time Order Flow Optimization": 5
}
selected_challenge = st.selectbox("🔍 Select Your Challenge:", list(challenges.keys()))

def get_difficulty_multiplier(challenge):
    return challenges[challenge] * 0.2

# AI Model Training
@st.cache_resource
def train_ml_model():
    np.random.seed(42)
    data_size = 2000
    X = np.random.randint(50, 250, (data_size, 3))
    y = np.random.randint(5, 25, data_size)  # Wait time prediction
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    return model, error

model, model_error = train_ml_model()

def predict_peak_traffic(date):
    np.random.seed(hash(date) % 1000)
    return np.random.randint(50, 250)

# Gemini 1.5 AI Insights
@st.cache_data
def get_gemini_insights(challenge, traffic, staff, menu):
    prompt = f"""
    Analyze restaurant management challenge: {challenge}.
    - Predicted peak traffic: {traffic} customers
    - Recommended staff: {staff} members
    - Suggested menu items: {menu}
    Provide AI-driven insights and improvements.
    """
    response = genai.chat(prompt)
    return response.text if response else "No insights available."

def adjust_staffing(predicted_traffic):
    return 5 + (predicted_traffic // 30)

def dynamic_menu_adjustment(predicted_traffic):
    return ["Fast-prep meals", "Combo offers"] if predicted_traffic > 180 else ["Regular menu", "Limited specials"]

def evaluate_performance(current_data, difficulty_multiplier):
    return sum([
        current_data["avg_wait_time"] <= 10 * (0.85 - difficulty_multiplier),
        current_data["table_turnover_rate"] >= 1.8 * (1.1 + difficulty_multiplier),
        current_data["customer_satisfaction"] >= 4.5,
        current_data["labor_cost_percentage"] <= 30,
        "Fast-prep meals" in current_data["menu_adjustments"]
    ])

# AI Dashboard Execution
today = datetime.today().strftime('%Y-%m-%d')
predicted_traffic = predict_peak_traffic(today)
recommended_staffing = adjust_staffing(predicted_traffic)
menu_suggestions = dynamic_menu_adjustment(predicted_traffic)
difficulty_multiplier = get_difficulty_multiplier(selected_challenge)

# Predict wait time using ML Model
predicted_wait_time = model.predict([[predicted_traffic, recommended_staffing, 30]])[0]

# Simulated Current Performance Data
current_data = {
    "avg_wait_time": round(predicted_wait_time, 2),
    "table_turnover_rate": round(np.random.uniform(1.7, 2.5), 2),
    "customer_satisfaction": round(np.random.uniform(4.2, 4.8), 1),
    "labor_cost_percentage": np.random.randint(27, 33),
    "menu_adjustments": menu_suggestions
}

# Gamification Score Calculation
score = evaluate_performance(current_data, difficulty_multiplier)

# AI Insights
ai_insights = get_gemini_insights(selected_challenge, predicted_traffic, recommended_staffing, menu_suggestions)

# Display Challenge Information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Predicted Traffic", value=f"{predicted_traffic} customers")
with col2:
    st.metric(label="👨‍🍳 Recommended Staff", value=f"{recommended_staffing} members")
with col3:
    st.metric(label="⌛ Predicted Wait Time", value=f"{predicted_wait_time:.2f} min", delta=f"±{model_error:.2f} min")

st.write(f"🍽 **Menu Adjustments:** {menu_suggestions}")

# AI Insights Section
st.subheader("🔍 AI-Generated Insights")
st.write(ai_insights)

# Performance Heatmap
data_heatmap = {
    "Metric": ["Avg Wait Time", "Table Turnover", "Satisfaction", "Labor Cost %"],
    "Value": [current_data["avg_wait_time"], current_data["table_turnover_rate"], current_data["customer_satisfaction"], current_data["labor_cost_percentage"]]
}
df_heatmap = pd.DataFrame(data_heatmap)
fig = px.bar(df_heatmap, x="Metric", y="Value", color="Metric", title="📊 Performance Heatmap")
st.plotly_chart(fig, use_container_width=True)

# Team-Based Multiplayer Leaderboard
st.sidebar.title("🏆 Leaderboard")
leaderboard_data = pd.DataFrame({
    "Manager ID": ["MGR_001", "MGR_002", "MGR_003", user_id if user_id else "You"],
    "Score": np.random.randint(50, 100, 4)
})
leaderboard_data.sort_values(by="Score", ascending=False, inplace=True)
st.sidebar.dataframe(leaderboard_data)

st.success("✅ Challenge Attempt Recorded. Track your progress in the sidebar!")

