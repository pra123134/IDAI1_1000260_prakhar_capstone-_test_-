import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import google.generativeai as genai
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# ✅ Configure API Key securely
api_key = st.secrets.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Add it in Streamlit Secrets.")
    st.stop()

# 🎮 AI Gamified Challenges for Restaurant Management
st.set_page_config(page_title="AI Restaurant Challenges", layout="wide")
st.title("🏆 AI-Powered Restaurant Management Challenges")
st.write("**Objective:** Solve real-time restaurant challenges using AI predictions to optimize operations, staffing, and menu decisions.")

# Firebase Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["FIREBASE_CREDENTIALS"])
    firebase_admin.initialize_app(cred, {'databaseURL': st.secrets["FIREBASE_DB_URL"]})

# Multiplayer Progress Tracking
st.sidebar.title("🏅 Progress Tracking")
user_id = st.sidebar.text_input("Enter Manager ID:")
if user_id:
    user_ref = db.reference(f'users/{user_id}/progress')
    past_scores = user_ref.get() or {}
    with st.sidebar.expander("📊 Your Past Challenges"):
        for record in past_scores.values():
            st.write(f"🗓 {record['date']}: {record['challenge']} - (Score: {record['score']}/5)")

# AI Chatbot Assistance
st.sidebar.title("🤖 AI Strategy Assistant")
chat_input = st.sidebar.text_area("Ask AI for help:")
if st.sidebar.button("Get AI Advice") and chat_input:
    response = genai.generate(chat_input)
    st.sidebar.write("💡 AI Suggests:", response)

# Challenge Selection
challenges = {
    "Dynamic AI Adjustments for Peak Hours": 5,
    "Inventory Optimization": 4,
    "Customer Personalization & Upselling": 3,
    "Energy Efficiency Management": 4,
    "Seasonal Menu Adaptation": 3
}
selected_challenge = st.selectbox("🔍 Select Your Challenge:", list(challenges.keys()))

def get_difficulty_multiplier(challenge):
    return challenges[challenge] * 0.2

# AI Model Training
@st.cache_resource
def train_ml_model():
    np.random.seed(42)
    data_size = 1000
    X = np.random.randint(50, 200, (data_size, 3))
    y = np.random.randint(8, 20, data_size)  # Wait time prediction
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    return model, error

model, model_error = train_ml_model()

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
    return 5 + (predicted_traffic // 40)

def dynamic_menu_adjustment(predicted_traffic):
    return ["Fast-prep meals", "Combo offers"] if predicted_traffic > 150 else ["Regular menu", "Limited specials"]

def evaluate_performance(current_data, difficulty_multiplier):
    return sum([
        current_data["avg_wait_time"] <= previous_month_data["avg_wait_time"] * (0.85 - difficulty_multiplier),
        current_data["table_turnover_rate"] >= previous_month_data["table_turnover_rate"] * (1.1 + difficulty_multiplier),
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
predicted_wait_time = model.predict([[predicted_traffic, recommended_staffing, previous_month_data["labor_cost_percentage"]]])[0]

# Simulated Current Performance Data
current_data = {
    "avg_wait_time": round(predicted_wait_time, 2),
    "table_turnover_rate": round(np.random.uniform(1.7, 2.2), 2),
    "customer_satisfaction": round(np.random.uniform(4.2, 4.7), 1),
    "labor_cost_percentage": np.random.randint(28, 34),
    "menu_adjustments": menu_suggestions
}

# Gamification Score Calculation
score = evaluate_performance(current_data, difficulty_multiplier)

# Store Progress in Firebase
if user_id:
    db.reference(f'users/{user_id}/progress/{today}').set({
        "challenge": selected_challenge,
        "score": score,
        "date": today
    })

# Display Challenge Information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Predicted Traffic", value=f"{predicted_traffic} customers")
with col2:
    st.metric(label="👨‍🍳 Recommended Staff", value=f"{recommended_staffing} members")
with col3:
    st.metric(label="⌛ Predicted Wait Time", value=f"{predicted_wait_time:.2f} min", delta=f"±{model_error:.2f} min")

st.write(f"🍽 **Menu Adjustments:** {menu_suggestions}")
st.success("✅ Challenge Attempt Recorded. Track your progress in the sidebar!")
