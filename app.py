import google.generativeai as genai
import pandas as pd
import streamlit as st
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Configure API Key for Gemini 1.5 Pro
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ✅ Configure Streamlit
st.title("📊 Dynamic AI Adjustments for Peak Hours")

def predict_customer_flow(history_data):
    """Use Gemini 1.5 Pro to predict customer flow during peak hours."""
    prompt = f"""
    Given the following historical restaurant data:
    {history_data}
    Predict the expected customer flow for today’s peak hours (12:00 PM - 2:00 PM & 6:00 PM - 8:00 PM).
    Provide estimates for customer count, expected wait times, and recommended staff allocation.
    """
    response = genai.generate_text(model="gemini-pro", prompt=prompt)
    return response.text

def optimize_staffing(staff_data, predicted_flow):
    """Use AI to recommend optimal staffing levels based on peak-hour predictions."""
    prompt = f"""
    Given staff availability {staff_data} and predicted customer flow {predicted_flow},
    recommend the best staff allocation to minimize labor costs while maintaining service quality.
    """
    response = genai.generate_text(model="gemini-pro", prompt=prompt)
    return response.text

def evaluate_performance(real_data, target_metrics):
    """Compare real-time restaurant performance against challenge objectives."""
    prompt = f"""
    Given real-time restaurant data {real_data} and performance targets {target_metrics},
    evaluate whether the restaurant has achieved its objectives in reducing wait time,
    improving table turnover rate, and optimizing labor costs.
    """
    response = genai.generate_text(model="gemini-pro", prompt=prompt)
    return response.text

# Sample data inputs
historical_data = pd.DataFrame({
    'Date': ['2025-03-18', '2025-03-19', '2025-03-20'],
    'Peak Hour Traffic': [120, 135, 150],
    'Average Wait Time (min)': [18, 20, 22],
    'Table Turnover Rate': [2.5, 2.8, 3.0]
})
staff_data = {"servers": 8, "cashiers": 2}
target_metrics = {"wait_time_reduction": 15, "turnover_increase": 10, "labor_cost_target": 30}
real_time_data = {"wait_time": 16, "turnover": 3.1, "labor_cost": 28}

# Running AI-driven adjustments
customer_flow_prediction = predict_customer_flow(historical_data)
staffing_plan = optimize_staffing(staff_data, customer_flow_prediction)
performance_evaluation = evaluate_performance(real_time_data, target_metrics)

# Display AI recommendations
st.subheader("🔍 AI Predictions & Insights")
st.write("### Customer Flow Prediction:")
st.write(customer_flow_prediction)

st.write("### Staffing Plan:")
st.write(staffing_plan)

st.write("### Performance Evaluation:")
st.write(performance_evaluation)

# 📊 Visualization
st.subheader("📊 Data Analysis & Trends")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Peak Hour Traffic
sns.lineplot(data=historical_data, x='Date', y='Peak Hour Traffic', marker='o', ax=ax[0])
ax[0].set_title("📈 Peak Hour Traffic Trends")
ax[0].set_ylabel("Customer Count")
ax[0].set_xlabel("Date")
ax[0].grid(True)

# Wait Time vs. Table Turnover Rate
sns.barplot(data=historical_data, x='Date', y='Average Wait Time (min)', ax=ax[1], color='red', label='Avg Wait Time')
sns.lineplot(data=historical_data, x='Date', y='Table Turnover Rate', marker='o', ax=ax[1], color='blue', label='Turnover Rate')
ax[1].set_title("⏳ Wait Time vs. Turnover Rate")
ax[1].set_ylabel("Time (min) / Turnover Rate")
ax[1].set_xlabel("Date")
ax[1].legend()
ax[1].grid(True)

st.pyplot(fig)

# 🎯 Reward System
st.subheader("🏆 Gamification & Rewards")
achievements = []
if real_time_data['wait_time'] <= (22 - target_metrics['wait_time_reduction'] * 0.01 * 22):
    achievements.append("✅ Reduced Wait Time Successfully")
if real_time_data['turnover'] >= (2.5 + target_metrics['turnover_increase'] * 0.01 * 2.5):
    achievements.append("✅ Increased Table Turnover Rate")
if real_time_data['labor_cost'] <= target_metrics['labor_cost_target']:
    achievements.append("✅ Optimized Labor Costs")

if len(achievements) == 3:
    reward = "🏆 Tier 1: Bonus + Company Recognition"
elif len(achievements) == 2:
    reward = "🥈 Tier 2: Half-Month Salary Bonus"
elif len(achievements) == 1:
    reward = "🥉 Tier 3: One-Week Salary Bonus"
else:
    reward = "❌ No Reward - Try Again Next Time"

st.write("### Achievements:")
st.write("\n".join(achievements))
st.write(f"### Reward Earned: {reward}")
