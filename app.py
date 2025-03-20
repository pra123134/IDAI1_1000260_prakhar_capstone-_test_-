import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import time

# ✅ Configure API Key securely
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Go to Streamlit Cloud → Settings → Secrets and add your API key.")
    st.stop()

# ✅ AI Response Generator with Caching and Retry Mechanism
@st.cache_data
def get_ai_response(prompt, fallback_message="⚠️ AI response unavailable. Please try again later.", retries=3, delay=5):
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text.strip():
                return response.text.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                return f"⚠️ AI Error: {str(e)}\n{fallback_message}"
    return fallback_message

# ✅ Gamification System for Restaurant Managers

data = {
    "hour": list(range(10, 23)),  
    "avg_wait_time": [15, 14, 12, 18, 10, 9, 8, 13, 16, 17, 11, 7, 6],  
    "table_turnover_rate": [1.2, 1.5, 1.0, 1.8, 1.3, 1.1, 0.9, 1.4, 1.7, 1.6, 1.0, 1.2, 1.3],  
    "customer_satisfaction": [4.2, 4.3, 4.0, 4.5, 4.6, 4.1, 4.4, 4.7, 4.8, 4.3, 4.2, 4.6, 4.5],  
    "staff_allocation": [5, 6, 4, 7, 5, 5, 6, 7, 8, 5, 4, 6, 7],  
    "revenue": [1500, 1700, 1600, 1800, 1400, 1300, 1200, 1900, 2000, 1750, 1600, 1550, 1650]  
}

df = pd.DataFrame(data)

peak_hours = [12, 13, 18, 19]
peak_df = df[df["hour"].isin(peak_hours)]

X = peak_df[["hour", "staff_allocation"]]
y = peak_df["avg_wait_time"]
model = LinearRegression()
model.fit(X, y)
peak_df["optimized_wait_time"] = model.predict(X) * 0.85  
peak_df["optimized_turnover_rate"] = peak_df["table_turnover_rate"] * 1.10
peak_df["optimized_satisfaction"] = peak_df["customer_satisfaction"].apply(lambda x: max(x, 4.5))
peak_df["optimized_staff_allocation"] = peak_df.apply(lambda row: min(row["staff_allocation"] + 1, row["revenue"] * 0.3 / 100), axis=1)

menu_prompts = "Suggest quick meal options and menu optimizations for high-efficiency service."
peak_df["menu_suggestion"] = peak_df["hour"].apply(lambda x: get_ai_response(menu_prompts + f" for {x}:00 hour"))

staffing_prompts = "Provide AI-driven recommendations to optimize restaurant staff allocation for peak hours."
peak_df["ai_staffing_insights"] = peak_df["hour"].apply(lambda x: get_ai_response(staffing_prompts + f" for {x}:00 hour"))

customer_prompts = "Analyze customer feedback and suggest improvements for a better dining experience."
peak_df["ai_customer_experience"] = peak_df["hour"].apply(lambda x: get_ai_response(customer_prompts + f" for {x}:00 hour"))

# Gamification Rewards System
def assign_rewards(row):
    achieved_objectives = sum([
        row["optimized_wait_time"] <= row["avg_wait_time"] * 0.85,
        row["optimized_turnover_rate"] >= row["table_turnover_rate"] * 1.10,
        row["optimized_satisfaction"] >= 4.5,
        row["optimized_staff_allocation"] <= row["staff_allocation"] + 1,
    ])
    
    reward_tiers = {
        4: "🏆 Tier 1: Bonus + Trip + Conference",
        3: "🥈 Tier 2: Half Bonus + Gift Card",
        2: "🥉 Tier 3: Small Bonus + Team Recognition",
        1: "🔹 Participation Recognition",
        0: "❌ No Reward"
    }
    
    return reward_tiers.get(achieved_objectives, "❌ No Reward")

peak_df["reward"] = peak_df.apply(assign_rewards, axis=1)

st.title("🚀 AI-Powered Gamification for Restaurant Managers")
st.subheader("Optimize Peak Hours and Reward Staff Performance")
st.dataframe(peak_df)

st.markdown("### AI Insights")

st.write("#### 📌 Menu Optimization Suggestions:")
for i, row in peak_df.iterrows():
    st.write(f"**{row['hour']}:00** - {row['menu_suggestion']}")

st.write("#### 📌 Staff Allocation Insights:")
for i, row in peak_df.iterrows():
    st.write(f"**{row['hour']}:00** - {row['ai_staffing_insights']}")

st.write("#### 📌 Customer Experience Enhancements:")
for i, row in peak_df.iterrows():
    st.write(f"**{row['hour']}:00** - {row['ai_customer_experience']}")

st.markdown("### 🎯 Gamification & Rewards System")
st.dataframe(peak_df[["hour", "reward"]])
