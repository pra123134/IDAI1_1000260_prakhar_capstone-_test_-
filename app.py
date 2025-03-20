import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai

# ✅ Configure API Key securely
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key is missing. Go to Streamlit Cloud → Settings → Secrets and add your API key.")
    st.stop()

# ✅ AI Response Generator
def get_ai_response(prompt, fallback_message="⚠️ AI response unavailable. Please try again later."):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") and response.text.strip() else fallback_message
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}\n{fallback_message}"

# ✅ Gamification System for Restaurant Managers

# Simulated historical data (for demonstration purposes)
data = {
    "hour": list(range(10, 23)),
    "avg_wait_time": np.random.randint(5, 20, size=13),
    "table_turnover_rate": np.round(np.random.uniform(0.8, 2.0, size=13), 2),
    "customer_satisfaction": np.round(np.random.uniform(3.5, 5.0, size=13), 1),
    "staff_allocation": np.random.randint(4, 8, size=13),
    "revenue": np.random.randint(1200, 2100, size=13)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define peak hours
peak_hours = [12, 13, 18, 19]
peak_df = df[df["hour"].isin(peak_hours)]

# AI-driven wait time reduction
X = peak_df[["hour", "staff_allocation"]]
y = peak_df["avg_wait_time"]
model = LinearRegression()
model.fit(X, y)
peak_df["optimized_wait_time"] = np.maximum(model.predict(X) * 0.85, 5)
peak_df["optimized_turnover_rate"] = peak_df["table_turnover_rate"] * 1.10
peak_df["optimized_satisfaction"] = peak_df["customer_satisfaction"].apply(lambda x: max(x, 4.5))
peak_df["optimized_staff_allocation"] = peak_df.apply(lambda row: min(row["staff_allocation"] + 1, row["revenue"] * 0.3 / 100), axis=1)

# AI-powered suggestions
menu_prompts = "Suggest quick meal options and menu optimizations for high-efficiency service."
staffing_prompts = "Provide AI-driven recommendations to optimize restaurant staff allocation for peak hours."
customer_prompts = "Analyze customer feedback and suggest improvements for a better dining experience."

peak_df["menu_suggestion"] = peak_df["hour"].apply(lambda x: get_ai_response(menu_prompts + f" for {x}:00 hour"))
peak_df["ai_staffing_insights"] = peak_df["hour"].apply(lambda x: get_ai_response(staffing_prompts + f" for {x}:00 hour"))
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

# Streamlit UI
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
