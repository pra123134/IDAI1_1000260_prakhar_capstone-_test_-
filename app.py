import streamlit as st
import random
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

# ✅ Streamlit UI Configuration

# Quiz questions
data = [
    {"question": "What is the best way to handle an unhappy customer?",
     "options": ["Ignore them", "Listen actively and resolve the issue", "Ask them to leave", "Give them a discount immediately"],
     "answer": "Listen actively and resolve the issue"},
    
    {"question": "Which of the following is a key factor in reducing food waste?",
     "options": ["Overordering stock", "Monitoring inventory regularly", "Guessing demand", "Ignoring expiration dates"],
     "answer": "Monitoring inventory regularly"},
    
    {"question": "What is an effective way to motivate restaurant staff?",
     "options": ["Strict rules and penalties", "Incentives and recognition", "Micromanagement", "Ignoring their concerns"],
     "answer": "Incentives and recognition"},
    
    {"question": "Which metric is crucial for tracking a restaurant's financial health?",
     "options": ["Customer complaints", "Gross Profit Margin", "Number of social media posts", "Kitchen cleanliness"],
     "answer": "Gross Profit Margin"}
]

# Shuffle questions
random.shuffle(data)

st.title("Restaurant Manager Quiz")
st.write("Test your knowledge in restaurant management!")

score = 0

for i, q in enumerate(data):
    st.subheader(f"Question {i+1}")
    user_answer = st.radio(q["question"], q["options"], key=i)
    if user_answer == q["answer"]:
        score += 1

if st.button("Submit Quiz"):
    st.write(f"You scored {score}/{len(data)}!")
    if score == len(data):
        st.success("Excellent! You're a top-tier restaurant manager!")
    elif score >= len(data) / 2:
        st.info("Good job! Keep improving your skills.")
    else:
        st.warning("You might want to review some restaurant management best practices.")

