import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import openai

# --- Load Model ---
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- API Configuration ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)
openai.api_key = OPENAI_API_KEY

# --- App Setup ---
st.set_page_config(page_title="BodyScope", page_icon="🩺", layout="centered")
st.title("🩺 BodyScope: Dual-AI Health Companion")
st.markdown("Answer a few lifestyle questions — get insights from **Dr. Gemi** (Gemini) and **Dr. Opie** (OpenAI) 🧠")

# --- Input Form ---
def get_user_input():
    st.header("🙋 Your Lifestyle Information")
    gender = st.radio("Gender", ['Female', 'Male', 'Others'])
    gender = {"Female": 0, "Male": 2, "Others": 1}[gender]

    age = st.slider("Age", 10, 100, 25)
    weight = st.number_input("Weight (kg)", 20, 200, 70)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.75, 0.01)
    bmi = round(weight / (height ** 2), 1)

    st.subheader("🧬 Habits & Lifestyle")
    fam_history = st.selectbox("Family history of overweight?", ['No', 'Yes'])
    fried = st.selectbox("Do you eat fried food frequently?", ['No', 'Yes'])
    veg = st.slider("Veggie consumption (1-Never to 3-Always)", 1, 3, 2)
    meals = st.slider("Main meals per day", 1, 4, 3)
    snack = st.selectbox("Do you snack between meals?", ['Never', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox("Do you smoke?", ['No', 'Yes'])

    water = st.selectbox("Daily water intake", ['Low (<1L)', 'Medium (1-2L)', 'High (>2L)'])
    calmon = st.selectbox("Do you monitor calorie intake?", ['No', 'Yes'])
    phys = st.slider("Physical activity (0-None to 3-High)", 0, 3, 1)

    screen = st.selectbox("Screen time usage", ['Low (<2h)', 'Medium (2-5h)', 'High (>5h)'])
    alc = st.selectbox("Alcohol consumption?", ['Never', 'Sometimes', 'Frequently', 'Always'])
    trans = st.selectbox("Primary transport?", ['Walking', 'Vehicle', 'Public Transport'])

    return {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'family_history_with_overweight': int(fam_history == 'Yes'),
        'Fried_Food_Consum': int(fried == 'Yes'),
        'Freq_of_Vegie_Consump': veg,
        'no_of_meals': meals,
        'snacking_freq': ['Never', 'Sometimes', 'Frequently', 'Always'].index(snack),
        'SMOKE': int(smoke == 'Yes'),
        'water_consumption': ['Low (<1L)', 'Medium (1-2L)', 'High (>2L)'].index(water),
        'calorie_monitoring': int(calmon == 'Yes'),
        'physical_activity': phys,
        'time_spend_on_tech': ['Low (<2h)', 'Medium (2-5h)', 'High (>5h)'].index(screen),
        'alcohol_consump': ['Never', 'Sometimes', 'Frequently', 'Always'].index(alc),
        'MTRANS': ['Walking', 'Vehicle', 'Public Transport'].index(trans)
    }

user_data = get_user_input()

# --- Prediction ---
required_cols = ['Gender', 'Age', 'Height', 'Weight', 'BMI',
                 'family_history_with_overweight', 'Fried_Food_Consum', 'Freq_of_Vegie_Consump',
                 'no_of_meals', 'snacking_freq', 'SMOKE', 'water_consumption',
                 'calorie_monitoring', 'physical_activity', 'time_spend_on_tech',
                 'alcohol_consump', 'MTRANS']

input_df = pd.DataFrame([user_data])[required_cols]
prediction = model.predict(input_df)[0]

st.success(f"🧾 Health Risk Category Predicted: **Class {prediction}**")

# --- AI Suggestions ---
st.subheader("🤖 AI Health Suggestions")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔵 Ask Dr. Gemi (Gemini)"):
        prompt_gemi = f"""
Hello! नमस्ते! मैं Dr. Gemi हूँ — आपकी AI स्वास्थ्य सहायक।
Here's the full patient profile:
{user_data}

Based on the above data and BMI = {user_data['BMI']}, provide 10 detailed friendly, practical health tips in English and Hindi to improve health and manage weight
And remember you are a female heartwarming and Avoid medical jargon. Keep it emotionally supportive, simple, and practical plus also motivate your patient so maintain tone accordingly .
"""
        gemi_response = genai.GenerativeModel("gemini-pro").generate_content(prompt_gemi, temperature=0.7).text
        st.markdown("#### 🩺 Dr. Gemi Says:")
        st.markdown(gemi_response, unsafe_allow_html=True)

with col2:
    if st.button("🟢 Ask Dr. Opie (OpenAI)"):
        prompt_opie = f"""
Hello and नमस्ते! I'm Dr. Opie — your AI health advisor.
Here’s the full lifestyle and health profile:
{user_data}

and remember you are a strict male experienced physician so maintain tone accordingly Please give 10 detailed personalized and specific, actionable health suggestions in English and Hindi, considering BMI = {user_data['BMI']}.
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Dr. Opie, an expert AI health assistant."},
                {"role": "user", "content": prompt_opie}
            ],
            temperature=0.7,
            max_tokens=500
        )
        st.markdown("#### 🩺 Dr. Opie Says:")
        st.markdown(response['choices'][0]['message']['content'], unsafe_allow_html=True)

# --- Health Warning ---
st.info("⚠️ These health tips are AI-generated for educational use and are not a substitute for professional medical advice.")
