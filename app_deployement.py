import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
import openai

# Load the model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# API configuration
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="BodyScope", layout="centered")
st.title("🩺 BodyScope: Your AI Health Companion")

# Input Form
def get_user_input():
    st.header("🙋 Enter Your Health Profile")

    gender = st.radio("Gender", ['Female', 'Male', 'Others'])
    gender = {"Female": 0, "Male": 2, "Others": 1}[gender]

    age = st.slider("How Old are You?", 10, 100, 25)

    st.markdown("#### Height")
    col1, col2 = st.columns(2)
    with col1:
        feet = st.number_input("Feet", min_value=2, max_value=8, value=5)
    with col2:
        inches = st.number_input("Inches", min_value=0, max_value=11, value=6)

    height_m = round(((feet * 12) + inches) * 0.0254, 2)

    weight = st.number_input("Weight (in kg)", 20, 200, 70)
    bmi = round(weight / (height_m ** 2), 1)
    st.markdown(f"**📊 Your BMI: {bmi}**")

    fam_history = st.selectbox("Family history of overweight?", ['No', 'Yes']) == 'Yes'
    fried = st.selectbox("Eat fried food often?", ['No', 'Yes']) == 'Yes'
    veg = st.slider("Veggies per meal (1-3)", 1, 3, 2)
    meals = st.slider("Main meals/day", 1, 4, 3)
    snacking = st.selectbox("Snacking frequency", ['Never', 'Sometimes', 'Frequently', 'Always'])
    snack_val = ['Never', 'Sometimes', 'Frequently', 'Always'].index(snacking)
    smoke = st.selectbox("Do you smoke?", ['No', 'Yes']) == 'Yes'

    water = st.selectbox("Water intake", ['Low (<1L)', 'Moderate (1-2L)', 'High (>2L)'])
    water_val = {'Low (<1L)': 1, 'Moderate (1-2L)': 2, 'High (>2L)': 3}[water]

    calorie = st.radio("Track calories?", ['No', 'Yes']) == 'Yes'
    activity = st.slider("Physical activity (0-3)", 0, 3, 1)

    screen = st.selectbox("Screen time", ['<2 hours', '2-5 hours', '>5 hours'])
    screen_val = {'<2 hours': 0, '2-5 hours': 1, '>5 hours': 2}[screen]

    alcohol = st.selectbox("Alcohol consumption", ['Never', 'Sometimes', 'Frequently', 'Always'])
    alcohol_val = ['Never', 'Sometimes', 'Frequently', 'Always'].index(alcohol)

    transport = st.selectbox("Primary transport", ['Walking', 'Vehicle', 'Public Transport'])
    transport_val = ['Walking', 'Vehicle', 'Public Transport'].index(transport)

    data = {
        'Gender': gender,
        'Age': age,
        'Height': height_m,
        'Weight': weight,
        'family_history_with_overweight': int(fam_history),
        'Fried_Food_Consump': int(fried),
        'Freq_of_Vegie_Consump': veg,
        'no_of_meals': meals,
        'snacking_freq': snack_val,
        'SMOKE': int(smoke),
        'water_consumption': water_val,
        'calorie_monitoring': int(calorie),
        'physical_activity': activity,
        'time_spend_on_tech': screen_val,
        'alcohol_consump': alcohol_val,
        'MTRANS': transport_val
    }

    return pd.DataFrame([data]), bmi, data

# Get input
input_df, bmi, user_data = get_user_input()

# Predict button
if st.button("🔍 Predict My Obesity Risk"):
    prediction = model.predict(input_df)[0]

    label_map = {
        0: "Underweight",
        1: "Healthy",
        2: "Overweight",
        3: "Obese"
    }

    prediction_label = label_map.get(prediction, "Unknown")
    st.success(f"🧬 **Obesity Category**: {prediction_label}")

# Horizontal layout for AI buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🧚‍♀️ Ask Dr. Gemi"):
        gemi_prompt = f"""
Hello! नमस्ते! मैं Dr. Gemi हूँ, आपकी AI हेल्थ साथी ❤️  
Please provide 10 sweet and heartwarming health suggestions in English & Hindi for the following profile:

{user_data}

Their BMI is {bmi}. Focus on motivation, warmth, and practical lifestyle suggestions.
"""
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        gemi_response = gemini_model.generate_content(gemi_prompt).text
        st.markdown(gemi_response, unsafe_allow_html=True)

with col2:
    if st.button("🧑‍⚕️ Ask Dr. Opie"):
        opie_prompt = f"""
User profile:  
{user_data}

BMI: {bmi}

You're Dr. Opie, a disciplined, no-nonsense AI doctor. Give 10 strict, practical health tips in **English & Hindi** to help them improve lifestyle and BMI.
"""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Dr. Opie, a strict and disciplined AI health expert."},
                {"role": "user", "content": opie_prompt}
            ],
            temperature=0.6,
            max_tokens=600
        )
        st.markdown(response.choices[0].message.content, unsafe_allow_html=True)

st.info("⚠️ These AI suggestions are educational and not a replacement for professional medical advice.")
