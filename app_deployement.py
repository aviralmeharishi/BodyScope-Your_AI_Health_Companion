import streamlit as st
import pandas as pd
import google.generativeai as genai
import pickle
from utils import (
    insert_to_sql,
    load_model,
    convert_height_to_meters
)
from bmi_utils import (
    calculate_bmi,
    get_bmi_category,
    get_bmi_color,
    get_bmi_message
)

# --- Prompt Generation ---
def generate_prompt(user_data):
    return f"""
User Profile:
- Gender: {user_data['Gender']}
- Age: {user_data['Age']} years
- Height: {user_data['Height']} meters
- Weight: {user_data['Weight']} kg
- Family history of overweight: {user_data['family_history_with_overweight']}
- Smoker: {user_data['SMOKE']}
- Alcohol Consumption: {user_data['alcohol_consump']}
- Fried Food Consumption more than thrice a week : {user_data['Fried_Food_Consump']}
- Vegetable Intake Rating Out of 3: {user_data['Freq_of_Vegie_Consump']}
- Meals per day: {user_data['no_of_meals']}
- Snacking Frequency: {user_data['snacking_freq']}
- Water Intake: {user_data['water_consumption']} liters
- Calorie Monitoring: {user_data['calorie_monitoring']}
- Physical Activity Rating out of 5: {user_data['physical_activity']}
- Technology Usage: {user_data['time_spend_on_tech']}
- Transport Mode: {user_data['MTRANS']}

Generate:
- 10 to 13 highly personalized tips
- 5 general healthy behavioral tips
Respond only in this language: LANG_PLACEHOLDER.
But Always start with a warm greeting and a small talk like ohh theats impressive if person is healthy and in another case you can motivate them and tell them the advantages of being healthy
"""

# --- Configurations ---
st.set_page_config(page_title="BodyScope | Your AI Health Companion")
st.markdown("**Answer A Few Questions And Get A Fully Personalised Suggestions**")
st.title("💪 BodyScope: Your AI Health Companion")
st.markdown("""
| Powered by Gemini 2.0 Flash |
---
""")

# --- Load Model ---
model = load_model("final_model.pkl")

# --- API Key ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Language Selector ---
language_map = {
    "English": "English",
    "Hindi": "Hindi",
    "Hinglish": "Hinglish",
    "Gujrati" : "Gujrati"
    
}
selected_lang = st.selectbox("Select Language for Advice", list(language_map.keys()))

# --- User Inputs (Unified Layout) ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (in years)", 1, 120,)
height_ft = st.number_input("Height (ft)", 1, 8)
height_in = st.number_input("Height (inches)", 0, 11)
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, step=0.1, format="%.1f")
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
alcohol = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
fried = st.selectbox("Do you eat fried food more than twice a week?", ["yes", "no"])
vegies = st.selectbox("Rate Your Freq. of Veggie Consumption", [0,1,2,3])
meals = st.slider("No. of Meals Per Day", 1, 4, 3)
snack = st.selectbox("Snacking Frequency", ["Nope", "Sometimes", "Frequently", "Always"])
water = st.slider("Water Intake (liters/day)", 0, 4, 1)
calories = st.selectbox("Do you monitor calorie intake?", ["yes", "no"])
activity = st.slider("Rate Your Physical Activity on the Scale of 5 Physical Activity, 0, 5, 2)
technology = st.selectbox("Technology Usage", ["Low Usage", "Moderate Usage", "High Usage"])
transport = st.selectbox("Mode of Transport", ["Walking", "Bike", "Public Transportation", "Automobile"])
family_history = st.selectbox("Family history with overweight?", ["yes", "no"])

submitted = st.button("💬 Ask Dr. Gemi")

if submitted:
    raw_data = {
        'Gender': gender,
        'Age': age,
        'Height_ft': height_ft,
        'Height_in': height_in,
        'Height': convert_height_to_meters(height_ft, height_in),
        'Weight': weight,
        'SMOKE': smoke,
        'alcohol_consump': alcohol,
        'Fried_Food_Consump': fried,
        'Freq_of_Vegie_Consump': vegies,
        'no_of_meals': meals,
        'snacking_freq': snack,
        'water_consumption': water,
        'calorie_monitoring': calories,
        'physical_activity': activity,
        'time_spend_on_tech': technology,
        'MTRANS': transport,
        'family_history_with_overweight': family_history
    }

    encoded = {
        'Gender': 1 if gender == 'Male' else 0,
        'Age': age,
        'Height': convert_height_to_meters(height_ft, height_in),
        'Weight': weight,
        'family_history_with_overweight': 1 if family_history == 'yes' else 0,
        'Fried_Food_Consump': 1 if fried == 'yes' else 0,
        'Freq_of_Vegie_Consump': vegies,
        'no_of_meals': meals,
        'snacking_freq': {'Nope': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[snack],
        'SMOKE': 1 if smoke == 'yes' else 0,
        'water_consumption': water,
        'calorie_monitoring': 1 if calories == 'yes' else 0,
        'physical_activity': activity,
        'time_spend_on_tech': {'Low Usage': 0, 'Moderate Usage': 1, 'High Usage': 2}[technology],
        'alcohol_consump': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[alcohol],
        'MTRANS': {'Walking': 0, 'Bike': 1, 'Public Transportation': 2, 'Automobile': 3}[transport]
    }

    ordered_cols = [
        'Gender', 'Age', 'Height', 'Weight',
        'family_history_with_overweight', 'Fried_Food_Consump',
        'Freq_of_Vegie_Consump', 'no_of_meals', 'snacking_freq',
        'SMOKE', 'water_consumption', 'calorie_monitoring',
        'physical_activity', 'time_spend_on_tech',
        'alcohol_consump', 'MTRANS'
    ]

    input_df = pd.DataFrame([encoded])[ordered_cols]

    prediction = model.predict(input_df)[0]

    bmi = calculate_bmi(weight, encoded["Height"])
    bmi_category = get_bmi_category(bmi)
    bmi_color = get_bmi_color(bmi_category)
    bmi_msg = get_bmi_message(bmi, bmi_category)
    bmi_color(bmi_msg)

    st.markdown(f"### 🧪 Obesity Risk Prediction: `{prediction}`")

    insert_to_sql(pd.DataFrame([raw_data]).drop(columns=["Height_ft", "Height_in"]))

    with st.spinner("Generating personalized advice from Dr. Gemi (Gemini 2.0 Flash)..."):
        prompt = generate_prompt(raw_data).replace("LANG_PLACEHOLDER", language_map[selected_lang])
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        suggestions = [line for line in response.text.strip().split("\n") if line.strip() != ""]

    st.markdown("#### 🗣️ Dr. Gemi's Advice")
    for tip in suggestions:
        st.markdown(f"- {tip}")

    st.info("Disclaimer: This is an AI-powered tool. Please consult a certified medical professional before making any medical decisions.")
    st.markdown("---")
    st.markdown("> © 2025 • An Aviral Meharishi Creation")
