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
    get_bmi_message,
    generate_prompt
)

# --- Configurations ---
st.set_page_config(page_title="BodyScope | Your AI Health Companion")
st.title("💪 BodyScope: Your AI Health Companion")
st.markdown("""
### Powered by Gemini 2.0 Flash | An Aviral Meharishi creation
---
""")

# --- Load Model ---
model = load_model("final_model.pkl")

# --- API Key ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

# --- User Inputs ---
with st.form("user_data_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (in years)", 1, 120)
        height_ft = st.number_input("Height (ft)", 1, 8)
        height_in = st.number_input("Height (inches)", 0, 11)
        weight = st.number_input("Weight (kg)", 20, 300)

    with col2:
        smoke = st.selectbox("Do you smoke?", ["yes", "no"])
        alcohol = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
        fried = st.selectbox("Do you eat fried food?", ["yes", "no"])
        vegies = st.selectbox("Freq. of Veggie Consumption (per week)", [0,1,2,3,4,5,6,7])
        meals = st.slider("No. of Meals Per Day", 1, 7, 3)

    with col3:
        snack = st.selectbox("Snacking Frequency", ["Nope", "Sometimes", "Frequently", "Always"])
        water = st.slider("Water Intake (liters/day)", 0, 10, 3)
        calories = st.selectbox("Do you monitor calorie intake?", ["yes", "no"])
        activity = st.slider("Physical Activity (hrs/week)", 0, 15, 2)
        technology = st.selectbox("Technology Usage", ["Low Usage", "Moderate Usage", "High Usage"])
        transport = st.selectbox("Mode of Transport", ["Walking", "Bike", "Public Transportation", "Automobile"])
        family_history = st.selectbox("Family history with overweight?", ["yes", "no"])

    submitted = st.form_submit_button("💬 Ask Dr. Gemi")

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
        full_prompt = generate_prompt(raw_data)
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(full_prompt + "\nPlease provide personalized suggestions in English, Hinglish and Hindi.")
        suggestions = response.text.split("\n\n")

    with st.expander("🗣️ Dr. Gemi's Advice - Multilingual"):
        for part in suggestions:
            st.write(part)

    st.info("Disclaimer: This is an AI-powered tool. Please consult a certified medical professional before making any medical decisions.")
    st.markdown("---")
    st.markdown("> © 2025 • An Aviral Meharishi Creation")
