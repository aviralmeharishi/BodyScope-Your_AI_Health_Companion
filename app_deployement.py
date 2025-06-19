# app.py
import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
from utils import insert_to_sql, load_model, convert_height_to_meters
from bmi_utils import calculate_bmi, get_bmi_category, get_bmi_message, get_bmi_color

# Load model
model = load_model("final_model.pkl")

# Configure Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def generate_prompt(data):
    return (
        f"The user is a {data['Gender']} aged {data['Age']} years, "
        f"weighs {data['Weight']} kg with a height of {data['Height']} meters (originally {data['Height_ft']}ft {data['Height_in']}in). "
        f"Smoking habit: {data['SMOKE']}, Alcohol consumption: {data['alcohol_consump']}, "
        f"Fried food: {data['Fried_Food_Consump']}, Veg intake: {data['Freq_of_Vegie_Consump']}x/day, Meals/day: {data['no_of_meals']}, "
        f"Water intake: {data['water_consumption']}L, Calorie monitoring: {data['calorie_monitoring']}, Activity: {data['physical_activity']}, "
        f"Tech usage: {data['time_spend_on_tech']}, Transport: {data['MTRANS']}, Family history: {data['family_history_with_overweight']}"
    )

st.set_page_config(page_title="BodyScope | AI Health Companion", layout="centered")
st.title("🧠 BodyScope - Your Dual-AI Health Companion")

st.subheader("📋 Enter Your Health & Lifestyle Info")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 25)
height_ft = st.slider("Height - Feet", 3, 8, 5)
height_in = st.slider("Height - Inches", 0, 11, 7)
weight = st.number_input("Weight (in kg)", 20.0, 200.0, 70.0)
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
alcohol = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
fried = st.selectbox("Fried Food Consumption?", ["yes", "no"])
vegies = st.slider("Vegetable consumption per day", 1, 3, 2)
meals = st.slider("Meals per day", 1, 4, 3)
snack = st.selectbox("Snacking frequency", ["Nope", "Sometimes", "Frequently", "Always"])
water = st.slider("Water intake (L/day)", 1, 3, 2)
calories = st.selectbox("Do you monitor calories?", ["yes", "no"])
activity = st.slider("Physical Activity Level", 0, 3, 1)
technology = st.selectbox("Technology usage", ["Low Usage", "Moderate Usage", "High Usage"])
transport = st.selectbox("Mode of Transport", ["Walking", "Bike", "Public Transportation", "Automobile"])
family_history = st.selectbox("Family history with overweight?", ["yes", "no"])

submitted = st.button("💬 Ask Dr. Gemi for Advice")

if submitted:
    # 1. Raw user input for Gemini and SQL
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

    # 2. Encoded data for prediction (matches model training)
    encoded = {
        'Gender': 1 if gender == 'Male' else 0,
        'Age': age,
        'Height': convert_height_to_meters(height_ft, height_in),
        'Weight': weight,
        'SMOKE': 1 if smoke == 'yes' else 0,
        'alcohol_consump': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[alcohol],
        'Fried_Food_Consump': 1 if fried == 'yes' else 0,
        'Freq_of_Vegie_Consump': vegies,
        'no_of_meals': meals,
        'snacking_freq': {'Nope': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[snack],
        'water_consumption': water,
        'calorie_monitoring': 1 if calories == 'yes' else 0,
        'physical_activity': activity,
        'time_spend_on_tech': {'Low Usage': 0, 'Moderate Usage': 1, 'High Usage': 2}[technology],
        'MTRANS': {'Walking': 0, 'Bike': 1, 'Public Transportation': 2, 'Automobile': 3}[transport],
        'family_history_with_overweight': 1 if family_history == 'yes' else 0
    }

    ordered_cols = list(encoded.keys())
    input_df = pd.DataFrame([encoded])[ordered_cols]

    # 3. Model Prediction
    prediction = model.predict(input_df)[0]

    # 4. BMI Logic
    bmi = calculate_bmi(weight, encoded["Height"])
    bmi_category = get_bmi_category(bmi)
    bmi_color = get_bmi_color(bmi_category)
    bmi_msg = get_bmi_message(bmi, bmi_category)
    bmi_color(bmi_msg)

    st.markdown(f"### 🧪 Obesity Risk Prediction: `{prediction}`")

    # 5. SQL Save
    insert_to_sql(pd.DataFrame([raw_data]).drop(columns=["Height_ft", "Height_in"]))

    # 6. Gemini Advice
    with st.spinner("Generating personalized advice from Dr. Gemi..."):
        prompt = generate_prompt(raw_data)
        response = genai.GenerativeModel("gemini-pro").generate_content(
            [{"parts": [prompt + " Now give me personalized suggestions in English, Hinglish and Hindi separately."]}]
        )
        suggestions = response.text.split("\n\n")

    with st.expander("🗣️ Dr. Gemi's Advice - Multilingual"):
        for part in suggestions:
            st.write(part)

    st.info("Disclaimer: This is an AI-powered tool. Please consult a certified medical professional before making any medical decisions.")
    st.markdown("---")

    st.markdown("> © 2025 • An Aviral Meharishi Creation")

