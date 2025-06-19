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
    data = {
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

    input_df = pd.DataFrame([data])
    bmi = calculate_bmi(weight, data['Height'])
    bmi_category = get_bmi_category(bmi)
    bmi_color = get_bmi_color(bmi_category)
    bmi_msg = get_bmi_message(bmi, bmi_category)

    bmi_color(bmi_msg)
    ordered_cols = [
        'Gender', 'Age', 'Height', 'Weight',
        'SMOKE', 'alcohol_consump', 'Fried_Food_Consump',
        'Freq_of_Vegie_Consump', 'no_of_meals', 'snacking_freq',
        'water_consumption', 'calorie_monitoring', 'physical_activity',
        'time_spend_on_tech', 'MTRANS', 'family_history_with_overweight'
    ]
    model_input = input_df[ordered_cols]
    prediction = model.predict(model_input)[0]
    st.markdown(f"### 🧪 Obesity Risk Prediction: `{prediction}`")

    insert_to_sql(input_df.drop(columns=['Height_ft', 'Height_in']))

    with st.spinner("Generating personalized advice from Dr. Gemi..."):
        prompt = generate_prompt(data)
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

