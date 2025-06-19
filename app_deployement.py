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
    prompt = (
        f"The user is a {data['Gender']} aged {data['Age']} years, "
        f"weighs {data['Weight']} kg with a height of {data['Height']} meters (originally {data['Height_ft']}ft {data['Height_in']}in). "
        f"They have {'a' if data['SMOKE'] == 'Yes' else 'no'} smoking habit, "
        f"alcohol consumption is {data['CALC']}, vegetable intake is {data['FCVC']}x/day, meals/day is {data['NCP']}, "
        f"water intake is {data['CH2O']}L, calorie tracking: {data['SCC']}, physical activity: {data['FAF']}, "
        f"tech usage: {data['TUE']}, transport: {data['MTRANS']}, family history: {data['family_history_with_overweight']}, high calorie food: {data['FAVC']}."
    )
    return prompt

st.set_page_config(page_title="BodyScope | AI Health Advisor", layout="centered")
st.title("🧠 BodyScope - Your Dual-AI Health Companion")

st.sidebar.header("Enter Your Health Info")

# Collect user inputs
data = {}
data['Gender'] = st.sidebar.selectbox("Gender", ["Male", "Female"])
data['Age'] = st.sidebar.slider("Age", 10, 100, 25)
data['Height_ft'] = st.sidebar.slider("Height - Feet", 3, 8, 5)
data['Height_in'] = st.sidebar.slider("Height - Inches", 0, 11, 7)
data['Height'] = convert_height_to_meters(data['Height_ft'], data['Height_in'])
data['Weight'] = st.sidebar.number_input("Weight (in kg)", 20.0, 200.0, 70.0)
data['FCVC'] = st.sidebar.slider("Vegetable consumption per day", 1, 3, 2)
data['NCP'] = st.sidebar.slider("Meals per day", 1, 4, 3)
data['CAEC'] = st.sidebar.selectbox("Snacking frequency", ["Nope", "Sometimes", "Frequently", "Always"])
data['SMOKE'] = st.sidebar.selectbox("Do you smoke?", ["Yes", "No"])
data['CH2O'] = st.sidebar.slider("Water intake (L/day)", 1, 3, 2)
data['SCC'] = st.sidebar.selectbox("Do you monitor calories?", ["yes", "no"])
data['FAF'] = st.sidebar.slider("Physical Activity Level", 0, 3, 1)
data['TUE'] = st.sidebar.selectbox("Technology usage", ["Low Usage", "Moderate Usage", "High Usage"])
data['MTRANS'] = st.sidebar.selectbox("Mode of Transport", ["Walking", "Bike", "Public Transportation", "Automobile"])
data['CALC'] = st.sidebar.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
data['family_history_with_overweight'] = st.sidebar.selectbox("Family history with overweight?", ["yes", "no"])
data['FAVC'] = st.sidebar.selectbox("Frequent high caloric food consumption?", ["yes", "no"])

if st.button("💬 Ask Dr. Gemi for Advice"):
    input_df = pd.DataFrame([data])

    # Calculate BMI
    bmi = calculate_bmi(data['Weight'], data['Height'])
    bmi_category = get_bmi_category(bmi)
    bmi_color = get_bmi_color(bmi_category)
    bmi_msg = get_bmi_message(bmi, bmi_category)

    # Display BMI Message
    with st.container():
        bmi_color(bmi_msg)

    # Make prediction
    prediction = model.predict(input_df.drop(columns=['Height_ft', 'Height_in']))[0]
    st.markdown(f"### 🧪 Obesity Risk Prediction: `{prediction}`")

    # Insert into SQL
    insert_to_sql(input_df.drop(columns=['Height_ft', 'Height_in']))

    # Generate AI Suggestions
    with st.spinner("Generating personalized advice from Dr. Gemi..."):
        prompt = generate_prompt(data)
        response = genai.GenerativeModel("gemini-pro").generate_content(
            [
                {"parts": [prompt + " Now give me personalized suggestions in English, Hinglish and Hindi separately."]}
            ]
        )
        suggestions = response.text.split("\n\n")

    with st.expander("🗣️ Dr. Gemi's Advice - Multilingual"):
        for part in suggestions:
            st.write(part)

    st.info("Disclaimer: This is an AI-powered tool. Please consult a certified medical professional before making any medical decisions.")
