import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import openai
import pickle
import altair as alt

# --- API Configuration ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)
openai.api_key = OPENAI_API_KEY

# --- Load Model ---
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- App Setup ---
st.set_page_config(page_title="BodyScope", page_icon="🩺", layout="centered")
st.title("🩺 BodyScope: Dual-AI Health Companion")
st.markdown("Answer a few personal and lifestyle questions — get tips from **Dr. Gemi** and **Dr. Opie** in both English & Hindi.")

# --- User Input ---
def get_user_input():
    st.header("🙋 Tell Us About Yourself")
    sex = st.radio("What’s your gender?", ['Female', 'Male', 'Others'])
    sex = {"Female": 0, "Male": 2, "Others": 1}[sex]
    age = st.slider("How old are you?", 10, 105, 25)
    weight = st.number_input("Weight (kg)", 20, 200, 70)
    height_m = st.number_input("Height (m)", 1.0, 2.5, 1.75, 0.01)
    bmi = round(weight / (height_m ** 2), 1)
    st.write(f"🔎 Your BMI is: **{bmi}**")

    st.subheader("Lifestyle & Habits")
    fam = st.selectbox("Do you have a family history of overweight?", ['No', 'Yes'])
    fry = st.selectbox("Do you eat fried/high-calorie food often?", ['No', 'Yes'])
    veg = st.slider("Veggies in meals? (1-Never to 3-Always)", 1, 3, 2)
    meals = st.slider("How many main meals/day?", 1, 4, 3)
    snack = st.selectbox("Do you snack between meals?", ['Never', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox("Do you smoke?", ['No', 'Yes'])
    
    water_level = st.radio("💧 How would you rate your daily water intake?", ['Low (<1L)', 'Moderate (1-2L)', 'High (>2L)'], horizontal=True)
    screen_level = st.radio("📱 How much screen time do you get daily?", ['Low (<2h)', 'Moderate (2-5h)', 'High (>5h)'], horizontal=True)
    water = ['Low (<1L)', 'Moderate (1-2L)', 'High (>2L)'].index(water_level) + 1
    screen = ['Low (<2h)', 'Moderate (2-5h)', 'High (>5h)'].index(screen_level)

    calmon = st.radio("Do you track calories?", ['No', 'Yes'], horizontal=True)
    phys = st.slider("Physical activity level (0-None to 3-High)", 0, 3, 1)
    alc = st.selectbox("Alcohol consumption?", ['Never', 'Sometimes', 'Frequently', 'Always'])
    trans = st.selectbox("Primary transport?", ['Walking', 'Vehicle', 'Public Transport'])

    return {
        'Gender': sex, 'Age': age, 'Height': height_m, 'Weight': weight,
        'BMI': bmi,
        'Family_History': int(fam == 'Yes'),
        'Fried_Food': int(fry == 'Yes'),
        'Veggy_Intake': veg,
        'Meals': meals,
        'Snacking': ['Never', 'Sometimes', 'Frequently', 'Always'].index(snack),
        'Smoking': int(smoke == 'Yes'),
        'Water': water,
        'Calories_Tracked': int(calmon == 'Yes'),
        'Physical_Activity': phys,
        'Screen_Time': screen,
        'Alcohol': ['Never', 'Sometimes', 'Frequently', 'Always'].index(alc),
        'Transport': ['Walking', 'Vehicle', 'Public Transport'].index(trans)
    }

user_data = get_user_input()

if st.button("🚀 Analyze My Health"):
    input_data = pd.DataFrame([user_data])
    prediction = model.predict(input_data)[0]

    profile_desc = f"""
The user is a {user_data['Age']}-year-old {"male" if user_data['Gender']==2 else "female" if user_data['Gender']==0 else "non-binary person"} with a BMI of {user_data['BMI']}. 
They {'do' if user_data['Family_History'] else "do not"} have a family history of overweight, {'frequently eat' if user_data['Fried_Food'] else "rarely eat"} fried food, 
consume vegetables at a level of {['Never', 'Sometimes', 'Always'][user_data['Veggy_Intake'] - 1]}, and typically have {user_data['Meals']} meals per day. 
They {['never','sometimes','frequently','always'][user_data['Snacking']]} snack between meals, and they {'do' if user_data['Smoking'] else "do not"} smoke.
Water intake is {['Low (<1L)','Moderate (1-2L)','High (>2L)'][user_data['Water'] - 1]}, and they {'do' if user_data['Calories_Tracked'] else "do not"} track their calories.
Their physical activity level is {['None','Low','Moderate','High'][user_data['Physical_Activity']]}, screen time is {['Low (<2h)','Moderate (2-5h)','High (>5h)'][user_data['Screen_Time']]}, 
and they {['never','sometimes','frequently','always'][user_data['Alcohol']]} consume alcohol. 
Their primary mode of transport is {['walking','personal vehicle','public transport'][user_data['Transport']]}.
"""

    # --- Dr. Gemi (Gemini) ---
    prompt_gemi = f"""
Hello! नमस्ते! मैं Dr. Gemi हूँ, आपकी AI डॉक्टर 👩‍⚕️.

Here is the patient's full health and lifestyle profile:
{profile_desc}

Their BMI is {user_data['BMI']} and the predicted obesity risk level is: **{prediction}**.

Please provide **10 practical, easy-to-follow, warm and bilingual (English + Hindi)** health improvement tips to manage or reduce obesity risk. The tone should be friendly and motivational.
"""
    response_g = genai.GenerativeModel("gemini-pro").generate_content(prompt_gemi, temperature=0.7).text

    # --- Dr. Opie (OpenAI) ---
    prompt_opie = f"""
Hello! नमस्ते! I'm Dr. Opie, your professional AI health advisor 🧑‍⚕️.

Here is the patient's lifestyle profile:
{profile_desc}

Their BMI is {user_data['BMI']} and the obesity risk level is predicted as: **{prediction}**.

Based on this, provide **10 detailed, personalized health suggestions** in a bilingual format (English + Hindi). The tips should be realistic, clear, and encouraging, keeping cultural context in mind.
"""
    response_o = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are Dr. Opie, a professional AI health advisor."},
                  {"role": "user", "content": prompt_opie}],
        temperature=0.7,
        max_tokens=400
    )['choices'][0]['message']['content']

    # --- Show Results ---
    tabs = st.tabs(["👩‍⚕️ Dr. Gemi", "🧑‍⚕️ Dr. Opie"])
    with tabs[0]:
        st.markdown(response_g, unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(response_o, unsafe_allow_html=True)

    # --- Health Disclaimer ---
    st.info("⚠️ These suggestions are for informational purposes only and are not a substitute for professional medical advice.\n⚠️ ये स्वास्थ्य सुझाव केवल जानकारी के लिए हैं और किसी चिकित्सक का विकल्प नहीं हैं।")
