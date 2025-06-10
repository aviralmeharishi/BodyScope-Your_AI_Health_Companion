import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
import openai

# --- Load Model ---
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- API Keys ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Streamlit Setup ---
st.set_page_config(page_title="BodyScope", layout="centered")
st.title("🩺 BodyScope: Your AI Health Companion")

# --- Input Form ---
def get_user_input():
    st.header("🙋 Enter Your Health Profile")

    gender = st.radio("Gender", ['Female', 'Male', 'Others'])
    gender = {"Female": 0, "Male": 2, "Others": 1}[gender]

    age = st.slider("Age", 10, 100, 25)
    st.markdown("#### Height")
    col1, col2 = st.columns(2)
    with col1:
        feet = st.number_input("Feet", min_value=2, max_value=8, value=5)
    with col2:
        inches = st.number_input("Inches", min_value=0, max_value=11, value=7)
    
    # Convert to meters
    height_m = round(((feet * 12) + inches) * 0.0254, 2)

    weight = st.number_input("Weight (in kg)", 20, 200, 70)
    bmi = round(weight / (height ** 2), 1)
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
        'Height': height,
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

input_df, bmi, user_data = get_user_input()

# --- Prediction ---
if st.button("🔍 Predict My Obesity Risk"):
    prediction = model.predict(input_df)[0]
    st.success(f"🧬 **Obesity Class**: {prediction}")

# --- Gemi Prompt ---
if st.button("🧚‍♀️ Ask Dr. Gemi"):
    gemi_prompt = f"""
Hello! नमस्ते! मैं Dr. Gemi हूँ, आपकी AI हेल्थ साथी ❤️  
Please provide 10 sweet and heartwarming health suggestions in English & Hindi for a person with the following profile:

{user_data}

BMI is approximately {bmi}. Focus on motivation, encouragement, and practical improvements.
"""
    response = genai.GenerativeModel("gemini-pro").generate_content(gemi_prompt, temperature=0.9).text
    st.markdown(response, unsafe_allow_html=True)

# --- Opie Prompt ---
if st.button("🧑‍⚕️ Ask Dr. Opie"):
    opie_prompt = f"""
User profile:  
{user_data}

BMI ~ {bmi}.  
You're Dr. Opie: strict, disciplined AI doctor. Give 10 direct, practical, and actionable health tips in **English & Hindi** to improve their lifestyle and BMI.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are Dr. Opie, a no-nonsense, strict health advisor."},
                  {"role": "user", "content": opie_prompt}],
        temperature=0.6, max_tokens=600
    )['choices'][0]['message']['content']
    st.markdown(response, unsafe_allow_html=True)


st.info("⚠️ These health tips are AI-generated for educational use and are not a substitute for professional medical advice.")

