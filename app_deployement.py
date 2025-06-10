import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import openai
import altair as alt

# --- API Configuration ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)
openai.api_key = OPENAI_API_KEY

# --- App Setup ---
st.set_page_config(page_title="BodyScope", page_icon="🩺", layout="centered")
st.title("🩺 BodyScope: Dual-AI Health Companion")
st.markdown("Answer a few personal and lifestyle questions — get tips from **Dr. Gemi** and **Dr. Opie** in both English & Hindi.")


# --- User Input Form ---
def get_user_input():
    st.header("🙋 Tell Us About Yourself")
    sex = st.radio("What’s your gender?", ['Female', 'Male', 'Others'])
    sex = {"Female":0, "Male":2, "Others":1}[sex]
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
    water = st.slider("Water intake (1-<1L,2-1-2L,3->2L)", 1, 3, 2)
    calmon = st.radio("Do you track calories?", ['No', 'Yes'], horizontal=True)
    phys = st.slider("Physical activity level (0-None to 3-High)", 0, 3, 1)
    screen = st.slider("Screen time (0-<2h,1-2–5h,2->5h)", 0, 2, 1)
    alc = st.selectbox("Alcohol consumption?", ['Never', 'Sometimes', 'Frequently', 'Always'])
    trans = st.selectbox("Primary transport?", ['Walking', 'Vehicle', 'Public Transport'])

    return {
        'Gender': sex, 'Age': age, 'Height': height_m, 'Weight': weight,
        'BMI': bmi,
        'Family_History': int(fam=='Yes'),
        'Fried_Food': int(fry=='Yes'),
        'Veggy_Intake': veg,
        'Meals': meals,
        'Snacking': ['Never', 'Sometimes', 'Frequently', 'Always'].index(snack),
        'Smoking': int(smoke=='Yes'),
        'Water': water,
        'Calories_Tracked': int(calmon=='Yes'),
        'Physical_Activity': phys,
        'Screen_Time': screen,
        'Alcohol': ['Never','Sometimes','Frequently','Always'].index(alc),
        'Transport': ['Walking','Vehicle','Public Transport'].index(trans)
    }

user_data = get_user_input()

if st.button("🚀 Analyze My Health"):
    bmi = user_data['BMI']
    # Log user input
    log_input(user_data)

    # --- Dr. Gemi Response ---
    prompt_gemi = f"""
Hello! नमस्ते! मैं Dr. Gemi हूँ, आपकी AI डॉक्टर 👩‍⚕️. I’ve reviewed your profile:
{user_data}
Could you please give 10 detailed friendly, practical tips in English & Hindi to improve health and manage BMI of {bmi}?
"""
    response_g = genai.GenerativeModel("gemini-pro").generate_content(prompt_gemi, temperature=0.7).text

    # --- Dr. Opie (OpenAI) Response ---
    prompt_opie = f"""
Hello and नमस्ते! I'm Dr. Opie, your AI health advisor 🧑‍⚕️.
Here’s the user’s lifestyle data:
{user_data}
Please give 10 detailed specific, actionable suggestions in English & Hindi, referencing BMI = {bmi}.
"""
    response_o = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are Dr. Opie, a professional AI health advisor."}, {"role":"user","content":prompt_opie}],
        temperature=0.7, max_tokens=400
    )['choices'][0]['message']['content']

    # --- Show Results ---
    tabs = st.tabs(["👩‍⚕️ Dr. Gemi", "🧑‍⚕️ Dr. Opie"])
    with tabs[0]:
        st.image("gemi_avatar.png", width=100)  # replace with your avatar
        st.markdown(response_g, unsafe_allow_html=True)
    with tabs[1]:
        st.image("opie_avatar.png", width=100)
        st.markdown(response_o, unsafe_allow_html=True)

    # --- BMI Trend Chart ---
    hist_df = pd.DataFrame(st.session_state.history)
    if 'BMI' in hist_df:
        chart = alt.Chart(hist_df).mark_line(point=True).encode(
            x=alt.X('index', title='Entry #'),
            y=alt.Y('BMI', title='BMI'),
            tooltip=['BMI','Age']
        ).properties(width=600, height=300, title="📈 BMI Trend Over Time")
        st.altair_chart(chart)

    # --- Health Disclaimer ---
    
    st.info("⚠️ These suggestions are for informational purposes only and are not a substitute for professional medical advice.\n⚠️ ये स्वास्थ्य सुझाव केवल जानकारी के लिए हैं और किसी चिकित्सक का विकल्प नहीं हैं।")
