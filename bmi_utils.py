# bmi_utils.py
def calculate_bmi(weight: float, height: float) -> float:
    return round(weight / (height ** 2), 2)

def get_bmi_category(bmi: float) -> str:
    if bmi < 16:
        return "Severe Underweight"
    elif 16 <= bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Healthy"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Obese"
    else:
        return "Severe Obese"

def get_bmi_message(bmi: float, category: str) -> str:
    return f"Your BMI is {bmi}, categorized as '{category}'."

def get_bmi_color(category: str):
    import streamlit as st
    if category in ["Severe Underweight", "Obese", "Severe Obese"]:
        return st.error
    elif category in ["Underweight", "Overweight"]:
        return st.warning
    elif category == "Healthy":
        return st.success
    else:
        return st.info
