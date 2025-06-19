# 🧠 BodyScope – Your Dual-AI Health Companion

**BodyScope** is an AI-powered health advisor that combines Machine Learning and Gemini AI to predict obesity risk, calculate BMI, and offer personalized, multilingual suggestions – all with SQL-backed logging and a professional UI.

🌐 [Live Demo](https://bodyscope.streamlit.app/)  |  👨‍⚕️ Built with ❤️ by **Aviral Meharishi**

---

## 🚀 Features

- 🧮 **BMI Calculator** with 6 Categories:
  - Severe Underweight (🔴)
  - Underweight (🟠)
  - Healthy (🟢)
  - Overweight (🟠)
  - Obese (🔴)
  - Severe Obese (🔴)

- 🧬 **Obesity Risk Prediction** using trained ML model
- 🧠 **Gemini-Powered Health Suggestions** in English, Hinglish, and Hindi
- 🗃️ **ETL + SQL Logging**: Automatically saves user inputs in MySQL database
- 📏 Height input in **Feet + Inches** for Indian users
- ⚠️ **Medical Disclaimer** displayed

---

## 🧠 ML Model Performance

| Metric         | Value  |
|----------------|--------|
| Accuracy       | **0.88** ✅ |
| F1-Score       | **0.85** |
| Model Type     | DecisionTreeClassifier |
| Input Features | 17 total lifestyle + health factors |

---

## 📥 Model Input Features

```txt
['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'MTRANS', 'CALC']
```

---

## 🧪 Tech Stack

- `Python`, `Streamlit`
- `Scikit-learn` (ML model)
- `Gemini API` via `google-generativeai`
- `MySQL` with `SQLAlchemy`
- **Deployed** on Streamlit Cloud

---

## 📁 Project Structure

```
📦 BodyScope/
├── app.py                  # Main Streamlit app logic
├── utils.py                # SQL insertions + model loader
├── bmi_utils.py            # BMI calculations + logic
├── final_model.pkl         # Trained ML model (DecisionTree)
├── requirements.txt        # Required libraries

```

---

## ⚙️ Installation

```bash
git clone https://github.com/aviralmeharishi/bodyscope.git
cd bodyscope
pip install -r requirements.txt
streamlit run app.py
```

Add your Gemini API key to:
```
.streamlit/secrets.toml
[general]
GOOGLE_API_KEY = "your_api_key_here"
```

---

## 📬 Connect with Me

**Aviral Meharishi**  
📧 aviralmeharishi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/aviralmeharishi/)  |  [GitHub](https://github.com/aviralmeharishi)

---

> "Prevention is better than cure — and BodyScope is your first step toward better health."
