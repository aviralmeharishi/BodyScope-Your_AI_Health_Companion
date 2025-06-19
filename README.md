# 🧠 BodyScope – Your Dual-AI Health Companion

---

🌐 [Live Demo](https://bodyscope.streamlit.app/)  |  👨‍⚕️ Built with ❤️ by **Aviral Meharishi**

---



**BodyScope** is a cutting-edge AI health advisor built by Aviral Meharishi that combines:


* ✅ Machine Learning (Decision Tree) for **obesity risk prediction**
* ✅ Gemini API for **smart multilingual suggestions**
* ✅ SQL-backed **ETL + logging**
* ✅ Indian-friendly UI with BMI calculator in Feet + Inches

---

## 📊 Data Overview

* **Dataset Size:** 10,000+ entries (synthetic + raw)
* **Target:** `NObeyesdad` (7-class Obesity Label)
* **Source:** `ObesityDataSet_raw_and_data_sinthetic.csv`

---

## 🔄 ETL Pipeline (SQL-Driven)

* Data ingested → cleaned → binned (via pandas + custom functions)
* Refined using **SQL update queries** (e.g., mapping `"MotorBike"` → `"Bike"`)
* MySQL + SQLAlchemy used to:

  * Create and push full dataset to `Obesity.Obesity_Data`
  * Log user interactions and predictions

---

## 📈 Feature Engineering & Stats

| Feature                      | Transformation                                 |
| ---------------------------- | ---------------------------------------------- |
| `FCVC`, `NCP`, `CH2O`, `FAF` | Ordinal binning                                |
| `TUE`                        | Categorized as `Low`, `Moderate`, `High Usage` |
| `SCC`, `SMOKE`, `FAVC`       | Normalized strings                             |
| `MTRANS`                     | Refined using SQL queries                      |

---

## 🤖 ML Model Comparison & Final Selection

Multiple models were evaluated to ensure optimal prediction accuracy and generalizability:

| Model                          | Accuracy | F1 Score | Why It Was Used                               |
| ------------------------------ | -------- | -------- | --------------------------------------------- |
| Logistic Regression            | 0.78     | 0.74     | Simple, interpretable baseline                |
| K-Nearest Neighbors (KNN)      | 0.84     | 0.82     | Easy to implement, performed decently         |
| Random Forest                  | 0.86     | 0.83     | Balanced bias-variance, better generalization |
| ✅ **Decision Tree Classifier** | **0.88** | **0.85** | Best accuracy + full interpretability         |

* **Why Decision Tree?**

  * Easy to explain to non-technical users
  * Captures non-linear relationships well
  * Quick training + minimal tuning needed

* Final model trained with 80-20 split

* Exported as `final_model.pkl` for production use

### 🎯 Input Features Used:

```python
['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC',
 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'MTRANS', 'CALC']
```

---

## 💡 AI Health Suggestions (Gemini-Powered)

* 🌐 Multilingual Tips: English, Hindi, Hinglish
* 💬 Context-aware prompts based on:

  * BMI category (e.g., Obese, Underweight, Healthy)
  * Age group & transport habits
  * Personalized, cultural sensitivity

### 🧠 Prompt Example

```python
"User BMI is 'Severely Obese', Age: 26, Transport: Public. Give 5 lifestyle tips in Hinglish."
```

---

## 🗃️ SQL Logging

* All user entries are recorded in **MySQL** for analytics and feedback loops
* Supports longitudinal health tracking and retraining

---

## 📁 Project Structure

```
📦 BodyScope/
├── app.py                         # Main Streamlit application
├── utils.py                       # DB logging and model interface
├── bmi_utils.py                   # BMI computation logic
├── final_model.pkl                # Exported DecisionTree model
├── Obesity Risk Prediction.ipynb  # EDA + training notebook
└── requirements.txt               # Python dependencies
```

---
## 📬 Connect with Me

**Aviral Meharishi**  
📧 aviralmeharishi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/aviralmeharishi/)  |  [GitHub](https://github.com/aviralmeharishi)

---
> *"Prevention is better than cure — and BodyScope is your first step toward better health."* ❤️

