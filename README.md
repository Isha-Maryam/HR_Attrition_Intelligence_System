# 🧠 HR Attrition Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3.0-blue?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **Predicting which employees are likely to quit — and WHY — using Machine Learning + SHAP Explainability**

---

## 📌 Problem Statement

Employee attrition costs companies **33% of an employee's annual salary** in hiring and training costs.
HR teams struggle to identify at-risk employees before they resign.

This project builds an **end-to-end ML pipeline** that:
- Predicts which employees are likely to leave
- Explains the top reasons driving their decision
- Provides HR with an actionable risk score per employee

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [IBM HR Analytics Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Rows | 1,470 employees |
| Columns | 35 features |
| Target | Attrition (Yes/No) |
| Class Balance | 83.9% Stayed / 16.1% Left |
| Missing Values | None |

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Scikit-Learn | Preprocessing & evaluation |
| XGBoost | Primary ML model |
| SHAP | Model explainability |
| Streamlit | Web application |
| Joblib | Model serialization |

---

## 🗂️ Project Structure

HR_Attrition_Intelligence_System/
│
├── 📁 data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── 📁 notebooks/
│   └── employee_attrition_predictor.ipynb
│
├── 📁 plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── model_summary.png
│   └── business_dashboard.png
│
├── 📁 models/
│   ├── xgboost_attrition_model.pkl
│   └── model_config.json
│
├── 📁 app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md

---

## ⚙️ Feature Engineering

7 custom features were engineered to capture hidden attrition signals:

| Feature | Formula | Business Insight |
|---|---|---|
| `Satisfaction_Score` | Avg of 4 satisfaction columns | Overall employee happiness |
| `Promotion_Lag` | YearsAtCompany − YearsSincePromotion | Career stagnation risk |
| `Tenure_Per_Company` | TotalWorkingYears / (NumCompanies+1) | Job hopper pattern |
| `WorkLife_Stress_Index` | OverTime × (5 − WorkLifeBalance) | Burnout risk |
| `Salary_Growth_Rate` | PercentSalaryHike / (YearsAtCompany+1) | Compensation fairness |
| `Experience_Age_Ratio` | TotalWorkingYears / Age | Career maturity |
| `Manager_Relationship_Risk` | (5−RelationSatisfaction) / (ManagerYears+1) | Manager conflict |

---

## 🤖 Model Pipeline

Raw Data
│
▼
EDA & Visualization
│
▼
Feature Engineering (7 new features)
│
▼
Encoding + Preprocessing
│
▼
XGBoost Classifier
│
▼
GridSearchCV (540 fits, 5-fold CV)
│
▼
Threshold Tuning (0.40 optimal)
│
▼
SHAP Explainability
---

## 📈 Model Performance

### Before vs After Tuning

| Metric | Default Model | Tuned Model |
|---|---|---|
| Accuracy | 81.63% | 77.55% |
| Precision | 42.22% | 33.03% |
| **Recall** | 40.43% | **76.60%** ✅ |
| F1 Score | 41.30% | 46.15% |
| AUC-ROC | 74.63% | 76.44% |

> **Recall improved from 40% → 76.6%** — model now catches 3 out of 4 employees who will quit.
> Threshold tuned to 0.40 to maximize HR value (catching quitters matters more than false alarms).

---

## 📉 Model Visualizations

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

### ROC Curve
![ROC Curve](plots/roc_curve.png)

### Feature Importance
![Feature Importance](plots/feature_importance.png)

### Model Summary
![Model Summary](plots/model_summary.png)

---

## 💡 Key Business Insights

🔴 Employees working OverTime are 3x more likely to quit
🔴 Sales Representatives have ~40% attrition rate
🔴 Employees aged 18-25 have the highest flight risk
🔴 Low Satisfaction Score (< 2.5) strongly predicts attrition
🔴 Employees with 7+ years and no promotion are high risk

### HR Recommendations
- 📌 Flag employees with `WorkLife_Stress_Index > 2` for immediate check-in
- 📌 Review compensation for employees with low `Salary_Growth_Rate`
- 📌 Prioritize promotion pipeline for employees with high `Promotion_Lag`
- 📌 Monitor Sales department retention quarterly

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-predictor.git
cd employee-attrition-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
cd app
streamlit run app.py
```

### 4. Open in browser
http://localhost:8501

---

## 🌐 Live Demo

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow?style=for-the-badge)](YOUR_HUGGINGFACE_LINK)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?style=for-the-badge&logo=kaggle)](YOUR_KAGGLE_LINK)

---

## 📁 How to Use the Model

```python
import joblib
import json
import pandas as pd

# Load model and config
model  = joblib.load('models/xgboost_attrition_model.pkl')
config = json.load(open('models/model_config.json'))

threshold = config['best_threshold']  # 0.40

# Predict on new data
proba      = model.predict_proba(X_new)[:, 1]
prediction = (proba >= threshold).astype(int)

print(f"Attrition Risk: {proba[0]*100:.1f}%")
print(f"Prediction: {'Will Leave' if prediction[0]==1 else 'Will Stay'}")
```

---

## 📋 Results Summary

✅ Dataset        : IBM HR Analytics (1470 employees)
✅ Features       : 52 total (35 original + 7 engineered)
✅ Model          : XGBoost (GridSearchCV tuned)
✅ Best Threshold : 0.40
✅ Recall         : 76.60% (catches 3/4 quitters)
✅ AUC-ROC        : 76.44%
✅ Deployment     : Streamlit + Hugging Face Spaces

---

## 👩‍💻 Author

**Isha Maryam**
BS Computer Science — Gomal University, DI Khan
Specialization: Machine Learning & Deep Learning

[![GitHub](https://img.shields.io/badge/GitHub-ishamaryam-black?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](YOUR_LINKEDIN)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?style=flat-square&logo=kaggle)](YOUR_KAGGLE)
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square)](YOUR_HF)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and build upon it.

---

<p align="center">
⭐ If you found this project helpful, please give it a star!
</p>