# Customer Churn Risk Analyzer
### End-to-End Machine Learning System for Predicting Customer Churn Probability and Risk Levels

This project predicts the **probability that a customer may churn** and converts that probability into **business-friendly risk categories**:

🟢 Low Risk  
🟡 Medium Risk  
🔴 High Risk  

Instead of only predicting **"Will churn / Won't churn"**, the system estimates **how likely** a customer is to churn, allowing businesses to prioritize retention strategies more effectively.

The final model is deployed as an interactive web application for real-time risk analysis.

---

## Live Application

Hugging Face Deployment:

https://huggingface.co/spaces/rb4450/Churn-Predictor

---

# Business Problem

Customer churn directly impacts long-term revenue and customer lifetime value.

Businesses often need to answer:

> Which customers are at risk of leaving, and how urgent is intervention?

Binary predictions alone may not provide enough insight.

A customer with:

- 15% churn probability
- 48% churn probability
- 82% churn probability

would all be treated differently in practice.

This project addresses that by converting churn probability into **risk levels**, making predictions more actionable.

---

# Project Objective

Develop a machine learning system capable of:

✔ Predicting customer churn probability  
✔ Estimating customer risk level  
✔ Comparing multiple models  
✔ Supporting business retention decisions  
✔ Deploying predictions through a real-time application  

---

# Workflow

The project follows a complete machine learning lifecycle:

```text
Business Problem
      ↓
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Feature Selection
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Probability Prediction
      ↓
Risk Classification
      ↓
Web Application
      ↓
Deployment
```

---

# Model Comparison

Multiple approaches were evaluated before selecting the final deployment model.

| Metric | XGBoost (Selected Model) | XGBoost + Backward Feature Selection |
|--------|---------------------------|--------------------------------------|
| Accuracy | **~79%** | ~72% |
| ROC-AUC | **~0.74** | ~0.73 |
| Recall | 0.68 | **0.79** |
| Precision | **Higher** | Lower |
| False Positives | Lower | Higher |
| Overall Stability | **Better** | Moderate |
| Probability Calibration | **Better** | Moderate |
| Deployment Choice | ✅ Selected | ❌ Not Selected |

---

# Why XGBoost Was Selected

Although backward feature selection improved recall, the standard XGBoost model produced stronger overall performance.

Reasons for deployment:

| Selection Factor | Business Impact |
|------------------|-----------------|
| Higher Accuracy | More reliable predictions |
| Better ROC-AUC | Improved separation between churners and non-churners |
| Better Precision | Reduces unnecessary retention efforts |
| Stable Probability Scores | Improves risk classification |
| Balanced Performance | Better suited for deployment |

Final decision:

**XGBoost was selected because it generated stronger and more stable probability estimates for customer risk prediction.**

---

# Final Model Performance

Selected Model:

**XGBoost**

| Metric | Score |
|--------|-------|
| Accuracy | ~79% |
| ROC-AUC | ~0.74 |
| Recall | ~68% |
| Precision | Higher than reduced-feature model |

---

# Risk Classification System

The application converts churn probability into risk levels using probability bands.

Example interpretation:

| Churn Probability | Risk Level | Business Meaning | Possible Action |
|-------------------|------------|------------------|----------------|
| Low Probability | 🟢 Low Risk | Customer likely to stay | Monitor |
| Moderate Probability | 🟡 Medium Risk | Potential churn risk | Engagement campaigns |
| High Probability | 🔴 High Risk | Customer likely to churn | Immediate retention action |

This transforms raw model output into **business decision support**.

---

# Example Application Output

Application output includes:

```text
Churn Probability: 0.16
Risk Level: 🟢 Low Risk
```

Meaning:

The customer currently shows a **16% probability of churn**, indicating relatively low retention concern.

---

# Real-World Business Usage

The system could support:

- Telecom churn monitoring
- Subscription services
- SaaS customer retention
- Banking customer loss prediction
- Insurance renewal prediction
- E-commerce loyalty analysis

Rather than replacing analysts, the model helps prioritize **which customers need attention first**.

---

# Application Features

Users can:

✔ Input customer information  
✔ Generate churn probability instantly  
✔ View customer risk level  
✔ Interpret predictions in business terms  
✔ Analyze churn risk through a web interface  

---

# Technologies Used

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Model Serialization
- Pickle / Joblib

### Deployment
- Hugging Face Spaces

### Application Layer
- Gradio / Streamlit *(replace with actual framework used)*

---

# Dataset

This project uses the **IBM Telco Customer Churn Dataset**, commonly used for churn prediction studies.

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset is **not included in this repository** to respect ownership/licensing considerations.

To reproduce experiments:

1. Download dataset
2. Place:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

inside the working directory

3. Run notebooks or application

---

# Repository Structure

```text
customer-churn-predictor/

├── app.py
├── models/
├── notebooks/
├── screenshots/
├── requirements.txt
└── README.md
```

---

# Run Locally

Clone repository:

```bash
git clone https://github.com/rb577/customer-churn-predictor.git
cd customer-churn-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
python app.py
```

---

# Key Outcome

This project was intentionally designed to move beyond binary classification and provide:

**Probability Prediction → Risk Interpretation → Business Action**

Instead of simply predicting churn, the system translates model outputs into customer risk insights that can support retention strategies.

---
