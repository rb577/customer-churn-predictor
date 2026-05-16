import gradio as gr
import joblib
import pandas as pd

# Load model and features
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("features.pkl")


# ---------- Preprocessing ----------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Example: binary columns (if you used drop_first or mapping)
    binary_cols = ['gender', 'Family', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Example: multi-category columns
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=multi_cols)

    # Align with training features
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df


# ---------- Prediction ----------
def predict_churn(
    gender, SeniorCitizen, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges, TenureGroup, Family
):
    
    input_data = {
        'gender': gender,
        'SeniorCitizen': int(SeniorCitizen),
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'TenureGroup': TenureGroup,
        'Family': int(Family)
    }

    df = preprocess_input(input_data)

    proba = model.predict_proba(df)[:, 1][0]

    # Probability bands
    if proba < 0.3:
        risk = "🟢 Low Risk"
    elif proba < 0.6:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    return f"Churn Probability: {proba:.2f}\nRisk Level: {risk}"


# ---------- UI ----------
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio([0,1], label="Senior Citizen"),
        gr.Radio(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["No", "Yes", "No phone service"], label="Multiple Lines"),

        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),

        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Radio(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method"),

        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),

        gr.Dropdown(["Short term", "Long term"], label="Tenure Group"),
        gr.Radio([0,1], label="Family")
    ],
    outputs="text",
    title="📡 Customer Churn Predictor",
    description="Predict churn risk using probability bands"
)

interface.launch()