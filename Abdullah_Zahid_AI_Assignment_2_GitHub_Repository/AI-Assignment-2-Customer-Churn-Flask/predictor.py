from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
model = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def customer_from_form(form):
    required_fields = [field for field in FEATURE_ORDER if field != "TotalCharges"]
    missing = [field for field in required_fields if not str(form.get(field, "")).strip()]
    if missing:
        raise ValueError("Please fill in all required fields.")

    try:
        senior_citizen = int(form["SeniorCitizen"])
        tenure = int(form["tenure"])
        monthly_charges = float(form["MonthlyCharges"])
        total_text = str(form.get("TotalCharges", "")).strip()
        total_charges = float(total_text) if total_text else np.nan
    except ValueError as exc:
        raise ValueError("Please enter valid numbers for tenure and charges.") from exc

    if senior_citizen not in (0, 1):
        raise ValueError("Senior Citizen must be Yes or No.")
    if tenure < 0 or tenure > 100:
        raise ValueError("Tenure must be between 0 and 100 months.")
    if monthly_charges < 0:
        raise ValueError("Monthly Charges cannot be negative.")
    if not np.isnan(total_charges) and total_charges < 0:
        raise ValueError("Total Charges cannot be negative.")

    values = {
        "gender": form["gender"],
        "SeniorCitizen": senior_citizen,
        "Partner": form["Partner"],
        "Dependents": form["Dependents"],
        "tenure": tenure,
        "PhoneService": form["PhoneService"],
        "MultipleLines": form["MultipleLines"],
        "InternetService": form["InternetService"],
        "OnlineSecurity": form["OnlineSecurity"],
        "OnlineBackup": form["OnlineBackup"],
        "DeviceProtection": form["DeviceProtection"],
        "TechSupport": form["TechSupport"],
        "StreamingTV": form["StreamingTV"],
        "StreamingMovies": form["StreamingMovies"],
        "Contract": form["Contract"],
        "PaperlessBilling": form["PaperlessBilling"],
        "PaymentMethod": form["PaymentMethod"],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([[values[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def make_prediction(form):
    customer = customer_from_form(form)
    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])
    return ("Churn" if prediction == 1 else "No Churn"), probability
