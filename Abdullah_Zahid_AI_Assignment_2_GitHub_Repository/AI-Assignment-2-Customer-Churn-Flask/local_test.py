from predictor import make_prediction

churn_customer = {
    "gender":"Female", "SeniorCitizen":"0", "Partner":"No", "Dependents":"No",
    "tenure":"2", "PhoneService":"Yes", "MultipleLines":"No",
    "InternetService":"Fiber optic", "OnlineSecurity":"No", "OnlineBackup":"No",
    "DeviceProtection":"No", "TechSupport":"No", "StreamingTV":"Yes",
    "StreamingMovies":"Yes", "Contract":"Month-to-month",
    "PaperlessBilling":"Yes", "PaymentMethod":"Electronic check",
    "MonthlyCharges":"85.0", "TotalCharges":"170.0"
}

no_churn_customer = {
    "gender":"Male", "SeniorCitizen":"0", "Partner":"Yes", "Dependents":"Yes",
    "tenure":"60", "PhoneService":"Yes", "MultipleLines":"Yes",
    "InternetService":"DSL", "OnlineSecurity":"Yes", "OnlineBackup":"Yes",
    "DeviceProtection":"Yes", "TechSupport":"Yes", "StreamingTV":"No",
    "StreamingMovies":"No", "Contract":"Two year",
    "PaperlessBilling":"No", "PaymentMethod":"Bank transfer (automatic)",
    "MonthlyCharges":"65.0", "TotalCharges":"3900.0"
}

for number, customer in enumerate([churn_customer, no_churn_customer], start=1):
    result, probability = make_prediction(customer)
    print(f"Customer {number}: {result} | Churn probability: {probability:.4f}")
