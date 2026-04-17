from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# allow your local Vite dev origin(s)
CORS(app,
     resources={r"/*": {"origins": [
         "http://localhost:5173",
         "http://127.0.0.1:5173",
         "https://spoofygoofy.xyz",
         "https://fancy-cactus-e1c04e.netlify.app/"
     ]}},
     supports_credentials=False)  # set to True only if you send cookies/auth

@app.route("/")
def home():
    return "Welcome to the ML API!"

# ---------------------------
# Final FULL feature list
# (probability + severity merged)
# ---------------------------
numeric_p1 = [
    "visits_last_year",
    "chronic_count",
    "risk_score",
    "mental_health",
    "arthritis",
    "diabetes",
    "smoker",       # ordinal encoded
    "hypertension",
    "age",
    "proc_surgery_count",
    "avg_num_of_days_per_hospitalizations_last_3yrs",
    "network_tier",   # ordinal encoded
    "provider_quality",
    "asthma",
    "cancer_history",
    "cardiovascular_disease",
    "policy_term_years",
    "medication_count",
    "proc_imaging_count",
    "income",
    "hba1c",
    "deductible",
    "bmi",
    "education",               # ordinal encoded
    "policy_changes_last_2yrs"
]


categorical_p1 = [
    "sex",
    "marital_status",
    "employment_status",
    "urban_rural",
    "region",
    "plan_type"
]

numeric_p2 = [
    "smoker",
    "avg_num_of_days_per_hospitalizations_last_3yrs",
    "risk_score",
    "visits_last_year",
    "chronic_count",
    "diabetes",
    "had_major_procedure",
    "proc_surgery_count",
    "bmi",
    "mental_health",
    "age",
    "copd",
    "income",
    "proc_imaging_count",
    "ldl",
    "medication_count",
    "network_tier",
    "provider_quality",
    "proc_consult_count",
    "household_size",
    "systolic_bp",
    "alcohol_freq",
    "hba1c",
    "hypertension",
    "proc_physio_count",
    "copay",
    "diastolic_bp",
    "cardiovascular_disease",
    "deductible",
    "policy_term_years",
    "education",
    "policy_changes_last_2yrs",
    "cancer_history",
    "asthma"
]


categorical_p2 = [
    "sex",
    "marital_status",
    "region",
    "plan_type",
    "employment_status"
]

ALL_P1 = numeric_p1 + categorical_p1
ALL_P2 = numeric_p2 + categorical_p2

pre_p1 = joblib.load("preprocessor_part1.pkl")
pre_p2 = joblib.load("preprocessor_part2.pkl")

clf_model = joblib.load("probability_model.pkl")   # Part 1 classifier
reg_model = joblib.load("severity_model.pkl")

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict/claim_amount/v2")
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # ----- PART 1 (probability) -----
    df_p1 = df[ALL_P1]
    X_prob = pre_p1.transform(df_p1)
    prob = clf_model.predict_proba(X_prob)[0, 1]

    # ----- PART 2 (severity) -----
    df_p2 = df[ALL_P2]
    X_sev = pre_p2.transform(df_p2)
    log_cost = reg_model.predict(X_sev)[0]
    severity = float(np.expm1(log_cost))

    expected_cost = prob * severity

    return {
        "probability_of_claim": float(prob),
        "severity_if_claim": severity,
        "expected_claim_amount_per_claim": float(expected_cost)
    }

# Load your pipeline (preprocessor + model)
pipeline = joblib.load("claims_amount_pipeline.pkl")

@app.route("/predict/claim_amount", methods=["POST"])
def predict_claims():
    try:
        data = request.get_json()

        # Convert request JSON to DataFrame
        df = pd.DataFrame([data])

        # Predict
        pred = pipeline.predict(df)[0]

        return jsonify({
            "predicted_claim_amount": float(pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)


