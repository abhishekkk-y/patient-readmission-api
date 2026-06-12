from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import sqlite3
from datetime import datetime
import os
from pathlib import Path

# ── Start the app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Patient Readmission Risk API",
    description="Predicts 30-day hospital readmission risk for diabetic patients",
    version="1.0.0"
)

# ── Load model from MLflow ─────────────────────────────────────────────────────
# This loads the model you marked as "champion" in the registry
# It runs once when the API starts up

MODEL_URI = os.getenv(
    "READMISSION_MODEL_URI",
    "models:/readmission-model@champion"
)
print(f"Loading model from {MODEL_URI}...")
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as exc:
    fallback_model_path = Path(__file__).resolve().parents[1] / "mlruns" / "1" / "models" / "m-883a2c92d3844206a9d7d90af3f7f57a" / "artifacts"
    print(f"Registry load failed: {exc}")
    print(f"Falling back to local model artifact at {fallback_model_path}")
    model = mlflow.sklearn.load_model(str(fallback_model_path))
print("Model loaded.")

# ── Feature columns ────────────────────────────────────────────────────────────
# The model expects features in this exact order — same as X_train.csv

FEATURE_COLUMNS = [
    'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed', 'glucose_tested', 'a1c_tested',
    'race_AfricanAmerican', 'race_Asian', 'race_Caucasian',
    'race_Hispanic', 'race_Other', 'race_Unknown'
]

AGE_MAP = {
    '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

# ── Input schema ───────────────────────────────────────────────────────────────
# Pydantic BaseModel defines exactly what data the API expects
# Default values mean you can test it without filling every field

class PatientData(BaseModel):
    race: str               = "Caucasian"   # Caucasian, AfricanAmerican, Asian, Hispanic, Other
    gender: str             = "Female"      # Male, Female
    age: str                = "[50-60)"     # age bracket e.g. [30-40)
    admission_type_id: int  = 1             # 1=Emergency, 2=Urgent, 3=Elective
    discharge_disposition_id: int = 1
    admission_source_id: int      = 7
    time_in_hospital: int         = 3       # days
    num_lab_procedures: int       = 40
    num_procedures: int           = 1
    num_medications: int          = 15
    number_outpatient: int        = 0
    number_emergency: int         = 0
    number_inpatient: int         = 0
    number_diagnoses: int         = 7
    insulin: str                  = "No"    # No, Steady, Up, Down
    change: str                   = "No"    # No, Ch (medication change)
    diabetesMed: str              = "Yes"   # No, Yes
    glucose_tested: bool          = False
    a1c_tested: bool              = False


# ── Preprocessing ──────────────────────────────────────────────────────────────
# Converts the raw API input into the exact format the model expects
# Must match the preprocessing done in preprocess.py

def preprocess_input(patient: PatientData) -> pd.DataFrame:

    # Start with a row of all zeros
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Gender: Male=1, Female=0
    row['gender'] = 1 if patient.gender == 'Male' else 0

    # Age: convert bracket to numeric midpoint
    row['age'] = AGE_MAP.get(patient.age, 55)

    # Numeric features: pass through directly
    row['admission_type_id']          = patient.admission_type_id
    row['discharge_disposition_id']   = patient.discharge_disposition_id
    row['admission_source_id']        = patient.admission_source_id
    row['time_in_hospital']           = patient.time_in_hospital
    row['num_lab_procedures']         = patient.num_lab_procedures
    row['num_procedures']             = patient.num_procedures
    row['num_medications']            = patient.num_medications
    row['number_outpatient']          = patient.number_outpatient
    row['number_emergency']           = patient.number_emergency
    row['number_inpatient']           = patient.number_inpatient
    row['number_diagnoses']           = patient.number_diagnoses

    # Medications: No=0, anything else=1
    row['insulin']      = 0 if patient.insulin == 'No' else 1
    row['change']       = 1 if patient.change == 'Ch' else 0
    row['diabetesMed']  = 1 if patient.diabetesMed == 'Yes' else 0

    # Tests
    row['glucose_tested'] = int(patient.glucose_tested)
    row['a1c_tested']     = int(patient.a1c_tested)

    # Race: one-hot encoding
    # Set the matching race column to 1, everything else stays 0
    race_col = f"race_{patient.race}"
    if race_col in FEATURE_COLUMNS:
        row[race_col] = 1
    else:
        row['race_Unknown'] = 1  # fallback for unexpected values

    # Return as DataFrame in the exact column order the model expects
    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ── Prediction logger ──────────────────────────────────────────────────────────
# Every prediction gets saved to a local SQLite database
# This is what Evidently AI will read later for monitoring

def log_to_db(patient_data: dict, probability: float, prediction: int):
    conn = sqlite3.connect("predictions.db")
    record = patient_data.copy()
    record['probability'] = probability
    record['prediction']  = prediction
    record['timestamp']   = datetime.utcnow().isoformat()
    pd.DataFrame([record]).to_sql(
        "predictions", conn, if_exists="append", index=False
    )
    conn.close()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick check that the API is running and the model is loaded."""
    return {"status": "ok", "model": "readmission-model@champion"}


@app.post("/predict")
def predict(patient: PatientData):
    """
    Accepts patient data and returns a 30-day readmission risk score.
    """
    try:
        # Step 1: convert input to model-ready features
        features = preprocess_input(patient)

        # Step 2: get probability from model
        probability = float(model.predict_proba(features)[0][1])
        prediction  = int(probability >= 0.5)

        # Step 3: assign a human-readable risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Step 4: log this prediction for monitoring
        log_to_db(patient.dict(), probability, prediction)

        # Step 5: return the result
        return {
            "readmission_risk_score": round(probability, 4),
            "risk_level": risk_level,
            "prediction": (
                "Likely readmission within 30 days"
                if prediction == 1
                else "Unlikely readmission within 30 days"
            ),
            "model_version": "readmission-model@champion"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))