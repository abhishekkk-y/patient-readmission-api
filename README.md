# Patient Readmission Risk API

An end-to-end machine learning system that predicts whether a diabetic patient
is likely to be readmitted to the hospital within 30 days of discharge. The
project covers the full lifecycle — from data preprocessing and model training
to experiment tracking, containerized deployment, and monitoring.

**Live API:** https://patient-readmission-api-psbw.onrender.com/docs

> Note: this is hosted on Render's free tier, which spins down after 15
> minutes of inactivity. The first request after a period of inactivity may
> take 30–60 seconds to respond while the service wakes up.

---

## Problem Statement

Hospital readmissions within 30 days are a major cost and quality-of-care
issue, and CMS (Centers for Medicare & Medicaid Services) financially
penalizes hospitals with high readmission rates. This project predicts
30-day readmission risk for diabetic patients using clinical and
administrative data, giving care teams an early signal to prioritize
follow-up care.

**Dataset:** [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
— 101,766 patient encounters across 130 US hospitals.

---

## Architecture

```
Raw data (UCI)
      │
      ▼
preprocess.py ──► clean train/val/test splits
      │
      ▼
train.py ──► XGBoost model ──► MLflow tracking & model registry
      │
      ▼
FastAPI app (app/main.py)
      │
      ├─► /health    – service health check
      └─► /predict   – returns readmission risk score
                │
                ▼
        Predictions logged to SQLite
                │
                ▼
        Evidently AI ──► data drift & model monitoring report
      │
      ▼
Docker container ──► deployed on Render (free tier)
```

---

## Tech Stack

| Layer              | Tool                                  |
|--------------------|---------------------------------------|
| Model              | XGBoost (scikit-learn API)            |
| Experiment tracking | MLflow (tracking + model registry)   |
| API serving        | FastAPI + Uvicorn                     |
| Containerization   | Docker (built via Rancher Desktop)    |
| Deployment         | Render (free tier, Docker runtime)    |
| Monitoring         | Evidently AI + SQLite prediction logs |
| Testing            | Pytest                                |

---

## Model Performance

Three XGBoost configurations were trained and tracked via MLflow. The best
run (200 trees, max depth 4, learning rate 0.1) was promoted to the model
registry under the alias `champion`:

| Metric     | Score  |
|------------|--------|
| ROC AUC    | 0.682  |
| F1 Score   | 0.277  |
| Precision  | 0.179  |
| Recall     | 0.609  |

Class imbalance (~11% of patients are readmitted within 30 days) was handled
using `scale_pos_weight`, tuned to roughly 8 — the ratio of negative to
positive cases — so the model is penalized more heavily for missing a true
readmission.

**Limitations:** the three `diag_*` (diagnosis) columns were dropped during
preprocessing because they contain 700+ unique ICD codes requiring more
sophisticated encoding. These are likely the most clinically predictive
features in the dataset, and re-incorporating them is the most promising path
to improving model performance.

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/abhishekkk-y/patient-readmission-api.git
cd patient-readmission-api

# Set up environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

Then open `http://localhost:8000/docs` for interactive API documentation.

### Running with Docker

```bash
docker build -t patient-readmission-api .
docker run -p 8000:8000 patient-readmission-api
```

---

## Project Structure

```
patient-readmission-api/
├── app/                # FastAPI application
│   └── main.py
├── src/                # Data preprocessing & training pipeline
│   ├── preprocess.py
│   └── train.py
├── monitoring/         # Evidently AI drift reports
├── tests/              # Pytest test suite
├── mlruns/             # MLflow experiment artifacts
├── mlflow.db           # MLflow tracking & model registry database
├── Dockerfile
└── requirements.txt
```

---

## API Usage

### Health check
```
GET /health
```
Returns service status and the model version currently loaded.

### Predict readmission risk
```
POST /predict
```

Example request body:
```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[50-60)",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 3,
  "num_lab_procedures": 40,
  "num_procedures": 1,
  "num_medications": 15,
  "number_outpatient": 0,
  "number_emergency": 0,
  "number_inpatient": 0,
  "number_diagnoses": 7,
  "insulin": "No",
  "change": "No",
  "diabetesMed": "Yes",
  "glucose_tested": false,
  "a1c_tested": false
}
```

Example response:
```json
{
  "readmission_risk_score": 0.3241,
  "risk_level": "Medium",
  "prediction": "Unlikely readmission within 30 days",
  "model_version": "readmission-model@champion"
}
```

---

## Monitoring

Every call to `/predict` is logged to a local SQLite database
(`predictions.db`), capturing the input features, predicted probability, and
timestamp. Evidently AI uses these logs to compare the distribution of
incoming requests against the training data distribution, flagging potential
data drift — for example, if the average patient age or admission type in
production shifts noticeably from what the model was trained on.

---

## What I'd Do Differently / Future Improvements

- Re-encode the `diag_1`, `diag_2`, `diag_3` diagnosis columns (likely the
  most predictive features) instead of dropping them
- Add authentication to the API before any real-world use
- Move model artifact storage to a remote store (e.g., S3) rather than
  committing `mlruns/` and `mlflow.db` to the repository
- Automate retraining triggers based on detected data drift
- Replace SQLite logging with a structured logging/observability pipeline
  suitable for production scale

---

## Acknowledgments

Dataset: Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S.,
Cios, K.J., & Clore, J.N. (2014). *Impact of HbA1c Measurement on Hospital
Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.*
BioMed Research International. UCI Machine Learning Repository.
