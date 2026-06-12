import streamlit as st
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Page setup ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Patient Readmission Risk", page_icon="🏥")
st.title("🏥 Patient Readmission Risk Predictor")
st.write(
    "Upload a CSV of patient records to get 30-day readmission risk "
    "predictions, or enter a single patient's details manually."
)

API_URL = "https://patient-readmission-api-psbw.onrender.com"

# ── Tabs: single patient vs batch CSV ───────────────────────────────────────

tab1, tab2 = st.tabs(["Single Patient", "Batch CSV Upload"])

# ── TAB 1: Single patient form ──────────────────────────────────────────────

with tab1:
    st.subheader("Enter patient details")

    col1, col2 = st.columns(2)

    with col1:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.selectbox("Age range", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
        ])
        time_in_hospital = st.number_input("Time in hospital (days)", 1, 14, 3)
        num_lab_procedures = st.number_input("Number of lab procedures", 0, 150, 40)
        num_medications = st.number_input("Number of medications", 0, 100, 15)

    with col2:
        number_diagnoses = st.number_input("Number of diagnoses", 1, 20, 7)
        number_inpatient = st.number_input("Prior inpatient visits", 0, 20, 0)
        number_emergency = st.number_input("Prior emergency visits", 0, 20, 0)
        number_outpatient = st.number_input("Prior outpatient visits", 0, 20, 0)
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        diabetesMed = st.selectbox("On diabetes medication", ["Yes", "No"])

    if st.button("Predict Risk", type="primary"):
        payload = {
            "race": race,
            "gender": gender,
            "age": age,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_medications": num_medications,
            "number_diagnoses": number_diagnoses,
            "number_inpatient": number_inpatient,
            "number_emergency": number_emergency,
            "number_outpatient": number_outpatient,
            "insulin": insulin,
            "diabetesMed": diabetesMed,
        }

        with st.spinner("Getting prediction... (may take ~30s if API is waking up)"):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=90, verify=False)
                result = response.json()

                risk = result["risk_level"]
                score = result["readmission_risk_score"]

                if risk == "High":
                    st.error(f"**Risk Level: {risk}**  (score: {score})")
                elif risk == "Medium":
                    st.warning(f"**Risk Level: {risk}**  (score: {score})")
                else:
                    st.success(f"**Risk Level: {risk}**  (score: {score})")

                st.write(result["prediction"])

            except Exception as e:
                st.error(f"Error calling API: {e}")

# ── TAB 2: Batch CSV upload ──────────────────────────────────────────────────

with tab2:
    st.subheader("Upload a CSV of patients")
    st.write(
        "The CSV should have columns matching the patient fields "
        "(race, gender, age, time_in_hospital, etc.)"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} patients.")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction", type="primary"):
            patients = df.to_dict(orient="records")

            with st.spinner(f"Scoring {len(patients)} patients... (may take a while on free tier)"):
                try:
                    response = requests.post(
                        f"{API_URL}/predict_batch", json=patients, timeout=300, verify=False
                    )
                    results = response.json()
                    results_df = pd.DataFrame(results)

                    output = pd.concat(
                        [df.reset_index(drop=True), results_df], axis=1
                    )

                    st.success(f"Scored {len(output)} patients.")
                    st.dataframe(output)

                    csv = output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "scored_patients.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Error calling API: {e}")