import streamlit as st
import pandas as pd
import requests
import urllib3
import plotly.graph_objects as go
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

    # Build payload (used by both prediction and PDF generation later)
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

    if st.button("Predict Risk", type="primary"):

        with st.spinner("Getting prediction... (may take ~30s if API is waking up)"):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=90, verify=False)
                result = response.json()

                # Store in session state so it survives reruns (needed for PDF export later)
                st.session_state["last_result"] = result
                st.session_state["last_payload"] = payload

            except Exception as e:
                st.error(f"Error calling API: {e}")
                st.session_state["last_result"] = None

    # ── Display results if we have them ─────────────────────────────────
    if st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        risk = result["risk_level"]
        score = result["readmission_risk_score"]

        st.markdown("---")
        st.subheader("Prediction Result")

        col_a, col_b = st.columns([1, 1])

        with col_a:
            if risk == "High":
                st.error(f"**Risk Level: {risk}**")
            elif risk == "Medium":
                st.warning(f"**Risk Level: {risk}**")
            else:
                st.success(f"**Risk Level: {risk}**")

            st.metric("Readmission Risk Score", f"{score:.1%}")
            st.write(result["prediction"])

        with col_b:
            # Probability distribution bar chart
            # Note: the model returns probability of readmission <30 days.
            # We split the remainder proportionally for illustration.
            prob_lt30 = score
            prob_remaining = 1 - score

            fig_bar = go.Figure(data=[
                go.Bar(
                    x=["Risk Score", "Remaining"],
                    y=[prob_lt30, prob_remaining],
                    marker_color=["#EF4444" if risk == "High" else "#F59E0B" if risk == "Medium" else "#10B981", "#374151"]
                )
            ])
            fig_bar.update_layout(
                title="Readmission Risk Score",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart of key clinical attributes
        radar_categories = [
            "Time in Hospital", "Lab Procedures", "Medications",
            "Diagnoses", "Inpatient Visits", "Emergency Visits", "Outpatient Visits"
        ]
        radar_values = [
            payload["time_in_hospital"],
            payload["num_lab_procedures"],
            payload["num_medications"],
            payload["number_diagnoses"],
            payload["number_inpatient"],
            payload["number_emergency"],
            payload["number_outpatient"],
        ]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself'
        ))
        fig_radar.update_layout(
            title="Patient Profile Overview",
            polar=dict(radialaxis=dict(visible=True)),
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

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