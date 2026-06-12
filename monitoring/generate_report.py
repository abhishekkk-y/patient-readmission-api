import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# ── Load data ──────────────────────────────────────────────────────────────

# Reference: what the model was trained on
reference = pd.read_csv("data/X_train.csv")

# Current: simulating "recent production traffic" using test data
# In a real system, this would be pulled from predictions.db
current = pd.read_csv("data/X_test.csv")

print(f"Reference data shape: {reference.shape}")
print(f"Current data shape:   {current.shape}")

# ── Generate the drift report ────────────────────────────────────────────

report = Report(metrics=[DataDriftPreset()])

result = report.run(reference_data=reference, current_data=current)

# ── Save as HTML ──────────────────────────────────────────────────────────

result.save_html("monitoring/drift_report.html")

print("\nReport saved to monitoring/drift_report.html")

# ── Simulate a drifted production dataset ───────────────────────────────────
# Imagine: a new hospital policy means patients are now older on average,
# and stay in the hospital longer

drifted = current.copy()
drifted['age'] = drifted['age'] + 15          # patients are now ~15 years older on average
drifted['time_in_hospital'] = drifted['time_in_hospital'] * 1.5  # longer stays

drift_report = Report(metrics=[DataDriftPreset()])
drift_result = drift_report.run(reference_data=reference, current_data=drifted)
drift_result.save_html("monitoring/drift_report_simulated.html")

print("Simulated drift report saved to monitoring/drift_report_simulated.html")