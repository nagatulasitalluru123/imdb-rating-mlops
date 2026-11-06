import os
import json
import pandas as pd

DATA_DRIFT_REPORT = "outputs/drift/data_drift_report.csv"
CONCEPT_DRIFT_REPORT = "outputs/drift/concept_drift_report.csv"
PREDICTION_DRIFT_REPORT = "outputs/drift/prediction_drift_report.csv"
PERFORMANCE_REPORT = "outputs/monitoring/performance_monitoring_report.csv"

OUTPUT_DIR = "outputs/monitoring"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "monitor_trigger_report.json")


def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_drift = load_csv(DATA_DRIFT_REPORT)
    concept_drift = load_csv(CONCEPT_DRIFT_REPORT)
    prediction_drift = load_csv(PREDICTION_DRIFT_REPORT)
    performance = load_csv(PERFORMANCE_REPORT)

    triggers = []

    if data_drift is not None:
        if "Drift_Detected" in data_drift.columns:
            if data_drift["Drift_Detected"].astype(bool).any():
                triggers.append("Data drift detected")

    if concept_drift is not None:
        if "R2" in concept_drift.columns and len(concept_drift) >= 2:
            baseline_r2 = concept_drift.iloc[0]["R2"]
            drift_r2 = concept_drift.iloc[1]["R2"]

            if baseline_r2 - drift_r2 > 0.10:
                triggers.append("Concept drift detected")

    if prediction_drift is not None:
        if "drift_detected" in prediction_drift.columns:
            if prediction_drift["drift_detected"].astype(bool).any():
                triggers.append("Prediction drift detected")

    if performance is not None:
        if "R2" in performance.columns:
            r2 = performance["R2"].iloc[0]

            if r2 < 0.60:
                triggers.append("Performance degradation detected")

    report = {
        "monitoring_triggered": len(triggers) > 0,
        "triggers": triggers,
        "action": "Retraining recommended" if triggers else "No action required"
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("[SUCCESS] Monitor trigger report saved to", OUTPUT_FILE)
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()