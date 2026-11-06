import os
import json
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp

FEATURES = [
    "Genre",
    "Director",
    "Year",
    "Runtime (Minutes)",
    "Votes",
    "Revenue (Millions)",
    "Metascore",
    "Movie_Age"
]

BASELINE_DATA = "data/raw/IMDB-Movie-Data.csv"
EVOLVED_DATA = "data/monitoring/drift/evolved_data.csv"

MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"

OUTPUT_DIR = "outputs/monitoring"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prediction_monitoring_report.json")


def prepare_features(df):
    df = df.copy()
    df["Movie_Age"] = 2016 - df["Year"]
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    baseline = pd.read_csv(BASELINE_DATA)
    evolved = pd.read_csv(EVOLVED_DATA)

    baseline = prepare_features(baseline)
    evolved = prepare_features(evolved)

    baseline_pred = model.predict(
        preprocessor.transform(baseline[FEATURES])
    )

    evolved_pred = model.predict(
        preprocessor.transform(evolved[FEATURES])
    )

    statistic, p_value = ks_2samp(baseline_pred, evolved_pred)

    drift_detected = p_value < 0.05

    report = {
        "baseline_mean_prediction": float(np.mean(baseline_pred)),
        "evolved_mean_prediction": float(np.mean(evolved_pred)),
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected)
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Prediction monitoring report saved to {OUTPUT_FILE}")
    print(json.dumps(report, indent=4))
    print(f"[INFO] Prediction drift detected: {drift_detected}")


if __name__ == "__main__":
    main()