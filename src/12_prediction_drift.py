import os
import joblib
import pandas as pd
from scipy.stats import ks_2samp

MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"
BASE_DATA = "data/raw/IMDB-Movie-Data.csv"
DRIFT_DATA = "data/monitoring/drift/evolved_data.csv"
OUTPUT = "outputs/drift/prediction_drift_report.csv"

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

def prepare_features(df):
    df = df.copy()
    latest_year = 2016
    df["Movie_Age"] = latest_year - df["Year"]
    return df

def predict(model, preprocessor, df):
    df = prepare_features(df)
    X = df[FEATURES].copy()
    return model.predict(preprocessor.transform(X))

def main():
    os.makedirs("outputs/drift", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    base = pd.read_csv(BASE_DATA)
    drift = pd.read_csv(DRIFT_DATA)

    a = predict(model, preprocessor, base)
    b = predict(model, preprocessor, drift)

    statistic, p_value = ks_2samp(a, b)
    drift_detected = p_value < 0.05

    report = pd.DataFrame([{
        "KS_Statistic": statistic,
        "P_Value": p_value,
        "Drift_Detected": drift_detected
    }])

    report.to_csv(OUTPUT, index=False)

    print(f"[SUCCESS] Prediction drift report saved to {OUTPUT}")
    print(report.to_string(index=False))
    print(f"[INFO] Prediction drift detected: {drift_detected}")

if __name__ == "__main__":
    main()