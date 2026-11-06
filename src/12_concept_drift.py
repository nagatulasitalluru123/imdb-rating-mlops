import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"
BASE_DATA = "data/raw/IMDB-Movie-Data.csv"
DRIFT_DATA = "data/monitoring/drift/evolved_labeled_data.csv"
OUTPUT = "outputs/drift/concept_drift_report.csv"

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

def metrics(model, preprocessor, df):
    df = prepare_features(df)
    X = df[FEATURES].copy()
    y = df["Rating"]

    X_processed = preprocessor.transform(X)
    predictions = model.predict(X_processed)

    return {
        "MAE": mean_absolute_error(y, predictions),
        "RMSE": mean_squared_error(y, predictions) ** 0.5,
        "R2": r2_score(y, predictions)
    }

def main():
    os.makedirs("outputs/drift", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    base = pd.read_csv(BASE_DATA)
    drift = pd.read_csv(DRIFT_DATA)

    baseline = metrics(model, preprocessor, base)
    evolved = metrics(model, preprocessor, drift)

    report = pd.DataFrame([
        {"Dataset": "Baseline", **baseline},
        {"Dataset": "Concept Drift", **evolved}
    ])

    report.to_csv(OUTPUT, index=False)

    print(f"[SUCCESS] Concept drift report saved to {OUTPUT}")
    print(report.to_string(index=False))

    r2_drop = baseline["R2"] - evolved["R2"]
    drift_detected = r2_drop > 0.05

    print(f"[INFO] R2 drop: {r2_drop:.4f}")
    print(f"[INFO] Concept drift detected: {drift_detected}")

if __name__ == "__main__":
    main()