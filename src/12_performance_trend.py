import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"
BASE_DATA = "data/raw/IMDB-Movie-Data.csv"
DRIFT_DATA = "data/monitoring/drift/evolved_labeled_data.csv"
OUTPUT = "outputs/monitoring/performance_trend_report.csv"

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
    df["Movie_Age"] = 2016 - df["Year"]
    return df

def score(model, preprocessor, df):
    df = prepare_features(df)

    X = df[FEATURES].copy()
    y = df["Rating"]

    pred = model.predict(preprocessor.transform(X))

    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    r2 = r2_score(y, pred)

    return mae, rmse, r2

def main():
    os.makedirs("outputs/monitoring", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    base = pd.read_csv(BASE_DATA)
    drift = pd.read_csv(DRIFT_DATA)

    base_mae, base_rmse, base_r2 = score(model, preprocessor, base)
    drift_mae, drift_rmse, drift_r2 = score(model, preprocessor, drift)

    report = pd.DataFrame([
        {
            "Dataset": "Baseline",
            "MAE": base_mae,
            "RMSE": base_rmse,
            "R2": base_r2
        },
        {
            "Dataset": "Concept Drift",
            "MAE": drift_mae,
            "RMSE": drift_rmse,
            "R2": drift_r2
        }
    ])

    report.to_csv(OUTPUT, index=False)

    print(f"[SUCCESS] Performance trend report saved to {OUTPUT}")
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()