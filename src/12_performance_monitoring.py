import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"
DATA_PATH = "data/monitoring/drift/evolved_labeled_data.csv"
OUTPUT = "outputs/monitoring/performance_monitoring_report.csv"

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

def main():
    os.makedirs("outputs/monitoring", exist_ok=True)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df = pd.read_csv(DATA_PATH)
    df = prepare_features(df)

    X = df[FEATURES].copy()
    y = df["Rating"]

    predictions = model.predict(preprocessor.transform(X))

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5
    r2 = r2_score(y, predictions)

    report = pd.DataFrame([{
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }])

    report.to_csv(OUTPUT, index=False)

    print(f"[SUCCESS] Performance monitoring report saved to {OUTPUT}")
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()