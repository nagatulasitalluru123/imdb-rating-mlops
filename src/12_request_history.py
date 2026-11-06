import os
import json
import pandas as pd
import joblib

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

DATA_FILE = "data/monitoring/drift/evolved_data.csv"
MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"

OUTPUT_DIR = "outputs/monitoring"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "request_history.json")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    df = df.copy()
    df["Movie_Age"] = 2016 - df["Year"]

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X = preprocessor.transform(df[FEATURES])
    predictions = model.predict(X)

    history = []

    for i, prediction in enumerate(predictions[:10]):
        history.append({
            "request_id": i + 1,
            "status": "success",
            "predicted_imdb_rating": round(float(prediction), 2)
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(history, f, indent=4)

    print("[SUCCESS] Request history saved to", OUTPUT_FILE)
    print("[INFO] Total requests recorded:", len(history))


if __name__ == "__main__":
    main()