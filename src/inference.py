
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from common import DATA_PATH

MODEL_NAME = "IMDb_Movie_Rating_Production_Model"

def load_production_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

def predict_movie_rating(movie):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    latest_year = pd.to_numeric(data["Year"], errors="coerce").max()
    row = dict(movie)
    row["Movie_Age"] = latest_year - float(row["Year"])
    features = pd.DataFrame([row], columns=[
        "Genre", "Director", "Year", "Runtime (Minutes)", "Votes",
        "Revenue (Millions)", "Metascore", "Movie_Age"
    ])
    model, preprocessor = load_production_model()
    transformed = preprocessor.transform(features)
    prediction = float(model.predict(transformed)[0])
    return round(prediction, 2)

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    sample = data.iloc[0]
    movie = {
        "Genre": sample["Genre"], "Director": sample["Director"], "Year": sample["Year"],
        "Runtime (Minutes)": sample["Runtime (Minutes)"], "Votes": sample["Votes"],
        "Revenue (Millions)": sample["Revenue (Millions)"], "Metascore": sample["Metascore"]
    }
    print("[INFO] Sample movie:", sample["Title"])
    print("[SUCCESS] Predicted IMDb rating:", predict_movie_rating(movie))
