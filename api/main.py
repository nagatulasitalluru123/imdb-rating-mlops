
import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="IMDb Movie Rating Prediction API", version="1.0.0")
MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"
METADATA_PATH = "deployment/model_metadata.json"
DATA_PATH = "data/raw/IMDB-Movie-Data.csv"

model = None
preprocessor = None
metadata = {}

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, metadata
    if not all(os.path.exists(p) for p in [MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH]):
        print("[WARNING] Deployment artifacts missing. Run Lab 11 first.")
        return
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

class MovieInput(BaseModel):
    Genre: str
    Director: str
    Year: float
    Runtime_Minutes: float = Field(alias="Runtime (Minutes)")
    Votes: float
    Revenue_Millions: float | None = Field(default=None, alias="Revenue (Millions)")
    Metascore: float | None = None

    class Config:
        populate_by_name = True

@app.get("/")
def root():
    return {"service": "IMDb Movie Rating Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy" if model is not None and preprocessor is not None else "not_ready",
            "model_loaded": model is not None, "preprocessor_loaded": preprocessor is not None}

@app.get("/metadata")
def get_metadata():
    return metadata or {"message": "Deployment metadata not found. Run Lab 11 first."}

@app.post("/predict")
def predict(movie: MovieInput):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run Lab 11 first.")
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=500, detail="IMDb dataset not found.")
    df = pd.read_csv(DATA_PATH)
    latest_year = pd.to_numeric(df["Year"], errors="coerce").max()
    row = {
        "Genre": movie.Genre,
        "Director": movie.Director,
        "Year": movie.Year,
        "Runtime (Minutes)": movie.Runtime_Minutes,
        "Votes": movie.Votes,
        "Revenue (Millions)": movie.Revenue_Millions,
        "Metascore": movie.Metascore,
        "Movie_Age": latest_year - movie.Year
    }
    features = pd.DataFrame([row], columns=[
        "Genre", "Director", "Year", "Runtime (Minutes)", "Votes",
        "Revenue (Millions)", "Metascore", "Movie_Age"
    ])
    try:
        transformed = preprocessor.transform(features)
        prediction = float(model.predict(transformed)[0])
        return {"predicted_imdb_rating": round(prediction, 2)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
