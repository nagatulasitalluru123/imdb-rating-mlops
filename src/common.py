import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "data/raw/IMDB-Movie-Data.csv"
TARGET = "Rating"
FEATURES = ["Genre", "Director", "Year", "Runtime (Minutes)", "Votes", "Revenue (Millions)", "Metascore", "Movie_Age"]
NUMERIC = ["Year", "Runtime (Minutes)", "Votes", "Revenue (Millions)", "Metascore", "Movie_Age"]
CATEGORICAL = ["Genre", "Director"]
RANDOM_STATE = 42


def load_clean_dataframe(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"IMDb dataset not found at {path}. Copy IMDB-Movie-Data.csv into data/raw/.")
    df = pd.read_csv(path)
    required = ["Genre", "Director", "Year", "Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)", "Metascore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.drop_duplicates().copy()
    for col in ["Year", "Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)", "Metascore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    latest_year = df["Year"].max()
    df["Movie_Age"] = latest_year - df["Year"]
    return df


def split_data(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, default=str)
