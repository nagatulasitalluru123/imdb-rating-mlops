import os, json, joblib
import numpy as np
from common import load_clean_dataframe, split_data, build_preprocessor, NUMERIC, CATEGORICAL, RANDOM_STATE

def run_preprocessing():
    print("[INFO] Starting IMDb preprocessing...")
    df = load_clean_dataframe()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)
    os.makedirs("data/processed", exist_ok=True); os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=float))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=float))
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    metadata = {
        "dataset_name": "IMDb Movie Data",
        "target": "Rating",
        "source_path": "data/raw/IMDB-Movie-Data.csv",
        "raw_rows_after_duplicate_removal": int(len(df)),
        "train_shape": list(X_train_final.shape), "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERIC, "categorical_features": CATEGORICAL,
        "test_size": 0.20, "random_state": RANDOM_STATE,
        "feature_engineering": "Movie_Age = max(Year) - Year",
        "missing_value_strategy": "median for numeric, most_frequent for categorical",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "scaling": "StandardScaler"
    }
    with open("data/processed/dataset_metadata.json", "w") as f: json.dump(metadata, f, indent=4)
    print("[SUCCESS] Preprocessing completed.")

if __name__ == "__main__": run_preprocessing()
