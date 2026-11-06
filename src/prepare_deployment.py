
import json
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "IMDb_Movie_Rating_Production_Model"

def prepare_deployment():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise RuntimeError("No Production model found. Complete Lab 6/8 first.")
    version = max(production, key=lambda v: int(v.version))
    model_uri = f"models:/{MODEL_NAME}/{version.version}"
    print(f"[INFO] Loading {MODEL_NAME}, version {version.version}")
    model = mlflow.sklearn.load_model(model_uri)
    preprocessor_path = "models/preprocessor.pkl"
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Missing preprocessing artifact: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    os.makedirs("deployment", exist_ok=True)
    joblib.dump(model, "deployment/model.pkl")
    joblib.dump(preprocessor, "deployment/preprocessor.pkl")
    run = client.get_run(version.run_id)
    metadata = {
        "service": "IMDb Movie Rating Prediction API",
        "version": "1.0.0",
        "model_name": MODEL_NAME,
        "model_version": int(version.version),
        "model_stage": version.current_stage,
        "run_id": version.run_id,
        "model_uri": model_uri,
        "model_loaded": True,
        "preprocessor_loaded": True,
        "target": "Rating",
        "task": "regression",
        "metrics": run.data.metrics
    }
    with open("deployment/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("[SUCCESS] Deployment bundle prepared.")
    print("[INFO] Files: deployment/model.pkl, deployment/preprocessor.pkl, deployment/model_metadata.json")

if __name__ == "__main__":
    prepare_deployment()
