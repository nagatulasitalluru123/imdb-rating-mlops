import json
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS = {
    "Linear_Regression": (LinearRegression(), {}),
    "Random_Forest": (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), {"n_estimators": 200, "random_state": 42}),
    "Gradient_Boosting": (GradientBoostingRegressor(n_estimators=200, random_state=42), {"n_estimators": 200, "random_state": 42})
}

def train_and_track():
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IMDb_Movie_Rating_Prediction")

    for name, (model, params) in MODELS.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_family", name)
            mlflow.log_params(params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            metrics = {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
            }

            mlflow.log_metrics(metrics)
            os.makedirs("artifacts", exist_ok=True)
            with open(f"artifacts/{name}_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            mlflow.log_artifact(f"artifacts/{name}_metrics.json", artifact_path="metrics")

            if os.path.exists("data/processed/dataset_metadata.json"):
                mlflow.log_artifact("data/processed/dataset_metadata.json", artifact_path="metadata")

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            print(f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

if __name__ == "__main__":
    train_and_track()
