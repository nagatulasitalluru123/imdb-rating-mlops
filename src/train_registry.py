import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_NAME = "IMDb_Movie_Rating_Production_Model"

def train_and_register():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IMDb_Movie_Rating_Registry")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}

    with mlflow.start_run(run_name="IMDb_Registry_Candidate") as run:
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("task", "regression")
        mlflow.log_param("dataset", "IMDb Movie Data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle"
        )

        print(f"[SUCCESS] Registered {MODEL_NAME} with R2={metrics['r2']:.4f}")
        return run.info.run_id

if __name__ == "__main__":
    train_and_register()
