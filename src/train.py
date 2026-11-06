import os, joblib, json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import build_preprocessor

MODELS = {
    "Linear_Regression": LinearRegression(),
    "Random_Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient_Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

def run_training():
    X_train=np.load("data/processed/X_train_final.npy"); X_test=np.load("data/processed/X_test_final.npy")
    y_train=np.load("data/processed/y_train.npy"); y_test=np.load("data/processed/y_test.npy")
    # The processed matrices are already transformed; models are trained directly on them.
    results=[]
    os.makedirs("models", exist_ok=True)
    for name, estimator in MODELS.items():
        print(f"[INFO] Training {name}...")
        estimator.fit(X_train, y_train)
        pred=estimator.predict(X_test)
        mae=mean_absolute_error(y_test,pred); rmse=float(np.sqrt(mean_squared_error(y_test,pred))); r2=r2_score(y_test,pred)
        joblib.dump(estimator, f"models/{name.lower()}.pkl")
        results.append({"Model":name,"MAE":mae,"RMSE":rmse,"R2 Score":r2})
    results=sorted(results,key=lambda x:x["R2 Score"],reverse=True)
    with open("artifacts/model_comparison.json","w") as f: json.dump(results,f,indent=4)
    best=results[0]
    joblib.dump(MODELS[best["Model"]], "models/best_model.pkl")
    with open("artifacts/best_model_metadata.json","w") as f: json.dump(best,f,indent=4)
    print(f"[SUCCESS] Best model: {best['Model']} | R2={best['R2 Score']:.4f}")

if __name__ == "__main__": run_training()
