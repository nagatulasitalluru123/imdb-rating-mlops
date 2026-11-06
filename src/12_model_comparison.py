import json
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = "outputs/monitoring/model_comparison.json"

def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"model": name, "mae": float(mean_absolute_error(y_test, pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, pred))), "r2": float(r2_score(y_test, pred))}

def main():
    X_train=np.load("data/processed/X_train_final.npy"); X_test=np.load("data/processed/X_test_final.npy"); y_train=np.load("data/processed/y_train.npy"); y_test=np.load("data/processed/y_test.npy")
    rows=[evaluate("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), X_train,y_train,X_test,y_test), evaluate("GradientBoosting", GradientBoostingRegressor(n_estimators=200, random_state=42), X_train,y_train,X_test,y_test)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT,"w") as f: json.dump(rows,f,indent=4)
    print("[SUCCESS] Model comparison completed")
    for r in rows: print(f"[INFO] {r['model']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} R2={r['r2']:.4f}")

if __name__ == "__main__": main()
