import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_evaluation():
    X_test=np.load("data/processed/X_test_final.npy"); y_test=np.load("data/processed/y_test.npy")
    model=joblib.load("models/best_model.pkl")
    pred=model.predict(X_test)
    metrics={"MAE":float(mean_absolute_error(y_test,pred)),"RMSE":float(np.sqrt(mean_squared_error(y_test,pred))),"R2 Score":float(r2_score(y_test,pred))}
    os.makedirs("artifacts",exist_ok=True)
    with open("artifacts/evaluation_report.json","w") as f: json.dump(metrics,f,indent=4)
    pd.DataFrame({"Actual_Rating":y_test,"Predicted_Rating":pred}).to_csv("outputs/predictions.csv",index=False)
    print("--- IMDb Evaluation ---")
    for k,v in metrics.items(): print(f"{k}: {v:.4f}")

if __name__ == "__main__": run_evaluation()
