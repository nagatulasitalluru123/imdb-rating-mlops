import os, json, joblib, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def run_deterministic_test():
    X_train=np.load("data/processed/X_train_final.npy"); X_test=np.load("data/processed/X_test_final.npy"); y_train=np.load("data/processed/y_train.npy"); y_test=np.load("data/processed/y_test.npy")
    params={"n_estimators":200,"random_state":42,"n_jobs":1}
    scores=[]; predictions=[]
    for _ in range(2):
        m=RandomForestRegressor(**params); m.fit(X_train,y_train); p=m.predict(X_test); predictions.append(p); scores.append(float(r2_score(y_test,p)))
    identical=bool(np.array_equal(predictions[0],predictions[1]))
    report={"test_name":"IMDb deterministic reproducibility validation","parameters":params,"execution_1_r2":scores[0],"execution_2_r2":scores[1],"predictions_identical":identical,"status":"PASSED" if identical else "FAILED"}
    os.makedirs("artifacts",exist_ok=True)
    with open("artifacts/reproducibility_report.json","w") as f: json.dump(report,f,indent=4)
    print("[SUCCESS] Reproducibility validation passed." if identical else "[ERROR] Reproducibility validation failed.")
    return identical
if __name__=="__main__":
    if not run_deterministic_test(): raise SystemExit(1)
