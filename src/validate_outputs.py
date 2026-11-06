import os, json, numpy as np

def validate_preprocessing_outputs():
    names=["X_train_final.npy","X_test_final.npy","y_train.npy","y_test.npy"]
    try: X_train=np.load("data/processed/X_train_final.npy"); X_test=np.load("data/processed/X_test_final.npy"); y_train=np.load("data/processed/y_train.npy"); y_test=np.load("data/processed/y_test.npy")
    except FileNotFoundError as e: print(e); return False
    errors=[]
    if not np.isfinite(X_train).all(): errors.append("Non-finite values in X_train")
    if not np.isfinite(X_test).all(): errors.append("Non-finite values in X_test")
    if X_train.shape[1]!=X_test.shape[1]: errors.append("Train/test feature dimension mismatch")
    if X_train.shape[0]!=y_train.shape[0] or X_test.shape[0]!=y_test.shape[0]: errors.append("Feature/target row mismatch")
    report={"validation_status":"PASSED" if not errors else "FAILED","matrix_dimensions":{"X_train":list(X_train.shape),"X_test":list(X_test.shape),"y_train":list(y_train.shape),"y_test":list(y_test.shape)},"errors":errors}
    os.makedirs("artifacts",exist_ok=True)
    with open("artifacts/preprocessing_summary_report.json","w") as f: json.dump(report,f,indent=4)
    print("[SUCCESS] Output validation passed." if not errors else "[ERROR] Output validation failed.")
    return not errors
if __name__=="__main__":
    if not validate_preprocessing_outputs(): raise SystemExit(1)
