import subprocess,sys
for script in ["src/preprocess.py","src/train_mlflow.py","src/validate_reproducibility.py"]:
    print(f"[INFO] ---> {script}"); r=subprocess.run([sys.executable,script]);
    if r.returncode: raise SystemExit(r.returncode)
print("[SUCCESS] Lab 4 MLflow workflow completed.")
