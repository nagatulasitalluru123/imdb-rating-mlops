import subprocess,sys
for script in ["src/preprocess.py","src/train.py","src/evaluate.py"]:
    print(f"[INFO] ---> {script}"); r=subprocess.run([sys.executable,script]);
    if r.returncode: raise SystemExit(r.returncode)
print("[SUCCESS] Lab 3 baseline workflow completed.")
