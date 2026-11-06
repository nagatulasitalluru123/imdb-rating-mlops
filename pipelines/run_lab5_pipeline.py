import subprocess,sys
for script in ["src/validate_data.py","src/preprocess_pipeline.py","src/validate_outputs.py"]:
    print(f"[INFO] ---> {script}"); r=subprocess.run([sys.executable,script]);
    if r.returncode: raise SystemExit(r.returncode)
print("[SUCCESS] Lab 5 production pipeline completed.")
