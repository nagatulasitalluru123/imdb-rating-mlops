import subprocess,sys
for script in ["src/train_registry.py","src/automate_lifecycle.py","src/generate_registry_report.py"]:
    print(f"[INFO] ---> {script}"); r=subprocess.run([sys.executable,script]);
    if r.returncode: raise SystemExit(r.returncode)
print("[SUCCESS] Lab 6 model registry workflow completed.")
