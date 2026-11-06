
import subprocess
import sys

for script in ["src/automate_lifecycle.py", "src/generate_registry_report.py", "src/quality_gate.py"]:
    print(f"[INFO] ---> {script}")
    result = subprocess.run([sys.executable, script])
    if result.returncode:
        raise SystemExit(result.returncode)
print("[SUCCESS] Lab 8 model lifecycle automation completed.")
