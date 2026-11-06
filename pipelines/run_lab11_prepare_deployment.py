
import subprocess
import sys

result = subprocess.run([sys.executable, "src/prepare_deployment.py"])
if result.returncode:
    raise SystemExit(result.returncode)
print("[SUCCESS] Lab 11 deployment preparation completed.")
