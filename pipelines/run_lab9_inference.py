
import subprocess
import sys

result = subprocess.run([sys.executable, "src/inference.py"])
if result.returncode:
    raise SystemExit(result.returncode)
print("[SUCCESS] Lab 9 inference completed.")
