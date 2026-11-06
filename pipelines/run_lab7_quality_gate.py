
import subprocess
import sys

result = subprocess.run([sys.executable, "src/quality_gate.py"])
if result.returncode:
    raise SystemExit(result.returncode)
print("[SUCCESS] Lab 7 quality gate completed.")
