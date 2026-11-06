
import subprocess
import sys

result = subprocess.run([sys.executable, "src/failure_injection.py"])
if result.returncode:
    raise SystemExit(result.returncode)
print("[SUCCESS] Lab 10 failure injection completed.")
