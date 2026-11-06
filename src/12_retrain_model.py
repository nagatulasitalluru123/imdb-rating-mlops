import subprocess
import sys

print("[INFO] Retraining triggered by monitoring...")
result = subprocess.run([sys.executable, "src/train_registry.py"])
raise SystemExit(result.returncode)
