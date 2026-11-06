import os
import pandas as pd

BASE = "data/monitoring/drift/evolved_data.csv"
OUT = "data/monitoring/drift/evolved_labeled_data.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df = pd.read_csv(BASE).copy()
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Rating"] = (df["Rating"] - 0.45).clip(0, 10)
df.to_csv(OUT, index=False)
print(f"[SUCCESS] Concept-drift labeled dataset saved to {OUT} ({len(df)} rows)")
