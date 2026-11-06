import os
import pandas as pd

BASE = "data/raw/IMDB-Movie-Data.csv"
OUT = "data/monitoring/drift/evolved_data.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df = pd.read_csv(BASE).copy()
for col, factor in {
    "Votes": 1.25,
    "Revenue (Millions)": 1.15,
    "Metascore": 0.92,
    "Runtime (Minutes)": 1.05,
}.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") * factor

df.to_csv(OUT, index=False)
print(f"[SUCCESS] Evolved dataset saved to {OUT} ({len(df)} rows)")
