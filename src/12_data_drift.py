import json
import os
import pandas as pd

BASE = "data/raw/IMDB-Movie-Data.csv"
EVOLVED = "data/monitoring/drift/evolved_data.csv"
OUT = "outputs/drift/data_drift_report.csv"

NUMERIC = ["Year", "Runtime (Minutes)", "Votes", "Revenue (Millions)", "Metascore", "Rating"]

def main():
    if not os.path.exists(EVOLVED):
        raise FileNotFoundError("Run 12_create_evolved_dataset.py first.")
    base = pd.read_csv(BASE)
    evolved = pd.read_csv(EVOLVED)
    rows = []
    for col in NUMERIC:
        if col not in base.columns or col not in evolved.columns:
            continue
        a = pd.to_numeric(base[col], errors="coerce").dropna()
        b = pd.to_numeric(evolved[col], errors="coerce").dropna()
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        std_a = float(a.std())
        std_b = float(b.std())
        relative_mean_shift = abs(mean_b - mean_a) / (abs(mean_a) + 1e-9)
        drift = relative_mean_shift > 0.05
        rows.append({
            "feature": col,
            "baseline_mean": mean_a,
            "evolved_mean": mean_b,
            "baseline_std": std_a,
            "evolved_std": std_b,
            "relative_mean_shift": relative_mean_shift,
            "drift_detected": drift,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    report = {"drift_detected": bool(any(r["drift_detected"] for r in rows)), "features_checked": len(rows)}
    with open("outputs/drift/data_drift_summary.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"[SUCCESS] Data drift report saved to {OUT}")
    print(f"[INFO] Data drift detected: {report['drift_detected']}")

if __name__ == "__main__":
    main()
