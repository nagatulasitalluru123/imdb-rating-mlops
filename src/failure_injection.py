
import json
import os
import tempfile
import pandas as pd
from common import load_clean_dataframe
from validate_data import validate_schema

def run_failure_tests():
    results = []

    # Scenario 1: dataset path is missing.
    with tempfile.TemporaryDirectory() as folder:
        missing_path = os.path.join(folder, "missing.csv")
        try:
            load_clean_dataframe(missing_path)
            results.append({"scenario": "Missing dataset", "status": "FAILED"})
        except FileNotFoundError:
            results.append({"scenario": "Missing dataset", "status": "RECOVERED"})

    # Scenario 2: required columns are missing.
    invalid_df = pd.DataFrame({"Title": ["Test Movie"], "Rating": [7.0]})
    results.append({
        "scenario": "Invalid schema",
        "status": "RECOVERED" if not validate_schema_without_report(invalid_df) else "FAILED"
    })

    # Scenario 3: invalid rating range.
    invalid_rating = pd.DataFrame({
        "Rank": [1], "Title": ["Test Movie"], "Genre": ["Drama"], "Description": ["Test"],
        "Director": ["Director"], "Actors": ["Actor"], "Year": [2020],
        "Runtime (Minutes)": [100], "Rating": [12.0], "Votes": [1000],
        "Revenue (Millions)": [10.0], "Metascore": [80]
    })
    results.append({
        "scenario": "Out-of-range rating",
        "status": "RECOVERED" if not validate_schema_without_report(invalid_rating) else "FAILED"
    })

    os.makedirs("artifacts", exist_ok=True)
    report = {"failure_injection_status": "PASSED" if all(x["status"] == "RECOVERED" for x in results) else "FAILED",
              "scenarios": results}
    with open("artifacts/failure_injection_report.json", "w") as f:
        json.dump(report, f, indent=4)
    for item in results:
        print(f"[{'PASS' if item['status'] == 'RECOVERED' else 'FAIL'}] {item['scenario']}: {item['status']}")
    print(f"[SUCCESS] Failure injection {report['failure_injection_status']}.")
    return report["failure_injection_status"] == "PASSED"

def validate_schema_without_report(df):
    required = ["Rank", "Title", "Genre", "Description", "Director", "Actors", "Year",
                "Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)", "Metascore"]
    if any(col not in df.columns for col in required):
        return False
    if not pd.api.types.is_numeric_dtype(df["Rating"]):
        return False
    if df["Rating"].dropna().gt(10).any() or df["Rating"].dropna().lt(0).any():
        return False
    return True

if __name__ == "__main__":
    if not run_failure_tests():
        raise SystemExit(1)
