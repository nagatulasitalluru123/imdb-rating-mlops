
import json
import os
import sys

REPORT_PATH = "artifacts/production_model_report.json"
MIN_R2 = 0.50
MAX_MAE = 0.60
MAX_RMSE = 0.75

def run_quality_gate():
    if not os.path.exists(REPORT_PATH):
        print(f"[ERROR] Missing {REPORT_PATH}. Run Lab 6 first.")
        return False
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)
    metrics = report.get("performance_metrics", {})
    checks = {
        "R2 >= 0.50": metrics.get("r2", float("-inf")) >= MIN_R2,
        "MAE <= 0.60": metrics.get("mae", float("inf")) <= MAX_MAE,
        "RMSE <= 0.75": metrics.get("rmse", float("inf")) <= MAX_RMSE,
        "Model stage is Production": report.get("model_lineage", {}).get("current_stage") == "Production",
    }
    print("[INFO] IMDb production model quality gate")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    result = {
        "quality_gate": "PASSED" if all(checks.values()) else "FAILED",
        "thresholds": {"minimum_r2": MIN_R2, "maximum_mae": MAX_MAE, "maximum_rmse": MAX_RMSE},
        "checks": checks,
        "metrics": metrics,
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/quality_gate_report.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"[{'SUCCESS' if all(checks.values()) else 'ERROR'}] Quality gate {result['quality_gate']}.")
    return all(checks.values())

if __name__ == "__main__":
    if not run_quality_gate():
        sys.exit(1)
