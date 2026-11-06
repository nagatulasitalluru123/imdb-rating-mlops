
# Lab 7 — Model Quality Gate

This lab checks the Production model report created in Lab 6.

Thresholds are selected for this existing IMDb regression model:
- R² must be at least 0.50.
- MAE must not exceed 0.60.
- RMSE must not exceed 0.75.
- The registered model must be in Production.

Run from the `imdb_project` folder:

`python pipelines/run_lab7_quality_gate.py`

Output: `artifacts/quality_gate_report.json`
