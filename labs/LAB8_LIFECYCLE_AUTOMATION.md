
# Lab 8 — Automated Model Lifecycle

This lab reuses the existing `src/automate_lifecycle.py` from Lab 6. It checks registered versions, compares their R² values, promotes the best eligible challenger to Production, archives the previous champion when appropriate, regenerates the registry report, and runs the quality gate.

Run:

`python pipelines/run_lab8_lifecycle.py`

Output: refreshed `artifacts/production_model_report.json` and `artifacts/quality_gate_report.json`.
