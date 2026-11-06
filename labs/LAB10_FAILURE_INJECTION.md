
# Lab 10 — Failure Injection and Recovery

This lab safely tests three expected failure conditions without changing the original dataset or model files:
1. Missing dataset path.
2. Missing required schema columns.
3. IMDb Rating outside the allowed 0–10 range.

Each injected failure is detected and recorded as recovered.

Run:

`python pipelines/run_lab10_failure_injection.py`

Output: `artifacts/failure_injection_report.json`.
