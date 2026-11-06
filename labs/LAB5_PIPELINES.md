# Lab 5 — Production ML Pipelines

The production pipeline mirrors the Customer Churn project's validation/preprocessing/output-validation pattern:

1. `src/validate_data.py` — validates the IMDb input schema and value constraints.
2. `src/preprocess_pipeline.py` — performs split, feature engineering, imputation, scaling, encoding, and artifact generation.
3. `src/validate_outputs.py` — validates dimensions and finite transformed values and writes `artifacts/preprocessing_summary_report.json`.

## Command
`python pipelines/run_lab5_pipeline.py`
