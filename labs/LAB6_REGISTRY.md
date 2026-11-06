# Lab 6 — Model Registry and Model Lifecycle Management

The registry workflow mirrors the Customer Churn project:

1. `train_registry.py` trains a Random Forest candidate and registers it as `IMDb_Movie_Rating_Production_Model`.
2. `automate_lifecycle.py` moves new versions to Staging, compares R², and promotes the best challenger to Production while archiving the defeated champion.
3. `generate_registry_report.py` writes a deployment-ready `artifacts/production_model_report.json` containing model lineage, metrics, parameters, and preprocessing dependency.

## Command
`python pipelines/run_lab6_registry.py`
