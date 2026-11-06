# Lab 4 — Experiment Tracking and Reproducibility

The MLflow workflow tracks all three notebook baseline models and logs:
- model family
- hyperparameters
- MAE
- RMSE
- R²
- dataset metadata
- serialized model artifacts

The reproducibility script trains a fixed Random Forest twice using identical parameters and checks that predictions are identical.

## Commands
`python pipelines/run_lab4_tracking.py`

To inspect runs:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
