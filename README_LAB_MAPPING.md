# IMDb ↔ Customer Churn Lab Mapping

| Customer Churn pattern | IMDb implementation |
|---|---|
| Raw dataset + validation | `data/raw/IMDB-Movie-Data.csv` + `src/validate_data.py` |
| Preprocessing | `src/preprocess.py`, `src/preprocess_pipeline.py` |
| Baseline model | `src/train.py` + `src/evaluate.py` |
| MLflow tracking | `src/train_mlflow.py` |
| Reproducibility | `src/validate_reproducibility.py` |
| Production pipeline | `pipelines/run_lab5_pipeline.py` |
| Registry training | `src/train_registry.py` |
| Champion/challenger lifecycle | `src/automate_lifecycle.py` |
| Registry report | `src/generate_registry_report.py` |
