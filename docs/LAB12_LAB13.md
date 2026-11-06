# IMDb Movie Rating Prediction — Labs 12 and 13

## Lab 12 — Production Monitoring

The project is a regression task predicting `Rating`. Monitoring uses MAE, RMSE and R2.

Scripts:
- `12_create_evolved_dataset.py`
- `12_create_concept_drift_dataset.py`
- `12_data_drift.py`
- `12_concept_drift.py`
- `12_prediction_drift.py`
- `12_performance_monitoring.py`
- `12_performance_trend.py`
- `12_prediction_monitoring.py`
- `12_model_comparison.py`
- `12_request_history.py`
- `12_monitor_trigger.py`
- `12_performance_monitoring_api.py`
- `12_retrain_model.py`

## Lab 13 — End-to-End MLOps

The workflow `.github/workflows/lab13_monitoring.yml` runs preprocessing, MLflow registration, lifecycle automation, deployment preparation, monitoring, Docker build and Kubernetes verification on a self-hosted Windows runner.

## Lab 11 Docker

1. Run `python pipelines/run_lab11_prepare_deployment.py`.
2. Build: `docker build -t imdb-rating-api:latest .`
3. Run: `docker run -d -p 8000:8000 --name imdb-api imdb-rating-api:latest`
4. Open `http://localhost:8000/docs`.
5. Test `/health`, `/metadata`, and `/predict`.
