
# IMDb Movie Rating Prediction — Additional MLOps Labs 7–11

This is an additive extension to the existing IMDb project. The original Labs 1–6 files, notebook, dataset, models, and artifacts are retained unchanged.

Run all commands from the `imdb_project` folder, after completing Labs 1–6.

| Lab | Topic | Command |
|---|---|---|
| 7 | Model quality gate | `python pipelines/run_lab7_quality_gate.py` |
| 8 | Lifecycle automation | `python pipelines/run_lab8_lifecycle.py` |
| 9 | Production inference | `python pipelines/run_lab9_inference.py` |
| 10 | Failure injection and recovery | `python pipelines/run_lab10_failure_injection.py` |
| 11 | Deployment preparation | `python pipelines/run_lab11_prepare_deployment.py` |

For API testing, follow `labs/LAB11_DEPLOYMENT.md`.

The IMDb task remains regression throughout: the target is `Rating`, and evaluation uses MAE, RMSE, and R². The extension does not convert the project into classification.
