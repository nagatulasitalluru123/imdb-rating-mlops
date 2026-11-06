# IMDb Movie Rating Prediction — MLOps Labs 1–6

This project reorganizes the supplied IMDb Movie Rating Prediction notebook into the same lab-oriented MLOps structure used by the Customer Churn project.

## Dataset
Copy `IMDB-Movie-Data.csv` to `data/raw/IMDB-Movie-Data.csv`. The supplied notebook describes a 1,000-row, 12-column IMDb movie dataset and uses `Rating` as the regression target.

## Labs

### Lab 1 — ML Project Initialization and Dataset Preparation
- repository structure
- data inspection and validation
- duplicate handling
- missing-value handling through preprocessing
- `Movie_Age` feature engineering
- reproducible 80/20 split with `random_state=42`
- numerical scaling and categorical one-hot encoding
- metadata generation

Run: `python src/preprocess.py`

### Lab 2 — Baseline Model Development and Evaluation
The notebook's regression approach is represented with Linear Regression, Random Forest Regression, and Gradient Boosting Regression. Metrics are MAE, RMSE, and R². Models and comparison artifacts are serialized.

Run: `python src/preprocess.py && python src/train.py && python src/evaluate.py`

### Lab 3 — Git-Based Version Control and Collaborative ML Workflows
Use the commands documented in `labs/LAB3_GIT_WORKFLOW.md` for meaningful commits, feature branches, merge/conflict resolution, rollback/revert, and tags.

### Lab 4 — Experiment Tracking and Reproducibility
MLflow logs parameters, MAE/RMSE/R², metadata, and model artifacts. A deterministic repeated-run test is also included.

Run: `python pipelines/run_lab4_tracking.py`

### Lab 5 — Production ML Pipelines
The production pipeline is: schema/data validation → preprocessing pipeline → processed-output validation.

Run: `python pipelines/run_lab5_pipeline.py`

### Lab 6 — Model Registry and Model Lifecycle Management
A Random Forest registry candidate is logged and registered, staging/production lifecycle is automated using R² as the comparison metric, and a deployment-ready registry report is generated.

Run: `python pipelines/run_lab6_registry.py`

## Project structure

```text
imdb-movie-rating/
├── data/raw/
├── data/processed/
├── models/
├── artifacts/
├── outputs/
├── logs/
├── src/
│   ├── common.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── validate_data.py
│   ├── preprocess_pipeline.py
│   ├── validate_outputs.py
│   ├── train_mlflow.py
│   ├── validate_reproducibility.py
│   ├── train_registry.py
│   ├── automate_lifecycle.py
│   └── generate_registry_report.py
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
├── notebooks/
├── labs/
├── requirements.txt
└── README.md
```

## Important
The uploaded notebook does not contain the CSV bytes; therefore the project does not fabricate or bundle a replacement dataset. Add the original CSV before running the executable labs.
