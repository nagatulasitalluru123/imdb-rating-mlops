# Lab 2 — Baseline Model Development and Evaluation

The baseline comparison follows the supplied IMDb notebook:
- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression

Metrics:
- MAE — lower is better
- RMSE — lower is better
- R² — higher is better

The best model is selected using the highest R² and saved as `models/best_model.pkl`.

## Command
`python src/preprocess.py && python src/train.py && python src/evaluate.py`
