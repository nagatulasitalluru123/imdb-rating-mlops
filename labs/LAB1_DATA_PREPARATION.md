# Lab 1 — ML Project Initialization and Dataset Preparation

## Objective
Convert the IMDb notebook's data-preparation workflow into a reproducible project structure.

## Workflow
1. Place `IMDB-Movie-Data.csv` in `data/raw/`.
2. Validate required columns.
3. Inspect missing values and duplicate records.
4. Remove duplicate rows.
5. Create `Movie_Age = max(Year) - Year`.
6. Select the notebook's eight predictive features.
7. Split data 80/20 with `random_state=42`.
8. Impute numerical features with median and categorical features with most frequent values.
9. Standard-scale numerical features.
10. One-hot encode `Genre` and `Director`.
11. Save processed matrices, preprocessor, and metadata.

## Command
`python src/preprocess.py`
