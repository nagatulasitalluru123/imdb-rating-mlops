
# Lab 11 — Deployment Preparation and API Testing

## 1. Prepare the deployment bundle

Run:

`python pipelines/run_lab11_prepare_deployment.py`

It exports the Production model, preprocessing object, and metadata into:
- `deployment/model.pkl`
- `deployment/preprocessor.pkl`
- `deployment/model_metadata.json`

## 2. Install API packages

`pip install -r api/requirements-api.txt`

## 3. Start the API

`python -m uvicorn api.main:app --reload`

Open `http://127.0.0.1:8000/docs` and test:
- `GET /health`
- `GET /metadata`
- `POST /predict`

Example request body:

```json
{
  "Genre": "Action, Drama",
  "Director": "Example Director",
  "Year": 2015,
  "Runtime (Minutes)": 120,
  "Votes": 150000,
  "Revenue (Millions)": 100,
  "Metascore": 70
}
```

The API returns `predicted_imdb_rating`, a numeric regression prediction.
