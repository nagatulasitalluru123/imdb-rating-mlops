FROM python:3.11-slim

WORKDIR /app

COPY api/requirements-api.txt /app/api/requirements-api.txt

RUN pip install --no-cache-dir -r /app/api/requirements-api.txt \
    pandas \
    numpy \
    scipy \
    scikit-learn==1.5.2 \
    joblib

COPY api /app/api
COPY deployment /app/deployment
COPY data/raw /app/data/raw

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]