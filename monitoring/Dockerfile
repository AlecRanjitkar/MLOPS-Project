FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir \
  "numpy<2.0" \
  fastapi uvicorn pandas \
  "evidently==0.4.22" \
  google-cloud-storage

COPY monitoring_api.py .
COPY reference_features.csv .

EXPOSE 8080
CMD ["uvicorn", "monitoring_api:app", "--host", "0.0.0.0", "--port", "8080"]
