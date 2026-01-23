import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = FastAPI()

FEATURE_COLS = ["avg_brightness", "contrast"]

REFERENCE_PATH = Path("reference_features.csv")
CURRENT_PATH = Path("prediction_log.csv")
LOG_URI = os.getenv("LOG_URI")  # gs://bucket/path/prediction_log.csv


def _download_from_gcs(gs_uri: str, dest: Path) -> None:
    parsed = urlparse(gs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"LOG_URI must be gs://..., got {gs_uri}")

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists(client):
        raise FileNotFoundError(f"GCS object not found: {gs_uri}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))


@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/drift-report"]}


@app.get("/drift-report", response_class=HTMLResponse)
def drift_report():
    if not REFERENCE_PATH.exists():
        raise HTTPException(status_code=500, detail="reference_features.csv missing in container")

    # If running in cloud, fetch latest logs from GCS
    if LOG_URI:
        _download_from_gcs(LOG_URI, CURRENT_PATH)

    if not CURRENT_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="prediction_log.csv not found. Set LOG_URI env var or provide local file.",
        )

    reference = pd.read_csv(REFERENCE_PATH)
    current_raw = pd.read_csv(CURRENT_PATH)

    missing = [c for c in FEATURE_COLS if c not in current_raw.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"prediction_log.csv missing columns: {missing}. Found: {list(current_raw.columns)}",
        )

    current = current_raw[FEATURE_COLS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report.get_html()
