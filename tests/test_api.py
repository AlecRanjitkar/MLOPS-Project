from __future__ import annotations

import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import io
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fashionmnist_classification_mlops.api as api_mod
from fashionmnist_classification_mlops.api import app


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    # We override the global MODEL_LOADED flag inside the app module for tests.
    # The app uses startup to load model; in tests we skip that and just force it.
    import fashionmnist_classification_mlops.api as api_mod

    api_mod.MODEL_LOADED = True  # pretend model is loaded
    yield TestClient(app)
    api_mod.MODEL_LOADED = False


def _make_dummy_image_bytes(size: int = 28) -> bytes:
    """Create a simple grayscale-ish image and return PNG bytes."""
    arr = (np.random.rand(size, size) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "classes" in data
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) == 10


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data


def test_classes(client: TestClient) -> None:
    r = client.get("/classes")
    assert r.status_code == 200
    data = r.json()
    assert data["num_classes"] == 10
    assert len(data["classes"]) == 10


def test_predict_requires_model_loaded() -> None:
    # Ensure 503 if not loaded
    import fashionmnist_classification_mlops.api as api_mod

    api_mod.MODEL_LOADED = False
    with TestClient(app) as c:
        img_bytes = _make_dummy_image_bytes()
        r = c.post("/predict", files={"file": ("img.png", img_bytes, "image/png")})
        assert r.status_code == 503


def test_predict_ok(client: TestClient) -> None:
    img_bytes = _make_dummy_image_bytes()
    r = client.post("/predict", files={"file": ("img.png", img_bytes, "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "all_probabilities" in data
    assert isinstance(data["all_probabilities"], dict)
    assert len(data["all_probabilities"]) == 10


def test_predict_invalid_file(client: TestClient) -> None:
    r = client.post("/predict", files={"file": ("not_image.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_batch_ok(client: TestClient) -> None:
    img1 = _make_dummy_image_bytes()
    img2 = _make_dummy_image_bytes()
    r = client.post(
        "/predict/batch",
        files=[
            ("files", ("img1.png", img1, "image/png")),
            ("files", ("img2.png", img2, "image/png")),
        ],
    )
    assert r.status_code == 200
    data = r.json()
    assert data["batch_size"] == 2
    assert "results" in data
    assert len(data["results"]) == 2


def test_metrics_endpoint(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_requests" in data
    assert "average_prediction_time_ms" in data
