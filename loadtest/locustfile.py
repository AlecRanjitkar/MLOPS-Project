from __future__ import annotations

import io
import os

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def make_image_bytes(size: int = 28) -> bytes:
    arr = (np.random.rand(size, size) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class APIUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.img = make_image_bytes()

    @task(5)
    def health(self):
        self.client.get("/health")

    @task(10)
    def predict(self):
        files = {"file": ("img.png", self.img, "image/png")}
        self.client.post("/predict", files=files)

    @task(2)
    def classes(self):
        self.client.get("/classes")
