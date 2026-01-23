import io
import time

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

app = FastAPI(title="Fashion-MNIST ONNX API")

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


stats = torch.load("data/processed/stats.pt")
MEAN = stats["mean"]
STD = stats["std"]

# Then update the transform:
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ]
)
session = ort.InferenceSession("models/model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    x = transform(img).unsqueeze(0).numpy()

    logits = session.run(None, {input_name: x})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    idx = int(probs.argmax())
    conf = float(probs[0, idx])

    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": conf,
        "latency_ms": round((time.time() - start) * 1000, 2),
    }
