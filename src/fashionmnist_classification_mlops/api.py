from __future__ import annotations

import io
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from fashionmnist_classification_mlops.model import FashionCNN

# Metrics tracking
request_count = 0
prediction_times = []

app = FastAPI(title="Fashion-MNIST Classifier API")

MODEL_PATH = Path("models/model.pth")
DEVICE = torch.device("cpu")
model = FashionCNN().to(DEVICE)

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

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


@app.on_event("startup")
async def load_model() -> None:
    try:
        if not MODEL_PATH.exists():
            print(f"WARNING: Model not found at {MODEL_PATH}")
            print(f"Current directory: {Path.cwd()}")

            if Path("models").exists():
                model_files = list(Path("models").glob("*"))
            else:
                model_files = "models/ does not exist"

            print(f"Files in models/: {model_files}")
            return

        print(f"Loading model from {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback

        traceback.print_exc()


@app.get("/")
async def root():
    return {
        "message": "Fashion-MNIST Classifier API",
        "model_loaded": MODEL_PATH.exists(),
        "classes": CLASS_NAMES,
    }


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy" if MODEL_PATH.exists() else "degraded",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.get("/classes")
async def get_classes():
    """Get all available class names."""
    return {"classes": CLASS_NAMES, "num_classes": len(CLASS_NAMES)}


@app.get("/metrics")
async def get_metrics():
    """Get API metrics for monitoring."""
    avg_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0
    return {
        "total_requests": request_count,
        "total_predictions": len(prediction_times),
        "average_prediction_time_ms": round(avg_time * 1000, 2),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of a single uploaded image."""
    global request_count
    request_count += 1

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")

    start_time = time.time()

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    elapsed = time.time() - start_time
    prediction_times.append(elapsed)

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "all_probabilities": {CLASS_NAMES[i]: float(probs[0, i]) for i in range(len(CLASS_NAMES))},
    }


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict classes for multiple uploaded images."""
    global request_count
    request_count += 1

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

    start_time = time.time()
    results = []

    for file in files:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            x = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(probs.argmax(dim=1).item())
                confidence = float(probs[0, pred_idx].item())

            results.append(
                {
                    "filename": file.filename,
                    "predicted_class": CLASS_NAMES[pred_idx],
                    "confidence": confidence,
                }
            )
        except Exception as e:
            results.append(
                {
                    "filename": file.filename,
                    "error": str(e),
                }
            )

    elapsed = time.time() - start_time
    prediction_times.append(elapsed)

    return {
        "batch_size": len(files),
        "results": results,
        "total_time_seconds": round(elapsed, 3),
    }
