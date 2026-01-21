from __future__ import annotations

import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from fashionmnist_classification_mlops.model import FashionCNN

app = FastAPI(title="Fashion-MNIST Classifier API")

# Load model at startup
MODEL_PATH = Path("models/model.pth")
DEVICE = torch.device("cpu")  # Use CPU in Docker
model = FashionCNN().to(DEVICE)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Preprocessing
transform = transforms.Compose(
    [transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Fashion-MNIST Classifier API", "model_loaded": MODEL_PATH.exists(), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.

    Args:
        file: Image file (28x28 grayscale preferred, but will be converted)

    Returns:
        Predicted class name and confidence scores
    """
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(confidence),
        "all_probabilities": {CLASS_NAMES[i]: float(probs[0, i]) for i in range(len(CLASS_NAMES))},
    }
