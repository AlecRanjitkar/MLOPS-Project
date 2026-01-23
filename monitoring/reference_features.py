import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)


def extract_features(x):
    img = x.squeeze().numpy()
    return {
        "avg_brightness": float(np.mean(img)),
        "contrast": float(np.std(img)),
    }


rows = []
for img, _ in dataset:
    rows.append(extract_features(img))

pd.DataFrame(rows).to_csv("reference_features.csv", index=False)
print("✅ reference_features.csv created")
