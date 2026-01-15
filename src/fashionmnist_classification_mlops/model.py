from __future__ import annotations

import torch
from torch import nn


class FashionCNN(nn.Module):
    """Small CNN for Fashion-MNIST (28x28 grayscale, 10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class FashionMLP(nn.Module):
    """Simple baseline MLP."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = FashionCNN()
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")
    dummy = torch.randn(1, 1, 28, 28)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
