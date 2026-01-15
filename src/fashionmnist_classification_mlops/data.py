from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import typer
from loguru import logger
from torchvision import datasets, transforms

app = typer.Typer(help="Download and preprocess Fashion-MNIST.")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_mean_std(images: torch.Tensor) -> Tuple[float, float]:
    """
    Compute mean/std for normalization.
    images: (N, 1, 28, 28) float tensor in [0,1]
    """
    mean = images.mean().item()
    std = images.std().item()
    return mean, std


@app.command()
def preprocess(
    raw_dir: Path = typer.Argument(..., help="Where raw data is downloaded/stored."),
    processed_dir: Path = typer.Argument(..., help="Where processed tensors are saved."),
) -> None:
    """
    Downloads Fashion-MNIST (if not present) and saves processed tensors:
      - train_images.pt, train_labels.pt
      - test_images.pt, test_labels.pt
      - stats.pt (mean/std used for normalization)
    """
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    logger.info(f"Raw dir: {raw_dir}")
    logger.info(f"Processed dir: {processed_dir}")

    # 1) Download via Torchvision (raw files end up under raw_dir/FashionMNIST/)
    # We load without normalization first, so we can compute dataset statistics.
    base_transform = transforms.ToTensor()

    train_ds = datasets.FashionMNIST(
        root=str(raw_dir),
        train=True,
        download=True,
        transform=base_transform,
    )
    test_ds = datasets.FashionMNIST(
        root=str(raw_dir),
        train=False,
        download=True,
        transform=base_transform,
    )

    # 2) Convert datasets to tensors (few files => good for DVC)
    # NOTE: This is small enough (~60MB) to keep in memory.
    train_images = torch.stack([train_ds[i][0] for i in range(len(train_ds))])  # (60000,1,28,28)
    train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)

    test_images = torch.stack([test_ds[i][0] for i in range(len(test_ds))])     # (10000,1,28,28)
    test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)

    logger.info(f"Loaded train: images={train_images.shape}, labels={train_labels.shape}")
    logger.info(f"Loaded test:  images={test_images.shape}, labels={test_labels.shape}")

    # 3) Normalize using train statistics (mean 0, std 1)
    mean, std = _compute_mean_std(train_images)
    logger.info(f"Train mean={mean:.6f}, std={std:.6f}")

    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    # 4) Save processed tensors
    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_labels, processed_dir / "train_labels.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_labels, processed_dir / "test_labels.pt")
    torch.save({"mean": mean, "std": std}, processed_dir / "stats.pt")

    logger.success("Saved processed Fashion-MNIST tensors + stats.")


def load_processed(processed_dir: Path = Path("data/processed")):
    """
    Convenience function for training/evaluation scripts.
    Returns TensorDatasets for train and test.
    """
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_labels = torch.load(processed_dir / "test_labels.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_labels)
    test_set = torch.utils.data.TensorDataset(test_images, test_labels)
    return train_set, test_set


if __name__ == "__main__":
    app()
