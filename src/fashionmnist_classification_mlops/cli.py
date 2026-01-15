from __future__ import annotations

from pathlib import Path

import typer

from fashionmnist_classification_mlops.data import preprocess
from fashionmnist_classification_mlops.train import train

app = typer.Typer(help="Fashion-MNIST Classification (MLOps) CLI")


@app.command()
def data(raw_dir: Path, processed_dir: Path) -> None:
    """Download and preprocess Fashion-MNIST into tensors."""
    preprocess(raw_dir=raw_dir, processed_dir=processed_dir)


@app.command()
def train_model(
    lr: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(64, help="Batch size"),
    epochs: int = typer.Option(5, help="Number of epochs"),
    model_type: str = typer.Option("cnn", help="Model type: cnn or mlp"),
) -> None:
    """Train a model on the processed Fashion-MNIST dataset."""
    train(lr=lr, batch_size=batch_size, epochs=epochs, model_type=model_type)


if __name__ == "__main__":
    app()
