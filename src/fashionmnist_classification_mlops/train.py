from __future__ import annotations

import cProfile
import os
import pstats
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb
from fashionmnist_classification_mlops.logging_utils import setup_logger
from fashionmnist_classification_mlops.model import FashionCNN, FashionMLP

# -----------------------------
# Constants / utilities
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def ensure_channel_dim(x: torch.Tensor) -> torch.Tensor:
    """Ensure images are shaped (N, 1, 28, 28) for CNN."""
    if x.ndim == 3:  # (N, 28, 28)
        return x.unsqueeze(1)
    return x


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def load_processed(processed_dir: Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load processed .pt tensors and return torch datasets."""
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_labels = torch.load(processed_dir / "test_labels.pt")

    # Ensure consistent shape for CNN
    train_images = ensure_channel_dim(train_images)
    test_images = ensure_channel_dim(test_images)

    train_set = torch.utils.data.TensorDataset(train_images, train_labels)
    test_set = torch.utils.data.TensorDataset(test_images, test_labels)
    return train_set, test_set


def make_model(model_type: str) -> torch.nn.Module:
    if model_type.lower() == "mlp":
        return FashionMLP()
    return FashionCNN()


def save_training_plots(stats: dict[str, list[float]], out_dir: Path) -> dict[str, Path]:
    """Save curves and return paths for potential W&B logging."""
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Train loss (per step)
    plt.figure(figsize=(10, 4))
    plt.plot(stats["train_loss"])
    plt.title("Train loss (per step)")
    p = out_dir / "train_loss.png"
    plt.savefig(p)
    plt.close()
    paths["train_loss"] = p

    # Train acc (per step)
    plt.figure(figsize=(10, 4))
    plt.plot(stats["train_acc"])
    plt.title("Train accuracy (per step)")
    p = out_dir / "train_accuracy.png"
    plt.savefig(p)
    plt.close()
    paths["train_accuracy"] = p

    # Test acc (per epoch)
    plt.figure(figsize=(6, 4))
    plt.plot(stats["test_acc"])
    plt.title("Test accuracy (per epoch)")
    p = out_dir / "test_accuracy.png"
    plt.savefig(p)
    plt.close()
    paths["test_accuracy"] = p

    return paths


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """Return accuracy on a dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# -----------------------------
# Training
# -----------------------------
def train(
    *,
    lr: float,
    batch_size: int,
    epochs: int,
    model_type: str,
    processed_dir: Path,
    model_out: Path,
    figures_dir: Path,
    log_every_steps: int = 200,
) -> float:
    """
    Train a model and return best test accuracy.
    Includes:
      - Loguru application logs
      - W&B experiment logging (metrics + plots + model artifact)
    """
    # Application logging setup
    setup_logger(level="INFO")

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Config: lr={lr}, batch_size={batch_size}, epochs={epochs}, model_type={model_type}")
    logger.info(f"Loading processed data from: {processed_dir.resolve()}")

    # Ensure output dirs exist
    model_out.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # W&B setup
    load_dotenv()  # optional; uses WANDB_* from .env if present
    run = wandb.init(
        project=None,  # can be set via WANDB_PROJECT env var
        entity=None,  # can be set via WANDB_ENTITY env var
        job_type="train",
        config={
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "model_type": model_type,
            "device": str(DEVICE),
        },
    )

    # Data
    train_set, test_set = load_processed(processed_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model
    model = make_model(model_type).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "test_acc": []}

    global_step = 0
    best_test_acc = 0.0

    logger.info("Training started.")
    for epoch in range(1, epochs + 1):
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            step_acc = accuracy_from_logits(logits.detach(), y)

            stats["train_loss"].append(loss.item())
            stats["train_acc"].append(step_acc)

            global_step += 1

            # Log every N steps
            if global_step % log_every_steps == 0:
                logger.info(
                    f"epoch={epoch} step={global_step} batch={batch_idx} loss={loss.item():.4f} acc={step_acc:.4f}"
                )
                wandb.log(
                    {"train/loss": loss.item(), "train/acc": step_acc, "epoch": epoch},
                    step=global_step,
                )

        # Epoch evaluation
        test_acc = evaluate(model, test_loader)
        stats["test_acc"].append(test_acc)

        logger.success(f"Epoch {epoch}/{epochs} done. test_acc={test_acc:.4f}")
        wandb.log(
            {"epoch/test_acc": test_acc, "epoch": epoch},
            step=global_step,
        )

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_out)
            logger.success(f"New best model saved to {model_out} (test_acc={best_test_acc:.4f})")

    # Save plots (local)
    plot_paths = save_training_plots(stats, figures_dir)
    logger.success(f"Saved training plots to: {figures_dir}")

    # Log plots to W&B
    wandb.log(
        {name: wandb.Image(str(path)) for name, path in plot_paths.items()},
        step=global_step,
    )

    # Log model artifact to W&B
    artifact = wandb.Artifact(
        name=f"fashionmnist-{model_type}",
        type="model",
        metadata={"best_test_acc": best_test_acc, "lr": lr, "batch_size": batch_size, "epochs": epochs},
    )
    artifact.add_file(str(model_out))
    run.log_artifact(artifact)

    logger.success(f"Training finished. best_test_acc={best_test_acc:.4f}")
    run.finish()

    return best_test_acc


# -----------------------------
# Hydra entrypoint
# -----------------------------
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra config expected (example):
      cfg.hyperparameters.learning_rate
      cfg.hyperparameters.batch_size
      cfg.hyperparameters.epochs
      cfg.model.type
    """
    # Optional: log full hydra config (useful for reproducibility)
    logger.info("Hydra config loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    # PROFILE=1 python -m fashionmnist_classification_mlops.train
    do_profile = os.getenv("PROFILE", "0") == "1"

    if do_profile:
        pr = cProfile.Profile()
        pr.enable()
        logger.info("Profiling enabled")

    train(
        lr=float(cfg.hyperparameters.learning_rate),
        batch_size=int(cfg.hyperparameters.batch_size),
        epochs=int(cfg.hyperparameters.epochs),
        model_type=str(cfg.model.type),
        processed_dir=Path("data/processed"),
        model_out=Path("models/model.pth"),
        figures_dir=Path("reports/figures"),
        log_every_steps=200,
    )

    if do_profile:
        pr.disable()

        logger.info("Profiling results (cumulative time):")
        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.print_stats(40)

        # Also useful to see "self time"
        logger.info("Profiling results (total time):")
        stats = pstats.Stats(pr).strip_dirs().sort_stats("tottime")
        stats.print_stats(40)

        # Save profile in the reports/profiler directory
        profile_path = Path("reports/profiler/train_profile.pstats")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        pstats.Stats(pr).dump_stats(str(profile_path))
        logger.success(f"Saved profile to: {profile_path}")


if __name__ == "__main__":
    main()
