from __future__ import annotations

from pathlib import Path
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import torch
from loguru import logger
import os
import cProfile
import pstats


from fashionmnist_classification_mlops.model import FashionCNN, FashionMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_processed(processed_dir: Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_labels = torch.load(processed_dir / "test_labels.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_labels)
    test_set = torch.utils.data.TensorDataset(test_images, test_labels)
    return train_set, test_set


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == y).float().mean()



def train(
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 5,
    model_type: str = "cnn",  # "cnn" or "mlp"
    processed_dir: Path = Path("data/processed"),
    model_out: Path = Path("models/model.pth"),
    fig_out: Path = Path("reports/figures/training_statistics.png"),
) -> None:
    """Train a Fashion-MNIST classifier."""
    logger.info(f"{lr=}, {batch_size=}, {epochs=}, {model_type=}")
    logger.info(f"Using device: {DEVICE}")

    # Ensure folders exist (M6 expects models/ and reports/figures/) :contentReference[oaicite:6]{index=6}
    model_out.parent.mkdir(parents=True, exist_ok=True)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    train_set, test_set = load_processed(processed_dir)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    if model_type.lower() == "mlp":
        model = FashionMLP().to(DEVICE)
    else:
        model = FashionCNN().to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            acc_t = accuracy_from_logits(logits, y)

            stats["train_loss"].append(loss.detach())
            stats["train_acc"].append(acc_t.detach())

            if i % 200 == 0:
                logger.info(
                    f"epoch={epoch} step={i} "
                    f"loss={loss.detach().item():.4f} acc={acc_t.detach().item():.4f}"
                )


        # Evaluate after each epoch
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        stats["test_acc"].append(test_acc)
        logger.success(f"Epoch {epoch} done. test_acc={test_acc:.4f}")

    # Save model checkpoint (M6 requirement) :contentReference[oaicite:7]{index=7}
    torch.save(model.state_dict(), model_out)
    logger.success(f"Saved model to: {model_out}")

    train_loss = [x.item() for x in stats["train_loss"]]
    train_acc = [x.item() for x in stats["train_acc"]]


    # Save training curves (M6 requirement) :contentReference[oaicite:8]{index=8}
    plt.figure(figsize=(12, 4))
    plt.plot(train_loss)
    plt.title("Train loss")
    plt.savefig(fig_out.parent / "train_loss.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(train_acc)
    plt.title("Train accuracy (per step)")
    plt.savefig(fig_out.parent / "train_accuracy.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(stats["test_acc"])
    plt.title("Test accuracy (per epoch)")
    plt.savefig(fig_out.parent / "test_accuracy.png")
    plt.close()

    logger.success(f"Saved plots to: {fig_out.parent}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Turn profiling on/off with an env var:
    # PROFILE=1 python src/.../train.py
    do_profile = os.getenv("PROFILE", "0") == "1"

    if do_profile:
        pr = cProfile.Profile()
        pr.enable()

    train(
        lr=cfg.hyperparameters.learning_rate,
        batch_size=cfg.hyperparameters.batch_size,
        epochs=cfg.hyperparameters.epochs,
        model_type=cfg.model.type,
        processed_dir=Path("data/processed"),
        model_out=Path("models/model.pth"),
        fig_out=Path("reports/figures/training_statistics.png"),
    )

    if do_profile:
        pr.disable()

        # Print top functions by cumulative time (M13)
        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.print_stats(40)

        # Also useful to see "self time"
        stats = pstats.Stats(pr).strip_dirs().sort_stats("tottime")
        stats.print_stats(40)

        # Save profile in the CURRENT working dir (Hydra run dir)
        pstats.Stats(pr).dump_stats("profile.pstats")
        print("Saved profile to: profile.pstats")




if __name__ == "__main__":
    main()
