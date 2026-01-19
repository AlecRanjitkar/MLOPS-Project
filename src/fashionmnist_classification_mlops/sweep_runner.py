from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--model", type=str, required=True, choices=["cnn", "mlp"])
    args = p.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "fashionmnist_classification_mlops.train",
        f"hyperparameters.learning_rate={args.lr}",
        f"hyperparameters.batch_size={args.batch_size}",
        f"hyperparameters.epochs={args.epochs}",
        f"model.type={args.model}",
    ]

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
