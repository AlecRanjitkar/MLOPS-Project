# src/fashionmnist_classification_mlops/evaluate.py
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from fashionmnist_classification_mlops.model import FashionCNN, FashionMLP

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def make_model(model_type: str = "cnn"):
    return FashionMLP() if model_type.lower() == "mlp" else FashionCNN()


def plot_confusion_matrix_normalized(cm, class_names, out_path):
    cm = cm.astype(np.float64)
    cm = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("Confusion matrix (row-normalized, %)", fontsize=16)
    plt.colorbar(label="Fraction")

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right", fontsize=11)
    plt.yticks(ticks, class_names, fontsize=11)

    # Show all cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j,
                i,
                f"{val * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val > 0.5 else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_classification_report(y_true, y_pred, class_names, out_path):
    """Create a visual classification report and top confusions."""
    from sklearn.metrics import precision_recall_fscore_support

    # Get metrics per class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    # --- Top subplot: Metrics table ---
    metrics_data = []
    for i, name in enumerate(class_names):
        metrics_data.append([name, f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", f"{support[i]}"])

    # Add summary rows
    accuracy = (y_true == y_pred).float().mean().item()
    metrics_data.append(["", "", "", "", ""])  # Empty row
    metrics_data.append(["accuracy", "", "", f"{accuracy:.4f}", f"{len(y_true)}"])
    metrics_data.append(
        ["macro avg", f"{precision.mean():.4f}", f"{recall.mean():.4f}", f"{f1.mean():.4f}", f"{support.sum()}"]
    )

    # Create table
    ax1.axis("tight")
    ax1.axis("off")
    table = ax1.table(
        cellText=metrics_data,
        colLabels=["Class", "Precision", "Recall", "F1-Score", "Support"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style summary rows
    for i in range(len(class_names) + 1, len(metrics_data)):
        for j in range(5):
            table[(i, j)].set_facecolor("#E7E6E6")

    ax1.set_title("Classification Report", fontsize=16, weight="bold", pad=20)

    # --- Bottom subplot: Top confusions ---
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    np.fill_diagonal(cm_norm, 0.0)
    flat = np.dstack(np.unravel_index(np.argsort(cm_norm.ravel())[::-1], cm_norm.shape))[0]

    confusion_data = []
    for idx, (i, j) in enumerate(flat[:8], 1):  # Top 8 confusions
        confusion_data.append([f"{idx}.", class_names[i], "→", class_names[j], f"{cm_norm[i, j] * 100:.1f}%"])

    ax2.axis("tight")
    ax2.axis("off")
    confusion_table = ax2.table(
        cellText=confusion_data,
        colLabels=["#", "True Label", "", "Predicted As", "Error Rate"],
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.25, 0.08, 0.25, 0.15],
    )
    confusion_table.auto_set_font_size(False)
    confusion_table.set_fontsize(10)
    confusion_table.scale(1, 2)

    # Style header
    for i in range(5):
        confusion_table[(0, i)].set_facecolor("#C65911")
        confusion_table[(0, i)].set_text_props(weight="bold", color="white")

    ax2.set_title("Top Model Confusions", fontsize=16, weight="bold", pad=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main(model_type: str = "cnn"):
    # Load test data
    x = torch.load("data/processed/test_images.pt")
    y = torch.load("data/processed/test_labels.pt")

    if x.ndim == 3:
        x = x.unsqueeze(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = make_model(model_type).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(x.to(device)).argmax(dim=1).cpu()

    # Metrics
    acc = (preds == y).float().mean().item()
    print(f"\nTest accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y, preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y, preds)

    # Save confusion matrix
    plot_confusion_matrix_normalized(
        cm,
        CLASS_NAMES,
        out_path="reports/figures/confusion_matrix_normalized.png",
    )

    # Save classification report as image
    plot_classification_report(y, preds, CLASS_NAMES, out_path="reports/figures/classification_report.png")

    print("\nSaved figures:")
    print(" - reports/figures/confusion_matrix_normalized.png")
    print(" - reports/figures/classification_report.png")

    # Top confusions in console
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    np.fill_diagonal(cm_norm, 0.0)
    flat = np.dstack(np.unravel_index(np.argsort(cm_norm.ravel())[::-1], cm_norm.shape))[0]

    print("\nTop confusions (true → predicted):")
    for i, j in flat[:5]:
        print(f" - {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm_norm[i, j]:.2%}")


if __name__ == "__main__":
    main()
