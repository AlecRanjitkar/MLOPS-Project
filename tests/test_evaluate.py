import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fashionmnist_classification_mlops.evaluate import (
    CLASS_NAMES,
    make_model,
    plot_classification_report,
    plot_confusion_matrix_normalized,
)


def test_make_model_evaluate():
    """Test model creation in evaluate module."""
    # Test CNN creation
    model_cnn = make_model("cnn")
    assert model_cnn is not None
    assert hasattr(model_cnn, "forward")

    # Test MLP creation
    model_mlp = make_model("mlp")
    assert model_mlp is not None
    assert hasattr(model_mlp, "forward")


def test_plot_confusion_matrix():
    """Test confusion matrix plotting."""
    # Create dummy confusion matrix
    cm = np.random.randint(0, 100, (10, 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "cm.png"

        # Should not raise an error
        plot_confusion_matrix_normalized(cm, CLASS_NAMES, str(out_path))

        assert out_path.exists(), "Confusion matrix plot should be saved"


def test_plot_classification_report():
    """Test classification report plotting."""
    # Create dummy predictions
    y_true = torch.randint(0, 10, (100,))
    y_pred = torch.randint(0, 10, (100,))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.png"

        # Should not raise an error
        plot_classification_report(y_true, y_pred, CLASS_NAMES, str(out_path))

        assert out_path.exists(), "Classification report should be saved"


def test_class_names_length():
    """Test that CLASS_NAMES has correct length."""
    assert len(CLASS_NAMES) == 10, "Fashion-MNIST has 10 classes"
    assert "T-shirt/top" in CLASS_NAMES
    assert "Ankle boot" in CLASS_NAMES


def test_evaluation_pipeline():
    """Test full evaluation pipeline with dummy data."""
    from fashionmnist_classification_mlops.model import FashionCNN

    model = FashionCNN()
    model.eval()

    # Create dummy test data
    x = torch.randn(20, 1, 28, 28)

    with torch.no_grad():
        logits = model(x)

    # Check output shape
    assert logits.shape == (20, 10), "Model should output (batch_size, num_classes)"

    # Get predictions
    preds = logits.argmax(dim=1)

    # Check predictions are valid class indices
    assert all(0 <= p < 10 for p in preds), "Predictions should be valid class indices"
