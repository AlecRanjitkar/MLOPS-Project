import tempfile
from pathlib import Path

import pytest
import torch

from fashionmnist_classification_mlops.train import (
    accuracy_from_logits,
    ensure_channel_dim,
    evaluate,
    make_model,
    save_training_plots,
)


def test_accuracy_from_logits():
    """Test accuracy calculation from logits."""
    logits = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0], [1.0, 0.5, 2.0]])
    labels = torch.tensor([0, 1, 2])

    acc = accuracy_from_logits(logits, labels)
    assert acc == 1.0, "Perfect predictions should give 100% accuracy"

    # Test wrong predictions
    wrong_labels = torch.tensor([1, 0, 0])
    acc = accuracy_from_logits(logits, wrong_labels)
    assert acc == 0.0, "All wrong predictions should give 0% accuracy"


def test_ensure_channel_dim():
    """Test channel dimension handling."""
    # Test 3D input (N, H, W)
    x_3d = torch.randn(10, 28, 28)
    x_4d = ensure_channel_dim(x_3d)
    assert x_4d.shape == (10, 1, 28, 28), "Should add channel dimension"

    # Test 4D input (already has channel)
    x_already_4d = torch.randn(10, 1, 28, 28)
    x_out = ensure_channel_dim(x_already_4d)
    assert x_out.shape == (10, 1, 28, 28), "Should keep existing channel dimension"


def test_make_model():
    """Test model creation."""
    # Test CNN creation
    model_cnn = make_model("cnn")
    assert model_cnn is not None
    assert hasattr(model_cnn, "forward")

    # Test MLP creation
    model_mlp = make_model("mlp")
    assert model_mlp is not None
    assert hasattr(model_mlp, "forward")

    # Test default (should be CNN)
    model_default = make_model("unknown")
    assert model_default is not None


def test_save_training_plots():
    """Test saving training plots."""
    stats = {
        "train_loss": [1.0, 0.8, 0.6, 0.4],
        "train_acc": [0.5, 0.6, 0.7, 0.8],
        "test_acc": [0.6, 0.7, 0.75, 0.8],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        paths = save_training_plots(stats, out_dir)

        assert "train_loss" in paths
        assert "train_accuracy" in paths
        assert "test_accuracy" in paths

        assert paths["train_loss"].exists()
        assert paths["train_accuracy"].exists()
        assert paths["test_accuracy"].exists()


def test_evaluate_function():
    """Test evaluation function."""
    from fashionmnist_classification_mlops.model import FashionCNN
    from fashionmnist_classification_mlops.train import DEVICE

    model = FashionCNN().to(DEVICE)  # Move model to the same device

    # Create dummy dataset
    images = torch.randn(20, 1, 28, 28)
    labels = torch.randint(0, 10, (20,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    acc = evaluate(model, dataloader)

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1"


def test_load_processed_data():
    """Test loading processed data (mock)."""
    # This test would require actual data files
    # For now, just test the import
    from fashionmnist_classification_mlops.train import load_processed

    assert callable(load_processed)


def test_train_function_signature():
    """Test that train function exists and has correct signature."""
    import inspect

    from fashionmnist_classification_mlops.train import train

    sig = inspect.signature(train)
    params = list(sig.parameters.keys())

    # Check some expected parameters
    assert "lr" in params
    assert "batch_size" in params
    assert "epochs" in params
