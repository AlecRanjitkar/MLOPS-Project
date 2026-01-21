import time

import pytest
import torch

from fashionmnist_classification_mlops.model import FashionCNN, FashionMLP


@pytest.mark.parametrize(
    "model_cls, input_shape",
    [
        (FashionCNN, (4, 1, 28, 28)),
        (FashionMLP, (4, 1, 28, 28)),
    ],
)
def test_model_forward_shape(model_cls, input_shape):
    """Test that model forward pass returns correct output shape."""
    num_classes = 10
    model = model_cls(num_classes=num_classes)

    x = torch.randn(*input_shape)
    y = model(x)

    assert y.shape == (input_shape[0], num_classes), (
        f"Expected output shape ({input_shape[0]}, {num_classes}), got {tuple(y.shape)}"
    )


def test_model_parameters_exist():
    """Ensure model has trainable parameters."""
    model = FashionCNN()
    n_params = sum(p.numel() for p in model.parameters())

    assert n_params > 0, "Model has no parameters"


def test_model_accepts_float_tensor():
    """Model should accept float tensors."""
    model = FashionCNN()
    x = torch.randn(2, 1, 28, 28, dtype=torch.float32)

    y = model(x)
    assert y.dtype == torch.float32, "Output dtype mismatch"


def test_cnn_output_shape():
    """Test CNN model outputs correct shape."""
    model = FashionCNN()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"


def test_mlp_output_shape():
    """Test MLP model outputs correct shape."""
    model = FashionMLP()
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"


def test_model_inference_speed():
    """Test model inference completes within acceptable time."""
    model = FashionCNN()
    model.eval()
    x = torch.randn(1, 1, 28, 28)

    start = time.time()
    with torch.no_grad():
        _ = model(x)
    duration = time.time() - start

    assert duration < 1.0, f"Model inference took {duration:.3f}s, expected < 1.0s"


def test_model_no_nan_outputs():
    """Test model doesn't produce NaN values."""
    model = FashionCNN()
    model.eval()
    x = torch.randn(2, 1, 28, 28)

    with torch.no_grad():
        output = model(x)

    assert not torch.isnan(output).any(), "Model produced NaN values"
