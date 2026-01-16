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
        f"Expected output shape ({input_shape[0]}, {num_classes}), "
        f"got {tuple(y.shape)}"
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
