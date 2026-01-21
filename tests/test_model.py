import time

import torch

from fashionmnist_classification_mlops.model import FashionCNN, FashionMLP


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
