from pathlib import Path

import pytest
import torch

from fashionmnist_classification_mlops.data import (
    _compute_mean_std,
    _ensure_dir,
    load_processed,
)


def test_ensure_dir_creates_directory(tmp_path):
    new_dir = tmp_path / "nested"
    _ensure_dir(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_compute_mean_std_valid_input():
    images = torch.rand(10, 1, 28, 28)
    mean, std = _compute_mean_std(images)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0


@pytest.mark.skipif(
    not Path("data/processed").exists(),
    reason="Processed data not found",
)
def test_load_processed_returns_datasets():
    train_set, test_set = load_processed(Path("data/processed"))
    assert len(train_set) > 0
    assert len(test_set) > 0
