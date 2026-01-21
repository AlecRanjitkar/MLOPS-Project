# src/fashionmnist_classification_mlops/drift_detection.py
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.report import Report
from PIL import Image


def extract_image_features(image_tensor: torch.Tensor) -> dict[str, float]:
    """Extract statistical features from a Fashion-MNIST image tensor."""
    img_array = image_tensor.numpy()

    return {
        "mean_pixel": float(img_array.mean()),
        "std_pixel": float(img_array.std()),
        "max_pixel": float(img_array.max()),
        "min_pixel": float(img_array.min()),
        "contrast": float(img_array.max() - img_array.min()),
        "median_pixel": float(np.median(img_array)),
        "percentile_25": float(np.percentile(img_array, 25)),
        "percentile_75": float(np.percentile(img_array, 75)),
    }


def extract_features_from_dataset(data_path: Path, max_samples: int = 500) -> pd.DataFrame:
    """Extract features from processed Fashion-MNIST data."""
    # Load processed tensors
    images = torch.load(data_path / "test_images.pt")
    labels = torch.load(data_path / "test_labels.pt")

    features_list = []

    for idx in range(min(len(images), max_samples)):
        img = images[idx]
        label = int(labels[idx].item())

        # Extract features
        features = extract_image_features(img)
        features["label"] = label
        features_list.append(features)

    return pd.DataFrame(features_list)


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: Path = Path("reports/drift_report.html"),
) -> None:
    """Generate drift detection report using Evidently."""
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(str(output_path))

    print(f"Drift report saved to {output_path}")


if __name__ == "__main__":
    print("📊 Extracting features from training data...")
    reference_data = extract_features_from_dataset(Path("data/processed"), max_samples=1000)

    print("📊 Extracting features from test data...")
    current_data = extract_features_from_dataset(Path("data/processed"), max_samples=500)

    print("📈 Generating drift report...")
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    generate_drift_report(reference_data, current_data, output_dir / "drift_report.html")

    print("✨ Done! Open reports/drift_report.html to view the report.")
