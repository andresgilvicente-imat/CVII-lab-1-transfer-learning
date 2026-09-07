"""Tests for prediction and visualization utilities."""

import os
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline_cnn import BaseCNN
from src.utils import (
    predict,
    predict_features,
    save_augmentations_figure,
    save_predictions_figure,
)


def _loader() -> DataLoader:
    """Creates a small loader with synthetic images."""

    generator = torch.Generator().manual_seed(12)
    images = torch.randn(12, 3, 16, 16, generator=generator)
    targets = torch.tensor([0, 1] * 6)
    return DataLoader(TensorDataset(images, targets), batch_size=4)


def test_predict_returns_evaluation_output() -> None:
    """Neural predictions contain labels, probabilities, and loss."""

    output = predict(BaseCNN(), _loader())

    assert output.targets.shape == (12,)
    assert output.predictions.shape == (12,)
    assert output.probabilities.shape == (12, 2)
    assert output.loss >= 0


def test_predict_features_returns_evaluation_output() -> None:
    """Random-forest predictions use the features returned by the technique."""

    features = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    targets = np.array([0, 0, 1, 1])
    classifier = RandomForestClassifier(n_estimators=5, random_state=0)
    classifier.fit(features, targets)
    technique = Mock()
    technique.extract_dataset_features.return_value = features, targets

    output = predict_features(classifier, technique, _loader())

    assert output.targets.shape == (4,)
    assert output.predictions.shape == (4,)
    assert output.probabilities.shape == (4, 2)
    assert output.loss >= 0


def test_save_predictions_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Correct predictions are green and incorrect predictions are red."""

    images = torch.zeros(9, 3, 8, 8)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
    loader = DataLoader(TensorDataset(images, targets), batch_size=3)
    predictions = {
        "model_a": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]),
        "model_b": np.array([1, 1, 0, 0, 0, 1, 1, 1, 0]),
    }
    close_figure = plt.close
    monkeypatch.setattr("src.utils.plt.close", Mock())

    save_predictions_figure(loader, predictions, str(tmp_path))

    figure = plt.gcf()
    colors = [text.get_color() for axis in figure.axes for text in axis.texts]
    assert os.path.exists(os.path.join(tmp_path, "test_predictions.png"))
    assert "green" in colors
    assert "red" in colors
    close_figure(figure)


def test_save_augmentations_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figure contains an original image and eight augmentations."""

    figure_path = tmp_path / "data_augmentation.png"
    close_figure = plt.close
    monkeypatch.setattr("src.utils.plt.close", Mock())

    save_augmentations_figure(_loader(), str(figure_path))

    figure = plt.gcf()
    titles = [axis.get_title() for axis in figure.axes]
    assert figure_path.exists()
    assert titles == ["Original"] + [f"Augmentation {index}" for index in range(1, 9)]
    close_figure(figure)
