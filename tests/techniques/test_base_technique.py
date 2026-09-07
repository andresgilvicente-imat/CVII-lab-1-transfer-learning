"""Tests for the fine-tuning technique interface."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseImageClassifier
from src.models.baseline_cnn import BaseCNN
from src.techniques.base_technique import FineTuningTechnique, PredictionOutput


class DummyTechnique(FineTuningTechnique):
    """Minimal concrete technique used to test the interface."""

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> BaseImageClassifier:
        """Returns the stored model."""

        return self.base_model


def _empty_loader() -> DataLoader:
    """Creates an empty but fully typed image loader."""

    images = torch.empty(0, 3, 8, 8)
    targets = torch.empty(0, dtype=torch.long)
    return DataLoader(TensorDataset(images, targets))


def test_technique_stores_and_returns_base_model() -> None:
    """A technique keeps the model received during construction."""

    model = BaseCNN()
    technique = DummyTechnique(model)
    loader = _empty_loader()

    assert technique.base_model is model
    assert technique.fit(loader, loader, epochs=1) is model


def test_prediction_output_stores_evaluation_arrays() -> None:
    """Prediction output keeps targets, predictions, probabilities, and loss."""

    output = PredictionOutput(
        targets=np.array([0, 1]),
        predictions=np.array([0, 1]),
        probabilities=np.array([[0.8, 0.2], [0.1, 0.9]]),
        loss=0.2,
    )

    assert output.targets.shape == (2,)
    assert output.predictions.shape == (2,)
    assert output.probabilities.shape == (2, 2)
    assert output.loss == 0.2
