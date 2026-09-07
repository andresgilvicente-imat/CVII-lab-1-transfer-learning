"""Tests for the experiment runner."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.compare import EvaluationResult
from src.main import run_experiments
from src.techniques.base_technique import PredictionOutput


def _loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates three small loaders without downloading data."""

    generator = torch.Generator().manual_seed(12)
    images = torch.randn(12, 3, 16, 16, generator=generator)
    targets = torch.tensor([0, 1] * 6)
    dataset = TensorDataset(images, targets)
    return (
        DataLoader(dataset, batch_size=4, shuffle=True),
        DataLoader(dataset, batch_size=4),
        DataLoader(dataset, batch_size=4),
    )


def test_run_base_experiment(monkeypatch) -> None:
    """The selected base experiment is run and returned."""

    expected_result = EvaluationResult(
        name="base_cnn",
        architecture="BaseCNN",
        technique="end_to_end",
        loss=0.5,
        accuracy=0.75,
        macro_f1=0.7,
    )
    expected_predictions = PredictionOutput(
        targets=np.array([0, 1] * 6),
        predictions=np.array([0, 1] * 6),
        probabilities=np.tile([0.5, 0.5], (12, 1)),
        loss=0.5,
    )
    train_model = Mock(return_value=(expected_result, expected_predictions))
    save_predictions = Mock()
    save_augmentations = Mock()
    monkeypatch.setattr("src.main.get_data_loaders", _loaders)
    monkeypatch.setattr("src.main._train_model", train_model)
    monkeypatch.setattr("src.main.save_predictions_figure", save_predictions)
    monkeypatch.setattr("src.main.save_augmentations_figure", save_augmentations)

    results = run_experiments(
        experiments=("base_cnn",),
        epochs=1,
        pretrained_weights=None,
    )

    assert results == [expected_result]
    train_model.assert_called_once()
    save_predictions.assert_called_once()
    save_augmentations.assert_called_once()


def test_unknown_experiment_raises_error(monkeypatch) -> None:
    """An unknown experiment name is rejected."""

    monkeypatch.setattr("src.main.get_data_loaders", _loaders)

    with pytest.raises(ValueError, match="Unknown experiment"):
        run_experiments(
            experiments=("unknown",),
            epochs=1,
            pretrained_weights=None,
        )
