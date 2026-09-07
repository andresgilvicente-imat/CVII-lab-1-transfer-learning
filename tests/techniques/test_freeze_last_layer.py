"""Tests for fine-tuning the final convolutional layer."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseImageClassifier
from src.techniques.freeze_last_layer import FreezeLastLayer
from src.train.train import Trainer


class TinyModel(BaseImageClassifier):
    """Small convolutional model used by the tests."""

    def __init__(self) -> None:
        super().__init__(num_classes=2)
        self.cnn = self._get_cnn()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(6, 2))

    @staticmethod
    def _get_cnn(*_args: object, **_kwargs: object) -> nn.Sequential:
        """Creates two convolutional layers."""

        return nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 6, kernel_size=3, padding=1),
            nn.ReLU(),
        )


def _empty_loader() -> DataLoader:
    """Creates an empty but fully typed image loader."""

    images = torch.empty(0, 3, 8, 8)
    targets = torch.empty(0, dtype=torch.long)
    return DataLoader(TensorDataset(images, targets))


def _class_weights(_loader: DataLoader) -> torch.Tensor:
    """Returns fixed class weights without reading the empty loader."""

    return torch.ones(2)


def test_prepare_model_selects_requested_layers() -> None:
    """Only the final convolution and classifier remain trainable."""

    model = TinyModel()
    FreezeLastLayer(model).prepare_model()
    convolutions = [
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    ]

    assert all(
        not parameter.requires_grad for parameter in convolutions[0].parameters()
    )
    assert all(parameter.requires_grad for parameter in convolutions[-1].parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_fit_returns_trainer_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The technique prepares the model and delegates its training."""

    model = TinyModel()
    loader = _empty_loader()
    monkeypatch.setattr(
        "src.techniques.freeze_last_layer.get_class_weights", _class_weights
    )

    def fake_fit(
        self: Trainer,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        *,
        path_weights: str,
        path_figure: str,
    ) -> BaseImageClassifier:
        del train_loader, validation_loader, epochs, path_weights, path_figure
        assert self.criterion.criterion.weight is not None
        return self.model

    monkeypatch.setattr(Trainer, "fit", fake_fit)

    trained_model = FreezeLastLayer(model).fit(loader, loader, epochs=1)

    assert trained_model is model
