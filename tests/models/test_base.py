"""Tests for the base image classifier."""

# pylint: disable=protected-access

import torch
from torch import nn

from src.models.base import BaseImageClassifier


class TinyClassifier(BaseImageClassifier):
    """Small concrete classifier used to test the base class."""

    def __init__(self) -> None:
        super().__init__(num_classes=3)
        self.cnn = self._get_cnn()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._get_classifier(
            num_classes=3,
            dropout=0.0,
            in_features=3,
            hidden_dimensions=[4],
        )

    @staticmethod
    def _get_cnn(*_args: object, **_kwargs: object) -> nn.Sequential:
        """Returns an identity feature extractor."""

        return nn.Sequential(nn.Identity())


def test_get_classifier_accepts_multiple_hidden_dimensions() -> None:
    """One linear block is created for each hidden dimension."""

    classifier = TinyClassifier._get_classifier(
        num_classes=3,
        dropout=0.2,
        in_features=32,
        hidden_dimensions=[128, 64],
    )
    assert [type(layer) for layer in classifier] == [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    ]
    linear_layers = [layer for layer in classifier if isinstance(layer, nn.Linear)]
    dropout_layers = [layer for layer in classifier if isinstance(layer, nn.Dropout)]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (32, 128),
        (128, 64),
        (64, 3),
    ]
    assert all(layer.p == 0.2 for layer in dropout_layers)


def test_forward_and_predictions() -> None:
    """The base class returns logits, probabilities, and class indices."""

    model = TinyClassifier()
    inputs = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        logits = model(inputs)
        probabilities = model.predict_proba(inputs)
        predictions = model.predict(inputs)

    assert logits.shape == (2, 3)
    assert probabilities.shape == (2, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(predictions, probabilities.argmax(dim=1))
