"""Tests for the baseline CNN."""

import torch
from torch import nn

from src.models.baseline_cnn import BaseCNN


def test_baseline_cnn_architecture_and_output() -> None:
    """The baseline uses three convolutions and returns one logit per class."""

    model = BaseCNN(num_classes=3)
    assert isinstance(model.cnn, nn.Sequential)
    convolutions = [layer for layer in model.cnn if isinstance(layer, nn.Conv2d)]

    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    assert [layer.out_channels for layer in convolutions] == [32, 64, 128]
    assert output.shape == (2, 3)
