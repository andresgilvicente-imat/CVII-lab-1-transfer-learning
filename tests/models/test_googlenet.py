"""Tests for the pretrained GoogLeNet wrapper."""

import torch
from torch import nn

from src.models.googlenet import PretrainedGoogLeNet


def test_pretrained_googlenet_features_and_output() -> None:
    """GoogLeNet produces 1024 features and the requested output size."""

    model = PretrainedGoogLeNet(num_classes=3, weights=None)
    model.eval()
    inputs = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        features = model.cnn(inputs)
        features = torch.flatten(model.avgpool(features), start_dim=1)
        output = model(inputs)
        main_logits, auxiliary_logits = model.forward_with_auxiliary(inputs)

    first_classifier_layer = model.classifier[0]
    assert features.shape == (2, 1024)
    assert output.shape == (2, 3)
    assert main_logits.shape == (2, 3)
    assert not auxiliary_logits
    assert isinstance(first_classifier_layer, nn.Linear)
    assert first_classifier_layer.in_features == 1024
