"""Tests for classification with frozen image features."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseImageClassifier
from src.techniques.use_features import UseFeatures


class TinyModel(BaseImageClassifier):
    """Small feature extractor used by the tests."""

    def __init__(self) -> None:
        super().__init__(num_classes=2)
        self.cnn = self._get_cnn()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(6, 2))

    @staticmethod
    def _get_cnn(*_args: object, **_kwargs: object) -> nn.Sequential:
        """Creates a six-channel feature extractor."""

        return nn.Sequential(nn.Conv2d(3, 6, kernel_size=3, padding=1), nn.ReLU())


def _loader() -> DataLoader:
    """Creates a small binary-classification loader."""

    generator = torch.Generator().manual_seed(4)
    images = torch.randn(12, 3, 8, 8, generator=generator)
    targets = torch.tensor([0, 1] * 6)
    images[targets == 1] += 1.0
    return DataLoader(TensorDataset(images, targets), batch_size=4)


def test_extract_features_and_fit_classifier() -> None:
    """Features are flattened and used to train the returned random forest."""

    technique = UseFeatures(TinyModel(), n_estimators=5)
    features, targets = technique.extract_dataset_features(_loader())
    classifier = technique.fit(_loader(), _loader(), epochs=1)

    assert features.shape == (12, 6)
    assert targets.shape == (12,)
    assert targets.dtype == np.int64
    assert classifier is technique.classifier
    assert classifier.predict(features).shape == (12,)
