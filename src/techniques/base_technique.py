"""Base class for fine-tuning techniques."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from src.models.base import BaseImageClassifier


@dataclass(frozen=True)
class PredictionOutput:
    """Predictions used to evaluate a model."""

    targets: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    loss: float


class FineTuningTechnique(ABC):
    """Common interface for fine-tuning techniques."""

    def __init__(self, base_model: BaseImageClassifier) -> None:
        """Stores the model used by the technique.

        Args:
            base_model: Base model used for fine-tuning.
        """

        self.base_model = base_model

    @abstractmethod
    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> BaseImageClassifier | BaseEstimator:
        """Trains and returns a model.

        Args:
            train_loader: Train loader.
            val_loader: Validation loader.
            epochs: Number of epochs to train.

        Returns:
            Trained model.
        """
