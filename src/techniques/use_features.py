"""Train a random forest with features from a pretrained model."""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from src.config import config
from src.models.base import BaseImageClassifier
from src.techniques.base_technique import FineTuningTechnique


class UseFeatures(FineTuningTechnique):
    """Use frozen neural features to train a random forest."""

    def __init__(
        self,
        base_model: BaseImageClassifier,
        n_estimators: int = config.experiments.random_forest_estimators,
        max_depth: int | None = config.experiments.random_forest_max_depth,
    ) -> None:
        """Constructor of the class.

        Args:
            base_model: Pretrained CNN.
            n_estimators: Number of estimators of the random forest.
            max_depth: Maximum depth of the random forest.
        """

        super().__init__(base_model)

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1
        )

    @torch.no_grad()
    def extract_dataset_features(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts the features and targets from a data loader and concatenates them.

        Args:
            loader: Loader.

        Returns:
            Features and labels of the loader. Dimensions: [n_samples,
            channels * height * width] and [n_samples].
        """

        # TODO

        self.base_model.eval()

        all_features: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        device = next(self.base_model.parameters()).device

        for images, batch_targets in loader:
            images = images.to(device)
            features = self.base_model.cnn(images)
            features = self.base_model.avgpool(features)
            features = torch.flatten(features, start_dim=1)
            
            all_features.append(features.cpu().numpy())
            all_targets.append(np.asarray(batch_targets, dtype=np.int64))

        return np.concatenate(all_features, axis=0), np.concatenate(all_targets, axis=0)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> RandomForestClassifier:
        """Trains and returns a model.

        Args:
            train_loader: Train loader.
            val_loader: Validation loader.
            epochs: Number of epochs to train.

        Returns:
            Trained model.
        """

        # TODO

        train_features, train_targets = self.extract_dataset_features(train_loader)
        self.classifier.fit(train_features, train_targets)

        return self.classifier
