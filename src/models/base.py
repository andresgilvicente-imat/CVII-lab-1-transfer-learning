"""Script to define the base class for the models used."""

import torch
from torch import nn

from src.config import config


class BaseImageClassifier(nn.Module):
    """Base class for the image classification models."""

    num_classes: int
    cnn: nn.Module
    avgpool: nn.AdaptiveAvgPool2d
    classifier: nn.Sequential

    def __init__(self, num_classes: int) -> None:
        """Constructor of the class.

        Args:
            num_classes: Number of output classes.
        """

        super().__init__()

        self.num_classes = num_classes

    @staticmethod
    def _get_classifier(
        num_classes: int,
        dropout: float,
        in_features: int = 512,
        hidden_dimensions: (
            list[int] | tuple[int, ...]
        ) = config.model.classifier_hidden_dimensions,
    ) -> nn.Sequential:
        """Build and return the classifier head. All layers are linear + ReLU + dropout
        except the last one, which is only a linear.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability.
            in_features: Flattened feature size from the CNN + pooling stage.
            hidden_dimensions: Number of neurons in each hidden layer.

        Returns:
            Classifier MLP.
        """

        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO

    def forward_with_auxiliary(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with auxiliary classifiers. Note that by default they are not
        used.

        Args:
            x: Input tensor. Dimensions: [batch, 3, height, width].

        Returns:
            Main logits and logits for each auxiliary classifier, all with dimensions
            [batch, num_classes].
        """

        return self(x), []

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts class probabilities for input images.

        Args:
            x: Input tensor. Dimensions: [batch, 3, height, width].

        Returns:
            Class probabilities. Dimensions: [batch, num_classes].
        """

        self.eval()
        logits = self(x)

        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts class for input images.

        Args:
            x: Input tensor. Dimensions: [batch, 3, height, width].

        Returns:
            Class for each image. Dimensions: [batch].
        """

        return torch.argmax(self.predict_proba(x), dim=1)
