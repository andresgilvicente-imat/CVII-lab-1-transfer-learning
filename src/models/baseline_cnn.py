"""A compact convolutional baseline for image classification."""

from torch import nn

from src.models.base import BaseImageClassifier


class BaseCNN(BaseImageClassifier):
    """Convolutional feature extractor followed by a small MLP classifier."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        """Constructor of the class. It uses a 4x4 adaptative average pool.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout for the classifier.
        """

        super().__init__(num_classes)

        raise NotImplementedError  # TODO

    @staticmethod
    def _get_cnn() -> nn.Sequential:
        """Obtains the CNN.

        Returns:
            Convolutional layers.
        """

        # TODO
