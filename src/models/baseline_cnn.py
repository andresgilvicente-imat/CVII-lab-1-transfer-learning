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

        # TODO

        avgpool_output_size: int = 4  # Specified in this function's docstring
        output_feature_maps: int = 128  # Showed in the README diagram

        self.cnn = self._get_cnn()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=avgpool_output_size)

        self.classifier = self._get_classifier(
            num_classes=num_classes,
            dropout=dropout,
            in_features=output_feature_maps*(avgpool_output_size)**2
        )  # The classifier is used after the flatten, so the in_features dimension is the result of multiplying: o x h x w = 128 x 4 x 4

    @staticmethod
    def _get_cnn() -> nn.Sequential:
        """Obtains the CNN.

        Returns:
            Convolutional layers.
        """

        # TODO

        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        return cnn
