"""Pretrained GoogLeNet adapted to the classification task."""

from torch import nn
from torchvision.models import GoogLeNet_Weights, googlenet

from src.models.base import BaseImageClassifier


class PretrainedGoogLeNet(BaseImageClassifier):
    """Pretrained GoogLeNet with a compact classifier."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        weights: GoogLeNet_Weights | None = GoogLeNet_Weights.DEFAULT,
    ) -> None:
        """Constructor of the class. It uses a 1x1 adaptative average pool.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout for the classifier.
            weights: Torchvision weights to load, or 'None' for random weights.
        """

        super().__init__(num_classes)

        self.cnn = self._get_cnn(weights)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._get_classifier(num_classes, dropout, in_features=1024)

    @staticmethod
    def _get_cnn(weights: GoogLeNet_Weights | None) -> nn.Sequential:
        """Obtains the pretrained GoogLeNet feature extractor.

        Args:
            weights: Torchvision weights to load, or 'None' for random weights.

        Returns:
            GoogLeNet convolution and Inception blocks.
        """

        backbone = googlenet(weights=weights, init_weights=False)
        return nn.Sequential(
            backbone.conv1,
            backbone.maxpool1,
            backbone.conv2,
            backbone.conv3,
            backbone.maxpool2,
            backbone.inception3a,
            backbone.inception3b,
            backbone.maxpool3,
            backbone.inception4a,
            backbone.inception4b,
            backbone.inception4c,
            backbone.inception4d,
            backbone.inception4e,
            backbone.maxpool4,
            backbone.inception5a,
            backbone.inception5b,
        )
