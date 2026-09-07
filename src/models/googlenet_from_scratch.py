"""GoogLeNet implemented from scratch."""

from dataclasses import dataclass

import torch
from torch import nn

from src.models.base import BaseImageClassifier


class AuxiliaryClassifier(nn.Module):
    """Class to define an auxiliary classifier."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Constructor of the class.

        Args:
            in_channels: Number of channels received from the CNN.
            num_classes: Number of output classes.
        """

        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = self._get_classifier(in_channels, num_classes)

    @staticmethod
    def _get_classifier(in_channels: int, num_classes: int) -> nn.Sequential:
        """Obtains the classifier.

        Args:
            in_channels: Number of channels received from the CNN.
            num_classes: Number of output classes.

        Returns:
            Auxiliary classifier.
        """

        hidden_dimension = 128
        return nn.Sequential(
            nn.Linear(in_channels * 4 * 4, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        return self.classifier(x)


class ConvBlock(nn.Sequential):
    """Convolution followed by batch normalization and ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, **kwargs
    ) -> None:
        """Constructor of the class.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
        """

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


@dataclass(frozen=True)
class InceptionBlockConfig:
    """Data needed to define an Inception block."""

    in_channels: int
    channels_1x1: int
    channels_3x3_reduce: int
    channels_3x3: int
    channels_5x5_reduce: int
    channels_5x5: int
    pool_projection: int


class InceptionBlock(nn.Module):
    """Four parallel convolution and pooling branches."""

    def __init__(self, inception_block_config: InceptionBlockConfig) -> None:
        """Constructor of the class.

        Args:
            inception_block_config: Configuration for the block.
        """

        super().__init__()

        raise NotImplementedError  # TODO

    @staticmethod
    def _get_branch_1(in_channels: int, channels_1x1: int) -> nn.Sequential:
        """Obtains the first branch.

        Args:
            in_channels: Number of input channels.
            channels_1x1: Number of output channels.

        Returns:
            First Inception block branch.
        """

        # TODO

    @staticmethod
    def _get_branch_2(
        in_channels: int, channels_3x3_reduce: int, channels_3x3: int
    ) -> nn.Sequential:
        """Obtains the second branch.

        Args:
            in_channels: Number of input channels.
            channels_3x3_reduce: Number of output of channels of the 1x1 convolution.
            channels_3x3: Number of output channels of the branch.

        Returns:
            Second Inception block branch.
        """

        # TODO

    @staticmethod
    def _get_branch_3(
        in_channels: int, channels_5x5_reduce: int, channels_5x5: int
    ) -> nn.Sequential:
        """Obtains the third branch.

        Args:
            in_channels: Number of input channels.
            channels_5x5_reduce: Number of output of channels of the 1x1 convolution.
            channels_5x: Number of output channels of the branch.

        Returns:
            Third Inception block branch.
        """

        # TODO

    @staticmethod
    def _get_branch_4(in_channels: int, pool_projection: int) -> nn.Sequential:
        """Obtains the fourth branch.

        Args:
            in_channels: Number of input channels.
            pool_projection: Pool projection.

        Returns:
            Fourth Inception block branch.
        """

        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, channels_1x1 + channels_3x3 +
            channels_5x5 + pool_projection], height, width].
        """

        # TODO


class GoogLeNetFromScratch(BaseImageClassifier):
    """Compact GoogLeNet built with four Inception blocks."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        """Constructor of the class. It uses a 1x1 adaptative average pool.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout for the classifier.
        """

        super().__init__(num_classes)

        raise NotImplementedError  # TODO

    @staticmethod
    def _get_cnn_blocks() -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:
        """Obtains the CNN blocks separated by auxiliary classifier outputs.

        Returns:
            CNN divided in blocks.
        """

        # TODO

    def forward_with_auxiliary(
        self, x: torch.Tensor, *, use_auxiliary: bool = True
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass with optional auxiliary classifiers.

        Args:
            x: Input tensor. Dimensions: [batch, 3, height, width].
            use_auxiliary: Whether to obtain the auxiliary predictions.

        Returns:
            Main logits and auxiliary logits.
        """

        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        return self.forward_with_auxiliary(x, use_auxiliary=False)[0]
