"""Script to define the loss function."""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class GoogLeNetLoss(nn.Module):
    """Class to define the loss used during training."""

    def __init__(
        self, class_weights: torch.Tensor, weight_aux_clfs: float = 0.3
    ) -> None:
        """Constructor of the class.

        Args:
            class_weights: Weight assigned to each class.
            weight_aux_clfs: Weight given to the loss of the auxiliary classifiers.
        """

        super().__init__()

        self.criterion = CrossEntropyLoss(weight=class_weights)
        self.weight_aux_clfs = weight_aux_clfs

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_clfs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the main and auxiliary classification losses. If `y_pred_clfs` is
        empty, only the main loss is computed.

        Args:
            y_true: True labels. Dimensions: [batch].
            y_pred: Logits of the final classifier. Dimensions: [batch, num_classes].
            y_pred_clfs: List with the prediction of each auxiliary classifier. Each
                prediction has dimensions [batch, num_classes].

        Returns:
            GoogLeNet loss.
        """

        # TODO
