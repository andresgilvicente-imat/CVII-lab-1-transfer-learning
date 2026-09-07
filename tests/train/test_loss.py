"""Tests for the GoogLeNet loss."""

import torch
from torch import nn

from src.train.loss import GoogLeNetLoss


def _outputs() -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Creates main logits, auxiliary logits, and class labels."""

    main_logits = torch.tensor([[2.0, -1.0], [0.5, 0.0], [-1.0, 2.0]])
    auxiliary_logits = [
        torch.tensor([[1.0, 0.0], [0.2, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.5, -0.5], [0.0, 0.0], [-0.5, 0.5]]),
    ]
    targets = torch.tensor([0, 0, 1])
    return main_logits, auxiliary_logits, targets


def test_loss_adds_weighted_auxiliary_losses() -> None:
    """Both auxiliary losses are added with the configured weight."""

    main_logits, auxiliary_logits, targets = _outputs()
    class_weights = torch.tensor([0.75, 1.5])
    criterion = GoogLeNetLoss(class_weights, weight_aux_clfs=0.3)
    cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    loss = criterion(main_logits, targets, auxiliary_logits)
    expected = cross_entropy(main_logits, targets)
    for logits in auxiliary_logits:
        expected = expected + 0.3 * cross_entropy(logits, targets)

    assert torch.allclose(loss, expected)


def test_loss_accepts_models_without_auxiliary_classifiers() -> None:
    """An empty auxiliary list produces the regular cross-entropy loss."""

    main_logits, _, targets = _outputs()
    class_weights = torch.ones(2)
    loss = GoogLeNetLoss(class_weights)(main_logits, targets, [])
    expected = nn.CrossEntropyLoss(weight=class_weights)(main_logits, targets)

    assert torch.allclose(loss, expected)
