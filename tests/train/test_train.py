"""Tests for neural training and its evolution figure."""

# pylint: disable=not-callable,protected-access

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseImageClassifier
from src.train.loss import GoogLeNetLoss
from src.train.train import Trainer


class TinyClassifier(BaseImageClassifier):
    """Small classifier used to test the training loop."""

    def __init__(self) -> None:
        """Creates a two-layer classifier for two-dimensional inputs."""

        super().__init__(num_classes=2)
        self.network = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    @staticmethod
    def _get_cnn(*_args: object, **_kwargs: object) -> nn.Sequential:
        """Returns the empty CNN required by the base interface."""

        return nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for the synthetic inputs."""

        return self.network(x)


class TinyAuxiliaryClassifier(TinyClassifier):
    """Small classifier with one auxiliary head."""

    def __init__(self) -> None:
        """Creates the main and auxiliary classifiers."""

        super().__init__()
        self.auxiliary_classifier = nn.Linear(2, 2)

    def forward_with_auxiliary(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns the main logits and one auxiliary prediction."""

        return self(x), [self.auxiliary_classifier(x)]


def _loader(samples: int = 8, batch_size: int = 3) -> DataLoader:
    """Creates a small loader for a linearly separable problem."""

    inputs = torch.tensor([[-1.0, -1.0], [-0.8, -0.5], [1.0, 1.0], [0.7, 0.9]]).repeat(
        (samples + 3) // 4, 1
    )[:samples]
    targets = torch.tensor([0, 0, 1, 1]).repeat((samples + 3) // 4)[:samples]
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def _model() -> TinyClassifier:
    """Creates the classifier used by the trainer tests."""

    return TinyClassifier()


def _criterion() -> GoogLeNetLoss:
    """Creates the loss used by the trainer tests."""

    return GoogLeNetLoss(torch.ones(2))


def test_training_updates_parameters_but_validation_does_not() -> None:
    """Only the gradient-enabled epoch mutates model parameters."""

    model = _model()
    trainer = Trainer(model, _criterion())
    before_train = [parameter.detach().clone() for parameter in model.parameters()]

    model.eval()
    train_loss, train_accuracy, auxiliary_losses = trainer._train_one_epoch(_loader())
    after_train = [parameter.detach().clone() for parameter in model.parameters()]

    assert model.training

    model.train()
    validation_loss, validation_accuracy, validation_auxiliary_losses = (
        trainer._valid_one_epoch(_loader())
    )
    after_validation = [parameter.detach().clone() for parameter in model.parameters()]

    assert not model.training
    assert train_loss >= 0
    assert not auxiliary_losses
    assert not validation_auxiliary_losses
    assert validation_loss >= 0
    assert 0 <= train_accuracy <= 1
    assert 0 <= validation_accuracy <= 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_train, after_train)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(after_train, after_validation)
    )


def test_training_updates_auxiliary_classifier() -> None:
    """Training and validation return auxiliary classifier losses."""

    model = TinyAuxiliaryClassifier()
    trainer = Trainer(model, _criterion())
    before = model.auxiliary_classifier.weight.detach().clone()

    _, _, auxiliary_losses = trainer._train_one_epoch(_loader())
    _, _, validation_auxiliary_losses = trainer._valid_one_epoch(_loader())

    assert not torch.equal(before, model.auxiliary_classifier.weight)
    assert len(auxiliary_losses) == 1
    assert auxiliary_losses[0] >= 0
    assert len(validation_auxiliary_losses) == 1
    assert validation_auxiliary_losses[0] >= 0


def test_fit_keeps_criterion_and_saves_outputs(tmp_path: Path) -> None:
    """Fitting keeps the configured criterion and saves its outputs."""

    model = _model()
    class_weights = torch.tensor([1.5, 0.75])
    criterion = GoogLeNetLoss(class_weights)
    trainer = Trainer(model, criterion)
    weights_path = tmp_path / "best.pt"
    figure_path = tmp_path / "training.png"
    train_loader = _loader()
    trained_model = trainer.fit(
        train_loader,
        _loader(),
        epochs=3,
        path_weights=str(weights_path),
        path_figure=str(figure_path),
    )

    assert trained_model is model
    assert trainer.criterion is criterion
    assert weights_path.exists()
    assert figure_path.exists()


def test_validation_uses_weighted_loss() -> None:
    """Validation uses the same class-weighted criterion as training."""

    model = _model()
    class_weights = torch.tensor([2.0, 0.5])
    trainer = Trainer(model, GoogLeNetLoss(class_weights))
    inputs = torch.tensor([[-1.0, -1.0], [-0.8, -0.5], [-0.4, -0.7], [1.0, 1.0]])
    targets = torch.tensor([0, 0, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    with torch.no_grad():
        expected_loss = trainer.criterion(model(inputs), targets, []).item()
    validation_loss, _, _ = trainer._valid_one_epoch(loader)

    assert validation_loss == pytest.approx(expected_loss)


def test_training_figure_accepts_auxiliary_losses(tmp_path: Path) -> None:
    """Auxiliary losses are added to the history and training figure."""

    names = (
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "aux_train_1",
        "aux_train_2",
        "aux_valid_1",
        "aux_valid_2",
    )
    history: dict[str, list[float]] = {name: [] for name in names}
    figure_path = tmp_path / "training_with_auxiliary.png"

    Trainer._update_history(history, (1.0, 0.5, [0.9, 1.0]), (1.1, 0.4, [1.0, 1.1]))
    Trainer._update_history(history, (0.8, 0.7, [0.7, 0.8]), (0.9, 0.6, [0.8, 0.9]))
    Trainer.save_training_figure(history, str(figure_path))

    assert history["aux_train_1"] == [0.9, 0.7]
    assert history["aux_train_2"] == [1.0, 0.8]
    assert history["aux_valid_1"] == [1.0, 0.8]
    assert history["aux_valid_2"] == [1.1, 0.9]
    assert figure_path.exists()
