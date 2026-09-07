"""Tests for checkpoint tracking and patience."""

import torch
from torch import nn

from src.train.early_stopping import EarlyStopping


def test_early_stopping_saves_best_weights_and_stops(tmp_path) -> None:
    """The best parameters are saved and patience eventually stops training."""

    model = nn.Linear(2, 2)
    checkpoint = tmp_path / "best.pt"
    stopping = EarlyStopping(patience=2, delta=0.1)

    stopping(1.0, model.state_dict(), str(checkpoint))
    best_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(3.0)

    stopping(0.95, model.state_dict(), str(checkpoint))
    stopping(1.2, model.state_dict(), str(checkpoint))

    assert stopping.apply_early_stop
    assert checkpoint.exists()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(model.weight, best_weight)


def test_improvement_resets_patience_counter(tmp_path) -> None:
    """A genuine improvement clears consecutive bad epochs."""

    model = nn.Linear(1, 2)
    checkpoint = tmp_path / "best.pt"
    stopping = EarlyStopping(patience=2, delta=0.0)

    stopping(2.0, model.state_dict(), str(checkpoint))
    stopping(2.1, model.state_dict(), str(checkpoint))
    assert stopping.epochs_without_improvement == 1
    stopping(1.5, model.state_dict(), str(checkpoint))

    assert stopping.epochs_without_improvement == 0
    assert stopping.best_validation_loss == 1.5
    assert not stopping.apply_early_stop


def test_reset_clears_early_stopping_state(tmp_path) -> None:
    """Reset starts a new early-stopping run."""

    model = nn.Linear(1, 2)
    stopping = EarlyStopping(patience=1)
    stopping(1.0, model.state_dict(), str(tmp_path / "best.pt"))
    stopping(2.0, model.state_dict(), str(tmp_path / "best.pt"))

    stopping.reset()

    assert stopping.epochs_without_improvement == 0
    assert not stopping.apply_early_stop
