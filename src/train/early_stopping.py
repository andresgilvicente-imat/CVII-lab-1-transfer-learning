"""Script to define early stopping."""

import numpy as np
import torch


class EarlyStopping:
    """Class to implement early stopping during training."""

    def __init__(self, patience: int, delta: float = 1e-4) -> None:
        """Constructor of the class.

        Args:
            patience: Epochs allowed without improving validation loss until the
                training is stopped.
            delta: Minimum change in the validation loss to consider an improvement.
        """

        self.patience = patience
        self.delta = delta
        self.epochs_without_improvement = 0
        self.best_validation_loss = np.inf

    @property
    def apply_early_stop(self) -> bool:
        """Flag to decide if early stopping should be applied or not.

        Returns:
            True if early stopping should be applied, False otherwise.
        """

        return self.epochs_without_improvement >= self.patience

    def __call__(
        self, val_loss: float, model_state_dict: dict[str, torch.Tensor], path: str
    ) -> None:
        """Call method.

        Args:
            val_loss: New value of the validation loss.
            model_state_dict: State dict of the model.
            path: Path where the parameters of the model are saved if the new validation
                loss is the best one obtained.
        """

        if val_loss >= self.best_validation_loss - self.delta:  # we do not improve
            self.epochs_without_improvement += 1
        else:  # we improve
            self.best_validation_loss = val_loss
            self.epochs_without_improvement = 0
            self.save_parameters(model_state_dict, path)

    def save_parameters(
        self, model_state_dict: dict[str, torch.Tensor], path: str
    ) -> None:
        """Saves the parameters of the model.

        Args:
            model_state_dict: State dict of the model.
            path: Path where the parameters of the model are saved.
        """

        torch.save(model_state_dict, path)

    def reset(self) -> None:
        """Resets the early stopping status."""

        self.epochs_without_improvement = 0
        self.best_validation_loss = np.inf
