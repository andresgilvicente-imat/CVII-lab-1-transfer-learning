"""Script to train and evaluate an image classifier."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.models.base import BaseImageClassifier
from src.train.early_stopping import EarlyStopping
from src.train.loss import GoogLeNetLoss


class Trainer:
    """Train, validate and evaluate a classification model."""

    def __init__(self, model: BaseImageClassifier, criterion: GoogLeNetLoss) -> None:
        """Constructor of the class.

        Args:
            model: Model to train.
            criterion: Loss used for training and validation.
        """

        self.device = torch.device(config.training.device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = config.training.optimizer(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.training.learning_rate,
        )
        self.early_stopping = EarlyStopping(
            config.training.patience, delta=config.training.delta
        )

    def _add_auxiliary_losses(
        self,
        total_losses: list[float],
        auxiliary_logits: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> None:
        """Adds the auxiliary losses of one batch to their totals.

        Args:
            total_losses: Accumulated auxiliary losses.
            auxiliary_logits: Logits from the auxiliary classifiers.
            targets: True labels.
        """

        if not total_losses:
            total_losses.extend([0.0] * len(auxiliary_logits))

        for index, auxiliary_logit in enumerate(auxiliary_logits):
            auxiliary_loss = self.criterion.criterion(auxiliary_logit, targets)
            total_losses[index] += auxiliary_loss.item()

    def _train_one_epoch(self, loader: DataLoader) -> tuple[float, float, list[float]]:
        """Runs one training epoch.

        Args:
            loader: Train loader.

        Returns:
            Loss, accuracy and auxiliary classifier losses.
        """

        total_loss = 0.0
        total_correct = 0
        total_auxiliary_losses: list[float] = []

        # TODO

    @torch.no_grad()
    def _valid_one_epoch(self, loader: DataLoader) -> tuple[float, float, list[float]]:
        """Runs one validation epoch.

        Args:
            loader: Validation loader.

        Returns:
            Loss, accuracy and auxiliary classifier losses.
        """

        total_loss = 0.0
        total_correct = 0
        total_auxiliary_losses: list[float] = []

        # TODO

    @staticmethod
    def _update_training_progress(
        epoch: int, epochs: int, history: dict[str, list[float]]
    ) -> None:
        """Prints on screen the current train and validation losses and accuracies.

        Args:
            epoch: Current epoch.
            epochs: Total number of epochs.
            history: Dictionary with train and validation losses and accuracies.
        """

        save_every = max(1, epochs // 5)

        if not (epoch == 1 or epoch == epochs or epoch % save_every == 0):
            return

        last_loss_train = history["train_loss"][-1]
        last_loss_val = history["val_loss"][-1]
        last_acc_train = history["train_acc"][-1]
        last_acc_val = history["val_acc"][-1]

        print(
            f"Epoch: {epoch}. Loss train: {last_loss_train:.3f}; Accuracy train: "
            f"{last_acc_train:.3f} | Loss validation: {last_loss_val:.3f}; Accuracy "
            f"validation: {last_acc_val:.3f}"
        )

    @staticmethod
    def _update_history(
        history: dict[str, list[float]],
        train_results: tuple[float, float, list[float]],
        validation_results: tuple[float, float, list[float]],
    ) -> None:
        """Adds the results of one epoch to the training history.

        Args:
            history: Training history.
            train_results: Training loss, accuracy and auxiliary losses.
            validation_results: Validation loss, accuracy and auxiliary losses.
        """

        loss_train, acc_train, train_auxiliary_losses = train_results
        loss_valid, acc_valid, validation_auxiliary_losses = validation_results

        history["train_loss"].append(loss_train)
        history["train_acc"].append(acc_train)
        history["val_loss"].append(loss_valid)
        history["val_acc"].append(acc_valid)

        for name, loss in zip(("aux_train_1", "aux_train_2"), train_auxiliary_losses):
            history[name].append(loss)
        for name, loss in zip(
            ("aux_valid_1", "aux_valid_2"), validation_auxiliary_losses
        ):
            history[name].append(loss)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        *,
        path_weights: str,
        path_figure: str,
    ) -> BaseImageClassifier:
        """Trains the model and restores its best parameters.

        Args:
            train_loader: Train loader.
            validation_loader: Validation loader.
            epochs: Number of epochs to train.
            path_weights: Path where the weights are saved.
            path_figure: Path where the evolution of the training is saved.
        """

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "aux_train_1": [],
            "aux_train_2": [],
            "aux_valid_1": [],
            "aux_valid_2": [],
        }
        self.early_stopping.reset()

        # TODO

        self.save_training_figure(history, path_figure)
        parameters = torch.load(
            path_weights, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(parameters)

        return self.model

    @staticmethod
    def save_training_figure(history: dict[str, list[float]], path: str) -> None:
        """Saves the evolution of the losses and accuracies.

        Args:
            history: Train and validation losses and accuracies.
            path: Path where the figure is saved.
        """

        epochs = range(1, len(history["train_loss"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"], label="Validation")
        auxiliary_curves = (
            ("aux_train_1", "Train auxiliary 1"),
            ("aux_train_2", "Train auxiliary 2"),
            ("aux_valid_1", "Validation auxiliary 1"),
            ("aux_valid_2", "Validation auxiliary 2"),
        )
        for name, label in auxiliary_curves:
            if history[name]:
                axes[0].plot(epochs, history[name], label=label)
        axes[0].set_title("Loss")

        axes[1].plot(epochs, history["train_acc"], label="Train")
        axes[1].plot(epochs, history["val_acc"], label="Validation")
        axes[1].set_title("Accuracy")

        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.grid()
            axis.legend()

        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
