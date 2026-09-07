"""Prediction and visualization utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

from src.config import config
from src.data.data_augmentation import build_train_transform
from src.models.base import BaseImageClassifier
from src.techniques.base_technique import PredictionOutput
from src.techniques.use_features import UseFeatures


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    """Restores normalized images to values between zero and one.

    Args:
        images: Images to denormalize. Dimensions: [batch, channels, height, width].

    Returns:
        Denormalized images. Dimensions: [batch, channels, height, width].
    """

    mean = torch.tensor(config.data.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(config.data.imagenet_std).view(3, 1, 1)

    return (images.cpu() * std + mean).clamp(0, 1)


@torch.no_grad()
def predict(model: BaseImageClassifier, loader: DataLoader) -> PredictionOutput:
    """Obtains labels, predictions, probabilities and loss for a model.

    Args:
        model: Model used for the predictions.
        loader: Loader containing images and targets.

    Returns:
        Corresponding `PredictionOutput` object.
    """

    device = config.training.device
    model.to(device)
    model.eval()
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    for images, batch_targets in loader:
        images = images.to(device)
        batch_targets = batch_targets.to(device)
        logits = model(images)
        batch_probabilities = torch.softmax(logits, dim=1)

        batch_size = batch_targets.size(0)
        total_loss += criterion(logits, batch_targets).item() * batch_size
        total_samples += batch_size
        targets.append(batch_targets.cpu())
        predictions.append(batch_probabilities.argmax(dim=1).cpu())
        probabilities.append(batch_probabilities.cpu())

    return PredictionOutput(
        targets=torch.cat(targets).numpy(),
        predictions=torch.cat(predictions).numpy(),
        probabilities=torch.cat(probabilities).numpy(),
        loss=total_loss / total_samples,
    )


def predict_features(
    classifier: RandomForestClassifier, technique: UseFeatures, loader: DataLoader
) -> PredictionOutput:
    """Obtains predictions from a classifier trained with image features.

    Args:
        classifier: Trained random forest.
        technique: Technique used to extract the image features.
        loader: Loader containing images and targets.

    Returns:
        Corresponding 'PredictionOutput' object.
    """

    features, targets = technique.extract_dataset_features(loader)
    probabilities = classifier.predict_proba(features)

    return PredictionOutput(
        targets=targets,
        predictions=classifier.predict(features),
        probabilities=probabilities,
        loss=float(log_loss(targets, probabilities, labels=classifier.classes_)),
    )


def save_predictions_figure(
    loader: DataLoader, model_predictions: dict[str, np.ndarray], output_directory: str
) -> None:
    """Saves nine test images with the predictions of every model.

    Args:
        loader: Test loader.
        model_predictions: Predictions of each experiment.
        output_directory: Folder where the figure is saved.
    """

    number_samples = len(next(iter(model_predictions.values())))
    sample_indices = np.linspace(0, number_samples - 1, 9).astype(int)
    samples = Subset(loader.dataset, sample_indices.tolist())
    images, targets = next(iter(DataLoader(samples, batch_size=9)))
    images = _denormalize(images)

    figure, axes = plt.subplots(3, 3, figsize=(12, 15))
    for index, axis in enumerate(axes.flat):
        target = int(targets[index])

        axis.imshow(images[index].permute(1, 2, 0).numpy())
        axis.set_title(f"True: {config.data.class_names[target]}", fontsize=9)
        for line, (name, values) in enumerate(model_predictions.items()):
            prediction = int(values[sample_indices[index]])
            axis.text(
                0.5,
                -0.08 - line * 0.06,
                f"{name}: {config.data.class_names[prediction]}",
                color="green" if prediction == target else "red",
                fontsize=8,
                ha="center",
                transform=axis.transAxes,
            )
        axis.axis("off")

    figure.tight_layout(h_pad=5)
    figure.savefig(
        os.path.join(output_directory, "test_predictions.png"), bbox_inches="tight"
    )
    plt.close(figure)


def save_augmentations_figure(loader: DataLoader, path: str) -> None:
    """Saves one original image and eight augmented versions.

    Args:
        loader: Validation or test loader with deterministic transformations.
        path: Path where the figure is saved.
    """

    images, _ = next(iter(loader))
    original = _denormalize(images[0])
    original_image = to_pil_image(original)
    train_transform = build_train_transform(config.data.image_size)

    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes[0, 0].imshow(original.permute(1, 2, 0).numpy())
    axes[0, 0].set_title("Original")

    for index, axis in enumerate(axes.flat[1:], start=1):
        transformed = _denormalize(train_transform(original_image))
        axis.imshow(transformed.permute(1, 2, 0).numpy())
        axis.set_title(f"Augmentation {index}")

    for axis in axes.flat:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
