"""Tests for stochastic and deterministic preprocessing pipelines."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.data_augmentation import build_eval_transform, build_train_transform


def _sample_image() -> Image.Image:
    """Creates a random image."""

    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    return Image.fromarray(pixels, mode="RGB")


def test_train_transform_contains_requested_augmentations() -> None:
    """Training preprocessing includes geometry, color, normalization, and erasing."""

    pipeline = build_train_transform(image_size=32)
    layer_types = tuple(type(layer) for layer in pipeline.transforms)

    assert transforms.RandomResizedCrop in layer_types
    assert transforms.RandomHorizontalFlip in layer_types
    assert transforms.RandomRotation in layer_types
    assert transforms.ColorJitter in layer_types
    assert transforms.Normalize in layer_types
    assert transforms.RandomErasing in layer_types
    assert pipeline(_sample_image()).shape == (3, 32, 32)


def test_eval_transform_is_deterministic() -> None:
    """Validation and test preprocessing returns repeatable normalized tensors."""

    pipeline = build_eval_transform(image_size=32)
    first = pipeline(_sample_image())
    second = pipeline(_sample_image())

    assert first.shape == (3, 32, 32)
    assert torch.equal(first, second)
    assert not any(
        isinstance(layer, transforms.RandomResizedCrop) for layer in pipeline.transforms
    )
