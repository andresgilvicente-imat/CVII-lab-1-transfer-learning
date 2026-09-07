"""Offline tests for the Oxford Pets dataset and loaders."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler, Subset
from torchvision import transforms

import src.data.dataset as dataset_module
from src.data.dataset import OxfordPetsDataset, get_class_weights, get_data_loaders


class FakeOxfordPets:
    """Small replacement for torchvision's downloaded dataset."""

    def __init__(self, split: str, transform=None, **_kwargs) -> None:
        self.length = 20 if split == "trainval" else 8
        self.transform = transform
        self._bin_labels = (
            [0] * 6 + [1] * 14 if split == "trainval" else [0] * 2 + [1] * 6
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        pixels = np.full((40, 48, 3), index, dtype=np.uint8)
        image = Image.fromarray(pixels)
        if self.transform is not None:
            image = self.transform(image)
        return image, self._bin_labels[index]


def test_dataset_returns_transformed_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper delegates loading and applies the requested transform."""

    monkeypatch.setattr(dataset_module.datasets, "OxfordIIITPet", FakeOxfordPets)
    dataset = OxfordPetsDataset(
        "unused",
        "trainval",
        transforms.Compose([transforms.ToTensor()]),
        download=False,
    )

    image, label = dataset[6]

    assert len(dataset) == 20
    assert image.shape == (3, 40, 48)
    assert label == 1


def test_loaders_split_and_shuffle_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The random subsets have the requested sizes and do not overlap."""

    monkeypatch.setattr(dataset_module.datasets, "OxfordIIITPet", FakeOxfordPets)
    train, validation, test = get_data_loaders(
        root="unused",
        batch_size=2,
        image_size=32,
        train_samples=10,
        validation_samples=4,
        test_samples=6,
        seed=7,
        download=False,
    )

    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(validation.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)

    train_data = train.dataset
    validation_data = validation.dataset
    test_data = test.dataset
    assert isinstance(train_data, Subset)
    assert isinstance(validation_data, Subset)
    assert isinstance(test_data, Subset)
    assert len(train_data) == 10
    assert len(validation_data) == 4
    assert len(test_data) == 6
    assert set(train_data.indices).isdisjoint(validation_data.indices)

    for loader in (train, validation, test):
        images, labels = next(iter(loader))
        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)


def test_random_split_is_reproducible_and_class_weights_use_training_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seed fixes the indices and weights only use the training subset."""

    monkeypatch.setattr(dataset_module.datasets, "OxfordIIITPet", FakeOxfordPets)
    loaders = get_data_loaders(
        root="unused",
        batch_size=2,
        image_size=32,
        train_samples=10,
        validation_samples=4,
        test_samples=6,
        seed=7,
        download=False,
    )
    repeated_loaders = get_data_loaders(
        root="unused",
        batch_size=2,
        image_size=32,
        train_samples=10,
        validation_samples=4,
        test_samples=6,
        seed=7,
        download=False,
    )

    train_data = loaders[0].dataset
    repeated_train_data = repeated_loaders[0].dataset
    assert isinstance(train_data, Subset)
    assert isinstance(repeated_train_data, Subset)
    assert train_data.indices == repeated_train_data.indices
    assert isinstance(train_data.dataset, OxfordPetsDataset)

    labels = torch.tensor(
        [train_data.dataset.labels[index] for index in train_data.indices]
    )
    counts = torch.bincount(labels, minlength=2).float()
    expected_weights = counts.sum() / (len(counts) * counts)

    assert counts.tolist() == [4, 6]
    assert torch.allclose(get_class_weights(loaders[0]), expected_weights)
