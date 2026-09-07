"""Oxford-IIIT Pet dataset and data loaders."""

import random
from typing import Literal, cast

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config import config
from src.data.data_augmentation import build_eval_transform, build_train_transform


class OxfordPetsDataset(Dataset):
    """Small PyTorch wrapper for binary cat-versus-dog classification."""

    def __init__(
        self,
        root: str,
        split: Literal["trainval", "test"],
        transform: transforms.Compose,
        download: bool = True,
    ) -> None:
        """Constructor of the class.

        Args:
            root: Path to the folder to save the data.
            split: Split of the dataset.
            transform: Transformation used when obtaining an image.
            download: True to download the data in local, False otherwise.
        """

        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="binary-category",
            transform=transform,
            download=download,
        )
        self.labels: list[int] = self.dataset._bin_labels

    def __len__(self) -> int:
        """Length method.

        Returns:
            Number of images in the dataset.
        """

        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Returns one transformed image and its binary label.

        Args:
            index: Index of the image and label to obtain.

        Returns:
            Image and label corresponding to the index.
        """

        image, label = self.dataset[index]
        return image, label


def _get_random_splits(
    total_samples: int, first_split_samples: int, second_split_samples: int
) -> tuple[list[int], list[int]]:
    """Creates two non-overlapping random sets of indices.

    Args:
        total_samples: Total number of available samples.
        first_split_samples: Number of samples in the first split.
        second_split_samples: Number of samples in the second split.

    Returns:
        Indices for the first and second random splits.
    """

    indices = random.sample(  # nosec B311
        range(total_samples), first_split_samples + second_split_samples
    )
    first_split = indices[:first_split_samples]
    second_split = indices[first_split_samples:]

    return first_split, second_split


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Calculates inverse-frequency weights from the training labels.

    Args:
        train_loader: Training loader returned by `get_data_loaders`.

    Returns:
        Weight of each class.
    """

    subset = cast(Subset, train_loader.dataset)
    dataset = cast(OxfordPetsDataset, subset.dataset)
    train_labels = torch.tensor([dataset.labels[index] for index in subset.indices])
    counts = torch.bincount(
        train_labels, minlength=len(config.data.class_names)
    ).float()

    return counts.sum() / (len(counts) * counts)


def get_data_loaders(
    root: str = config.paths.data_dir,
    batch_size: int = config.data.batch_size,
    image_size: int = config.data.image_size,
    train_samples: int = config.data.train_samples,
    validation_samples: int = config.data.validation_samples,
    test_samples: int = config.data.test_samples,
    seed: int = config.data.seed,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the train, validation and test loaders.

    Args:
        root: Path to the folder to save the data.
        batch_size: Batch size.
        image_size: Size to which the image is resized.
        train_samples: Number of training samples.
        validation_samples: Number of validation samples.
        test_samples: Number of test samples.
        seed: Random seed used to select the samples.
        download: True to download the data in local, False otherwise.

    Returns:
        Train, validation and test loaders.
    """

    # Transformations
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)

    # Datasets
    train_data = OxfordPetsDataset(root, "trainval", train_transform, download)
    validation_data = OxfordPetsDataset(root, "trainval", eval_transform, download)
    test_data = OxfordPetsDataset(root, "test", eval_transform, download)

    # Select random subsets
    random.seed(seed)
    train_indices, validation_indices = _get_random_splits(
        len(train_data), train_samples, validation_samples
    )
    test_indices, _ = _get_random_splits(len(test_data), test_samples, 0)
    train_subset = Subset(train_data, train_indices)
    validation_subset = Subset(validation_data, validation_indices)
    test_subset = Subset(test_data, test_indices)

    # Loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader
