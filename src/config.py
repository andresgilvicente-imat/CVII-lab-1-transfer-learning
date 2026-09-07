"""Script to define the configuration of the project."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the project."""

    data_dir: str = "data/"
    checkpoints_dir: str = "weights/"
    results_dir: str = "images/results/"
    augmentations_figure: str = "images/results/data_augmentation.png"


@dataclass(frozen=True)
class DataConfig:
    """Oxford-IIIT Pet preprocessing and loading configuration."""

    image_size: int = 200
    batch_size: int = 32
    train_samples: int = 1_000
    validation_samples: int = 500
    test_samples: int = 500
    seed: int = 42
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    class_names: tuple[str, str] = ("cat", "dog")


@dataclass(frozen=True)
class ModelConfig:
    """Shared model configuration."""

    classifier_hidden_dimensions: tuple[int, ...] = (256,)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and early-stopping settings."""

    # Normal training
    epochs: int = 30
    learning_rate: float = 1e-3
    optimizer = torch.optim.Adam
    # Fine-tuning
    fine_tune_learning_rate: float = 1e-4
    classifier_learning_rate: float = 1e-3
    # Early stopping
    patience: int = 5
    delta: float = 0.005
    # Device
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ExperimentsConfig:
    """Selection and classical-classifier settings for the four experiments."""

    experiments: tuple[str, ...] = (
        "base_cnn",
        "googlenet_scratch",
        "googlenet_finetune",
        "googlenet_features",
    )
    random_forest_estimators: int = 100
    random_forest_max_depth: int | None = 7


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    experiments: ExperimentsConfig = ExperimentsConfig()
    quick: bool = False


config = Config()
