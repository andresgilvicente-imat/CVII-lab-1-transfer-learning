"""Script to run the Project 1 experiments."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNet_Weights

from src.config import config
from src.data.dataset import get_class_weights, get_data_loaders
from src.evaluation.compare import (
    EvaluationResult,
    compare_results,
    evaluate_predictions,
)
from src.models.base import BaseImageClassifier
from src.models.baseline_cnn import BaseCNN
from src.models.googlenet import PretrainedGoogLeNet
from src.models.googlenet_from_scratch import GoogLeNetFromScratch
from src.techniques.base_technique import PredictionOutput
from src.techniques.freeze_last_layer import FreezeLastLayer
from src.techniques.use_features import UseFeatures
from src.train.loss import GoogLeNetLoss
from src.train.train import Trainer
from src.utils import (
    predict,
    predict_features,
    save_augmentations_figure,
    save_predictions_figure,
)


def _train_model(
    name: str,
    architecture: str,
    model: BaseImageClassifier,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    epochs: int,
) -> tuple[EvaluationResult, PredictionOutput]:
    """Trains a model and evaluates its best checkpoint.

    Args:
        name: Name of the experiment.
        architecture: Name of the model architecture.
        model: Model to train.
        loaders: Train, validation and test loaders.
        epochs: Number of epochs to train.

    Returns:
        Evaluation result and test predictions.
    """

    train_loader, validation_loader, test_loader = loaders
    weights_path = f"{config.paths.checkpoints_dir}{name}.pt"
    figure_path = f"{config.paths.results_dir}{name}_training.png"

    criterion = GoogLeNetLoss(get_class_weights(train_loader))
    trainer = Trainer(model, criterion)
    trained_model = trainer.fit(
        train_loader,
        validation_loader,
        epochs,
        path_weights=weights_path,
        path_figure=figure_path,
    )
    predictions = predict(trained_model, test_loader)

    result = evaluate_predictions(name, predictions, architecture, "end_to_end")
    return result, predictions


def _fine_tune_googlenet(
    model: PretrainedGoogLeNet,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    epochs: int,
) -> tuple[EvaluationResult, PredictionOutput]:
    """Fine-tunes the last convolution and classifier of GoogLeNet.

    Args:
        model: Model to train.
        loaders: Train, validation and test loaders.
        epochs: Number of epochs to train.

    Returns:
        Evaluation result and test predictions.
    """

    train_loader, validation_loader, test_loader = loaders
    technique = FreezeLastLayer(model)
    trained_model = technique.fit(train_loader, validation_loader, epochs)
    predictions = predict(trained_model, test_loader)

    result = evaluate_predictions(
        "googlenet_finetune",
        predictions,
        "Pretrained GoogLeNet",
        "last_conv_and_classifier",
    )
    return result, predictions


def _classify_googlenet_features(
    model: PretrainedGoogLeNet,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
) -> tuple[EvaluationResult, PredictionOutput]:
    """Trains a random forest with fixed GoogLeNet features.

    Args:
        model: Model to train.
        loaders: Train, validation and test loaders.

    Returns:
        Evaluation result and test predictions.
    """

    train_loader, validation_loader, test_loader = loaders
    technique = UseFeatures(
        model,
        n_estimators=(
            10 if config.quick else config.experiments.random_forest_estimators
        ),
        max_depth=config.experiments.random_forest_max_depth,
    )
    classifier = technique.fit(train_loader, validation_loader, epochs=1)
    predictions = predict_features(classifier, technique, test_loader)

    result = evaluate_predictions(
        "googlenet_features",
        predictions,
        "Pretrained GoogLeNet",
        "use_features",
    )
    return result, predictions


def run_experiments(
    experiments: tuple[str, ...] = config.experiments.experiments,
    epochs: int = 1 if config.quick else config.training.epochs,
    pretrained_weights: GoogLeNet_Weights | None = GoogLeNet_Weights.DEFAULT,
) -> list[EvaluationResult]:
    """Runs the selected experiments.

    Args:
        experiments: Names of the experiments to run.
        epochs: Maximum number of training epochs.
        pretrained_weights: GoogLeNet weights, or 'None' to avoid loading weights.

    Returns:
        Evaluation results for all the experiments.

    Raises:
        ValueError: If the name of the experiment is not correct.
    """

    torch.manual_seed(42)
    device = config.training.device
    loaders = get_data_loaders()

    results: list[EvaluationResult] = []
    model_predictions: dict[str, np.ndarray] = {}
    for experiment in experiments:
        print(f"Running {experiment} on {device}...")
        if experiment == "base_cnn":
            result, predictions = _train_model(
                experiment,
                "BaseCNN",
                BaseCNN(),
                loaders,
                epochs,
            )
        elif experiment == "googlenet_scratch":
            result, predictions = _train_model(
                experiment,
                "GoogLeNet scratch",
                GoogLeNetFromScratch(),
                loaders,
                epochs,
            )
        elif experiment == "googlenet_finetune":
            model = PretrainedGoogLeNet(weights=pretrained_weights)
            result, predictions = _fine_tune_googlenet(model, loaders, epochs)
        elif experiment == "googlenet_features":
            model = PretrainedGoogLeNet(weights=pretrained_weights)
            result, predictions = _classify_googlenet_features(model, loaders)
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
        results.append(result)
        model_predictions[experiment] = predictions.predictions

    save_predictions_figure(loaders[2], model_predictions, config.paths.results_dir)
    save_augmentations_figure(loaders[2], config.paths.augmentations_figure)

    return results


def main() -> None:
    """Runs all experiments and saves their comparison."""

    results = run_experiments()
    compare_results(results, config.paths.results_dir)

    for result in results:
        print(
            f"{result.name}: accuracy={result.accuracy:.3f}, "
            f"macro F1={result.macro_f1:.3f}"
        )
    print(f"Results saved to {config.paths.results_dir}")


if __name__ == "__main__":
    main()
