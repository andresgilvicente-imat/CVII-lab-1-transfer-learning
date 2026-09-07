"""Evaluate the experiments and compare their results."""

import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.techniques.base_technique import PredictionOutput


@dataclass
class EvaluationResult:
    """Results obtained by one experiment."""

    name: str
    architecture: str
    technique: str
    loss: float
    accuracy: float
    macro_f1: float


def evaluate_predictions(
    name: str, output: PredictionOutput, architecture: str, technique: str
) -> EvaluationResult:
    """Calculates the metrics of one experiment.

    Args:
        name: Name of experiment.
        output: Prediction output for the test set.
        architecture: Name of the architecture used.
        technique: Name of the fine-tuning technique used.

    Returns:
        Corresponding 'EvaluationResult' object.
    """

    return EvaluationResult(
        name=name,
        architecture=architecture,
        technique=technique,
        loss=output.loss,
        accuracy=float(accuracy_score(output.targets, output.predictions)),
        macro_f1=float(f1_score(output.targets, output.predictions, average="macro")),
    )


def _plot_results(
    results: list[EvaluationResult], labels: list[str], path: str, title: str
) -> None:
    """Plot accuracy and macro F1 for a group of experiments.

    Args:
        results: Results for each architecture and fine-tuning technique.
        labels: Labels shown under the bars.
        path: Path where the figure is saved.
        title: Figure title.
    """

    positions = np.arange(len(results))
    width = 0.35

    figure, axis = plt.subplots()
    axis.bar(
        positions - width / 2,
        [result.accuracy for result in results],
        width,
        label="Accuracy",
    )
    axis.bar(
        positions + width / 2,
        [result.macro_f1 for result in results],
        width,
        label="Macro F1",
    )

    axis.set_xticks(positions, labels, rotation=15)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def compare_results(results: list[EvaluationResult], output_directory: str) -> None:
    """Save the metrics and the two project comparison plots.

    Args:
        results: Results for each architecture and fine-tuning technique.
        output_directory: Folder where the figures will be saved.
    """

    metrics_path = os.path.join(output_directory, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["name", "architecture", "technique", "loss", "accuracy", "macro_f1"]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.architecture,
                    result.technique,
                    result.loss,
                    result.accuracy,
                    result.macro_f1,
                ]
            )

    architecture_results = [
        result for result in results if result.technique != "use_features"
    ]
    architecture_path = os.path.join(output_directory, "architecture_comparison.png")
    _plot_results(
        architecture_results,
        [result.architecture for result in architecture_results],
        architecture_path,
        "Architecture comparison",
    )

    technique_results = [
        result for result in results if result.architecture == "Pretrained GoogLeNet"
    ]
    technique_path = os.path.join(
        output_directory, "pretrained_googlenet_technique_comparison.png"
    )

    _plot_results(
        technique_results,
        [result.technique for result in technique_results],
        technique_path,
        "Pretrained GoogLeNet techniques",
    )
