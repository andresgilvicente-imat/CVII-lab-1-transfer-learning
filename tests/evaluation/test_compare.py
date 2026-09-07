"""Tests for metrics and the two comparison views."""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.compare import compare_results, evaluate_predictions
from src.techniques.base_technique import PredictionOutput


def _output() -> PredictionOutput:
    """Creates a fake 'PredictionOutput' object."""

    return PredictionOutput(
        targets=np.array([0, 0, 1, 1]),
        predictions=np.array([0, 1, 1, 1]),
        probabilities=np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]),
        loss=0.4,
    )


def test_evaluate_predictions_calculates_common_metrics() -> None:
    """Known labels yield the expected accuracy and macro F1."""

    result = evaluate_predictions("known", _output(), "test", "end_to_end")

    assert result.accuracy == pytest.approx(0.75)
    assert result.macro_f1 == pytest.approx(0.7333333333)
    assert result.loss == pytest.approx(0.4)


def test_comparison_report_writes_csv_and_both_plots(
    tmp_path: Path,
) -> None:
    """Four runs become a three-model view and a two-technique view."""

    specifications = [
        ("base_cnn", "BaseCNN", "end_to_end"),
        ("googlenet_scratch", "GoogLeNet scratch", "end_to_end"),
        (
            "googlenet_finetune",
            "Pretrained GoogLeNet",
            "last_conv_and_classifier",
        ),
        ("googlenet_features", "Pretrained GoogLeNet", "use_features"),
    ]
    results = [
        evaluate_predictions(name, _output(), architecture, technique)
        for name, architecture, technique in specifications
    ]

    output_directory = str(tmp_path)
    compare_results(results, output_directory)

    metrics_path = os.path.join(output_directory, "metrics.csv")
    architecture_path = os.path.join(output_directory, "architecture_comparison.png")
    technique_path = os.path.join(
        output_directory, "pretrained_googlenet_technique_comparison.png"
    )

    assert os.path.exists(metrics_path)
    assert os.path.exists(architecture_path)
    assert os.path.exists(technique_path)
    with open(metrics_path, encoding="utf-8", newline="") as csv_file:
        assert len(list(csv.DictReader(csv_file))) == 4
