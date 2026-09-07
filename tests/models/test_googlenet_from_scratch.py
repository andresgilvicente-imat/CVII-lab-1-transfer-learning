"""Tests for GoogLeNet implemented from scratch."""

import torch
from torch import nn

from src.models.baseline_cnn import BaseCNN
from src.models.googlenet_from_scratch import (
    AuxiliaryClassifier,
    ConvBlock,
    GoogLeNetFromScratch,
    InceptionBlock,
    InceptionBlockConfig,
)


def _inception_block() -> InceptionBlock:
    """Returns the first canonical GoogLeNet Inception block."""

    return InceptionBlock(InceptionBlockConfig(192, 64, 96, 128, 16, 32, 32))


def test_inception_block_uses_expected_branches() -> None:
    """The branches use the expected reductions, kernels, and pooling."""

    block = _inception_block()
    branch_1_convolution = block.branch_1[0]
    branch_2_reduction_block = block.branch_2[0]
    branch_2_convolution_block = block.branch_2[1]
    branch_3_reduction_block = block.branch_3[0]
    branch_3_convolution_block = block.branch_3[1]
    branch_4_pool = block.branch_4[0]
    branch_4_convolution_block = block.branch_4[1]

    assert isinstance(branch_1_convolution, nn.Conv2d)
    assert isinstance(branch_2_reduction_block, ConvBlock)
    assert isinstance(branch_2_convolution_block, ConvBlock)
    assert isinstance(branch_3_reduction_block, ConvBlock)
    assert isinstance(branch_3_convolution_block, ConvBlock)
    assert isinstance(branch_4_pool, nn.MaxPool2d)
    assert isinstance(branch_4_convolution_block, ConvBlock)

    branch_2_reduction = branch_2_reduction_block[0]
    branch_2_convolution = branch_2_convolution_block[0]
    branch_3_reduction = branch_3_reduction_block[0]
    branch_3_convolution = branch_3_convolution_block[0]
    branch_4_convolution = branch_4_convolution_block[0]

    assert isinstance(branch_2_reduction, nn.Conv2d)
    assert isinstance(branch_2_convolution, nn.Conv2d)
    assert isinstance(branch_3_reduction, nn.Conv2d)
    assert isinstance(branch_3_convolution, nn.Conv2d)
    assert isinstance(branch_4_convolution, nn.Conv2d)
    assert branch_4_pool.kernel_size == 3

    convolutions = (
        branch_1_convolution,
        branch_2_reduction,
        branch_2_convolution,
        branch_3_reduction,
        branch_3_convolution,
        branch_4_convolution,
    )
    assert [layer.kernel_size for layer in convolutions] == [
        (1, 1),
        (1, 1),
        (3, 3),
        (1, 1),
        (5, 5),
        (1, 1),
    ]
    assert [layer.out_channels for layer in convolutions] == [
        64,
        96,
        128,
        16,
        32,
        32,
    ]


def test_inception_block_concatenates_its_branches() -> None:
    """The four branches are concatenated along the channel dimension."""

    block = _inception_block()
    block.eval()

    with torch.no_grad():
        output = block(torch.randn(2, 192, 16, 16))

    assert output.shape == (2, 256, 16, 16)


def test_googlenet_from_scratch_architecture_and_output() -> None:
    """The compact model returns main and auxiliary class logits."""

    model = GoogLeNetFromScratch(num_classes=3)
    cnn_blocks = (model.cnn_block_1, model.cnn_block_2, model.cnn_block_3)
    inception_blocks = [
        layer
        for block in cnn_blocks
        for layer in block
        if isinstance(layer, InceptionBlock)
    ]
    convolution_blocks = [
        layer for block in cnn_blocks for layer in block if isinstance(layer, ConvBlock)
    ]
    pooling_layers = [
        layer for layer in model.modules() if isinstance(layer, nn.MaxPool2d)
    ]

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
        _, evaluation_auxiliary_logits = model.forward_with_auxiliary(
            torch.randn(2, 3, 64, 64)
        )

    model.train()
    main_logits, auxiliary_logits = model.forward_with_auxiliary(
        torch.randn(2, 3, 64, 64)
    )

    assert all(isinstance(block, nn.Sequential) for block in cnn_blocks)
    assert len(inception_blocks) == 4
    assert len(convolution_blocks) == 3
    first_convolution = convolution_blocks[0][0]
    first_classifier_layer = model.classifier[0]
    assert isinstance(first_convolution, nn.Conv2d)
    assert isinstance(first_classifier_layer, nn.Linear)
    assert first_convolution.out_channels == 32
    assert first_classifier_layer.in_features == 160
    assert all(not layer.ceil_mode for layer in pooling_layers)
    assert output.shape == (2, 3)
    assert main_logits.shape == (2, 3)
    assert [logits.shape for logits in auxiliary_logits] == [(2, 3), (2, 3)]
    assert [logits.shape for logits in evaluation_auxiliary_logits] == [
        (2, 3),
        (2, 3),
    ]


def test_auxiliary_classifier_reduces_spatial_dimensions() -> None:
    """The auxiliary head pools its input before applying the classifier."""

    classifier = AuxiliaryClassifier(in_channels=96, num_classes=3)

    with torch.no_grad():
        output = classifier(torch.randn(2, 96, 14, 14))

    first_linear_layer = classifier.classifier[0]
    assert isinstance(first_linear_layer, nn.Linear)
    assert first_linear_layer.in_features == 96 * 4 * 4
    assert first_linear_layer.out_features == 128
    assert output.shape == (2, 3)


def test_scratch_models_have_similar_parameter_counts() -> None:
    """The two models trained from scratch have similar parameter counts."""

    baseline_parameters = sum(parameter.numel() for parameter in BaseCNN().parameters())
    googlenet_parameters = sum(
        parameter.numel() for parameter in GoogLeNetFromScratch().parameters()
    )

    relative_difference = (
        abs(baseline_parameters - googlenet_parameters) / baseline_parameters
    )

    assert relative_difference < 0.02
