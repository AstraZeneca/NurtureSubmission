"""Tests for ``segmentation_training_utils``."""

from torch import eye, full, tensor  # pylint: disable=no-name-in-module

from nurture_stain.segmentation_training_utils import binary_seg_loss_weights


def test_with_balanced_fraction():
    """Test with equal numbers of positive and negatives."""
    targets = full((10, 32, 32), 1)
    targets[:, :, 16:] = 0

    targets = eye(2)[targets].permute(0, 3, 1, 2).float()

    weights = binary_seg_loss_weights(targets)

    assert (weights == 1.0).all(), "Weights should be one with balanced props."

    assert weights.shape == (10, 1, 32, 32)


def test_with_one_quarter_positive_fraction():
    """Test with equal numbers of positive and negatives."""
    targets = full((10, 32, 32), 0)
    targets[:, 16:, 16:] = 1

    targets = eye(2)[targets].permute(0, 3, 1, 2).float()

    weights = binary_seg_loss_weights(targets)

    msg = "Weights should be three with balanced props."

    assert (weights[:, :, 16:, 16:] == 3.0).all(), msg

    assert (weights[:, :, :-16, :-16] == 1.0).all(), "Otherwise 1"

    assert weights.shape == (10, 1, 32, 32)


def test_with_three_quarters_positive_fraction():
    """Test with equal numbers of positive and negatives."""
    targets = full((10, 64, 64), 1)
    targets[:, 32:, 32:] = 0

    targets = eye(2)[targets].permute(0, 3, 1, 2).float()

    weights = binary_seg_loss_weights(targets)

    msg = "Weights should be 1.0 / 3.0 with balanced props."

    is_correct = weights[:, :, :-32, :-32].isclose(
        tensor(0.3333),
        atol=0.0001,
        rtol=0.001,
    )

    assert is_correct.all(), msg

    assert (weights[:, :, 32:, 32:] == 1.0).all(), "Otherwise 1"
