"""Tests for ``src.nurture_stain.labels``."""

from nurture_stain.labels import binary_labels


def test_create_binary_labels_return_cols():
    """Test the correct number of columns gets returned."""
    labels = binary_labels(real=True, batch_size=10)
    assert labels.ndim == 2
    assert labels.shape[1] == 1

    labels = binary_labels(real=False, batch_size=20)
    assert labels.ndim == 2
    assert labels.shape[1] == 1


def test_create_binary_labels_return_rows():
    """Test the correct number of rows is returned."""
    labels = binary_labels(real=True, batch_size=10)
    assert labels.shape[0] == 10

    labels = binary_labels(real=False, batch_size=20)
    assert labels.shape[0] == 20


def test_create_binary_labels_return_values_no_smoothing():
    """Test the correct values are being returned."""
    for real in [True, False]:
        labels = binary_labels(
            real=real,
            batch_size=10,
            low_clip=0.0,
            high_clip=1.0,
        )
        assert (labels == real).all()


def test_create_binary_labels_return_values_with_smoothing():
    """Test the correct values are being returned."""
    clips = [0.05, 0.95]

    for real in [True, False]:
        labels = binary_labels(
            real=real,
            batch_size=10,
            low_clip=0.05,
            high_clip=0.95,
        )
        assert (labels == clips[int(real)]).all()
