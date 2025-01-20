"""Label-creating utilities."""

from torch import Tensor, full, float32  # pylint: disable=no-name-in-module


def binary_labels(
    real: bool,
    batch_size: int,
    low_clip: float = 0.05,
    high_clip: float = 0.95,
) -> Tensor:
    """Create real or fake labels.

    Parameters
    ----------
    real : bool
        Boolean real/fake status of the labels.
    batch_size : int
        The batch size of the labels.
    low_clip : float
        Minimum value to clip the labels at.
    high_clip : float
        Maximum value to clip the labels at

    Returns
    -------
    Tensor
        The labels.

    """
    return full((batch_size, 1), int(real), dtype=float32).clip(low_clip, high_clip)
