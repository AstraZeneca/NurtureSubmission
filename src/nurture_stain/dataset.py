"""CycleGAN dataset object."""

from typing import List, Tuple
from pathlib import Path

from numpy import floor  # pylint: disable=no-name-in-module
from numpy.random import default_rng


from torch import Tensor  # pylint: disable=no-name-in-module
from torch.utils.data import Dataset
from torchvision.transforms import Compose  # type: ignore


class CycleGanDataset(Dataset):
    """PyTorch dataset object for training CycleGAN.

    Parameters
    ----------
    src_paths : List[Path]
        A list of paths to the source images.
    tgt_paths : List[Path]
        A list of paths to the target images.
    tfms : Compose
        A composition of transforms to prepare the images.
    rng_seed : int, optional
        A seed for Numpy's default random number generator.

    Notes
    -----
    When training CycleGAN, it is unlikely that you have the same number of
    images in the source and target distributions, so this dataset can be
    used to iterate over the source distribution in the usual way, and will
    return a randomly-chosen target image with each source image.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        src_paths: List[Path],
        tgt_paths: List[Path],
        src_tfms: Compose,
        tgt_tfms: Compose,
        rng_seed: int = 123,
    ):
        """Build ``CycleGanDataset``."""
        self._src = tuple(src_paths)
        self._tgt = tuple(tgt_paths)
        self._src_tfms = src_tfms
        self._tgt_tfms = tgt_tfms
        self._rng = default_rng(seed=rng_seed)

    def __len__(self) -> int:
        """Return the length of the data set.

        Returns
        -------
        int
            The number of items in the source domain.

        """
        return len(self._src)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return source image of index ``idx`` and a random target image.

        Parameters
        ----------
        idx : int
            The index of the item to return.

        Returns
        -------
        src_img : Tensor
            Image from the source domain.
        tgt_img : Tensor
            Image from the target domain.

        """
        src_img = self._src_tfms(self._src[idx])
        tgt_img = self._tgt_tfms(self._tgt[self._rng.integers(len(self._tgt))])
        return src_img, tgt_img


class BalancedDataset(Dataset):
    """A dataset that samples from multiple sources with equal probability.

    Parameters
    ----------
    src_paths : Tuple[Tuple[Path, ...], ...]
        The paths to the images from each source domain: each tuple should
        contain all of the patches from a single source.
    tgt_paths : Tuple[Tuple[Path, ...], ...]
        The paths to the images from each target source: each tuple should
        contain all of the patches from a single source.
    src_tfms : Compose
        A composition of transforms to apply to the source images.
    tgt_tfms : Compose
        A composition of transforms to apply to the target images.
    rng_seed : int, optional
        A seed for Numpy's default random number generator.
    length : int, optional
        A 'dummy' length value for the '__len__' method to return.

    Notes
    -----
    The rationale of this dataset is such that if you are training on images
    from multiple source hospitals, you will have a lot of staining and
    scanning heterogeneity, and therefore may wish to sample images from each
    source with equal probabilities, rather than letting the sources with more
    images dominate.

    This dataset assumes _unpaired_ images and targets.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        src_paths: Tuple[Tuple[Path, ...], ...],
        tgt_paths: Tuple[Tuple[Path, ...], ...],
        src_tfms: Compose,
        tgt_tfms: Compose,
        rng_seed: int = 123,
        length: int = 1000,
    ):
        """Build ``BalancedDataset``."""
        self._src_paths = src_paths
        self._tgt_paths = tgt_paths

        self._src_tfms = src_tfms
        self._tgt_tfms = tgt_tfms

        self._rng = default_rng(seed=rng_seed)

        self._length = length

    def _sample_one_item(self, paths: Tuple[Tuple[Path, ...], ...]) -> Path:
        """Sample one input item from one of the sources with equal prob.

        Parameters
        ----------
        paths : Tuple[Tuple[Path], ...]
            The paths from each of the sources.

        Returns
        -------
        Path
            The path to the sampled image.

        """
        src_idx = int(floor(self._rng.random() * len(paths)))
        patch_idx = int(floor(self._rng.random() * len(paths[src_idx])))

        return paths[src_idx][patch_idx]

    def __len__(self) -> int:
        """Return the (dummy) length of the dataset.

        Returns
        -------
        int
            The 'length' of the dataset.

        """
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return an input-target pair.

        Parameters
        ----------
        idx : int
            The item in the dataset to sample. Note, we actually sample
            randomly, so this variable is just to be consistent with PyTorch
            datasets.

        Returns
        -------
        Tensor
            A source image.
        Tensor
            A target image.

        Notes
        -----
        The inputs and targets are unpaired.

        """
        src_item = self._sample_one_item(self._src_paths)
        tgt_item = self._sample_one_item(self._tgt_paths)

        return self._src_tfms(src_item), self._tgt_tfms(tgt_item)
