"""Image-loadin transforms."""

from typing import List, Callable, Union, Tuple


from skimage.io import imread

from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    randn,
    randint,
    rand,
    randn_like,
)
from torch import from_numpy  # pylint: disable=no-name-in-module

from torchvision.transforms import Compose, ToTensor, RandomCrop  # type: ignore

from torchvision.transforms.functional import adjust_gamma  # type: ignore
from torchvision.transforms.functional import rotate  # type: ignore
from torchvision.transforms import ColorJitter  # type: ignore
from torchvision.transforms import GaussianBlur  # type: ignore
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


from torch_tools.torch_utils import target_from_mask_img


class RandomApplyAug:  # pylint: disable=too-few-public-methods
    """A callable object for randomly apply augmentations.

    Parameters
    ----------
    prob : float, optional
        The probability with which each agumentation is applied.

    Notes
    -----
    ``torchvision.transforms.RandomApply`` applies either all or none of
    one's augmentations. Here, we apply each agumentation with a given
    probality.

    """

    def __init__(self, apply_prob: float = 0.5):
        """Build ``RandomApplyAug``."""
        self._augs: List[Callable] = []
        self.apply_prob = apply_prob

        self._augs.append(
            lambda x: augment_gamma_uniform(
                x,
                lower=0.1,
                upper=1.9,
            )
        )

        self._augs.append(
            ColorJitter(
                hue=0.05,
                saturation=0.25,
                brightness=0.25,
            )
        )

        self._augs.append(GaussianBlur(kernel_size=5))
        self._augs.append(lambda x: gaussian_noise(x, 0.001))

    def __call__(self, image: Tensor) -> Tensor:
        """Randomly augment the tensor ``image``.

        Parameters
        ----------
        image : Tensor
            An RGB image of shape (C, H, W).

        Returns
        -------
        image : Tensor
            An augmented version of ``image``.

        """
        for aug, rand_num in zip(self._augs, rand(len(self._augs))):

            if rand_num.item() < self.apply_prob:
                image = aug(image)

        return image


def spatial_scale_jitter(
    inputs: Tensor,
    targets: Tensor,
    scale: float = 0.3,
) -> Tuple[Tensor, Tensor]:
    """Jitter the spatial scale of ``inputs`` and ``targets``.

    Parameters
    ----------
    inputs : Tensor
        A mini-batch of images.
    targets : Tensor
        A mini-batch of segmentation masks.
    scale : float, optional
        The scale factor to jitter the images by.

    Returns
    -------
    new_inputs : Tensor
        A resized version of ``inputs``.
    new_targets : Tensor
        A resized version of ``targets``.

    """
    scale = (1.0 - scale) + (2.0 * scale * rand((1,)).item())

    _, _, height, width = inputs.shape

    new_height = round(scale * height)
    new_width = round(scale * width)

    new_inputs = resize(
        inputs,
        (new_height, new_width),
        interpolation=InterpolationMode.BILINEAR,
    )

    new_targets = resize(
        targets,
        (new_height, new_width),
        interpolation=InterpolationMode.NEAREST,
    )

    return new_inputs, new_targets


def augment_gamma_normal(
    img_tensor: Tensor,
    std_dev: float = 0.1,
    lower: float = 0.5,
    upper: float = 1.5,
) -> Tensor:
    """Adjust gamma as a form of augmentation.

    Parameters
    ----------
    img_tensor : Tensor
        The image to augment.
    std_dev : float
        The standard deviation of the normal distribution we sample gamma
        from.
    lower : float, optional
        The lower limit of the gamma range.
    upper : float, optional
        The upper limit of the gamma range.

    Returns
    -------
    Tensor
        Gamma-augmented version of ``img_tensor``.

    """
    gamma = (randn(1) * std_dev) + 1.0
    gamma = gamma.clip(lower, upper)
    return adjust_gamma(img=img_tensor, gamma=gamma.item())


def augment_gamma_uniform(
    img_tensor: Tensor,
    lower: float = 0.5,
    upper: float = 1.5,
) -> Tensor:
    """Adjust gamma as a form of augmentation.

    Parameters
    ----------
    img_tensor : Tensor
        The image to augment.
    lower : float, optional
        The lower limit of the gamma range.
    upper : float, optional
        The upper limit of the gamma range.

    Returns
    -------
    Tensor
        Gamma-augmented version of ``img_tensor``.

    """
    gamma = (rand(1) * (upper - lower)) + lower

    return adjust_gamma(img=img_tensor, gamma=gamma.item())


def gaussian_noise(img_tensor: Tensor, var: float) -> Tensor:
    """Add gaussian noise to ``image``.

    Parameters
    ----------
    img_tensor : Tensor
        The image to be noised.
    var : float
        The variance of the noise.

    Returns
    -------
    Tensor
        A noisy version of ``image``.

    """
    noise = randn_like(img_tensor) * (var**0.5)

    return img_tensor + noise


def random_on_axis_rotation(img: Tensor) -> Tensor:
    """Rotate ``img`` by a random integer multiple of 90 degrees.

    Parameters
    ----------
    img : Tensor
        The image, to be rotated.

    Returns
    -------
    Tensor
        A rotated version of ``img``.

    """
    return rotate(img=img, angle=randint(4, (1,)).item() * 90.0)


def cycle_gan_transforms(training: bool, gamma_aug: bool = False) -> Compose:
    """Get image-loading and augmenting transforms.

    Parameters
    ----------
    training : bool
        Are these training (``True``) or inference (``False``) transforms?
    gamma_aug : bool, optional
        Should we include gamma augmentation?

    Returns
    -------
    Compose
        A composition of transforms.

    """
    tfm_list: List[Callable] = [imread, ToTensor()]

    if training is True:
        tfm_list.append(RandomCrop(256))
        tfm_list.append(random_on_axis_rotation)
        if gamma_aug is True:
            tfm_list.append(augment_gamma_normal)

    return Compose(tfm_list)


def segmentation_input_transforms(training: bool) -> Compose:
    """Return transforms for the segmentation inputs.

    Paramaters
    ----------
    training : bool
        Are these training or validation transforms?

    Returns
    -------
    Compose
        A composition of transforms for the inputs.

    """
    tfm_list: List[Callable] = [imread, ToTensor()]

    if training is True:
        tfm_list.append(RandomApplyAug())

    return Compose(tfm_list)


def _binary_target_from_mask(
    grey_mask: Tensor,
    low_clip: float = 0.01,
    high_clip: float = 0.99,
) -> Tensor:
    """Get one-hot-encoded target from ``grey_mask`` image.

    Parameters
    ----------
    grey_mask : Tensor
        Grey binary mask.
    low_clip : float, optional
        Low clip to appy for label smoothing.
    high_clip : float, optional
        High clip to apply for label smoothing.

    Returns
    -------
    Tensor
        3D version of mask with one-hot-encoding.

    """
    grey_mask = grey_mask.clip(0, 1)

    return target_from_mask_img(grey_mask, 2).clip(low_clip, high_clip)


def segmentation_target_transforms() -> Compose:
    """Return transforms for segmentation targets.

    Returns
    -------
    Compose
        Segmentation target transforms.

    """
    # pylint: disable=cell-var-from-loop,line-too-long
    tfm_list: List[Callable] = [
        imread,
        from_numpy,
        _binary_target_from_mask,
    ]
    return Compose(tfm_list)


def segmentation_both_transforms(training: bool) -> Union[Compose, None]:
    """Return transforms to apply to both the inputs and targets.

    Parameters
    ----------
    training : int
        Are these training (or validation) transforms.
    crop_size : int
        The size to crop the inputs and targets to.

    Returns
    -------
    Compose
        A composition of transforms, or None if validating.

    """
    if training is True:
        return Compose(
            [
                random_on_axis_rotation,
                RandomResizedCrop(
                    512,
                    scale=(0.666666, 1.33333),
                    ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

    return None
