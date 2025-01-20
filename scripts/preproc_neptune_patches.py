#!/usr/bin/env python
"""Pre-process the neptune test patches."""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pathlib import Path


from skimage.io import imread, imsave
from skimage.transform import rescale  # pylint: disable=no-name-in-module
from skimage.util import img_as_ubyte


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Preprocess the npetune test patches.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "image_dir",
        help="Directory holding the image data.",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        help="Directory to save the images in.",
        type=Path,
        default="processed-test-patches",
    )

    parser.add_argument(
        "--scale",
        help="Scale factor to apply to patches and masks.",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def preprocess_patches(args: Namespace):
    """Preprocess all patches.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    patch_iter = args.image_dir.glob("*.tif")
    patch_iter = filter(lambda x: "mask" not in x.name, patch_iter)

    for patch_path in patch_iter:

        mask_path = patch_path.with_name(f"{patch_path.stem}_mask.png")
        mask_path = Path(str(mask_path).replace("/patches/", "/masks/"))

        patch = imread(patch_path)
        mask = imread(mask_path)

        patch = img_as_ubyte(rescale(patch, args.scale, channel_axis=2))
        mask = img_as_ubyte(rescale(mask, args.scale, order=0).astype(bool))

        for image, folder in zip([patch, mask], ["he-patches", "masks"]):
            save_path = args.out_dir / f"{folder}/{patch_path.stem}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            imsave(save_path, image)


if __name__ == "__main__":
    preprocess_patches(_parse_command_line())
