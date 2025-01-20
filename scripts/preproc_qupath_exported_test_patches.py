#!/usr/bin/env python
"""Pre-process the neptune test patches."""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pathlib import Path


from skimage.io import imread, imsave
from skimage.util import img_as_ubyte


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Preprocess the kpmp test patches.",
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
        "--positive-tol",
        help="Minimum fraction of the pixels which should be positive.",
        type=float,
        default=0.2,
    )

    return parser.parse_args()


def preprocess_patches(args: Namespace):
    """Preprocess all patches.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    patch_iter = args.image_dir.glob("*.jpg")
    patch_iter = filter(lambda x: "mask" not in x.name, patch_iter)

    for patch_path in patch_iter:

        mask_path = patch_path.with_name(f"{patch_path.stem}.png")
        mask_path = Path(str(mask_path).replace("/patches/", "/masks/"))

        patch = imread(patch_path)
        mask = imread(mask_path).clip(0, 1).any(axis=2)

        if mask.mean() < args.positive_tol:
            continue

        for image, folder in zip([patch, mask], ["he-patches", "masks"]):
            save_path = args.out_dir / f"{folder}/{patch_path.stem}.png"
            save_path.parent.mkdir(exist_ok=True, parents=True)
            imsave(save_path, img_as_ubyte(image))


if __name__ == "__main__":
    preprocess_patches(_parse_command_line())
