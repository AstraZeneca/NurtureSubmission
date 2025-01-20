#!/usr/bin/env python
"""Extract patches from the neptune paper data."""
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from pathlib import Path

from pandas import DataFrame

from skimage.io import imread

from nurture_stain.patch_extraction_utils import extract_from_non_wsi


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    The command-line arguments.

    """
    parser = ArgumentParser(
        description="Extract glomerular patches.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "img_dir",
        help="Directory holding the images and masks.",
        type=Path,
    )

    parser.add_argument(
        "output_dir",
        help="Directory to save the patches in.",
        type=Path,
    )

    parser.add_argument(
        "--scale-factor",
        help="Down- or up-scaling factor to apply to the patches.",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--patch-size",
        help="Size of the patches to extract.",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--stride",
        help="Stride to use in the patch extraction.",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--positive-frac",
        help="The minimum fraction of positive pixels in each patch.",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def _get_image_metadata(img_dir: Path) -> DataFrame:
    """Get the image and mask metadata.

    Parameters
    ----------
    img_dir : Path
        Path to the directory holding the images and masks.

    """
    metadata = DataFrame()

    all_imgs = img_dir.glob("*")

    metadata["img_path"] = list(filter(lambda x: "mask" not in x.name, all_imgs))

    metadata["mask_path"] = metadata.img_path.apply(
        lambda x: x.with_name(f"{x.stem}_mask_capsule.png")
    )

    where_exist = metadata.img_path.apply(lambda x: x.is_file())
    where_exist = where_exist & metadata.mask_path.apply(lambda x: x.is_file())

    return metadata.loc[where_exist].reset_index(drop=True)


def extract_from_all_images(args: Namespace):
    """Extract patches from all slides.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.output_dir /= "neptune"

    metadata = _get_image_metadata(args.img_dir)

    print(f"There are {len(metadata)} image-mask pairs.")

    for row in metadata.itertuples():

        image = imread(row.img_path)  # type: ignore
        mask = imread(row.mask_path)  # type: ignore

        extract_from_non_wsi(image, mask, row.img_path.name, args)  # type: ignore


if __name__ == "__main__":
    extract_from_all_images(_parse_command_line())
