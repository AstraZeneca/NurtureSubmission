#!/usr/bin/env python
"""Extract patches from the Hubmap patches."""
from pathlib import Path
from time import perf_counter

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from numpy import zeros, ndarray, array

from pandas import DataFrame, read_json

from skimage.io import imread, imsave
from skimage.draw import polygon2mask  # pylint: disable=no-name-in-module

from nurture_stain.patch_extraction_utils import extract_from_non_wsi


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Extract patches from the hubmap kidney data set.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "image_dir",
        help="Directory containing the Hubmap images.",
        type=Path,
    )

    parser.add_argument(
        "output_dir",
        help="Directory to save the patches in.",
        type=Path,
    )

    parser.add_argument(
        "--patch-size",
        help="Size of the patches to extract.",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--stride",
        help="Stride to use when sampling the patches.",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--positive-frac",
        help="The minimum fraction of positive pixels in each patch.",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def create_mask(mask_path: Path, height: int, width: int, img_name: str) -> ndarray:
    """Create a segmentation mask from the json file.

    Parameters
    ----------
    mask_path : Path
        Path to the json mask file.
    height : int
        Height of the image.
    width : int
        Width of the image.
    img_name : str
        name of the image file

    Returns
    -------
    mask : ndarray
        Binary segmentation mask.

    """
    start_time = perf_counter()

    mask_json = DataFrame(read_json(mask_path))

    mask = zeros((height, width), dtype=bool)

    for row in mask_json.itertuples():  # type: ignore

        for poly in row.geometry["coordinates"]:  # type: ignore

            mask = mask | polygon2mask((height, width), array(poly)[:, ::-1])  # type: ignore

    stop_time = perf_counter()

    print(f"Created mask for '{img_name}' in {stop_time - start_time:.6f} seconds.")

    return mask


def extract_patches_from_image_mask_pair(
    save_dir: Path,
    image_path: Path,
    mask: ndarray,
    case_id: str,
    patch_df: DataFrame,
):
    """Extract patches from ``image``.

    Parameters
    ----------
    save_dir : Path
        The directory to save the image in.
    image_path : Path or ndarray
        The path to the RGB image.
    case_id : str
        The ID of the case.
    patch_df : DataFrame
        Data frame holding the patch coords.

    """
    img = imread(image_path).squeeze()

    patch_df["size"] = patch_df.right - patch_df.left

    for row in patch_df.itertuples():
        file_name = (
            f"{case_id}---[x={row.left},y={row.top},h={row.size},w={row.size}].png"
        )

        img_patch = img[row.top : row.bottom, row.left : row.right]  # type: ignore
        mask_patch = mask[row.top : row.bottom, row.left : row.right]  # type: ignore

        glom_dir = "with-glomeruli" if mask_patch.sum() != 0 else "without-glomeruli"

        for patch, parent in zip([img_patch, mask_patch], ["images", "masks"]):
            save_path = save_dir / f"{parent}/{glom_dir}/{file_name}"
            save_path.parent.mkdir(exist_ok=True, parents=True)

            imsave(save_path, patch, check_contrast=False)


def extract_patches(args: Namespace):
    """Extract patches from the hubmap slides.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.output_dir /= "hubmap"

    metadata = DataFrame(
        columns=["img_path"],
        data=list(args.image_dir.glob("train/*.tiff")),
    )

    print(metadata)

    for row in metadata.itertuples():

        image = imread(row.img_path)  # type: ignore

        mask_path = row.img_path.with_suffix(".json")  # type: ignore

        mask = create_mask(
            mask_path,
            image.shape[0],
            image.shape[1],
            row.img_path.name,  # type: ignore
        )

        extract_from_non_wsi(image, mask, row.img_path.name, args)  # type: ignore


if __name__ == "__main__":
    extract_patches(_parse_command_line())
