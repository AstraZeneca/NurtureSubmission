#!/usr/bin/env python
"""Create the glomeruli-only segmentation masks for the KPMP patches."""
from typing import Dict, Any
import gc

from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from multiprocessing import Pool

from numpy import array, where

from skimage.io import imsave
from skimage.util import img_as_ubyte

from pandas import read_csv


from tiffslide import TiffSlide

from nurture_stain.kpmp_misc import bad_slides, classes


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Extract the KPMP segmentation masks.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "patch_dir",
        help="Parent directory the patches are written to.",
        type=Path,
    )

    parser.add_argument(
        "kpmp_metadata",
        help="Path to csv file with the KPMP metadata.",
        type=Path,
    )

    parser.add_argument(
        "mask_dir",
        help="Parent directory holding the WSI-level masks.",
        type=Path,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the patch-level masks in.",
        type=Path,
        default="mask-patches/",
    )

    parser.add_argument(
        "--mag",
        help="Magnification to save the patches at.",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--workers",
        help="Number of workers to extract the patches with.",
        type=int,
        default=6,
    )

    return parser.parse_args()


def write_mask_patch(patch_dict: Dict[str, Any]):
    """Write a single mask patch to file.

    Parameters
    ----------
    patch_dict : Dict[str, Any]
        Dictionary containing the parameters of the mask patch to be extracted.

    """
    with TiffSlide(patch_dict["wsi_mask_path"]) as slide:
        region = (
            slide.read_region(
                (patch_dict["left"], patch_dict["top"]),
                patch_dict["level"],
                (patch_dict["patch_size"], patch_dict["patch_size"]),
                as_array=True,
            )
            .clip(0, 1)
            .astype(bool)
        )
    slide.close()

    chans = tuple(where(list(map(lambda x: "glomeruli" in x, classes)))[0])
    region = img_as_ubyte(region[:, :, chans].any(axis=2))

    save_path = Path(
        patch_dict["base_dir"], "with-glom" if region.any() else "without-glom"
    )
    save_path /= patch_dict["sub_dir"]
    save_path /= patch_dict["patch_name"]

    save_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(save_path, region, check_contrast=False)
    gc.collect()


def _process_one_slide(  # pylint: disable=too-many-arguments
    slide_name: str,
    patch_dir: Path,
    wsi_mask_path: Path,
    mag: float,
    save_dir: Path,
    workers: int,
):
    """Extratch the patch-level masks from a single slide.

    Parameters
    ----------
    slide_name : str
        Name of the whole-slide image.
    patch_dir : Path
        Path to the parent directory of the patch file structure.
    wsi_mask_path : Path
        Path to the binary WSI-level mask.
    mag : float
        Magnification the patches were taken at.
    save_dir : Path
        Directory to save the patches in.
    workers : int
        Number of workers to extract the patches with.

    """
    slide_df = read_csv(
        patch_dir / str(slide_name) / f"patches-mag_{mag:.1f}.csv",
    )

    downsample = float((slide_df.slide_mag / slide_df.patch_mag).unique().squeeze())

    slide_df["wsi_mask_path"] = wsi_mask_path

    with TiffSlide(wsi_mask_path) as slide:
        diff = abs(array(slide.level_downsamples) - downsample)
        print(diff.argmin(), slide_name, wsi_mask_path.name, diff.min())

        slide_df["level"] = abs(array(slide.level_downsamples) - downsample).argmin()

    slide_df["patch_size"] = (slide_df.right - slide_df.left) // (2**slide_df.level)
    print(slide_df.patch_size.unique())

    # pylint: disable=line-too-long
    slide_df["base_dir"] = save_dir
    slide_df["sub_dir"] = Path(slide_name, "masks", f"mag_{mag:.1f}")

    slide_df["patch_name"] = slide_df.apply(
        lambda x: f"{slide_name}---[x={x.left},y={x.top},w={x.patch_size *(2 ** x.level)},h={x.patch_size * (2 ** x.level)}].png",
        axis=1,
    )

    keys = [
        "left",
        "top",
        "patch_size",
        "level",
        "slide_name",
        "base_dir",
        "sub_dir",
        "patch_name",
        "wsi_mask_path",
    ]
    use_dicts = slide_df[keys].apply(dict, axis=1).to_list()

    with Pool(workers) as pool:
        pool.map(write_mask_patch, use_dicts)


def extract_mask_patches(args: Namespace):
    """Extract mask patches corresponding to the KPMP patches.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    kpmp_metadata = read_csv(args.kpmp_metadata)

    for row in kpmp_metadata.itertuples():
        print(row.wsi_name, row.mask_name)
        if str(row.wsi_name) in bad_slides:
            continue

        _process_one_slide(
            str(row.wsi_name),
            args.patch_dir,
            args.mask_dir / row.mask_name,
            args.mag,
            args.save_dir,
            args.workers,
        )


if __name__ == "__main__":
    extract_mask_patches(_parse_command_line())
