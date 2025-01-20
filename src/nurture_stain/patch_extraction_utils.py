"""Patch-extraction utility functions for non-WSIs."""

from time import perf_counter
from itertools import product
from typing import Tuple
from pathlib import Path

from argparse import Namespace

from numpy import ndarray, where

from pandas import DataFrame

from skimage.util import img_as_ubyte
from skimage.transform import rescale  # pylint: disable=no-name-in-module
from skimage.io import imsave

from tiffslide import TiffSlide

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _correct_rows(coord_df: DataFrame, height: int):
    """Correct the row coordinates which go off the image's edge.

    Parameters
    ----------
    coord_df : DataFrame
        Data frame holding the coords.
    height : int
        Height of the image.

    """
    row_diffs = coord_df.bottom % height
    row_diffs = row_diffs * (row_diffs != coord_df.bottom).astype(int)
    for coord in ["top", "bottom"]:
        coord_df[coord] -= row_diffs


def _correct_cols(coord_df: DataFrame, width: int):
    """Correct the row coordinates which go off the image's edge.

    Parameters
    ----------
    coord_df : DataFrame
        Data frame holding the coords.
    width : int
        Height of the image.

    """
    col_diffs = coord_df.right % width
    col_diffs = col_diffs * (col_diffs != coord_df.right).astype(int)
    for coord in ["left", "right"]:
        coord_df[coord] -= col_diffs


def _create_coord_df(
    width: int,
    height: int,
    patch_size: int,
    stride: int,
) -> DataFrame:
    """Create a data frame holding the coordinates.

    Parameters
    ----------
    width : int
        The total width of the image (in pixels).
    height : int
        The total height of the image (in pixels).
    patch_size : int
        The patch size (in pixels).
    stride : int
        The stride when sampling the patches  (in pixels).

    Returns
    -------
    coord_df : DataFrame
        The coordinates of the patches.

    """
    coord_df = DataFrame(
        columns=["top", "left"],
        data=product(range(0, height, stride), range(0, width, stride)),
    )

    coord_df["bottom"] = coord_df["top"] + patch_size
    coord_df["right"] = coord_df["left"] + patch_size

    _correct_rows(coord_df, height)
    _correct_cols(coord_df, width)

    return coord_df.astype(int)


def extract_from_non_wsi(  # pylint: disable=too-many-locals
    image: ndarray,
    mask: ndarray,
    file_name: str,
    args: Namespace,
):
    """Extract patches from a single image.

    Parameters
    ----------
    image : ndarray
        The histological image.
    mask : ndarray
        The binary segmentation mask.
    file_name : str
        Name of the image file.
    args : Namespace
        Command-line arguments.

    """
    start_time = perf_counter()

    coord_df = _create_coord_df(
        image.shape[1],
        image.shape[0],
        args.patch_size,
        args.stride,
    )

    coord_df["frac"] = coord_df.apply(
        lambda row: _non_wsi_patch_positive_frac(
            mask,
            row.left,
            row.right,
            row.top,
            row.bottom,
        ),
        axis=1,
    )

    coord_df = coord_df.loc[coord_df.frac >= args.positive_frac]

    for row in coord_df.itertuples():

        left = row.left  # type: ignore
        top = row.top  # type: ignore

        patch_name = f"{file_name}---[x={left},y={top},h={args.patch_size},w={args.patch_size}].png"

        has_glom = bool(row.frac > 0.0)  # type: ignore

        patch = image[row.top : row.bottom, row.left : row.right]  # type: ignore
        mask_patch = mask[row.top : row.bottom, row.left : row.right]  # type: ignore

        mask_patch = mask_patch.any(axis=2) if mask.ndim == 3 else mask_patch
        mask_patch = mask_patch.astype(bool)

        if hasattr(args, "scale_factor"):
            patch = rescale(patch, args.scale_factor, channel_axis=2, order=1)
            mask_patch = rescale(mask_patch, args.scale_factor, order=0)

        _save_patch_mask_pair(
            patch,
            mask_patch,
            has_glom,
            args.output_dir,
            patch_name,
        )

    stop_time = perf_counter()

    print(
        f"Extracted patches in '{file_name}' in {stop_time - start_time:.6f} seconds.",
        flush=True,
    )


def _save_patch_mask_pair(
    patch: ndarray,
    mask: ndarray,
    has_glom: bool,
    save_dir: Path,
    patch_name: str,
):
    """Save a patch and its mask.

    Parameters
    ----------
    patch : ndarray
        An RGB patch.
    mask : ndarray
        The glomerular mask.
    has_glom : bool
        Whether the patch has a glomerulus or not.
    save_dir : Path
        Directory to save in.
    patch_name : str
        File name to save to.


    """
    glom_dir = "with-glom" if has_glom else "without-glom"

    save_path = save_dir / f"patches/{glom_dir}/{patch_name}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(save_path, img_as_ubyte(patch), check_contrast=False)

    save_path = save_dir / f"masks/{glom_dir}/{patch_name}"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(save_path, img_as_ubyte(mask), check_contrast=False)


def _get_wsi_mag(slide: TiffSlide) -> float:
    """Return the magnification of ``slide``.

    Parameters
    ----------
    slide : TiffSlide
        The WSI image object.

    Returns
    -------
    float
        The magnification of ``slide`` in the level-zero reference frame.

    """
    return float(slide.properties["tiffslide.objective-power"])


def _wsi_patch_positive_frac(  # pylint: disable=too-many-arguments
    mask_slide: TiffSlide,
    left: int,
    right: int,
    top: int,
    bottom: int,
    channel_inds: Tuple[int, ...],
):
    """Get the fraction of the mask patch which is positive.

    Parameters
    ----------
    mask_slide : TiffSlide
        The WSI mask.
    left : int
        The left-hand patch coord (L0 reference frame).
    right : int
        The right-hand patch coord (L0 reference frame).
    top : int
        The top patch coord (L0 reference frame).
    bottom : int
        The bottom patch coord (L0 reference frame).
    channel_inds : Tuple[int, ...]
        Tuple with the glomerulus-containing channels.

    """
    width = right - left
    height = bottom - top

    mask_patch = mask_slide.read_region(
        location=(left, top),
        level=0,
        size=(width, height),
        as_array=True,
    )

    mask_patch = mask_patch[:, :, channel_inds].clip(0, 1).astype(bool)
    mask_patch = mask_patch.any(axis=2)

    return mask_patch.mean()


def _non_wsi_patch_positive_frac(
    mask_img: ndarray,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> float:
    """Get the fraction of the mask containing positives.

    Parameters
    ----------
    mask_img : ndarray
        The mask image.
    left : int
        The left-hand patch coord.
    right : int
        The right-hand patch coord.
    top : int
        The top of the patch.
    bottom : int
        The bottom of the patch.

    """
    mask_patch = mask_img[top:bottom, left:right].clip(0, 1).astype(bool)

    if mask_patch.ndim == 3:
        mask_patch = mask_patch.any(axis=2)

    return mask_patch.mean()


def _size_check(to_check: ndarray, patch_size: int):
    """Raise an exception if ``to_check`` is the wrong size.

    Parameters
    ----------
    to_check : ndarray
        The image-like object to check.
    patch_size : int
        The length of a square patch.

    Raises
    ------
    RuntimeError
        If ``to_check`` is not of ``patch_size``.

    """
    expected = (patch_size, patch_size)
    if not to_check.shape[:2] == expected:
        msg = f"Patch spatial dims should be {(patch_size, patch_size)}, got {to_check.shape}."
        raise RuntimeError(msg)


def extract_from_wsi(  # pylint: disable=too-many-locals
    slide_path: Path,
    mask_path: Path,
    args: Namespace,
    classes: Tuple[str, ...],
):
    """Extract patches from the WSI at ``slide_path``.

    Parameters
    ----------
    slide_path : Path
        Path to the WSI.
    mask_path : Path
        Path to the WSI mask.
    args : Namespace
        Command-line arguments.
    indices : Tuple[int, ...]
        The indices to take the union over to create a binary mask.

    """
    wsi = TiffSlide(slide_path)
    wsi_mask = TiffSlide(mask_path)

    scale_factor = _get_wsi_mag(wsi) / args.target_mag

    chans = tuple(where(list(map(lambda x: "glomeruli" in x, classes)))[0])

    width, height = wsi.dimensions
    coord_df = _create_coord_df(
        width,
        height,
        int(args.patch_size * scale_factor),
        int(args.stride * scale_factor),
    )

    coord_df["frac"] = coord_df.apply(
        lambda row: _wsi_patch_positive_frac(
            wsi_mask,
            row.left,
            row.right,
            row.top,
            row.bottom,
            chans,
        ),
        axis=1,
    )

    coord_df = coord_df.loc[coord_df.frac >= args.positive_frac]

    level_zero_size = (
        int(args.patch_size * scale_factor),
        int(args.patch_size * scale_factor),
    )

    for row in coord_df.itertuples():

        mask_patch = wsi_mask.read_region(  # type: ignore
            location=(row.left, row.top),
            level=0,
            size=level_zero_size,
            as_array=True,
        )

        # pylint: disable=unsubscriptable-object
        mask_patch = mask_patch[:, :, chans].any(axis=2)

        has_glom = row.frac != 0.0

        patch = wsi.read_region(  # type: ignore
            location=(row.left, row.top),
            level=0,
            size=level_zero_size,
            as_array=True,
        )

        patch = rescale(patch, scale_factor**-1.0, channel_axis=2)
        mask_patch = rescale(mask_patch, scale_factor**-1.0, order=0)

        patch = img_as_ubyte(patch)
        mask_patch = img_as_ubyte(mask_patch)

        _size_check(patch, args.patch_size)
        _size_check(mask_patch, args.patch_size)

        _save_patch_mask_pair(
            patch,
            mask_patch,
            has_glom,
            args.output_dir,
            f"{slide_path.name}---[x={row.left},y={row.right},h={level_zero_size[0]},w={level_zero_size[1]}].png",  # pylint: disable=line-too-long
        )

    wsi.close()
    wsi_mask.close()
