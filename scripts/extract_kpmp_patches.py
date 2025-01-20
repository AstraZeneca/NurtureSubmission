#!/usr/bin/env python
"""Extract patches from the Hubmap patches."""
from pathlib import Path
from time import perf_counter

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter


from nurture_stain.kpmp_misc import bad_slides, classes
from nurture_stain.patch_extraction_utils import extract_from_wsi


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Extract patches from the KPMP data set.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "image_dir",
        help="Directory containing the KPMP images.",
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
        "--target-mag",
        help="Magnification to extract the patches at.",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--stride",
        help="Stride to use when sampling the patches.",
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


def extract_patches(args: Namespace):
    """Extract patches from the hubmap slides.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.output_dir /= "KPMP"

    slide_paths = list(args.image_dir.glob("wsi/*.svs"))

    for slide_path in slide_paths:

        if slide_path.name in bad_slides:
            continue

        start_time = perf_counter()

        mask_path = slide_path.parent.with_name("masks")
        mask_path /= slide_path.with_suffix(".ome.tif").name

        if not (slide_path.is_file() and mask_path.is_file()):
            continue

        extract_from_wsi(slide_path, mask_path, args, classes)

        stop_time = perf_counter()

        print(
            f"Processed {slide_path.name} in {stop_time - start_time:.6f} seconds.",
            flush=True,
        )


if __name__ == "__main__":
    extract_patches(_parse_command_line())
