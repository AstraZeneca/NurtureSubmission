#!/usr/bin/env python
"""Produce a plot of the virtual staining on the validation images."""

from itertools import product

from pathlib import Path

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from skimage.io import imread

from torch import from_numpy  # pylint: disable=no-name-in-module
from torch import concat as torch_concat  # pylint: disable=no-name-in-module
from torchvision.utils import make_grid  # type: ignore

from numpy import concat

import matplotlib.pyplot as plt

from nurture_stain.plotting import figure_cleanup


def _parse_args() -> Namespace:
    """Parse the commnad-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Plot an image showing VS on validation imgs.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "img_dir",
        help="Directory holding all of the images.",
        type=Path,
    )

    return parser.parse_args()


def produce_plot(args: Namespace):
    """Produce the visualisation.

    Parameters
    ----------
    Namespace
        Command-line arguments.

    """
    structures = ["glomerulus", "tubule"]
    img_paths = dict(zip(structures, ({}, {})))  # type: ignore

    for struct, stain in product(structures, ["HE", "PAS"]):
        folder = args.img_dir / f"{struct}/{stain}"
        img_paths[struct][stain] = tuple(folder.glob("*.png"))

    for struct in structures:
        img_paths[struct]["S-PAS"] = tuple(
            map(
                lambda x: Path(str(x).replace("/HE/", "/virtual-pas/")),
                img_paths[struct]["HE"],
            )
        )

    figure, axes = plt.subplots(2, 1, figsize=(8.3, 5.525))

    for struct, axis in zip(structures, axes.ravel()):

        grids = []
        for stain in ["HE", "S-PAS", "PAS"]:
            paths = img_paths[struct][stain]

            imgs = from_numpy(
                concat(list(map(lambda x: imread(x)[None, :], paths)), axis=0)
            )
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = make_grid(imgs, nrow=len(paths)).permute(1, 2, 0)

            grids.append(imgs)

        full_grid = torch_concat(grids, dim=0)
        axis.imshow(full_grid, extent=(0.0, 10.0, 0.0, 3.0))
        axis.set_xticks([])
        axis.set_yticks([])

    labels = [r"(a) --- Glomeruli", r"(b) --- Tubuli"]

    for axis, label in zip(axes.ravel(), labels):
        axis.set_title(label, fontsize=14)

        axis.set_yticks([0.5, 1.5, 2.5])
        axis.set_yticklabels(["(iii) PAS", "(ii) VPAS", r"(i) H\&E"])

    figure.tight_layout(pad=0.01, h_pad=0.025)

    figure.savefig("valid-visuals.png", dpi=256)
    figure_cleanup(axes)


if __name__ == "__main__":
    produce_plot(_parse_args())
