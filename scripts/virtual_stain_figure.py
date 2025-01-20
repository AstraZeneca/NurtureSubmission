#!/usr/bin/env python
"""Create a figure visualising virtual staining."""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from string import ascii_lowercase

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # type: ignore


from skimage.io import imread

from nurture_stain.plotting import figure_cleanup


plt.switch_backend("TkAgg")


# pylint: disable=line-too-long


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Produce a visualisation of virtual staining.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "real_dir",
        help="Directory holding the real images.",
        type=Path,
    )

    parser.add_argument(
        "synth_dir",
        help="Directory holding the real images.",
        type=Path,
    )

    parser.add_argument(
        "--file-names",
        help="Names of the files to use in the figure.",
        type=str,
        default=[
            "4e536765-d18e-4167-a3a5-cb37dd6b8fb2_S-2311-011828_HE_1of2.svs---[x=65536,y=6144,w=2048,h=2048].png",
            "82247cf5-8177-4962-b287-45168f9042c4_S-2303-008864_HE_2of2.svs---[x=22528,y=8192,w=2048,h=2048].png",
            "6aea6322-b49c-4d07-bc72-34c56f081665_S-2311-014796_HE_2of2.svs---[x=10240,y=47104,w=2048,h=2048].png",
            "6c0a35e2-995c-42bc-b019-5068eb4205b2_S-2304-003076_HE_2of2.svs---[x=57344,y=6144,w=2048,h=2048].png",
            "7c542fa8-07ce-40d4-a993-be47180302f9_S-2006-003965_HE_2of2.svs---[x=16384,y=16384,w=2048,h=2048].png",
        ],
        nargs="*",
    )

    return parser.parse_args()


def produce_figure(args: Namespace):
    """Produce figure visualising virtual staining.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    real_files = list(map(lambda x: args.real_dir / x, args.file_names))
    synth_files = list(map(lambda x: args.synth_dir / x, args.file_names))

    figure, axes = plt.subplots(
        2,
        len(real_files),
        figsize=(8.27 + 0.2, 2.0 * 8.27 / len(real_files)),
    )

    for idx, (real, synth) in enumerate(zip(real_files, synth_files)):

        axes[0, idx].imshow(imread(real))
        axes[1, idx].imshow(imread(synth))

    for axis, letter in zip(axes.ravel(), ascii_lowercase):
        axis.set_xticks([])
        axis.set_yticks([])

        axis.text(
            0.05,
            0.875,
            f"({letter})",
            fontsize=13,
            transform=axis.transAxes,
            bbox={"facecolor": "white", "boxstyle": "round"},
        )

        scalebar = AnchoredSizeBar(
            axis.transData,
            50.0 / (2.0 * 0.2527),
            r"$50\ \mu\mathrm{m}$",
            "lower left",
            frameon=True,
            fontproperties={"size": 13},
            size_vertical=2,
        )
        axis.add_artist(scalebar)

    axes[0, 0].set_ylabel(r"H\&E", fontsize=12)
    axes[1, 0].set_ylabel(r"Synthetic PAS", fontsize=12)

    figure.tight_layout(pad=0.05, h_pad=0.0, w_pad=0.0)

    figure.savefig("virtual-stain-visuals.png", dpi=250)

    figure_cleanup(axes)


if __name__ == "__main__":
    produce_figure(_parse_command_line())
