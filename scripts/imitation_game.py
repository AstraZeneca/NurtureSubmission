#!/usr/bin/env python
"""Make a grid of real and synthetic images to play the imitation game."""
from string import ascii_uppercase

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

import pandas as pd

from skimage.io import imread, imsave
from skimage.util import img_as_ubyte

import torch

from torchvision.transforms.functional import to_tensor  # type: ignore
from torchvision.utils import make_grid  # type: ignore

import matplotlib.pyplot as plt


def _parse_command_line() -> Namespace:
    """Parse the coomand-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Create a grid of real and fakes for the imitation game.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "real_dir",
        help="Directory holding the real image patches.",
        type=Path,
    )

    parser.add_argument(
        "fake_dir",
        help="Directory holding the fake image patches.",
        type=Path,
    )

    parser.add_argument(
        "--rng-seed",
        help="Random seed to use.",
        type=int,
        default=12321,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the output files int.",
        type=Path,
        default="imitation-game/",
    )

    parser.add_argument(
        "--ppr",
        help="Patches per row in the grid.",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--dpi",
        help="DPI to use saving the matplotlib figure.",
        type=int,
        default=1024,
    )

    return parser.parse_args()


def _prepare_metadata(args: Namespace) -> pd.DataFrame:
    """Prepare the files' metadata.

    Parameters
    ----------
    real_dir : Path
        The directory holding the real images.
    fake_dir : Path
        The directory holding the fake images.

    Returns
    -------
    DataFrame
        The patch-level metadata.

    """
    real = pd.DataFrame(
        columns=["patch_path"],
        data=list(args.real_dir.glob("*.png")),
    )
    real["fake"] = False

    fake = pd.DataFrame(
        columns=["patch_path"],
        data=list(args.fake_dir.glob("*.png")),
    )
    fake["fake"] = True

    metadata = pd.concat([real, fake], axis=0, ignore_index=True)
    metadata["patch_name"] = metadata.patch_path.apply(lambda x: x.name)

    metadata = metadata.sort_values(by="patch_name")

    return metadata.sample(frac=1.0, random_state=args.rng_seed).reset_index(drop=True)


def _save_single_img_grid(metadata: pd.DataFrame, args: Namespace):
    """Save the grid as a single image.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.
    args : Namespace
        Command-line arguments.

    """
    batch = torch.concat(  # pylint: disable=no-member
        metadata.patch_path.apply(
            lambda x: to_tensor(imread(x)).unsqueeze(0),
        ).to_list()
    )

    grid = img_as_ubyte(make_grid(batch, nrow=args.ppr).permute(1, 2, 0).numpy())

    imsave(args.save_dir / "images.png", grid)


def _save_matplotlib_grid(metadata: pd.DataFrame, args: Namespace):
    """Save the images as a matplotlib grid.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.
    args : Namespace
        Command-line arguments.

    """
    num_cols = args.ppr
    num_rows = len(metadata) // num_cols

    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols, num_rows),
    )

    for axis, df_row in zip(axes.ravel(), metadata.itertuples()):

        axis.imshow(imread(df_row.patch_path))

        axis.set_xticks([])
        axis.set_yticks([])

        axis.text(
            0.05,
            0.05,
            f"{df_row.ss_row}{df_row.ss_col }",
            transform=axis.transAxes,
        )

    figure.tight_layout(pad=0.0)
    figure.savefig(args.save_dir / "images.pdf", dpi=args.dpi)


def produce_grid(args: Namespace):
    """Produce the grid for the imitation game.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.save_dir.mkdir(exist_ok=True, parents=True)

    metadata = _prepare_metadata(args)

    metadata["ss_row"] = list(ascii_uppercase[: args.ppr] * (len(metadata) // args.ppr))
    metadata["ss_col"] = range(len(metadata))
    metadata["ss_col"] = metadata["ss_col"].floordiv(args.ppr) + 1

    metadata["real"] = ~metadata.fake

    _save_single_img_grid(metadata, args)

    _save_matplotlib_grid(metadata, args)

    metadata.to_csv(args.save_dir / "metadata.csv", index=False)


if __name__ == "__main__":
    produce_grid(_parse_command_line())
