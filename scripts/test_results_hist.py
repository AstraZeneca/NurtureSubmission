#!/usr/bin/env python
"""Plot a histogram of the test results."""

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pandas import read_csv

from numpy import histogram, linspace, diff


import matplotlib.pyplot as plt


from nurture_stain.plotting import figure_cleanup


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Plot a histogram of the test performance.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "test_dir",
        help="directory holding the output test data.",
        type=Path,
    )

    parser.add_argument(
        "--stain",
        help="Stain whose results we should look at.",
        type=str,
        default="VPAS",
    )

    parser.add_argument(
        "--metrics",
        help="Name of the metrics to use.",
        type=str,
        default=["precision", "recall", "dice"],
        nargs="*",
    )

    parser.add_argument(
        "--bins",
        help="Number of bins to use in the histogram.",
        type=int,
        default=50,
    )

    return parser.parse_args()


# pylint: disable=too-many-locals
def produce_plot(args: Namespace):
    """Produce a histogram of the performance metrics.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    sources = list(args.test_dir.glob("*"))
    colours = ["r", "k", "white"]

    labels = {"KPMP": "KPMP", "neptune": "Jayapandian et al.", "nurture": "NURTuRE"}

    figure, axes = plt.subplots(3, 1, figsize=(3.0, 4.5))

    for source_dir, colour in zip(sources, colours):

        metrics = read_csv(source_dir / "test-metrics.csv")
        metrics = metrics.loc[metrics.stain == args.stain]

        for axis, metric_name in zip(axes.ravel(), metrics):

            edges = linspace(0.0, 1.0, args.bins + 1)
            middles = edges[1:] - (0.5 * edges[1])

            print(
                f"Removed {metrics[metric_name].isna().sum()} from {source_dir.stem} {metric_name}"
            )

            to_hist = metrics[metric_name].loc[~metrics[metric_name].isna()]

            prob_density, _ = histogram(to_hist, bins=edges, density=True)

            axis.bar(
                middles,
                prob_density,
                width=edges[1],
                alpha=0.25,
                fc=colour,
                ec="k",
                lw=0.5,
                label=f"{labels[source_dir.stem]} (Mean: {to_hist.mean():.2f})",
            )

            axis.set_xlabel(
                metric_name.capitalize(),  # type: ignore
                labelpad=0.05,
            )
            axis.set_ylabel("Probability density")

            axis.set_xlim(left=0.0, right=1.0)

            axis.set_aspect(0.5 * diff(axis.get_xlim()) / diff(axis.get_ylim()))

            axis.legend()

    for axis, letter in zip(axes.ravel(), ["a", "b", "c"]):
        axis.text(0.1, 0.1, f"({letter})", transform=axis.transAxes)

    figure.tight_layout(pad=0.05)

    figure.savefig("test-hist.jpg", dpi=512)

    figure_cleanup(axes)


if __name__ == "__main__":
    produce_plot(_parse_command_line())
