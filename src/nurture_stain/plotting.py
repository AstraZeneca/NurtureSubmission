"""Plotting functions."""

from typing import Union

import shutil
from pathlib import Path

from string import ascii_lowercase

from pandas import read_csv
from numpy import array, sqrt, ndarray, diff

from torch import no_grad  # pylint: disable=no-name-in-module
from torch.nn import Module

from torch.utils.data import Dataset, DataLoader

from torchvision.utils import make_grid  # type: ignore

from skimage.util import img_as_ubyte  # type: ignore
from skimage.io import imsave  # type: ignore
from skimage.measure import find_contours  # pylint: disable=no-name-in-module


import matplotlib.pyplot as plt
from matplotlib import rcParams

from matplotlib.pyplot import Axes

from nurture_stain.cycle_gan_utils import DEVICE


plt.switch_backend("agg")
plt.style.use(Path(__file__).with_name("matplotlibrc"))
rcParams["text.usetex"] = bool(shutil.which("latex"))


_colors = array([(221, 28, 119), (201, 148, 199), (231, 225, 239)]) / 255.0


def plot_losses(csv_path: Path):  # pylint: disable=too-many-locals
    """Plot the losses.

    Parameters
    ----------
    csv_path : Path
        Path to the csv file holding the loss data.

    """
    data = read_csv(csv_path)

    splits = ["train", "valid"]
    keys = ["fwd", "bwd"]

    labels = [
        "disc. loss",
        "adv. loss",
        "ID loss",
        "cycle",
    ]

    figure, axes = plt.subplots(2, 4, figsize=(8.5, 4))

    # pylint: disable=cell-var-from-loop

    for split, axes_row in zip(splits, axes):
        split_df = data[
            list(filter(lambda x: f"_{split}" in x, data.keys())) + ["valid_source"]
        ]

        for idx, key in enumerate(keys):
            # Plot the discriminator losses
            axes_row[0].plot(
                data.num_steps,
                split_df[f"{key}_disc_{split}"],
                "-o",
                color=_colors[idx],
                label=key.capitalize(),
            )
            # Plot the adversarial losses
            axes_row[1].plot(
                data.num_steps,
                split_df[f"{key}_adv_{split}"],
                "-o",
                color=_colors[idx],
                label=key.capitalize(),
            )
            # Plot the ID loss
            axes_row[2].plot(
                data.num_steps,
                split_df[f"{key}_id_{split}"],
                "-o",
                color=_colors[idx],
                label=key.capitalize(),
            )

            # Plot the pixel-level cycle loss
            axes_row[3].plot(
                data.num_steps,
                split_df[f"{key}_cycle_{split}"],
                "-o",
                color=_colors[idx],
                label=key.capitalize(),
            )

        for axis, label in zip(axes_row.ravel(), labels):
            axis.set_ylabel(f"{split.capitalize()} {label}")
            axis.set_xlim(left=0, right=data.num_steps.max())

    for axis in axes[1].ravel():
        axis.set_xlabel("Step", labelpad=0)

    for axis in axes.ravel():
        axis.legend()
        axis.set_ylim(bottom=0.0)
        axis.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        x_lims, y_lims = axis.get_xlim(), axis.get_ylim()
        axis.set_aspect(abs(x_lims[0] - x_lims[1]) / abs(y_lims[0] - y_lims[1]))

    figure.tight_layout(pad=0.1)
    figure.savefig(csv_path.with_name("metrics.pdf"), dpi=500)

    figure_cleanup(axes)


@no_grad()
def save_preds(
    model: Module,
    data_set: Dataset,
    save_dir: Path,
    num_batches: int,
    batch_size: int,
):
    """Save ``num_batches`` worth of predictions.

    Parameters
    ----------
    model : Module
        The generator model.
    data_set : Dataset
        The image-yielding data set.
    save_dir : Path
        The directory to save the patches in.
    num_batches : int
        The number of batches to save to file.
    batch_size : int
        The mini-batch size.

    """
    model.eval()
    model.to(DEVICE)
    save_dir.mkdir(exist_ok=True, parents=True)

    loader = DataLoader(
        data_set,
        num_workers=0,
        shuffle=False,
        batch_size=batch_size,
    )

    idx = 0
    n_row = int(sqrt(batch_size)) if (sqrt(batch_size) % 1) == 0 else 8

    for batch, _ in loader:
        pred = model(batch.to(DEVICE)).cpu().clip(0.0, 1.0)

        input_grid = make_grid(batch, nrow=n_row).permute(1, 2, 0).numpy()
        pred_grid = make_grid(pred, nrow=n_row).permute(1, 2, 0).numpy()

        _input_prediction_compare(
            input_grid,
            pred_grid,
            save_dir / f"{idx}.pdf",
        )

        idx += 1

        if not idx < num_batches:
            break


def _input_prediction_compare(
    input_grid: ndarray,
    pred_grid: ndarray,
    save_path: Path,
):
    """Save a figure comparing the inputs and the predictions.

    Parameters
    ----------
    input_grid : ndarray
        An image grid on the inputs.
    pred_grid : ndarray
        An image grid of the predictions.
    save_path : Path
        The target path to save the figure to.

    """
    figure, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(input_grid)
    axes[1].imshow(pred_grid)

    labels = ["(a) — Input", "(b) — Prediction"]
    for label, axis in zip(labels, axes.ravel()):
        axis.text(
            0.1,
            0.9,
            label,
            transform=axis.transAxes,
            bbox={
                "facecolor": "white",
            },
        )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout(pad=0.05, w_pad=0.5)
    figure.savefig(save_path, dpi=500)

    figure_cleanup(axes)


def save_batch_visuals(
    data_set: Dataset,
    save_dir: Path,
    num_batches: int,
    batch_size: int,
):
    """Save visualisations of the batches in ``data_loader``.

    Parameters
    ----------
    data_set : Dataset
        Image-yielding data set.
    save_dir : Path
        Directory to save the images in.
    num_batches : int
        The number of batches to save to file.
    batch_size : int
        Mini-batch size

    """
    save_dir.mkdir(exist_ok=True, parents=True)

    loader = DataLoader(
        data_set,
        num_workers=0,
        shuffle=True,
        batch_size=batch_size,
    )

    idx = 0

    n_row = int(sqrt(batch_size)) if (sqrt(batch_size) % 1) == 0 else 8

    for _, batch in loader:
        grid = make_grid(batch, nrow=n_row).permute(1, 2, 0).numpy()
        grid = img_as_ubyte(grid)
        imsave(save_dir / f"{idx}.png", grid, check_contrast=False)
        idx += 1

        if not idx < num_batches:
            break


def plot_segmentation_metrics(csv_path: Path):
    """Plot the segmentation metrics.

    Parameters
    ----------
    csv_path : Path
        Path to the metrics-containing csv file.

    """
    metrics = read_csv(csv_path)

    metric_names = ["loss", "precision", "recall", "dice"]
    splits = ["train", "valid"]

    figure, axes = plt.subplots(2, len(metric_names), figsize=(2.25 * 4, 2 * 2.25))

    for metric_idx, metric_name in enumerate(metric_names):

        for split_idx, split in enumerate(splits):

            for idx, (fold, frame) in enumerate(metrics.groupby(by="fold")):

                axes[split_idx, metric_idx].plot(
                    frame.epoch,
                    frame[f"{metric_name}_{split}"],
                    "-o",
                    color=_colors[idx],
                    label=f"{fold.capitalize()}",  # type: ignore
                )

            axes[split_idx, 0].set_ylabel(split.capitalize())
            axes[split_idx, metric_idx].set_ylabel(metric_name.capitalize())

    for idx, axis in enumerate(axes.ravel()):
        axis.text(
            0.1,
            0.9,
            f"({ascii_lowercase[idx]}) --- {splits[int(idx >= len(metric_names))].capitalize()}",
            transform=axis.transAxes,
        )

    for axis in axes.ravel():
        axis.set_ylim(bottom=0.0)
        axis.legend()

    for axis in axes[0, :].ravel():
        axis.set_xticklabels([])

    for axis in axes[1, :].ravel():
        axis.set_xlabel("Epoch")

    for axis in axes[:, 1:].ravel():
        axis.set_ylim(bottom=0.0, top=1.0)
    for axis in axes.ravel():
        axis.set_aspect(diff(axis.get_xlim()) / diff(axis.get_ylim()))

    figure.tight_layout(pad=0.5, w_pad=0.5)
    figure.savefig(csv_path.with_suffix(".pdf"), dpi=500)

    figure_cleanup(axes)


def plot_segmentation_ground_truths(
    loader: DataLoader, num_batches: int, save_dir: Path
):
    """Overlay the segmentation ground truths on the images.

    Parameters
    ----------
    loader : DataLoader
        Image-target yielding data loader.
    num_batches : int
        The number of batches to save to file.
    save_dir : Path
        The directory to save the images in.

    """
    save_dir.mkdir(exist_ok=True, parents=True)

    for idx, (batch, tgts) in enumerate(loader):

        if not idx < num_batches:
            break

        img_grid = make_grid(batch, nrow=len(batch)).permute(1, 2, 0)
        target_grid = make_grid(tgts, nrow=len(batch)).argmax(dim=0).squeeze().numpy()

        height = 1.0
        width = len(batch)

        figure, axis = plt.subplots(1, 1, figsize=(width, height))
        axis.imshow(img_grid.clip(0.0, 1.0))

        for poly in find_contours(target_grid):
            axis.plot(poly[:, 1], poly[:, 0], "-", color="yellow", lw=1.0)

        axis.set_xticks([])
        axis.set_yticks([])

        save_path = save_dir / f"batch-{idx}.pdf"

        figure.tight_layout(pad=0.01)
        figure.savefig(save_path, dpi=250)

        figure_cleanup(axis)


def plot_segmentation_predictions(  # pylint: disable=too-many-locals
    loader: DataLoader,
    model: Module,
    num_batches: int,
    save_dir: Path,
):
    """Plot the segmentation predictions and targets.

    Parameters
    ----------
    loader : DataLoader
        Image-target yielding data loader.
    model : Module
        The segmentation model.
    num_batches : int
        The number of batches to generate images for.
    save_dir : Path
        The directory to save the images in.

    """
    model.eval()
    model.to("cpu")

    save_dir.mkdir(exist_ok=True, parents=True)

    for idx, (batch, tgts) in enumerate(loader):

        if not idx < num_batches:
            break

        img_grid = make_grid(batch, nrow=len(batch)).permute(1, 2, 0)
        tgt_grid = make_grid(tgts, nrow=len(batch)).argmax(dim=0).squeeze().numpy()
        pred_grid = make_grid(model(batch).softmax(dim=1), nrow=len(batch))
        pred_grid = pred_grid.argmax(dim=0).numpy()

        height = 1.0
        width = len(batch)

        figure, axis = plt.subplots(1, 1, figsize=(width, height))
        axis.imshow(img_grid.clip(0.0, 1.0))

        for binary, style in zip([tgt_grid, pred_grid], ["-g", ":r"]):
            for poly in find_contours(binary):
                axis.plot(poly[:, 1], poly[:, 0], style, lw=1.0)

        axis.set_xticks([])
        axis.set_yticks([])

        save_path = save_dir / f"batch-{idx}.pdf"

        figure.tight_layout(pad=0.01)
        figure.savefig(save_path, dpi=250)

        figure_cleanup(axis)


def figure_cleanup(ax_obj: Union[Axes, ndarray]):
    """Clean up after matplotlib.

    Parameters
    ----------
    ax_obj : Axis or ndarray
        A single axis object or an array of them.

    """
    if isinstance(ax_obj, Axes):
        ax_obj.cla()
        ax_obj.remove()
    else:
        for axis in ax_obj.ravel():
            axis.cla()
            axis.remove()

    plt.close("all")


if __name__ == "__main__":
    plot_segmentation_metrics(Path("segmentation-output-data/metrics.csv"))
