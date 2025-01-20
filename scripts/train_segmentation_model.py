#!/usr/bin/env python
"""Train a simple glomeruli-detection model."""
from time import perf_counter
from typing import Optional, List, Dict

from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from torch import save
from torch.nn import Module

from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR

from torch.utils.data import DataLoader

from numpy import linspace

from pandas import DataFrame

from nurture_stain.segmentation_training_utils import (
    segmentation_dataloader,
    one_seg_epoch,
    DEVICE,
)

from nurture_stain.transforms import spatial_scale_jitter

from nurture_stain.plotting import (
    plot_segmentation_metrics,
    plot_segmentation_ground_truths,
    plot_segmentation_predictions,
)

from nurture_stain.hubmap_misc import frozen_img_names

from nurture_stain.models import SegmentationModel


def _parse_command_line() -> Namespace:
    """Parse the command-line argument.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Kidney segmentation experiment.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "patch_dir",
        help="Directory containing the patches and masks.",
        type=Path,
    )

    parser.add_argument(
        "--valid-frac",
        help="Fraction of slides to put in the validation set.",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--subset-frac",
        help="Fraction of patches per image to use.",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the data in.",
        type=Path,
        default="segmentation-output-data/",
    )

    parser.add_argument(
        "--bs",
        help="Mini-batch size to use.",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--loader-workers",
        help="Number of workers each data loader should use.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        help="Learning rate.",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--wd",
        help="Weight decay.",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--epochs",
        help="Number of training intervals to run for.",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--plot-batches",
        help="Number of batches to plot.",
        type=int,
        default=4,
    )

    return parser.parse_args()


def get_metadata(
    patch_dir: Path,
    subset_frac: Optional[float] = None,
) -> DataFrame:
    """Get the patch and mask metadata.

    Parameters
    ----------
    parent_dir : Path
        Parent directory of the patch file tree.
    subset_frac : float, optional
        The fraction of patches per parent image to use.

    Returns
    -------
    metadata : DataFrame
        Patch- and mask-level data.

    """
    metadata = DataFrame(
        columns=["patch_path"],
        data=list(patch_dir.glob("*/patches/*/*.png")),
    )

    metadata["mask_path"] = metadata.patch_path.apply(
        lambda x: Path(str(x).replace("/patches/", "/masks/"))  # type: ignore
    )

    metadata["source"] = metadata.patch_path.apply(
        lambda x: x.parent.parent.parent.name
    )

    metadata["parent_img"] = metadata.patch_path.apply(lambda x: x.name.split("---")[0])

    if subset_frac is not None:
        metadata = metadata.groupby("parent_img").sample(
            frac=subset_frac,
            random_state=123,
        )

    return metadata


def save_checkpoint(model: Module, target_path: Path):
    """Save the model's parameters.

    Parameters
    ----------
    model : Module
        The model to be checkpointed.
    target_path : Path
        The file path to save the weights to.

    """
    target_path.parent.mkdir(exist_ok=True, parents=True)
    save(model.state_dict(), target_path)


def _save_metrics(
    train_metrics: List[Dict[str, float]],
    valid_metrics: List[Dict[str, float]],
    save_dir: Path,
    fold_name: str,
):
    """Save and plot the segmentation performance.

    Parameters
    ----------
    train_metrics : List[Dict[str, float]]
        The training metrics.
    valid_metrics : List[Dict[str, float]]
        The validation metrics.
    save_dir : Path
        Path to the directory in which the files should be saved.
    fold_name : str
        Name of the cross-validation fold.

    """
    metrics = DataFrame(train_metrics).join(
        DataFrame(valid_metrics), lsuffix="_train", rsuffix="_valid"
    )

    metrics["epoch"] = linspace(1, len(metrics), len(metrics), dtype=int)
    metrics["fold"] = fold_name

    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / "metrics.csv"

    if save_path.exists():
        # Save the last row ony
        metrics.iloc[-1:].to_csv(
            save_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        metrics.to_csv(save_path, index=False)
    plot_segmentation_metrics(save_dir / "metrics.csv")


def train_single_cv_fold(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    args: Namespace,
    fold_name: str,
):
    """Train the model for a single cross-validation fold.

    Parameters
    ----------
    train_loader : DataLoader
        Training data loader.
    valid_loader : DataLoader
        Validation data loader.
    args : Namespace
        Command-line arguments.
    fold_name : str
        Name of the cross-validation fold.

    """
    save_dir = args.save_dir / fold_name

    model = SegmentationModel()
    model.to(DEVICE)
    print(model)

    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = PolynomialLR(
        optimizer=optimiser,
        total_iters=args.epochs,
        power=1,
    )

    train_mets, valid_metrics = [], []
    for epoch in range(args.epochs):

        print(scheduler.get_last_lr())

        model.to(DEVICE)

        start_time = perf_counter()

        train_mets.append(
            one_seg_epoch(
                model,
                train_loader,
                optim=optimiser,
                batch_tfms=spatial_scale_jitter,
            )
        )
        valid_metrics.append(one_seg_epoch(model, valid_loader))

        scheduler.step()

        for ldr, splt in zip([train_loader, valid_loader], ["train", "valid"]):

            plot_segmentation_predictions(
                ldr,
                model,
                args.plot_batches,
                save_dir / f"{splt}-preds/epoch-{epoch + 1}",
            )

        stop_time = perf_counter()
        print(
            f"Epoch {epoch + 1} time: {stop_time - start_time:.6f} seconds.",
            flush=True,
        )

        _save_metrics(train_mets, valid_metrics, save_dir.parent, fold_name)

        save_checkpoint(model, save_dir / f"checkpoints/{epoch + 1}.pth")


def train_model(args: Namespace):
    """Train a glomerular segmentation model.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.save_dir.mkdir(exist_ok=True, parents=True)

    metadata = get_metadata(args.patch_dir, subset_frac=args.subset_frac)

    metadata.to_csv(args.save_dir / "patches.csv", index=False)

    print(metadata.groupby(by="source").patch_path.nunique())

    for valid_fold, valid_split in metadata.groupby("source"):

        train_split = metadata.loc[metadata.source != valid_fold]
        where_frozen = valid_split.parent_img.isin(frozen_img_names)
        valid_split = valid_split.loc[~where_frozen]

        (args.save_dir / f"{valid_fold}").mkdir(exist_ok=True, parents=True)

        train_split.to_csv(
            args.save_dir / f"{valid_fold}/train_patches.csv",
            index=False,
        )

        valid_split.to_csv(
            args.save_dir / f"{valid_fold}/valid_patches.csv",
            index=False,
        )

        print("Training data")
        print(train_split.groupby(by=["source"]).patch_path.nunique())
        print("\n")

        print("Validation data")
        print(valid_split.groupby(by=["source"]).patch_path.nunique())

        train_loader = segmentation_dataloader(
            metadata=train_split,
            training=True,
            args=args,
        )

        valid_loader = segmentation_dataloader(
            metadata=valid_split,
            training=False,
            args=args,
        )

        iterator = zip([train_loader, valid_loader], ["train", "valid"])
        for loader, split in iterator:
            plot_segmentation_ground_truths(
                loader,
                args.plot_batches,
                args.save_dir / f"{valid_fold}/{split}-batches/",
            )

        train_single_cv_fold(
            train_loader,
            valid_loader,
            args,
            valid_fold,  # type: ignore
        )


if __name__ == "__main__":
    train_model(_parse_command_line())
