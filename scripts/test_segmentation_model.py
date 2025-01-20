#!/usr/bin/env python
"""Apply the glomerular segmentation model to the test set."""
from time import perf_counter

from typing import Dict

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pandas import DataFrame

from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.measure import find_contours  # pylint: disable=no-name-in-module


from torch.cuda import is_available
from torch import load, no_grad, Tensor, concat  # pylint: disable=no-name-in-module


from torchvision.transforms import Compose  # type: ignore

from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore


from numpy import ndarray, nan

import matplotlib.pyplot as plt

from nurture_stain.models import SegmentationModel
from nurture_stain.transforms import (
    segmentation_target_transforms,
    segmentation_input_transforms,
)

from nurture_stain.plotting import figure_cleanup


DEVICE = "cuda" if is_available() else "cpu"


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Apply the glomerular segmentation model to test data.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "test_dir",
        help="Dir containing the 'patches/' and 'masks/'.",
        type=Path,
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Directory holding the saved checkpoints.",
        type=Path,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the output data in.",
        type=Path,
        default="test-output",
    )

    parser.add_argument(
        "--folds",
        help="Names of the cross validation folds.",
        type=str,
        nargs="*",
        default=["KPMP", "hubmap", "neptune"],
    )

    return parser.parse_args()


def _get_metadata(parent_dir: Path) -> DataFrame:
    """Get the metadata for the patches and masks.

    Parameters
    ----------
    parent_dir : Path
        Directory holding the patche and mask directories.

    Returns
    -------
    metadata : DataFrame
        The patch-level metadata.

    """
    metadata = DataFrame(
        columns=["patch_path"],
        data=list((parent_dir).glob("*/*")),
    )

    where_mask = metadata.patch_path.apply(lambda x: "/masks/" in str(x))
    metadata = metadata.loc[~where_mask].reset_index(drop=True)

    metadata["stain"] = metadata.patch_path.apply(
        lambda x: x.parent.name.split("-")[0].upper()
    )

    metadata["mask_path"] = metadata.patch_path.apply(
        lambda x: x.with_name(f"{x.stem}.png")
    )
    metadata.mask_path = metadata.mask_path.apply(
        lambda x: Path(str(x).replace(f"/{x.parent.name}/", "/masks/"))  # type: ignore
    )

    where_exist = metadata.apply(
        lambda x: x.patch_path.is_file() and x.mask_path.is_file(), axis=1
    )

    metadata = metadata.loc[where_exist].reset_index(drop=True)

    return metadata


def _load_mask(mask_path: Path) -> ndarray:
    """Load the mask image.

    Parameters
    ----------
    mask_path : Path
        Path to the mask image.

    Returns
    -------
    mask : ndarray
        The mask image.

    """
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = mask.any(axis=2).astype(float)

    return mask


@no_grad()
def _make_prediction(
    model: SegmentationModel,
    patch_path: Path,
    tfms: Compose,
) -> Tensor:
    """Make a prediction with the segmentation model.

    Parameters
    ----------
    model : SegmentationModel
        The segmentation model.
    patch_path : Path
        Path to the input image.
    tfms : Compose
        Input transforms

    Returns
    -------
    Tensor
        The prediction.

    """
    model.eval()
    patch_tensor = tfms(patch_path).unsqueeze(0).to(DEVICE)
    return model(patch_tensor).squeeze(0).argmax(0).cpu()


def _plot_target_and_prediction(
    img_path: ndarray,
    target_path: ndarray,
    pred: Tensor,
    dice: float,
    save_path: Path,
):
    """Plot the target and the prediction.

    Parameters
    ----------
    img_path : ndarray
        The input image.
    target_path : ndarray
        The ground truth.
    pred : Tensor
        The predicted mask.
    dice : float
        The dice score.
    save_dir : Path
        Directory to save the images in.

    """
    figure, axis = plt.subplots(1, 1, figsize=(2.0, 2.0))

    axis.imshow(imread(img_path))

    pred_arr = pred.squeeze(0).numpy()

    for name, mask, marker in zip(
        ["Ground truth", "Prediction"],
        [imread(target_path), pred_arr],
        ["-r", "--g"],
    ):

        contours = find_contours(mask)
        for contour in contours[:-1]:
            axis.plot(
                contour[:, 1],
                contour[:, 0],
                marker,
                lw=1.0,
                label="__nolegend__",
            )

        if not len(contours) == 0:
            label = name if "Ground" in name else f"{name} (Dice {dice:.2f})"
            axis.plot(
                contours[-1][:, 1],
                contours[-1][:, 0],
                marker,
                lw=1.0,
                label=label,
            )

    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=True, loc="upper right")

    figure.tight_layout(pad=0.0)

    save_path.parent.mkdir(exist_ok=True, parents=True)

    figure.savefig(save_path, dpi=500.0)
    figure_cleanup(axis)


def _save_prediction(prediction: Tensor, save_path: Path):
    """Save the predicted mask as a binary image.

    Parameters
    ----------
    prediction : Tensor
        The predicted image.
    save_path : path
        The path to save the image to.

    """
    pred_img = img_as_ubyte(prediction.numpy().astype(bool))

    save_path.parent.mkdir(exist_ok=True, parents=True)

    imsave(save_path, pred_img, check_contrast=False)


def _endow_model_with_weights(model: SegmentationModel, weight_path: Path):
    """Endow the model with the weights saved at ``weight_path``.

    Parameters
    ----------
    model : SegmentationModel
        The instantiated segmentation model.
    weight_path : Path
        The path to the saved weights.

    """
    model.to("cpu")
    model.load_state_dict(load(weight_path, weights_only=True))
    model.to(DEVICE)
    model.eval()


def compute_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    """Compute the segmentation performance metrics.

    Parameters
    ----------
    pred : Tensor
        The prediction tensor.
    target : Tensor
        The target tensor.

    Returns
    -------
    Dict[str, float]
        The segmentation performance metrics

    """
    return {
        "precision": precision_score(
            target.flatten(),
            pred.flatten(),
            zero_division=nan,
        ),
        "recall": recall_score(
            target.flatten(),
            pred.flatten(),
            zero_division=nan,
        ),
        "dice": f1_score(
            target.flatten(),
            pred.flatten(),
            zero_division=nan,
        ),
    }


def test_model(args: Namespace):  # pylint: disable=too-many-locals
    """Test a segmentation model's performance.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    metadata = _get_metadata(args.test_dir)

    print(metadata)

    model = SegmentationModel()

    patch_tfms = segmentation_input_transforms(training=False)
    tgt_tfms = segmentation_target_transforms()

    args.save_dir.mkdir(exist_ok=True, parents=True)
    csv_path = args.save_dir / "test-metrics.csv"

    group_keys = ["patch_path", "stain"]

    for (patch_path, stain), frame in metadata.groupby(by=group_keys):

        tic = perf_counter()

        mask_path = frame.mask_path.item()

        tgt_tensor = tgt_tfms(mask_path).argmax(dim=0)
        preds = {}
        for fold in args.folds:
            checkpoint_path = args.checkpoint_dir / f"{fold}/22.pth"
            _endow_model_with_weights(model, checkpoint_path)
            preds[fold] = _make_prediction(model, patch_path, patch_tfms)

        preds["mode"] = (
            concat(
                tuple(map(lambda x: x.unsqueeze(0), preds.values())),
                dim=0,
            )
            .mode(dim=0)
            .values
        )

        metrics = []
        for fold, pred in preds.items():
            img_metrics = compute_metrics(pred, tgt_tensor)
            img_metrics["fold"] = fold
            img_metrics["img_name"] = patch_path.name
            img_metrics["stain"] = stain

            metrics.append(img_metrics)

            _save_prediction(
                pred,
                args.save_dir / f"pred-masks/{patch_path.stem}/{stain}/{fold}.png",
            )

            _plot_target_and_prediction(
                patch_path,
                mask_path,
                pred,
                img_metrics["dice"],
                args.save_dir / f"visuals/{patch_path.stem}/{stain}/{fold}.png",
            )

        if csv_path.exists():
            DataFrame(metrics).to_csv(
                csv_path,
                mode="a",
                index=False,
                header=False,
            )
        else:
            DataFrame(metrics).to_csv(csv_path, index=False)

        toc = perf_counter()

        print(f"Processed '{patch_path.name}' in {toc - tic :.6f} secs.")


if __name__ == "__main__":
    test_model(_parse_command_line())
