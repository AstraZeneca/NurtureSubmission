"""Segmentation training utilities."""

from typing import Dict, Optional, List, Callable
from argparse import Namespace

from torch import (  # pylint: disable=no-name-in-module
    no_grad,
    Tensor,
    set_grad_enabled,
    ones_like,
)
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda import is_available, empty_cache
from torch.nn.functional import binary_cross_entropy

from torch_tools import DataSet

from pandas import DataFrame

from numpy import nanmean, nan

from sklearn.metrics import precision_score, recall_score, f1_score  # type: ignore


from nurture_stain.transforms import (
    segmentation_input_transforms,
    segmentation_target_transforms,
    segmentation_both_transforms,
)


DEVICE = "cuda" if is_available() else "cpu"


metric_functions = {
    "precision": precision_score,
    "recall": recall_score,
    "dice": f1_score,
}


def binary_seg_loss_weights(target: Tensor) -> Tensor:
    """Weights to multiply loss by in one-hot-encoded semantic segmentation.

    Should be used in conjunction with the ``reduction='none'`` in PyTorch
    loss functions.

    Parameters
    ----------
    target : Tensor
        The one-hot-encoded segmentation mask.

    Returns
    -------
    weights : Tensor
        The weight to apply to each pixel in the loss tensor.

    """
    # Binary targets of shape (N, 1, H, W)
    target = target.argmax(dim=1).unsqueeze(1).float()

    positive_frac = target.mean()

    negative_frac = 1.0 - positive_frac

    weights = ones_like(target).float()
    weights[target == 1] *= negative_frac / positive_frac

    return weights


def segmentation_dataloader(
    metadata: DataFrame,
    training: bool,
    args: Namespace,
) -> DataLoader:
    """Create a segmentation data loader.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.
    training : bool
        Is this a training or inference dataloader.
    args : Namespace
        Command-line arguments.

    Returns
    -------
    DataLoader
        The image-target yielding data loader.

    """
    data_set = DataSet(
        inputs=metadata.patch_path.to_list(),
        targets=metadata.mask_path.to_list(),
        input_tfms=segmentation_input_transforms(training),
        target_tfms=segmentation_target_transforms(),
        both_tfms=segmentation_both_transforms(training),
    )

    return DataLoader(
        data_set,
        shuffle=training,
        batch_size=args.bs,
        num_workers=args.loader_workers,
    )


@no_grad()
def compute_metrics(preds: Tensor, targets: Tensor) -> Dict[str, float]:
    """Compute the segmentation metrics between ``preds`` and ``targets``.

    Parameters
    ----------
    preds : Tensor
        The model's predictions.
    targets : Tensor
        The ground truths.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary holding the segmentation metrics for each class.

    """
    metrics: Dict[str, float] = {}

    preds = preds.argmax(dim=1).cpu().flatten()
    targets = targets.cpu().argmax(dim=1).flatten()

    for metric_name, func in metric_functions.items():
        metrics[metric_name] = func(targets, preds, zero_division=nan)

    return metrics


def one_seg_epoch(
    model: Module,
    loader: DataLoader,
    optim: Optional[Adam] = None,
    batch_tfms: Optional[Callable] = None,
) -> Dict[str, float]:
    """Train or validate ``model`` for a single epoch.

    Parameters
    ----------
    model : Module
        The segmentation model.
    loader : DataLoader
        Image-target yielding data loader.
    optim : Adam, optional
        The optimiser to fit the model with. Only supply if training.
    batch_tfms : Callable, optional
        Callable object which applies transforms on the mini-batch level.

    Returns
    -------
    Dict[str, float]
        Mean segmentation metrics.

    """
    metrics: Dict[str, List[float]] = {"loss": []}
    for metric in metric_functions:
        metrics[metric] = []

    _ = model.train() if optim is not None else model.eval()

    for batch, targets in loader:

        if optim is not None:
            optim.zero_grad()
            empty_cache()

        if batch_tfms is not None:
            batch, targets = batch_tfms(batch, targets)

        batch, targets = batch.to(DEVICE), targets.to(DEVICE)

        with set_grad_enabled(optim is not None):
            preds = model(batch)

        loss = binary_cross_entropy(preds, targets)

        if optim is not None:
            loss.backward()
            optim.step()
            optim.zero_grad()

        metrics["loss"].append(loss.item())
        for key, val in compute_metrics(preds, targets).items():
            metrics[key].append(val)  # type: ignore

    return {key: float(nanmean(val)) for key, val in metrics.items()}
