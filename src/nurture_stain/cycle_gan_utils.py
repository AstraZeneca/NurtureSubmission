"""CycleGAN setup utils."""

from pathlib import Path
from typing import Dict, Optional, Union

from torch import save
from torch.nn import Module

from torch.utils.data import DataLoader

from torch.optim.optimizer import Optimizer
from torch.optim import RAdam
from torch.optim.lr_scheduler import PolynomialLR, LRScheduler

from torch.cuda import is_available


from pandas import DataFrame

from nurture_stain.models import Generator, PatchGanDisc
from nurture_stain.dataset import CycleGanDataset, BalancedDataset
from nurture_stain.transforms import cycle_gan_transforms


DEVICE = "cuda" if is_available() else "cpu"
print(f"Device set to {DEVICE}.")


def checkpoint_models(
    base_dir: Path,
    models: Dict[str, Module],
    file_name: str,
):
    """Save ``model``'s paremeters to file.

    Parameters
    ----------
    base_dir : Path
        The base directory to save the models in.
    models : Dict[str, Module]
        All of the models.
    file_name : str
        The file to save the parameters at.

    """
    for name, model in models.items():
        save_path = base_dir / f"{name}/{file_name}"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save(model.state_dict(), save_path)


def set_train_or_eval(models: Dict[str, Module], train: bool):
    """Set the models to train or eval model.

    Parameters
    ----------
    models : Dict[str, Module]
        All of the models.
    train : bool
        If ``True``, we call ``model.train()`` for each model, otherwise we
        call ``model.eval()``.

    """
    for _, model in models.items():
        if train is True:
            model.train()
        else:
            model.eval()


def zero_all_grads(optim: Optional[Dict[str, Optimizer]] = None):
    """Zero the gradients in all of the models.

    Parameters
    ----------
    optim : Dict[str, Optimizer], optional
        A dictionary holding the optimisers.

    """
    if optim is not None:
        for _, opt in optim.items():
            opt.zero_grad()


def zero_disc_grads(optim: Optional[Dict[str, Optimizer]] = None):
    """Zero the discriminators' gradients.

    Parameters
    ----------
    optim : Dict[str, Optimizer]
        A dictionary with all models' optimisers.

    """
    if optim is not None:
        for key, optimiser in optim.items():
            if "disc" in key:
                optimiser.zero_grad()


def create_cycle_gan_models() -> Dict[str, Module]:
    """Create the models.

    Returns
    -------
    models : Dict[str, Module]
        The models needed for the cycle-gan training.

    """
    models: Dict[str, Module] = {}

    for key in ["fwd_gen", "bwd_gen"]:
        models[key] = Generator()

    for key in ["fwd_disc", "bwd_disc"]:
        # models[key] = Discriminator()
        models[key] = PatchGanDisc()

    for _, model in models.items():
        model.to(DEVICE)

    return models


def create_optimisers(
    models: Dict[str, Module],
    learning_rate: float,
    weight_decay: float = 0.0,
) -> Dict[str, Optimizer]:
    """Create the optimisers for ``models``.

    Parameters
    ----------
    models : Dict[str, Module]
        Models to fit.
    learning_rate : float
        Learning rate for the optimiser.
    weight_decay : float, optional
        Weight decay for the optimisers.

    Returns
    -------
    Dict[str, Optimizer]
        Dictionary of optimisers.

    """
    optimisers: Dict[str, Optimizer] = {}

    optimisers["gens"] = RAdam(
        list(models["fwd_gen"].parameters()) + list(models["bwd_gen"].parameters()),
        learning_rate,
        weight_decay=weight_decay,
        betas=(0.5, 0.999),
    )

    optimisers["discs"] = RAdam(
        list(models["fwd_disc"].parameters()) + list(models["bwd_disc"].parameters()),
        learning_rate,
        weight_decay=weight_decay,
        betas=(0.5, 0.999),
    )

    return optimisers


def create_schedulers(
    optimsers: Dict[str, Optimizer],
    decay_iters: int,
) -> Dict[str, LRScheduler]:
    """Create learning-rate schedulers for each of the optimisers.

    Parameters
    ----------
    optimsers : Dict[str, RRAdam]
        Optimisers for each of the models.
    decay_iters : int
        The number of iterations to decay the learning rate over.

    Returns
    -------
    Dict[str, MultiStepLR]
        The schedulers for each of the optimers.

    """
    return {
        key: PolynomialLR(optim, total_iters=decay_iters, power=1)
        for key, optim in optimsers.items()
    }


def cycle_gan_data_set(
    src_frame: DataFrame,
    tgt_frame: DataFrame,
    training: bool,
) -> Union[CycleGanDataset, BalancedDataset]:
    """Create Pytorch dataset.

    Parameters
    ----------
    src_frame : DataFrame
        Patch-level metadata on the source domain.
    tgt_frame : DataFrame
        Patch-level metadata on the target domain.
    training : bool
        Whether or not we are training or validating.

    Returns
    -------
    DataSet
        Pytorch dataset object.

    """
    return BalancedDataset(
        src_paths=tuple(src_frame.groupby("source").patch_path.apply(tuple)),
        tgt_paths=tuple(tgt_frame.groupby("source").patch_path.apply(tuple)),
        src_tfms=cycle_gan_transforms(training=training, gamma_aug=True),
        tgt_tfms=cycle_gan_transforms(training=training, gamma_aug=True),
    )


def cycle_gan_data_loader(
    data_set: Union[CycleGanDataset, BalancedDataset],
    training: bool,
    batch_size: int,
    workers: int,
) -> DataLoader:
    """Create a ``DataLoader``.

    Parameters
    ----------
    data_set : DataSet
        PyTorch image-yielding dataset.
    training : bool
        Has the data loader to be for training or inference?
    batch_size : int
        Size of the mini-batches.
    workers : int
        The number of workers the data loader should use.

    """
    return DataLoader(
        data_set,
        shuffle=True,
        num_workers=workers,
        drop_last=training,
        batch_size=batch_size,
    )
