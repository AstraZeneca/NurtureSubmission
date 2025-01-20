#!/usr/bin/env python
"""Cycle Gan training script."""
from typing import Dict, Union
from pathlib import Path
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
from time import perf_counter

from pandas import read_csv, DataFrame

from numpy import log10, logspace, floor  # pylint: disable=no-name-in-module

from torch import manual_seed

from nurture_stain.metadata_utils import prepare_metadata

from nurture_stain.plotting import plot_losses

from nurture_stain.cycle_gan_training import (
    cycle_gan_interval,
    generator_warmup,
    discriminator_warmup,
)

from nurture_stain.cycle_gan_utils import (
    create_cycle_gan_models,
    create_optimisers,
    create_schedulers,
    cycle_gan_data_set,
    cycle_gan_data_loader,
    checkpoint_models,
)

from nurture_stain.dataset import CycleGanDataset, BalancedDataset

manual_seed(123)


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Train the Cycle Gan model.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "metadata_csv",
        help="Path to the Nurture metadata csv.",
        type=Path,
    )

    parser.add_argument(
        "patch_dir",
        help="Path to the Nurture patch dir.",
        type=Path,
    )

    parser.add_argument(
        "save_dir",
        help="Directory save the training output in.",
        type=Path,
    )

    parser.add_argument(
        "--source-stain",
        help="The stain type to be transformed.",
        type=str,
        default="H&E",
    )

    parser.add_argument(
        "--target-stain",
        help="The stain type to be transformed.",
        type=str,
        default="PAS",
    )

    parser.add_argument(
        "--warmup-steps",
        help="The number of steps to warm up the discriminators for.",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--training-steps",
        help="Number of mini-batch gradient descent steps to perform.",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--decay-lr-steps",
        help="Final steps to train for with a decaying learning rate.",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--l1_decay_steps",
        help="The number of steps to decay the L1 regularisation over.",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--interval",
        help="Frequency, in training steps, to sample metrics at.",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--mag",
        help="The patch-level magnification.",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--loader-workers",
        help="The number of workers the loader.",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--bs",
        help="Mini-batch size",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--lr",
        help="Learning rate to use in training.",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--wd",
        help="Weight decay to use in training.",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--pixel-l1-start",
        help="Initial L1 regularisation parameterfor the pixel-level loss.",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--pixel-l1-stop",
        help="The value to stop the pixel-level L1 at.",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--snapshot-batches",
        help="Number of batches to save images of when we extract metrics.",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--valid-sources",
        help="Source(s) to use as the validation set.",
        type=str,
        nargs="*",
        default=["Birmingham BCH NURTuRE", "Birmingham UHB NURTuRE"],
    )

    return parser.parse_args()


def create_all_datasets(
    data_frames: Dict[str, DataFrame],
) -> Dict[str, Union[CycleGanDataset, BalancedDataset]]:
    """Create all of the data samplers.

    Parameters
    ----------
    data_frames : Dict[str, DataFrame]
        All of the source/target train/valid data frames.

    Returns
    -------
    loaders : Dict[str, DataSet]
        Dictionary of training/validation source/target domain samplers.

    """
    data_sets: Dict[str, Union[CycleGanDataset, BalancedDataset]] = {}

    for split in ["train", "valid"]:
        data_sets[split] = cycle_gan_data_set(
            src_frame=data_frames[f"{split}_src"],
            tgt_frame=data_frames[f"{split}_tgt"],
            training="train" == split,
        )

    return data_sets


def write_metrics(
    train_metrics: Dict[str, float],
    valid_metrics: Dict[str, float],
    num_steps: int,
    valid_source: str,
    save_dir: Path,
):
    """Write the metrics to file during training.

    Parameters
    ----------
    train_metrics : Dict[str, float]
        Training metrics
    valid_metrics : Dict[str, float]
        Validation metrics
    num_steps : int
        Number of training steps.
    valid_source : str
        Name of the validation source.
    save_dir : Path
        Base directory to save the metrics and plots in.

    """
    metrics = DataFrame([train_metrics]).join(
        DataFrame([valid_metrics]),
        lsuffix="_train",
        rsuffix="_valid",
    )

    metrics["num_steps"] = num_steps
    metrics["valid_source"] = valid_source

    save_path = save_dir / "metrics.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        metrics.to_csv(save_path, mode="a", header=False, index=False)
    else:
        metrics.to_csv(save_path, index=False)


def reg_gen(start: float, stop: float, num_decays: int):
    """Regularisation parameter generator.

    Parameters
    ----------
    start : float
        Start value of the parameter.
    stop : float
        Stop value of the parameter.
    num_decays : int
        Number of decay steps.

    """
    arr_vals = logspace(log10(start), log10(stop), num_decays, dtype=float)
    vals = iter(arr_vals)

    print(arr_vals)

    while True:
        try:
            next_val = next(vals)
        except StopIteration:
            next_val = stop
        yield next_val


def train_model(  # pylint: disable=too-many-locals
    data_frames: Dict[str, DataFrame],
    valid_source_str: str,
    args: Namespace,
):
    """Train a single cross-validation fold.

    Parameters
    ----------
    data_frame : Dict[str, DataFrame]
        Training and validation source and target data frames.
    valid_source : List[str]
        The sources being used as the validation split.
    args : Namespace
        Command-line arguments.

    """
    valid_source_str = valid_source_str.replace("/", "---")

    data_sets = create_all_datasets(data_frames)

    train_loader = cycle_gan_data_loader(
        data_sets["train"],
        training=True,
        batch_size=args.bs,
        workers=args.loader_workers,
    )

    valid_loader = cycle_gan_data_loader(
        data_sets["valid"],
        training=False,
        batch_size=args.bs,
        workers=args.loader_workers,
    )

    models = create_cycle_gan_models()
    optimisers = create_optimisers(models, args.lr, args.wd)
    schedulers = create_schedulers(optimisers, args.decay_lr_steps)

    num_decays = int(floor(args.l1_decay_steps / args.interval))
    pixel_regs = reg_gen(args.pixel_l1_start, args.pixel_l1_stop, num_decays)

    generator_warmup(
        models["fwd_gen"],
        models["bwd_gen"],
        train_loader,
        optimisers["gens"],
        args.warmup_steps,
        args.interval,
    )

    discriminator_warmup(
        models,
        train_loader,
        optimisers,
        args.warmup_steps,
        args.interval,
    )

    num_steps = 0

    checkpoint_models(
        args.save_dir / "model-checkpoints",
        models,
        f"params-{num_steps}.pth",
    )

    print(f"Training for {args.training_steps + args.decay_lr_steps} steps.")
    print(f"The lrs will decay to 0 in the final {args.decay_lr_steps} steps.")

    pixel_l1 = next(pixel_regs)

    print(f"Pixel L1 reg = {pixel_l1:.5f} at {num_steps} steps.")

    while num_steps < (args.training_steps + args.decay_lr_steps):
        start_time = perf_counter()
        train_metrics = cycle_gan_interval(
            models,
            train_loader,
            pixel_l1,
            args.interval,
            optimisers=optimisers,
            schedulers=None if num_steps < args.training_steps else schedulers,
        )

        valid_metrics = cycle_gan_interval(
            models,
            valid_loader,
            pixel_l1,
            args.interval,
        )

        stop_time = perf_counter()

        print(f"Interval time: {stop_time - start_time:.6f} seconds.")

        num_steps += args.interval

        pixel_l1 = next(pixel_regs)

        print(f"End of step {num_steps}.")
        print(f"Pixel L1 reg changes to = {pixel_l1:.5f}.")
        for key, sched in schedulers.items():
            print(f"Training step: {num_steps}")
            print(key)
            print(sched.get_last_lr())

        write_metrics(
            train_metrics,
            valid_metrics,
            num_steps,
            valid_source_str,
            args.save_dir,
        )

        plot_losses(args.save_dir / "metrics.csv")

        checkpoint_models(
            args.save_dir / "model-checkpoints",
            models,
            f"params-{num_steps}.pth",
        )

    plot_losses(args.save_dir / "metrics.csv")


def run_training_experiment(args: Namespace):
    """Train the cycle gan model.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    metadata = read_csv(args.metadata_csv)
    metadata = prepare_metadata(
        metadata,
        args.patch_dir,
        args.source_stain,
        args.target_stain,
        args.mag,
    )

    metadata = metadata.sample(frac=1.0, random_state=666)

    source_df = metadata.loc[metadata.stainBiomarker == args.source_stain]
    target_df = metadata.loc[metadata.stainBiomarker == args.target_stain]

    args.save_dir /= f"{args.source_stain}-{args.target_stain}"
    args.save_dir /= f"{args.pixel_l1_start}-{args.pixel_l1_stop}/"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    source_df.to_csv(args.save_dir / "source.csv", index=False)
    target_df.to_csv(args.save_dir / "target.csv", index=False)

    patch_dicts = {
        "train_src": source_df.loc[~source_df.source.isin(args.valid_sources)],
        "valid_src": source_df.loc[source_df.source.isin(args.valid_sources)],
        "train_tgt": target_df.loc[~target_df.source.isin(args.valid_sources)],
        "valid_tgt": target_df.loc[target_df.source.isin(args.valid_sources)],
    }

    for key, val in patch_dicts.items():
        print(f"There are {len(val)} in '{key}'.")

    start_time = perf_counter()

    train_model(patch_dicts, "---".join(args.valid_sources), args)

    stop_time = perf_counter()

    msg = f"Training time with valid sources '{args.valid_sources}': "
    msg += f"{stop_time - start_time:.6f} seconds."

    print(msg)


if __name__ == "__main__":
    run_training_experiment(_parse_command_line())
