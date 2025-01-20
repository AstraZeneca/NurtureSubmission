"""Training utility functions."""

from time import perf_counter

from typing import Dict, Optional, Tuple


from torch import Tensor, tensor, float32  # pylint: disable=no-name-in-module
from torch import set_grad_enabled  # pylint: disable=no-name-in-module

from torch.utils.data import DataLoader

from torch.nn import Module
from torch.nn.functional import l1_loss

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torchvision.transforms import GaussianBlur  # type: ignore

from nurture_stain import cycle_gan_utils as utils
from nurture_stain.cycle_gan_functional import (
    discrim_pass,
    get_adv_losses,
    get_id_losses,
    get_cycle_losses,
)


# pylint: disable=too-many-locals,too-many-statements,too-many-arguments


def discrim_step(
    real_src: Tensor,
    real_tgt: Tensor,
    fake_src: Tensor,
    fake_tgt: Tensor,
    models: Dict[str, Module],
    optimisers: Optional[Dict[str, Optimizer]] = None,
) -> Tuple[float, float]:
    """Perform one optimisation step for each of the discriminators.

    Parameters
    ----------
    real_src : Tensor
        A mini-batch of fake source images.
    real_tgt : Tensor
        A mini-batch of real target images.
    fake_src : Tensor
        A fake (generated) image on the source domain.
    fake_tgt : Tensor
        A fake (generated) image on the target domain.
    models : Dict[str, Module]
        Dictionary holding all the models.
    optimisers : Dict[str, RAdam], optional
        The optimisers for each model.

    Returns
    -------
    fwd_disc : float
        The forward discriminator's loss.
    bwd_disc : float
        The backward discriminator's loss.

    Notes
    -----
    This function calls ``detach`` on the generated images.

    """
    fwd_disc = discrim_pass(
        discrim=models["fwd_disc"],
        real=real_tgt,
        fake=fake_tgt.detach(),
    )

    bwd_disc = discrim_pass(
        discrim=models["bwd_disc"],
        real=real_src,
        fake=fake_src.detach(),
    )

    if optimisers is not None:
        total_disc_loss = fwd_disc + bwd_disc
        total_disc_loss.backward()
        optimisers["discs"].step()
        utils.zero_disc_grads(optimisers)

    return fwd_disc.item(), bwd_disc.item()


def generator_step(
    real_tgt: Tensor,
    real_src: Tensor,
    fake_tgt: Tensor,
    fake_src: Tensor,
    models: Dict[str, Module],
    cycle_reg: float,
    optimisers: Optional[Dict[str, Optimizer]] = None,
) -> Dict[str, float]:
    """Perform one optimisation step with the generators.

    Parameters
    ----------
    real_tgt : Tensor
        A mini-batch of real target images.
    real_src : Tensor
        A mini-batch of real source images.
    fake_tgt : Tensor
        A fake image on the target domain.
    fake_src : Tensor
        A fake image on the source domain.
    models : Dict[str, Module]
        All of the system's models.
    cycle_reg : float
        The reguluarisation to apply to the pixel-level L^{1} comparison.
    optimisers : Dict[str, Optimizer], optional
        The models' optimisers.

    Returns
    -------
    Dict[str, float]
        Dictionary with all of the disaggregated generator losses.

    """
    fwd_adv, bwd_adv = get_adv_losses(models, fake_src, fake_tgt)

    fwd_id, bwd_id = get_id_losses(models, real_src, real_tgt)

    fwd_cycle, bwd_cycle = get_cycle_losses(
        real_src,
        real_tgt,
        fake_src,
        fake_tgt,
        models,
    )

    fwd_cycle *= cycle_reg
    bwd_cycle *= cycle_reg

    if optimisers is not None:
        adv_loss = fwd_adv + bwd_adv
        id_loss = fwd_id + bwd_id
        cycle = fwd_cycle + bwd_cycle

        gen_loss = adv_loss + id_loss + cycle
        gen_loss.backward()
        optimisers["gens"].step()
        optimisers["gens"].zero_grad()

    return {
        "fwd_adv": fwd_adv.item(),
        "bwd_adv": bwd_adv.item(),
        "fwd_id": fwd_id.item(),
        "bwd_id": bwd_id.item(),
        "fwd_cycle": fwd_cycle.item(),
        "bwd_cycle": bwd_cycle.item(),
    }


def cycle_gan_interval(
    models: Dict[str, Module],
    loader: DataLoader,
    cycle_reg: float,
    interval: int,
    optimisers: Optional[Dict[str, Optimizer]] = None,
    schedulers: Optional[Dict[str, LRScheduler]] = None,
):
    """Train or validate for ``interval`` mini-batch steps.

    Parameters
    ----------
    models : Dict[str, Module]
        Dictionary holding all of the models.
    src_loader : DataLoader
        Data loader yielding source and target images.
    cycle_reg : float
        Pixel-level L1 regularisation parameter.
    steps : int
        The number of steps to train for before stopping.
    optimisers : Dict[str, RAdam], optional
        Dictionary holding the optimisers---only supplied if training (not
        validating).
    schedulers : Dict[str, MultiStepLR], optional
        The learning rate schedulers.

    """
    utils.set_train_or_eval(models=models, train=optimisers is not None)

    for _, model in models.items():
        model.to(utils.DEVICE)

    metrics = {
        "fwd_disc": 0.0,
        "bwd_disc": 0.0,
        "fwd_adv": 0.0,
        "bwd_adv": 0.0,
        "fwd_id": 0.0,
        "bwd_id": 0.0,
        "fwd_cycle": 0.0,
        "bwd_cycle": 0.0,
    }

    steps = 0
    while steps < interval:
        for real_src, real_tgt in loader:  # type: ignore
            utils.zero_all_grads(optimisers)

            real_src, real_tgt = real_src.to(utils.DEVICE), real_tgt.to(utils.DEVICE)

            with set_grad_enabled(optimisers is not None):
                fake_tgt = models["fwd_gen"](real_src)
                fake_src = models["bwd_gen"](real_tgt)

            with set_grad_enabled(optimisers is not None):
                fwd_disc, bwd_disc = discrim_step(
                    real_src,
                    real_tgt,
                    fake_src,
                    fake_tgt,
                    models,
                    optimisers,
                )

            metrics["fwd_disc"] += fwd_disc
            metrics["bwd_disc"] += bwd_disc

            with set_grad_enabled(optimisers is not None):
                gen_losses = generator_step(
                    real_tgt,
                    real_src,
                    fake_tgt,
                    fake_src,
                    models,
                    cycle_reg,
                    optimisers,
                )

            for loss_name, loss_val in gen_losses.items():
                metrics[loss_name] += loss_val

            ####### End enerator step #######

            if schedulers is not None:
                for _, scheduler in schedulers.items():
                    scheduler.step()

            steps += 1
            if steps == interval:
                break

    utils.zero_all_grads(optimisers)
    return {key: val / interval for key, val in metrics.items()}


def generator_warmup(
    fwd_gen: Module,
    bwd_gen: Module,
    loader: DataLoader,
    optimiser: Optimizer,
    num_steps: int,
    freq: int = 1000,
):
    """Warm-up the generators using the identity loss.

    Note: this function will not run for more han a single epoch reagrdless of
    ``num_steps``.

    Paramaters
    ----------
    fwd_gen : Module
        The forward generator model.
    bwd_gen : Module
        The backward generator model.
    loader : DataLoader
        Data loader yielding source and target images.
    optimiser : Optimizer
        The generators' optimiser.
    num_steps : int
        The number of steps to train for.
    freq : int
        The frequency to print and average the loss over.

    """
    print(f"Warming up the generators for {num_steps} steps.")
    fwd_gen.train()
    bwd_gen.train()

    fwd_gen.to(utils.DEVICE)
    bwd_gen.to(utils.DEVICE)

    running_loss = 0.0

    blur = GaussianBlur(kernel_size=11, sigma=(0.01, 2.0))

    start = perf_counter()

    counter = 0
    while counter < num_steps:
        for real_src, real_tgt in loader:
            optimiser.zero_grad()

            loss = tensor(0.0, dtype=float32, device=utils.DEVICE)
            for target in [real_src, real_tgt]:

                target = target.to(utils.DEVICE)
                batch = blur(target)

                for generator in [fwd_gen, bwd_gen]:
                    loss += l1_loss(generator(batch), batch)

            loss /= 4.0
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            counter += 1

            if (counter % freq) == 0:
                mean_loss = running_loss / freq
                running_loss = 0.0
                msg = f"L1 loss after {counter} warmup steps: {mean_loss:.4f}"
                print(msg)

            if counter == num_steps:
                break

    optimiser.zero_grad()

    stop = perf_counter()
    print(f"Generator warm-up time : {stop - start:.6f} seconds.")


def discriminator_warmup(
    models: Dict[str, Module],
    loader: DataLoader,
    optimisers: Dict[str, Optimizer],
    num_steps: int,
    freq: int = 1000,
):
    """Warm-up the discriminator models.

    Parameters
    ----------
    models : Dict[str, Module]
        Dictionary holding the models.
    loader : DataLoader
        Training data-loader.
    optimisers : Dict[str, RAdam]
        Models' optimisers.
    num_steps : int
        The number of steps to warm up for.
    freq : int
        The frequency at which the loss is printed and averaged.

    """
    print(f"Warming up the discriminators for {num_steps} steps.")
    utils.set_train_or_eval(models=models, train=True)

    loss = 0.0

    for _, model in models.items():
        model.to(utils.DEVICE)

    start = perf_counter()

    count = 0
    while count < num_steps:
        for real_src, real_tgt in loader:

            real_src, real_tgt = real_src.to(utils.DEVICE), real_tgt.to(utils.DEVICE)

            fake_tgt = models["fwd_gen"](real_src)
            fake_src = models["bwd_gen"](real_tgt)

            fwd, bwd = discrim_step(
                real_src,
                real_tgt,
                fake_src,
                fake_tgt,
                models,
                optimisers,
            )

            loss += (fwd + bwd) * 0.5

            count += 1

            if (count % freq) == 0:
                print(f"Mean discrim loss at {count} steps: {loss / freq}.")
                loss = 0.0

            if count == num_steps:
                break

    utils.zero_all_grads(optimisers)

    stop = perf_counter()
    print(f"Discriminator warm-up time : {stop - start:.6f} seconds.")
