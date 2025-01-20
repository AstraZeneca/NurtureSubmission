"""CycleGAN computations."""

from typing import Dict, Tuple

from torch import tensor, Tensor, float32  # pylint: disable=no-name-in-module

from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss


from nurture_stain.labels import binary_labels


def discrim_pass(discrim: Module, real: Tensor, fake: Tensor) -> Tensor:
    """Discriminator training step.

    Parameters
    ----------
    discrim : Module
        The discriminator model.
    real : Tensor
        A mini-batch of real image.
    fake : Tensor
        A mini-batch of fake image.

    Returns
    -------
    Tensor
        The discriminator loss.

    """
    loss = tensor(0.0, dtype=float32, device=fake.device)
    for batch, label in zip([real, fake], [True, False]):
        labels = binary_labels(real=label, batch_size=len(batch))
        preds = discrim(batch)
        loss += binary_cross_entropy_with_logits(preds, labels.to(batch.device))

    return loss * 0.5


def get_adv_losses(
    models: Dict[str, Module],
    fake_src: Tensor,
    fake_tgt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Adversarial step.

    Parameters
    ----------
    models[str, Module]
        Dictionary holding the models.
    fake_src : Tensor
        A mini-batch of fake source images.
    fake_tgt : Tensor
        A mini-batch of fake target images.

    Returns
    -------
    fwd_adv : Tensor
        The adversarial loss for the forward generator.
    bwd_adv : Tensor
        The adversarial loss for backward generator.

    """
    fwd_preds = models["fwd_disc"](fake_tgt)
    bwd_preds = models["bwd_disc"](fake_src)

    # Note: we create real labels because we are *lying*.
    real_labels = binary_labels(
        real=True,
        batch_size=len(fwd_preds),
    ).to(fwd_preds.device)
    fwd_adv = binary_cross_entropy_with_logits(fwd_preds, real_labels) * 0.5

    # Note: we create real labels because we are *lying*.
    real_labels = binary_labels(
        real=True,
        batch_size=len(bwd_preds),
    ).to(bwd_preds.device)
    bwd_adv = binary_cross_entropy_with_logits(bwd_preds, real_labels) * 0.5

    return fwd_adv, bwd_adv


def get_id_losses(
    models: Dict[str, Module], real_src: Tensor, real_tgt: Tensor
) -> Tuple[Tensor, Tensor]:
    """Get the ID losses.

    Parameters
    ----------
    models : Dict[str, Module]
        Dictionary holding all of the models.
    real_src : Tensor
        A mini-batch of real source-domain images.
    real_src : Tensor
        A mini-batch of real target domain images.

    Returns
    -------
    fwd_id : Tensor
        The forward ID loss.
    bwd_id : Tensor
        The backward ID loss.

    """
    fwd_id = l1_loss(models["fwd_gen"](real_tgt), real_tgt) * 0.5
    bwd_id = l1_loss(models["bwd_gen"](real_src), real_src) * 0.5

    return fwd_id, bwd_id


def get_cycle_losses(
    real_src: Tensor,
    real_tgt: Tensor,
    fake_src: Tensor,
    fake_tgt: Tensor,
    models: Dict[str, Module],
) -> Tuple[Tensor, Tensor]:
    """Return the cycle losses.

    Parameters
    ----------
    real_src : Tensor
        A mini-batch of real source images.
    models : Dict[str, Module]
        Dictionary holding all of the models.

    Returns
    -------
    fwd : Tensor
        Forward cycle pixel loss.
    bwd : Tensor
        Backward cycle pixel loss.

    """
    cycled_src = models["bwd_gen"](fake_tgt)
    cycled_tgt = models["fwd_gen"](fake_src)

    fwd = l1_loss(cycled_src, real_src) * 0.5

    bwd = l1_loss(cycled_tgt, real_tgt) * 0.5

    return fwd, bwd
