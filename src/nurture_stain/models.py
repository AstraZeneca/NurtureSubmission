"""Model-creating utilities."""

from torch import Tensor, concat, zeros  # pylint: disable=no-name-in-module
from torch.nn import Module, BatchNorm1d, BatchNorm2d, InstanceNorm1d, InstanceNorm2d


from torch_tools import ConvNet2d, UNet
from torch_tools.torch_utils import patchify_img_batch

# pylint: disable=cell-var-from-loop


class Generator(Module):
    """Image-generating model."""

    def __init__(
        self,
        out_chans: int = 3,
        apply_tanh: bool = True,
        batch_norm: bool = False,
    ):
        """Build the generator model.

        Parameters
        ----------
        out_chans : int, optional
            The number of output channels the model should produce.
        apply_tanh : bool, optional
            Should the hyperbolic tangent be applied to output?
        batch_norm : bool, optional
            Should we use batch normalisation? If ``False``, we use instance.

        """
        super().__init__()
        self._generator = UNet(
            in_chans=3,
            out_chans=out_chans,
            num_layers=6,
            features_start=64,
            block_style="conv_res",
            bilinear=True,
            pool_style="avg",
            dropout=0.0,
        )
        self._tanh = apply_tanh

        if batch_norm is False:
            self.apply(_batch_norm_to_instance_norm)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.

        Returns
        -------
        preds : Tensor
            The result of passing ``batch`` through the model.

        """
        preds = self._generator(batch)

        if self._tanh is True:
            preds = preds.tanh()

        return preds


class SegmentationModel(Module):
    """Image-generating model."""

    def __init__(self):
        """Build the generator model."""
        super().__init__()
        self._generator = UNet(
            in_chans=3,
            out_chans=2,
            num_layers=6,
            features_start=64,
            block_style="conv_res",
            bilinear=False,
            pool_style="max",
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of image-like inputs.

        Returns
        -------
        Tensor
            The result of passing ``batch`` through the model.

        """
        return self._generator(batch).softmax(dim=1)


class Discriminator(Module):
    """Discriminator model.

    Notes
    -----
    We don't apply an activation at the final layer.

    """

    def __init__(self):
        """Build ``Discriminator``."""
        super().__init__()
        self._cnn = ConvNet2d(
            out_feats=1,
            encoder_style="resnet18",
            pretrained=True,
        )

        self.apply(_batch_norm_to_instance_norm)

    def get_feats(self, batch: Tensor) -> Tensor:
        """Extract feature from ``batch``.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            Features extracted from ``batch``.

        """
        return self._cnn.get_features(batch)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        """
        return self._cnn(batch)


class PatchGanDisc(Module):
    """PatchGAN-like discriminator.

    Parameters
    ----------
    receptive_field : int, optional
        The size of the sub-patches the discrimintor sees.

    """

    def __init__(self, receptive_field: int = 64):
        """Build ``PatchGANDisc``."""
        super().__init__()
        self._cnn = ConvNet2d(
            out_feats=1,
            encoder_style="resnet18",
            pretrained=True,
        )

        self._ps = receptive_field
        self._off = self._ps // 2

        self.apply(_disable_batchnorm_stats)

    def _subpatch_image(self, img: Tensor) -> Tensor:
        """Extract sub-patches from the image.

        Parameters
        ----------
        img : Tensor
            A batch containing only one image.

        Returns
        -------
        Tensor
            Sub-patches extracted from ``img``.

        """
        on_axis = patchify_img_batch(img, self._ps)
        off_axis = patchify_img_batch(
            img[:, :, self._off : -self._off, self._off : -self._off],
            self._ps,
        )

        return concat((on_axis, off_axis), dim=0)

    def _single_item_forward(self, img: Tensor) -> Tensor:
        """Infer on the patches sampled from ``single_item``.

        Parameters
        ----------
        single_item : Tensor
            A single image.

        Returns
        -------
        Tensor
            The mean prediction over the sub-patches.

        """
        return self._cnn(self._subpatch_image(img.unsqueeze(0))).mean(axis=0)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing ``batch`` through the model.

        """
        preds = zeros((len(batch), 1), dtype=batch.dtype, device=batch.device)
        for idx, img in enumerate(batch):
            preds[idx] += self._single_item_forward(img)
        return preds


def _disable_batchnorm_stats(module: Module):
    """Disable the tracking of running statistics in batch norm layers.

    Parameters
    ----------
    module : Module
        A layer or module.

    """
    if isinstance(module, (BatchNorm1d, BatchNorm2d)):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None


def _batch_norm_to_instance_norm(layer: Module):
    """Turn the batch normalisation layers to instance normalisations.

    Parameters
    ----------
    layer : Module
        Layer in the network.

    """
    for name, module in layer.named_children():

        if isinstance(module, BatchNorm1d):
            setattr(layer, name, InstanceNorm1d(module.num_features))

        if isinstance(module, BatchNorm2d):
            setattr(layer, name, InstanceNorm2d(module.num_features))
