#!/usr/bin/env python
"""Script for generating images with a trained model."""
from typing import List
from pathlib import Path
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from torch import no_grad
from torch import load
from torch.cuda import is_available

from torch.utils.data import DataLoader


from torch_tools.file_utils import traverse_directory_tree
from torch_tools import DataSet

from skimage.io import imsave
from skimage.util import img_as_ubyte

from nurture_stain.transforms import cycle_gan_transforms
from nurture_stain.models import Generator

DEVICE = "cuda" if is_available() else "cpu"
print(f"Device set to '{DEVICE}'.")


def _parse_command_line() -> Namespace:
    """Parse the command-ling arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Infer using a trained model.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "patch_dir",
        help="Root directory of the patches.",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        help="Directory to save the restained patches in.",
        type=Path,
        default="restained",
    )

    parser.add_argument(
        "--weights",
        help="Model weights.",
        type=Path,
        default="params.pth",
    )

    parser.add_argument(
        "--patch-formats",
        help="The allowed file formats of the patches.",
        type=str,
        default=[".png", ".jpg"],
        nargs="*",
    )

    return parser.parse_args()


def _list_patches(patch_dir: Path, patch_formats: List[str]) -> List[Path]:
    """List the patches to infer on.

    Parameters
    ----------
    patch_dir : Path
        The parent directory of the patches.
    patch_formats : List[str]
        The allowed file extensions of the patches.

    Returns
    -------
    List[Path]
        List of the patches to infer on.

    """
    patch_gen = filter(
        lambda x: x.suffix in patch_formats,
        traverse_directory_tree(patch_dir),
    )

    patch_gen = filter(lambda x: ".zip" not in str(x), patch_gen)

    return list(patch_gen)


def run_inference(args: Namespace):
    """Run inference with a trained model.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    patches = _list_patches(args.patch_dir, args.patch_formats)

    model = Generator(out_chans=3, apply_tanh=True, batch_norm=False).to(DEVICE)

    weights = load(args.weights, map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    dataset = DataSet(
        inputs=patches,
        input_tfms=cycle_gan_transforms(training=False),
    )

    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)

    pather_iter = iter(patches)

    for batch in data_loader:
        batch = batch.to(DEVICE)

        with no_grad():
            pred = model(batch).cpu()  # pylint: disable=not-callable

        for img in pred:
            img_numpy = img_as_ubyte(img.permute(1, 2, 0).numpy())
            patch_path = next(pather_iter)

            target_path = Path(
                str(patch_path).replace(str(args.patch_dir), str(args.out_dir))
            )
            target_path.parent.mkdir(exist_ok=True, parents=True)

            imsave(target_path, img_numpy, check_contrast=False)


if __name__ == "__main__":
    run_inference(_parse_command_line())
