#!/usr/bin/env python
"""Create visualisation of CycleGAN-generated images."""

from typing import List, Tuple, Dict
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from string import ascii_lowercase

from pandas import read_csv, DataFrame

from torch import Tensor, no_grad, concat, load  # pylint: disable=no-name-in-module
from torch.nn import Module

from torch.cuda import is_available

from torchvision.utils import make_grid  # type: ignore


from numpy import ndarray

from torch_tools.file_utils import traverse_directory_tree

import matplotlib.pyplot as plt

from nurture_stain.metadata_utils import prepare_metadata
from nurture_stain.transforms import cycle_gan_transforms
from nurture_stain.models import Generator
from nurture_stain.plotting import figure_cleanup

from nurture_stain._nurture_misc import glom_valid_patches, tubule_valid_patches

DEVICE = "cuda" if is_available() else "cpu"


def _parse_args() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Create CycleGAN image visuals.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "patch_dir",
        help="Parent directory of the Nurture patches.",
        type=Path,
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Directory holding the saved weights.",
        type=Path,
    )

    parser.add_argument(
        "--patch-types",
        help="The type of patch: 'glomerular', 'tubular' or 'random'.",
        type=str,
        default=["glomerular", "tubular"],
        nargs="*",
    )

    parser.add_argument(
        "--metadata-csv",
        help="Path to the nurture metadata.",
        type=Path,
        default="nurture-metadata.csv",
    )

    parser.add_argument(
        "--src-stain",
        help="Stain to use as the source domain.",
        type=str,
        default="H&E",
    )

    parser.add_argument(
        "--tgt-stain",
        help="Stain to use as the target domain.",
        type=str,
        default="PAS",
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the inference images in.",
        type=Path,
        default="visuals",
    )

    parser.add_argument(
        "--mag",
        help="The patch-level magnification.",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--sources",
        help="Source(s) to infer on.",
        type=str,
        nargs="*",
        default=["Birmingham BCH NURTuRE", "Birmingham UHB NURTuRE"],
    )

    parser.add_argument(
        "--num-patches",
        help="Batch size of the image-grids to produce.",
        type=int,
        default=10,
    )

    return parser.parse_args()


def _get_checkpoints(checkpoint_dir: Path) -> DataFrame:
    """List the model checkpoints from file.

    Parameters
    ----------
    checkpoint_dir : Path
        Path to the directory holding the checkpoints.

    Returns
    -------
    DataFrame
        Path information for the model's checkpoints.

    """
    files = traverse_directory_tree(checkpoint_dir)

    checkpoints = DataFrame(
        columns=["path"],
        data=list(filter(lambda x: x.suffix == ".pth", files)),  # type: ignore
    )
    checkpoints["file_name"] = checkpoints["path"].apply(lambda x: str(x.stem))

    # pylint: disable=anomalous-backslash-in-string
    checkpoints["checkpoint"] = checkpoints.file_name.str.extract("(\d+)")

    checkpoints.checkpoint = checkpoints.checkpoint.astype(int)

    checkpoints = checkpoints.sort_values(by="checkpoint", ascending=True)

    return checkpoints.reset_index(drop=True)


def list_images(img_paths: List[Path]) -> List[Tensor]:
    """Load the images listed in ``img_paths``.

    Parameters
    ----------
    img_paths : Path
        List of paths to the images to be loaded.

    Returns
    -------
    List[Tensor]
        List of image tensors, where each image is a Tensor of mini-batch size
        one.

    """
    tfms = cycle_gan_transforms(training=False, gamma_aug=False)

    return list(map(lambda x: tfms(x).unsqueeze(0), img_paths))


@no_grad()
def generate_predictions(src_img: List[Tensor], model: Module) -> List[Tensor]:
    """Generate predictions from ``src_imgs``.

    Parameters
    ----------
    src_img : List[Tensor]
        List of the source images.

    Returns
    -------
    pred_imgs : List[Tensor]
        A list of the predicted images.

    """
    model.to(DEVICE)
    model.eval()
    pred_imgs = []
    for src in src_img:
        pred_imgs.append(model(src.to(DEVICE)).cpu())

    return pred_imgs


def _save_comparison_plot(img_dict: Dict[str, ndarray], save_path: Path):
    """Save a plot of the images to file.

    Parameters
    ----------
    img_dict : Dict[str, ndarray]
        Dictionary of images, to be plotted.
    save_path : Path
        Target path to save the file as.

    """
    rows, cols, _ = img_dict[list(img_dict.keys())[0]].shape
    width = 8.27
    height = width * (rows / cols) * len(img_dict)

    figure, axes = plt.subplots(len(img_dict), 1, figsize=(width, height))

    letters = iter(ascii_lowercase)

    for (content, img), axis in zip(img_dict.items(), axes.ravel()):

        axis.imshow(img)
        axis.set_ylabel(f"({next(letters)}) {content.capitalize()}")

        axis.set_xticks([])
        axis.set_yticks([])

    save_path.parent.mkdir(exist_ok=True, parents=True)

    figure.tight_layout(pad=0.75)
    figure.savefig(save_path, dpi=500)

    figure_cleanup(axes)


def produce_images(
    model: Module,
    src_paths: List[Path],
    tgt_paths: List[Path],
) -> ndarray:
    """Produce inference visualisation using ``src_paths`` and ``tgt_paths``.

    Parameters
    ----------
    model : Moduel
        The generator model, already endowed with the trained weights.
    src_paths : List[Path]
        List of the source images.
    tgt_paths : List[Path]
        List of the target images.

    Returns
    -------
    ndarray
        All of the images stacked in a grid.


    """
    src_imgs = list_images(src_paths)
    pred_imgs = generate_predictions(src_imgs, model)
    tgt_imgs = list_images(tgt_paths)

    all_imgs = concat(src_imgs + pred_imgs + tgt_imgs, dim=0).clip(0.0, 1.0)
    return make_grid(all_imgs, nrow=len(src_imgs)).permute(1, 2, 0).numpy()


def _get_patches_of_interest(
    metadata: DataFrame,
    args: Namespace,
    patch_dict: Dict[str, Tuple[str, ...]],
) -> Tuple[DataFrame, DataFrame]:
    """Get patches of interest from ``patch_dict`` for example visualisation.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.

    Returns
    -------
    src_df : DataFrame
        Data frame with patches from the source domain.
    tgt_df : DataFrame
        Data frame with patches for the target domain.

    """
    where_src = metadata.patch_path.apply(
        lambda x: x.name in patch_dict[args.src_stain]
    )

    where_tgt = metadata.patch_path.apply(
        lambda x: x.name in patch_dict[args.tgt_stain]
    )

    metadata = metadata.loc[where_src | where_tgt]

    src_df = metadata.loc[metadata.stainBiomarker == args.src_stain]
    tgt_df = metadata.loc[metadata.stainBiomarker == args.tgt_stain]

    return src_df, tgt_df


def _get_random_patches(
    metadata: DataFrame,
    args: Namespace,
) -> Tuple[DataFrame, DataFrame]:
    """Get patches at random.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.

    Returns
    -------
    src_df : DataFrame
        Source-domain patches.
    tgt_df : DataFrame
        Target-domain patches.

    """
    metadata = metadata.groupby(by=["source", "stainBiomarker"]).sample(
        (args.num_patches) // 2,
        random_state=123,
    )

    src_df = metadata.loc[metadata.stainBiomarker == args.src_stain]
    tgt_df = metadata.loc[metadata.stainBiomarker == args.tgt_stain]

    return src_df, tgt_df


def get_patch_paths(
    patch_type: str,
    args: Namespace,
    metadata: DataFrame,
    glom_patches: Dict[str, Tuple[str, ...]],
    tubule_patches: Dict[str, Tuple[str, ...]],
) -> Tuple[DataFrame, DataFrame]:
    """Return the patch paths which go into the plot.

    Parameters
    ----------
    patch_type : str
        The type of patch to return.
    args : Namespace
        Command-line arguments.
    metadata : DataFrame
        Patch-level metadata.
    glom_patches : Dict[str, Tuple[str, ...]]
        Different-stain glom patches.
    tubule_patches : Dict[str, Tuple[str, ...]]
        Different-stain tubule patches.

    Returns
    -------
    DataFrame
        The source-domain patches.
    DataFrame
        The target-domain patches.

    """
    if patch_type == "random":
        return _get_random_patches(metadata, args)

    if patch_type == "glomerular":
        return _get_patches_of_interest(
            metadata,
            args,
            glom_patches,  # type: ignore
        )

    if patch_type == "tubular":
        return _get_patches_of_interest(
            metadata,
            args,
            tubule_patches,  # type: ignore
        )

    msg = f"Options '{patch_type}' not understood."
    raise ValueError(msg)


def run_inference(args: Namespace):
    """Run inference on patches using different model checkpoints.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.save_dir /= f"{args.src_stain}-{args.tgt_stain}"
    metadata = read_csv(args.metadata_csv)
    metadata = prepare_metadata(
        metadata,
        args.patch_dir,
        args.src_stain,
        args.tgt_stain,
        args.mag,
    )

    metadata = metadata.loc[metadata.source.isin(args.sources)]
    metadata = metadata.loc[
        metadata.stainBiomarker.isin([args.src_stain, args.tgt_stain])
    ]

    checkpoints = _get_checkpoints(args.checkpoint_dir)
    model = Generator()

    for row in checkpoints.itertuples():
        state_dict = load(str(row.path))  # type: ignore
        model.load_state_dict(state_dict)

        overviews = {}

        for patch_type in args.patch_types:

            src_df, tgt_df = get_patch_paths(
                patch_type,
                args,
                metadata,
                glom_valid_patches,
                tubule_valid_patches,
            )

            overviews[patch_type] = produce_images(
                model,
                src_df.patch_path.to_list(),
                tgt_df.patch_path.to_list(),
            )

        _save_comparison_plot(
            overviews,
            args.save_dir / f"overviews/checkpoint-{row.checkpoint}.pdf",
        )


if __name__ == "__main__":
    run_inference(_parse_args())
