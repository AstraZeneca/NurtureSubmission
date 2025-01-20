"""Metadata-handling utils."""

from pathlib import Path

from pandas import DataFrame

from torch_tools.file_utils import traverse_directory_tree


def _find_all_patches(metadata: DataFrame, patch_dir: Path) -> DataFrame:
    """Get a ``DataFrame`` with metadata for all of the patches.

    Parameters
    ----------
    metadata : DataFrame
        Nurture dataset's metadata.
    patch_dir : Path
        The parent directory of the patches.

    Returns
    -------
    DataFrame
        ``metadata`` with the patch paths added.

    """
    scans = list(map(lambda x: x.stem, patch_dir.glob("*")))
    metadata = metadata.loc[metadata.imageId.isin(scans)].reset_index(drop=True)
    metadata["patch_path"] = metadata.imageId.apply(
        lambda x: traverse_directory_tree(patch_dir / f"{x}.svs/patches/")
    ).to_list()

    metadata = metadata.explode("patch_path").reset_index(drop=True)
    metadata["patch_name"] = metadata.patch_path.apply(lambda x: x.name).to_list()

    return metadata.sort_values(by=["imageId", "patch_name"])


def _extract_train_patches(metadata: DataFrame) -> DataFrame:
    """Extract the training patches.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.

    Returns
    -------
    DataFrame
        The metadata for the training patches only.

    """
    return metadata.loc[metadata.data_split == "train"].reset_index(drop=True)


def _extract_magnification(metadata: DataFrame, mag: float):
    """Extract patches at requested magnification ``mag``.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.
    mag : float
        Desired patcg-level magnification.

    Returns
    -------
    DataFrame
        Patch-level metadata at magnification ``mag``.

    """
    where = metadata.patch_path.apply(lambda x: f"mag_{mag:.1f}.zip" in str(x))
    return metadata.loc[where].reset_index(drop=True)


def _extract_stains(metadata: DataFrame, source_stain: str, target_stain: str):
    """Extract the source and target stains.

    Parameters
    ----------
    metadata : DataFrame
        Patch-level metadata.
    source_stain : str
        The source stain.
    target_stain : str
        The target stain.

    Returns
    -------
    DataFrame
        The patch-level metadata.

    """
    where = metadata.stainBiomarker.isin([source_stain, target_stain])
    return metadata.loc[where].reset_index(drop=True)


def prepare_metadata(
    metadata: DataFrame,
    patch_dir: Path,
    source_stain: str,
    target_stain: str,
    mag: float,
) -> DataFrame:
    """Prepare the metadata.

    Parameters
    ----------
    metadata : DataFrame
        Nurture metadata.
    patch_dir : Path
        Root directory of the patches.
    source_stain : str
        The staining domain we want to map from.
    target_stain : str
        The staining domain we want to map to.
    mag : float
        Magnification of the patches to use.

    Returns
    -------
    metadata : DataFrame
        Patch-level metadata.

    """
    metadata = _find_all_patches(metadata, patch_dir)

    metadata = _extract_train_patches(metadata)

    metadata = _extract_magnification(metadata, mag)

    metadata = _extract_stains(metadata, source_stain, target_stain)

    return metadata
