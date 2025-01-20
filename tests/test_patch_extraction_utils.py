"""Tests for the patch extraction utility functions."""

from pandas import DataFrame

from nurture_stain.patch_extraction_utils import (
    _correct_rows,
    _correct_cols,
    _create_coord_df,
)


def test_correct_rows_with_no_correction_required():
    """Test the row coords with no required correction."""
    coord_df = DataFrame()
    coord_df["top"] = [0, 256, 512, 768]
    coord_df["bottom"] = [256, 512, 768, 1024]

    _correct_rows(coord_df, height=1024)

    assert tuple(coord_df["top"]) == (0, 256, 512, 768)
    assert tuple(coord_df["bottom"]) == (256, 512, 768, 1024)


def test_correct_rows_with_correction_required():
    """Test the row coords with a correction required."""
    coord_df = DataFrame()
    coord_df["top"] = [0, 256, 512, 768]
    coord_df["bottom"] = [256, 512, 768, 1024]

    height = 1000

    _correct_rows(coord_df, height=height)

    assert tuple(coord_df.top)[:-1] == (0, 256, 512)
    assert tuple(coord_df.bottom)[:-1] == (256, 512, 768)

    assert coord_df.top.iloc[-1] == 768 - (1024 - height)
    assert coord_df.bottom.iloc[-1] == height


def test_correct_cols_with_no_correction_required():
    """Test the col coords with no required correction."""
    coord_df = DataFrame()
    coord_df["left"] = [0, 256, 512, 768]
    coord_df["right"] = [256, 512, 768, 1024]

    _correct_cols(coord_df, width=1024)

    assert tuple(coord_df["left"]) == (0, 256, 512, 768)
    assert tuple(coord_df["right"]) == (256, 512, 768, 1024)


def test_correct_cols_with_correction_required():
    """Test the col coords with a correction required."""
    coord_df = DataFrame()
    coord_df["left"] = [0, 256, 512, 768]
    coord_df["right"] = [256, 512, 768, 1024]

    width = 1024

    _correct_cols(coord_df, width=width)

    assert tuple(coord_df.left)[:-1] == (0, 256, 512)
    assert tuple(coord_df.right)[:-1] == (256, 512, 768)

    assert coord_df.left.iloc[-1] == 768 - (1024 - width)
    assert coord_df.right.iloc[-1] == width


def test_create_coord_df_number_of_rows():
    """Test the number of coordinates in data frame."""
    width, height = 1024, 1024

    coord_df = _create_coord_df(width, height, patch_size=256, stride=256)
    assert len(coord_df) == 16

    coord_df = _create_coord_df(width, height, patch_size=256, stride=128)
    assert len(coord_df) == 64


def test_coord_df_max_and_min():
    """Test the max and min values in each of the coord fields."""
    width, height = 2048, 1024

    coord_df = _create_coord_df(width, height, patch_size=256, stride=256)
    assert coord_df.right.max() == width
    assert coord_df.left.max() == 1792
    assert coord_df.top.max() == 768
    assert coord_df.bottom.max() == height

    coord_df = _create_coord_df(width, height, patch_size=256, stride=128)
    assert coord_df.right.max() == width
    assert coord_df.left.max() == 1792
    assert coord_df.top.max() == 768
    assert coord_df.bottom.max() == height
