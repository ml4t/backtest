# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and functions for flexible indexing.

!!! info
    For default settings, see `vectorbtpro._settings.indexing`.
"""

from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = [
    "flex_select_1d_nb",
    "flex_select_1d_pr_nb",
    "flex_select_1d_pc_nb",
    "flex_select_nb",
    "flex_select_row_nb",
    "flex_select_col_nb",
    "flex_select_2d_row_nb",
    "flex_select_2d_col_nb",
]

_rotate_rows = settings["indexing"]["rotate_rows"]
_rotate_cols = settings["indexing"]["rotate_cols"]


@register_jitted(cache=True)
def flex_choose_i_1d_nb(arr: tp.FlexArray1d, i: int) -> int:
    """Choose a position in a flexible 1-dimensional array as if it were broadcast against rows or columns.

    Args:
        arr (FlexArray1d): Flexible 1-dimensional array.
        i (int): Index used for position selection.

    Returns:
        int: Computed index based on the array shape.

    !!! note
        Array must be one-dimensional.
    """
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    return int(flex_i)


@register_jitted(cache=True)
def flex_select_1d_nb(arr: tp.FlexArray1d, i: int) -> tp.Scalar:
    """Select an element from a flexible 1-dimensional array as if it were broadcast against rows or columns.

    Args:
        arr (FlexArray1d): Flexible 1-dimensional array.
        i (int): Index used for selecting the element.

    Returns:
        Scalar: Element selected from the array.

    !!! note
        Array must be one-dimensional.
    """
    flex_i = flex_choose_i_1d_nb(arr, i)
    return arr[flex_i]


@register_jitted(cache=True)
def flex_choose_i_pr_1d_nb(arr: tp.FlexArray1d, i: int, rotate_rows: bool = _rotate_rows) -> int:
    """Choose a position in a flexible 1-dimensional array as if it were broadcast along rows.

    Args:
        arr (FlexArray1d): Flexible 1-dimensional array.
        i (int): Index used for row selection.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.

    Returns:
        int: Computed index considering rotational indexing if enabled.

    !!! note
        Array must be one-dimensional.
    """
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    if rotate_rows:
        return int(flex_i) % arr.shape[0]
    return int(flex_i)


@register_jitted(cache=True)
def flex_choose_i_pr_nb(arr: tp.FlexArray2d, i: int, rotate_rows: bool = _rotate_rows) -> int:
    """Choose a position in a flexible 2-dimensional array as if it were broadcast along rows.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        i (int): Index used for row selection.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.

    Returns:
        int: Computed row index considering rotational indexing if enabled.

    !!! note
        Array must be two-dimensional.
    """
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    if rotate_rows:
        return int(flex_i) % arr.shape[0]
    return int(flex_i)


@register_jitted(cache=True)
def flex_select_1d_pr_nb(
    arr: tp.FlexArray1d, i: int, rotate_rows: bool = _rotate_rows
) -> tp.Scalar:
    """Select an element from a flexible 1-dimensional array as if it were broadcast along rows.

    Args:
        arr (FlexArray1d): Flexible 1-dimensional array.
        i (int): Index used for selecting the element.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.

    Returns:
        Scalar: Element selected from the array.

    !!! note
        Array must be one-dimensional.
    """
    flex_i = flex_choose_i_pr_1d_nb(arr, i, rotate_rows=rotate_rows)
    return arr[flex_i]


@register_jitted(cache=True)
def flex_choose_i_pc_1d_nb(arr: tp.FlexArray1d, col: int, rotate_cols: bool = _rotate_cols) -> int:
    """Choose a position in a flexible 1-dimensional array as if it were broadcast along columns.

    Args:
        arr (FlexArray1d): Flexible 1-dimensional array.
        col (int): Index used for column selection.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        int: Computed column index considering rotational indexing if enabled.

    !!! note
        Array must be one-dimensional.
    """
    if arr.shape[0] == 1:
        flex_col = 0
    else:
        flex_col = col
    if rotate_cols:
        return int(flex_col) % arr.shape[0]
    return int(flex_col)


@register_jitted(cache=True)
def flex_choose_i_pc_nb(arr: tp.FlexArray2d, col: int, rotate_cols: bool = _rotate_cols) -> int:
    """Choose a position in a flexible 2-dimensional array as if it were broadcast along columns.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        col (int): Index used for column selection.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        int: Computed column index considering rotational indexing if enabled.

    !!! note
        Array must be two-dimensional.
    """
    if arr.shape[1] == 1:
        flex_col = 0
    else:
        flex_col = col
    if rotate_cols:
        return int(flex_col) % arr.shape[1]
    return int(flex_col)


@register_jitted(cache=True)
def flex_select_1d_pc_nb(
    arr: tp.FlexArray1d, col: int, rotate_cols: bool = _rotate_cols
) -> tp.Scalar:
    """Select an element from a flexible 1-dimensional array as if it were broadcast along columns.

    Args:
        arr (FlexArray1d): Flexible 1-dimensional array.
        col (int): Index used for selecting the element.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        Scalar: Element selected from the array.

    !!! note
        Array must be one-dimensional.
    """
    flex_col = flex_choose_i_pc_1d_nb(arr, col, rotate_cols=rotate_cols)
    return arr[flex_col]


@register_jitted(cache=True)
def flex_choose_i_and_col_nb(
    arr: tp.FlexArray2d,
    i: int,
    col: int,
    rotate_rows: bool = _rotate_rows,
    rotate_cols: bool = _rotate_cols,
) -> tp.Tuple[int, int]:
    """Choose positions in a flexible 2-dimensional array as if it were broadcast along rows and columns.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        i (int): Index used for row selection.
        col (int): Index used for column selection.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        Tuple[int, int]: Tuple containing the computed row and column indices.

    !!! note
        Array must be two-dimensional.
    """
    if arr.shape[0] == 1:
        flex_i = 0
    else:
        flex_i = i
    if arr.shape[1] == 1:
        flex_col = 0
    else:
        flex_col = col
    if rotate_rows and rotate_cols:
        return int(flex_i) % arr.shape[0], int(flex_col) % arr.shape[1]
    if rotate_rows:
        return int(flex_i) % arr.shape[0], int(flex_col)
    if rotate_cols:
        return int(flex_i), int(flex_col) % arr.shape[1]
    return int(flex_i), int(flex_col)


@register_jitted(cache=True)
def flex_select_nb(
    arr: tp.FlexArray2d,
    i: int,
    col: int,
    rotate_rows: bool = _rotate_rows,
    rotate_cols: bool = _rotate_cols,
) -> tp.Scalar:
    """Select an element from a flexible 2-dimensional array as if it were broadcast along rows and columns.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        i (int): Index used for row selection.
        col (int): Index used for column selection.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        Scalar: Element selected from the array.

    !!! note
        Array must be two-dimensional.
    """
    flex_i, flex_col = flex_choose_i_and_col_nb(
        arr,
        i,
        col,
        rotate_rows=rotate_rows,
        rotate_cols=rotate_cols,
    )
    return arr[flex_i, flex_col]


@register_jitted(cache=True)
def flex_select_row_nb(arr: tp.FlexArray2d, i: int, rotate_rows: bool = _rotate_rows) -> tp.Array1d:
    """Select a row from a flexible 2-dimensional array, returning a 1-dimensional array.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        i (int): Index used for row selection.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.

    Returns:
        Array1d: Selected row as a 1-dimensional array.

    !!! note
        Array must be two-dimensional.
    """
    flex_i = flex_choose_i_pr_nb(arr, i, rotate_rows=rotate_rows)
    return arr[flex_i]


@register_jitted(cache=True)
def flex_select_col_nb(
    arr: tp.FlexArray2d, col: int, rotate_cols: bool = _rotate_cols
) -> tp.Array1d:
    """Select a column from a flexible 2-dimensional array, returning a 1-dimensional array.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        col (int): Index used for column selection.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        Array1d: Selected column as a 1-dimensional array.

    !!! note
        Array must be two-dimensional.
    """
    flex_col = flex_choose_i_pc_nb(arr, col, rotate_cols=rotate_cols)
    return arr[:, flex_col]


@register_jitted(cache=True)
def flex_select_2d_row_nb(
    arr: tp.FlexArray2d, i: int, rotate_rows: bool = _rotate_rows
) -> tp.Array2d:
    """Select a row from a flexible 2-dimensional array, returning a 2-dimensional array.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        i (int): Index used for row selection.
        rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.

    Returns:
        Array2d: 2-dimensional array containing the selected row.

    !!! note
        Array must be two-dimensional.
    """
    flex_i = flex_choose_i_pr_nb(arr, i, rotate_rows=rotate_rows)
    return arr[flex_i : flex_i + 1]


@register_jitted(cache=True)
def flex_select_2d_col_nb(
    arr: tp.FlexArray2d, col: int, rotate_cols: bool = _rotate_cols
) -> tp.Array2d:
    """Select a column from a flexible 2-dimensional array, returning a 2-dimensional array.

    Args:
        arr (FlexArray2d): Flexible 2-dimensional array.
        col (int): Index used for column selection.
        rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

    Returns:
        Array2d: 2-dimensional array containing the selected column.

    !!! note
        Array must be two-dimensional.
    """
    flex_col = flex_choose_i_pc_nb(arr, col, rotate_cols=rotate_cols)
    return arr[:, flex_col : flex_col + 1]
