# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for iterative use."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted(cache=True)
def iter_above_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Return whether the value in `arr1` is above the value in `arr2`
    at the specified row and column.

    Args:
        arr1 (FlexArray2d): 2D array from which to retrieve the current value.
        arr2 (FlexArray2d): 2D array from which to retrieve the current value.
        i (int): Current row index.
        col (int): Column index.

    Returns:
        bool: True if the current value in `arr1` is greater than the corresponding value in `arr2`
            and both values are valid, otherwise False.
    """
    if i < 0:
        return False
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_now > arr2_now


@register_jitted(cache=True)
def iter_below_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Return whether the value in `arr1` is below the value in `arr2`
    at the specified row and column.

    Args:
        arr1 (FlexArray2d): 2D array from which to retrieve the current value.
        arr2 (FlexArray2d): 2D array from which to retrieve the current value.
        i (int): Current row index.
        col (int): Column index.

    Returns:
        bool: True if the current value in `arr1` is less than the corresponding value in `arr2`
            and both values are valid, otherwise False.
    """
    if i < 0:
        return False
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_now < arr2_now


@register_jitted(cache=True)
def iter_crossed_above_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Return whether the value in `arr1` has crossed above the value in `arr2`
    at the specified row and column.

    Args:
        arr1 (FlexArray2d): 2D array from which to retrieve consecutive values.
        arr2 (FlexArray2d): 2D array from which to retrieve consecutive values.
        i (int): Current row index.
        col (int): Column index.

    Returns:
        bool: True if, at the previous row, the value in `arr1` was below that in `arr2`
            and at the current row it is above, provided all values are valid; otherwise, False.
    """
    if i < 0 or i - 1 < 0:
        return False
    arr1_prev = flex_select_nb(arr1, i - 1, col)
    arr2_prev = flex_select_nb(arr2, i - 1, col)
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_prev) or np.isnan(arr2_prev) or np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_prev < arr2_prev and arr1_now > arr2_now


@register_jitted(cache=True)
def iter_crossed_below_nb(arr1: tp.FlexArray2d, arr2: tp.FlexArray2d, i: int, col: int) -> bool:
    """Return whether the value in `arr1` has crossed below the value in `arr2`
    at the specified row and column.

    Args:
        arr1 (FlexArray2d): 2D array from which to retrieve consecutive values.
        arr2 (FlexArray2d): 2D array from which to retrieve consecutive values.
        i (int): Current row index.
        col (int): Column index.

    Returns:
        bool: True if, at the previous row, the value in `arr1` was above that in `arr2`
            and at the current row it is below, provided all values are valid; otherwise, False.
    """
    if i < 0 or i - 1 < 0:
        return False
    arr1_prev = flex_select_nb(arr1, i - 1, col)
    arr2_prev = flex_select_nb(arr2, i - 1, col)
    arr1_now = flex_select_nb(arr1, i, col)
    arr2_now = flex_select_nb(arr2, i, col)
    if np.isnan(arr1_prev) or np.isnan(arr2_prev) or np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_prev > arr2_prev and arr1_now < arr2_now
