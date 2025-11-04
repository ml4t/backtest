# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for iterative portfolio simulation."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.generic.nb.iter_ import (
    iter_above_nb as _iter_above_nb,
)
from vectorbtpro.generic.nb.iter_ import (
    iter_below_nb as _iter_below_nb,
)
from vectorbtpro.generic.nb.iter_ import (
    iter_crossed_above_nb as _iter_crossed_above_nb,
)
from vectorbtpro.generic.nb.iter_ import (
    iter_crossed_below_nb as _iter_crossed_below_nb,
)
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted
def select_nb(
    c: tp.NamedTuple,
    arr: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> tp.Scalar:
    """Get the current element using flexible indexing.

    Args:
        c (NamedTuple): Context.
        arr (FlexArray2d): Array used for element selection.
        i (Optional[int]): Row index; if None, uses `c.i`.
        col (Optional[int]): Column index; if None, uses `c.col`.

    Returns:
        Scalar: Selected element from the array.
    """
    if i is None:
        _i = c.i
    else:
        _i = i
    if col is None:
        _col = c.col
    else:
        _col = col
    return flex_select_nb(arr, _i, _col)


@register_jitted
def select_from_col_nb(
    c: tp.NamedTuple,
    col: int,
    arr: tp.FlexArray2d,
    i: tp.Optional[int] = None,
) -> tp.Scalar:
    """Get the current element from a specified column using flexible indexing.

    Args:
        c (NamedTuple): Context.
        col (int): Column index.
        arr (FlexArray2d): Array from which to select the element.
        i (Optional[int]): Row index; if None, uses `c.i`.

    Returns:
        Scalar: Selected element from the array.
    """
    if i is None:
        _i = c.i
    else:
        _i = i
    return flex_select_nb(arr, _i, col)


@register_jitted
def iter_above_nb(
    c: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_above_nb` using the provided context.

    Args:
        c (NamedTuple): Context.
        arr1 (FlexArray2d): First array for comparison.
        arr2 (FlexArray2d): Second array for comparison.
        i (Optional[int]): Row index; if None, uses `c.i`.
        col (Optional[int]): Column index; if None, uses `c.col`.

    Returns:
        bool: True if `arr1` is above `arr2`, otherwise False.
    """
    if i is None:
        _i = c.i
    else:
        _i = i
    if col is None:
        _col = c.col
    else:
        _col = col
    return _iter_above_nb(arr1, arr2, _i, _col)


@register_jitted
def iter_below_nb(
    c: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_below_nb` using the provided context.

    Args:
        c (NamedTuple): Context.
        arr1 (FlexArray2d): First array for comparison.
        arr2 (FlexArray2d): Second array for comparison.
        i (Optional[int]): Row index; if None, uses `c.i`.
        col (Optional[int]): Column index; if None, uses `c.col`.

    Returns:
        bool: True if `arr1` is below `arr2`, otherwise False.
    """
    if i is None:
        _i = c.i
    else:
        _i = i
    if col is None:
        _col = c.col
    else:
        _col = col
    return _iter_below_nb(arr1, arr2, _i, _col)


@register_jitted
def iter_crossed_above_nb(
    c: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_crossed_above_nb` using the provided context.

    Args:
        c (NamedTuple): Context.
        arr1 (FlexArray2d): First array for the crossing comparison.
        arr2 (FlexArray2d): Second array for the crossing comparison.
        i (Optional[int]): Row index; if None, uses `c.i`.
        col (Optional[int]): Column index; if None, uses `c.col`.

    Returns:
        bool: True if `arr1` has crossed above `arr2`, otherwise False.
    """
    if i is None:
        _i = c.i
    else:
        _i = i
    if col is None:
        _col = c.col
    else:
        _col = col
    return _iter_crossed_above_nb(arr1, arr2, _i, _col)


@register_jitted
def iter_crossed_below_nb(
    c: tp.NamedTuple,
    arr1: tp.FlexArray2d,
    arr2: tp.FlexArray2d,
    i: tp.Optional[int] = None,
    col: tp.Optional[int] = None,
) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_crossed_below_nb` using the provided context.

    Args:
        c (NamedTuple): Context.
        arr1 (FlexArray2d): First array for the crossing comparison.
        arr2 (FlexArray2d): Second array for the crossing comparison.
        i (Optional[int]): Row index; if None, uses `c.i`.
        col (Optional[int]): Column index; if None, uses `c.col`.

    Returns:
        bool: True if `arr1` has crossed below `arr2`, otherwise False.
    """
    if i is None:
        _i = c.i
    else:
        _i = i
    if col is None:
        _col = c.col
    else:
        _col = col
    return _iter_crossed_below_nb(arr1, arr2, _i, _col)
