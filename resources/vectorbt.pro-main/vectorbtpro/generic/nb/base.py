# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for base operations."""

import numpy as np
from numba import prange
from numba.core.types import Omitted, Type
from numba.extending import overload
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_nb, flex_select_col_nb
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch


def _select_indices_1d_nb(arr, indices, fill_value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(arr, indices, fill_value):
        out = np.empty(indices.shape, dtype=dtype)
        for i in range(indices.shape[0]):
            if 0 <= indices[i] <= arr.shape[0] - 1:
                out[i] = arr[indices[i]]
            else:
                out[i] = fill_value
        return out

    if not nb_enabled:
        return _impl(arr, indices, fill_value)

    return _impl


overload(_select_indices_1d_nb)(_select_indices_1d_nb)


@register_jitted(cache=True)
def select_indices_1d_nb(arr: tp.Array1d, indices: tp.Array1d, fill_value: tp.Scalar) -> tp.Array1d:
    """Select elements from a 1-D array based on provided indices.

    Args:
        arr (Array1d): 1-dimensional input array.
        indices (Array1d): Array of integer indices for selection.
        fill_value (Scalar): Value to assign for indices that are out of bounds.

    Returns:
        Array1d: Array containing the selected elements.
    """
    return _select_indices_1d_nb(arr, indices, fill_value)


def _select_indices_nb(arr, indices, fill_value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(arr, indices, fill_value):
        out = np.empty(indices.shape, dtype=dtype)
        for col in range(indices.shape[1]):
            for i in range(indices.shape[0]):
                if 0 <= indices[i, col] <= arr.shape[0] - 1:
                    out[i, col] = arr[indices[i, col], col]
                else:
                    out[i, col] = fill_value
        return out

    if not nb_enabled:
        return _impl(arr, indices, fill_value)

    return _impl


overload(_select_indices_nb)(_select_indices_nb)


@register_jitted(cache=True)
def select_indices_nb(arr: tp.Array2d, indices: tp.Array2d, fill_value: tp.Scalar) -> tp.Array2d:
    """Select elements from a 2-D array based on provided indices.

    Args:
        arr (Array2d): 2-dimensional input array.
        indices (Array2d): Array of integer indices for selection.
        fill_value (Scalar): Value to assign for indices that are out of bounds.

    Returns:
        Array2d: Array containing the selected elements.
    """
    return _select_indices_nb(arr, indices, fill_value)


@register_jitted(cache=True)
def shuffle_1d_nb(arr: tp.Array1d, seed: tp.Optional[int] = None) -> tp.Array1d:
    """Shuffle elements in a 1-D array.

    Args:
        arr (Array1d): Input array to shuffle.
        seed (Optional[int]): Random seed for deterministic output.

    Returns:
        Array1d: Shuffled array.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(arr)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), seed=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def shuffle_nb(arr: tp.Array2d, seed: tp.Optional[int] = None) -> tp.Array2d:
    """Shuffle each column in a 2-D array independently.

    Args:
        arr (Array2d): 2-dimensional input array.
        seed (Optional[int]): Random seed for deterministic output.

    Returns:
        Array2d: Array with each column's elements shuffled.

    !!! tip
        This function is parallelizable.
    """
    if seed is not None:
        np.random.seed(seed)
    out = np.empty_like(arr, dtype=arr.dtype)

    for col in range(arr.shape[1]):
        out[:, col] = np.random.permutation(arr[:, col])
    return out


def _set_by_mask_1d_nb(arr, mask, value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(arr, mask, value):
        out = arr.astype(dtype)
        out[mask] = value
        return out

    if not nb_enabled:
        return _impl(arr, mask, value)

    return _impl


overload(_set_by_mask_1d_nb)(_set_by_mask_1d_nb)


@register_jitted(cache=True)
def set_by_mask_1d_nb(arr: tp.Array1d, mask: tp.Array1d, value: tp.Scalar) -> tp.Array1d:
    """Set elements in a 1-D array to a specified value using a boolean mask.

    Args:
        arr (Array1d): Input array.
        mask (Array1d): Boolean mask indicating which elements to update.
        value (Scalar): Value to assign where the mask is True.

    Returns:
        Array1d: Array with updated values.
    """
    return _set_by_mask_1d_nb(arr, mask, value)


def _set_by_mask_nb(arr, mask, value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(arr, mask, value):
        out = arr.astype(dtype)
        for col in range(arr.shape[1]):
            out[mask[:, col], col] = value
        return out

    if not nb_enabled:
        return _impl(arr, mask, value)

    return _impl


overload(_set_by_mask_nb)(_set_by_mask_nb)


@register_jitted(cache=True)
def set_by_mask_nb(arr: tp.Array2d, mask: tp.Array2d, value: tp.Scalar) -> tp.Array2d:
    """Set elements in a 2-D array to a specified value using a boolean mask.

    Args:
        arr (Array2d): 2-dimensional input array.
        mask (Array2d): Boolean mask indicating which elements to update.
        value (Scalar): Value to assign where the mask is True.

    Returns:
        Array2d: Array with updated values.
    """
    return _set_by_mask_nb(arr, mask, value)


def _set_by_mask_mult_1d_nb(arr, mask, values):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(values.dtype)
    else:
        a_dtype = arr.dtype
        value_dtype = values.dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(arr, mask, values):
        out = arr.astype(dtype)
        out[mask] = values[mask]
        return out

    if not nb_enabled:
        return _impl(arr, mask, values)

    return _impl


overload(_set_by_mask_mult_1d_nb)(_set_by_mask_mult_1d_nb)


@register_jitted(cache=True)
def set_by_mask_mult_1d_nb(arr: tp.Array1d, mask: tp.Array1d, values: tp.Array1d) -> tp.Array1d:
    """Set elements in a 1-D array to corresponding values from another array using a boolean mask.

    Args:
        arr (Array1d): Input array.
        mask (Array1d): Boolean mask indicating which elements to update.
        values (Array1d): Array of values to assign where the mask is True; must be the same shape as `arr`.

    Returns:
        Array1d: Array with updated values.
    """
    return _set_by_mask_mult_1d_nb(arr, mask, values)


def _set_by_mask_mult_nb(arr, mask, values):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(values.dtype)
    else:
        a_dtype = arr.dtype
        value_dtype = values.dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(arr, mask, values):
        out = arr.astype(dtype)
        for col in range(arr.shape[1]):
            out[mask[:, col], col] = values[mask[:, col], col]
        return out

    if not nb_enabled:
        return _impl(arr, mask, values)

    return _impl


overload(_set_by_mask_mult_nb)(_set_by_mask_mult_nb)


@register_jitted(cache=True)
def set_by_mask_mult_nb(arr: tp.Array2d, mask: tp.Array2d, values: tp.Array2d) -> tp.Array2d:
    """Set elements in a 2-D array to corresponding values from another array using a boolean mask.

    Args:
        arr (Array2d): 2-dimensional input array.
        mask (Array2d): Boolean mask indicating which elements to update.
        values (Array2d): Array of values to assign where the mask is True; must be the same shape as `arr`.

    Returns:
        Array2d: Array with updated values.
    """
    return _set_by_mask_mult_nb(arr, mask, values)


@register_jitted(cache=True)
def first_valid_index_1d_nb(arr: tp.Array1d, check_inf: bool = True) -> int:
    """Return the index of the first valid element in a 1D array.

    Args:
        arr (Array1d): Input array to search.
        check_inf (bool): Whether to treat infinity as an invalid value.

    Returns:
        int: Index of the first valid element, or -1 if no valid element is found.
    """
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]) and (not check_inf or not np.isinf(arr[i])):
            return i
    return -1


@register_jitted(cache=True)
def first_valid_index_nb(arr: tp.Array2d, check_inf: bool = True) -> tp.Array1d:
    """Return an array of indices for the first valid elements from each column in a 2D array.

    Args:
        arr (Array2d): 2D array to search column-wise.
        check_inf (bool): Whether to treat infinity as an invalid value.

    Returns:
        Array1d: Array of indices where each element represents the first valid value index
            in the corresponding column, or -1 if none is found.
    """
    out = np.empty(arr.shape[1], dtype=int_)
    for col in range(arr.shape[1]):
        out[col] = first_valid_index_1d_nb(arr[:, col], check_inf=check_inf)
    return out


@register_jitted(cache=True)
def last_valid_index_1d_nb(arr: tp.Array1d, check_inf: bool = True) -> int:
    """Return the index of the last valid element in a 1D array.

    Args:
        arr (Array1d): Input array to search.
        check_inf (bool): Whether to treat infinity as an invalid value.

    Returns:
        int: Index of the last valid element, or -1 if no valid element is found.
    """
    for i in range(arr.shape[0] - 1, -1, -1):
        if not np.isnan(arr[i]) and (not check_inf or not np.isinf(arr[i])):
            return i
    return -1


@register_jitted(cache=True)
def last_valid_index_nb(arr: tp.Array2d, check_inf: bool = True) -> tp.Array1d:
    """Return an array of indices for the last valid elements from each column in a 2D array.

    Args:
        arr (Array2d): 2D array to search column-wise.
        check_inf (bool): Whether to treat infinity as an invalid value.

    Returns:
        Array1d: Array of indices where each element represents the last valid value index
            in the corresponding column, or -1 if none is found.
    """
    out = np.empty(arr.shape[1], dtype=int_)
    for col in range(arr.shape[1]):
        out[col] = last_valid_index_1d_nb(arr[:, col], check_inf=check_inf)
    return out


@register_jitted(cache=True)
def fillna_1d_nb(arr: tp.Array1d, value: tp.Scalar) -> tp.Array1d:
    """Return a 1D array with NaN values replaced by a specified value.

    Numba equivalent to `pd.Series(arr).fillna(value)`.

    Args:
        arr (Array1d): Input array in which to replace NaN values.
        value (Scalar): Value to use for replacing NaNs.

    Returns:
        Array1d: Resulting array with NaNs replaced.
    """
    return set_by_mask_1d_nb(arr, np.isnan(arr), value)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), value=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def fillna_nb(arr: tp.Array2d, value: tp.Scalar) -> tp.Array2d:
    """Return a 2D array with NaN values replaced by a specified value column-wise.

    Args:
        arr (Array2d): Input 2D array in which to replace NaN values.
        value (Scalar): Value to use for replacing NaNs.

    Returns:
        Array2d: Resulting array with NaNs replaced.

    !!! tip
        This function is parallelizable.
    """
    return set_by_mask_nb(arr, np.isnan(arr), value)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def fbfill_nb(arr: tp.Array2d) -> tp.Array2d:
    """Return a 2D array with NaN values filled using forward and backward fill.

    Args:
        arr (Array2d): Input 2D array to process.

    Returns:
        Array2d: Array with NaN values filled by forward and backward fill.

    !!! note
        If there are no NaN (or any) values, will return `arr`.

    !!! tip
        This function is parallelizable.
    """
    if arr.size == 0:
        return arr
    need_fbfill = False
    for col in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            if np.isnan(arr[i, col]):
                need_fbfill = True
                break
        if need_fbfill:
            break
    if not need_fbfill:
        return arr

    out = np.empty_like(arr)
    for col in prange(arr.shape[1]):
        last_valid = np.nan
        for i in range(arr.shape[0]):
            if not np.isnan(arr[i, col]):
                last_valid = arr[i, col]
            out[i, col] = last_valid
        if np.isnan(out[0, col]):
            last_valid = np.nan
            for i in range(arr.shape[0] - 1, -1, -1):
                if not np.isnan(arr[i, col]):
                    last_valid = arr[i, col]
                if np.isnan(out[i, col]):
                    out[i, col] = last_valid
    return out


def _bshift_1d_nb(arr, n, fill_value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _impl(arr, n, fill_value):
        out = np.empty(arr.shape[0], dtype=dtype)
        for i in range(out.shape[0]):
            if i + n <= out.shape[0] - 1:
                out[i] = arr[i + n]
            else:
                out[i] = fill_value
        return out

    if not nb_enabled:
        return _impl(arr, n, fill_value)

    return _impl


overload(_bshift_1d_nb)(_bshift_1d_nb)


@register_jitted(cache=True)
def bshift_1d_nb(arr: tp.Array1d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array1d:
    """Return a 1D array shifted backward by n positions, with missing elements filled by a specified value.

    Numba equivalent to `pd.Series(arr).shift(-n)`.

    Args:
        arr (Array1d): Input array to shift.
        n (int): Number of positions to shift.
        fill_value (Scalar): Value used to fill positions that exceed the array boundaries.

    Returns:
        Array1d: Shifted array.

    !!! warning
        This operation looks ahead.
    """
    return _bshift_1d_nb(arr, n, fill_value)


def _bshift_nb(arr, n, fill_value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _impl(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        for col in range(arr.shape[1]):
            out[:, col] = bshift_1d_nb(arr[:, col], n=n, fill_value=fill_value)
        return out

    if not nb_enabled:
        return _impl(arr, n, fill_value)

    return _impl


overload(_bshift_nb)(_bshift_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None, fill_value=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def bshift_nb(arr: tp.Array2d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array2d:
    """Return a 2D array with each column shifted backward by n positions,
    with missing elements filled by a specified value.

    Args:
        arr (Array2d): Input 2D array to shift.
        n (int): Number of positions to shift.
        fill_value (Scalar): Value used to fill positions that exceed the array boundaries.

    Returns:
        Array2d: Resulting 2D array with backward shift applied to each column.

    !!! tip
        This function is parallelizable.
    """
    return _bshift_nb(arr, n, fill_value)


def _fshift_1d_nb(arr, n, fill_value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _impl(arr, n, fill_value):
        out = np.empty(arr.shape[0], dtype=dtype)
        for i in range(out.shape[0]):
            if i - n >= 0:
                out[i] = arr[i - n]
            else:
                out[i] = fill_value
        return out

    if not nb_enabled:
        return _impl(arr, n, fill_value)

    return _impl


overload(_fshift_1d_nb)(_fshift_1d_nb)


@register_jitted(cache=True)
def fshift_1d_nb(arr: tp.Array1d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array1d:
    """Return a 1D array shifted forward by n positions, with missing elements filled by a specified value.

    Numba equivalent to `pd.Series(arr).shift(n)`.

    Args:
        arr (Array1d): Input array to shift.
        n (int): Number of positions to shift.
        fill_value (Scalar): Value used to fill positions that go out-of-bound.

    Returns:
        Array1d: Shifted array.
    """
    return _fshift_1d_nb(arr, n, fill_value)


def _fshift_nb(arr, n, fill_value):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _impl(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        for col in range(arr.shape[1]):
            out[:, col] = fshift_1d_nb(arr[:, col], n=n, fill_value=fill_value)
        return out

    if not nb_enabled:
        return _impl(arr, n, fill_value)

    return _impl


overload(_fshift_nb)(_fshift_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None, fill_value=None),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def fshift_nb(arr: tp.Array2d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array2d:
    """Return a 2-dimensional shifted version of the input array.

    Args:
        arr (Array2d): Input 2-dimensional array.
        n (int): Number of positions to shift the array.
        fill_value (Scalar): Value used to fill the newly created empty positions.

    Returns:
        Array2d: Shifted array.

    !!! tip
        This function is parallelizable.
    """
    return _fshift_nb(arr, n, fill_value)


@register_jitted(cache=True)
def diff_1d_nb(arr: tp.Array1d, n: int = 1) -> tp.Array1d:
    """Compute the discrete difference for a 1-dimensional array.

    Numba-optimized equivalent of `pd.Series(arr).diff()`.

    Args:
        arr (Array1d): Input array.
        n (int): Interval over which to compute the difference.

    Returns:
        Array1d: Array of discrete differences.
    """
    out = np.empty(arr.shape[0], dtype=float_)
    for i in range(out.shape[0]):
        if i - n >= 0:
            out[i] = arr[i] - arr[i - n]
        else:
            out[i] = np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def diff_nb(arr: tp.Array2d, n: int = 1) -> tp.Array2d:
    """Return a 2-dimensional array of discrete differences computed column-wise.

    Args:
        arr (Array2d): Input 2-dimensional array.
        n (int): Interval over which to compute the difference.

    Returns:
        Array2d: 2-dimensional array where each column represents the discrete differences
            of the corresponding input column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = diff_1d_nb(arr[:, col], n=n)
    return out


@register_jitted(cache=True)
def pct_change_1d_nb(arr: tp.Array1d, n: int = 1) -> tp.Array1d:
    """Compute the percentage change for a 1-dimensional array.

    Numba-optimized equivalent of `pd.Series(arr).pct_change()`.

    Args:
        arr (Array1d): Input array.
        n (int): Period over which to compute the percentage change.

    Returns:
        Array1d: Array of percentage changes.
    """
    out = np.empty(arr.shape[0], dtype=float_)
    for i in range(out.shape[0]):
        if i - n >= 0:
            out[i] = (arr[i] - arr[i - n]) / arr[i - n]
        else:
            out[i] = np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), n=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pct_change_nb(arr: tp.Array2d, n: int = 1) -> tp.Array2d:
    """Return a 2-dimensional array of percentage changes computed column-wise.

    Args:
        arr (Array2d): Input 2-dimensional array.
        n (int): Period over which to compute the percentage change.

    Returns:
        Array2d: 2-dimensional array where each column represents the percentage changes
            of the corresponding input column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = pct_change_1d_nb(arr[:, col], n=n)
    return out


@register_jitted(cache=True)
def bfill_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Fill NaN values in a 1-dimensional array by backward propagation of the next valid observation.

    Numba-optimized equivalent of `pd.Series(arr).fillna(method='bfill')`.

    Args:
        arr (Array1d): Input array with potential NaN values.

    Returns:
        Array1d: Array with NaN values filled using backward propagation.

    !!! warning
        This operation looks ahead.
    """
    out = np.empty_like(arr, dtype=arr.dtype)
    lastval = arr[-1]
    for i in range(arr.shape[0] - 1, -1, -1):
        if np.isnan(arr[i]):
            out[i] = lastval
        else:
            lastval = out[i] = arr[i]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bfill_nb(arr: tp.Array2d) -> tp.Array2d:
    """Return a 2-dimensional array with NaN values filled by backward propagation computed column-wise.

    Args:
        arr (Array2d): Input 2-dimensional array with potential NaN values.

    Returns:
        Array2d: 2-dimensional array where each column has been filled using
            backward propagation via `bfill_1d_nb`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[:, col] = bfill_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def ffill_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Fill NaN values in a 1-dimensional array using forward propagation of the last valid observation.

    Numba-optimized equivalent of `pd.Series(arr).fillna(method='ffill')`.

    Args:
        arr (Array1d): Input array with potential NaN values.

    Returns:
        Array1d: Array with NaN values filled using forward propagation.
    """
    out = np.empty_like(arr, dtype=arr.dtype)
    lastval = arr[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            out[i] = lastval
        else:
            lastval = out[i] = arr[i]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ffill_nb(arr: tp.Array2d) -> tp.Array2d:
    """Return a 2-dimensional array with NaN values filled by forward propagation computed column-wise.

    Args:
        arr (Array2d): Input 2-dimensional array with potential NaN values.

    Returns:
        Array2d: 2-dimensional array where each column has been filled using
            forward propagation via `ffill_1d_nb`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[:, col] = ffill_1d_nb(arr[:, col])
    return out


def _nanprod_nb(arr):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _impl(arr):
        out = np.empty(arr.shape[1], dtype=dtype)
        for col in prange(arr.shape[1]):
            out[col] = np.nanprod(arr[:, col])
        return out

    if not nb_enabled:
        return _impl(arr)

    return _impl


overload(_nanprod_nb)(_nanprod_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True)
def nanprod_nb(arr: tp.Array2d) -> tp.Array1d:
    """Compute the product of array elements over axis 0, ignoring NaN values,
    using a Numba-optimized algorithm.

    Args:
        arr (Array2d): Input 2-dimensional array.

    Returns:
        Array1d: Array containing the product for each column.

    !!! tip
        This function is parallelizable.
    """
    return _nanprod_nb(arr)


def _nancumsum_nb(arr):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _impl(arr):
        out = np.empty(arr.shape, dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = np.nancumsum(arr[:, col])
        return out

    if not nb_enabled:
        return _impl(arr)

    return _impl


overload(_nancumsum_nb)(_nancumsum_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def nancumsum_nb(arr: tp.Array2d) -> tp.Array2d:
    """Compute the cumulative sum of array elements over axis 0, ignoring NaN values,
    with Numba optimization.

    Args:
        arr (Array2d): Input 2-dimensional array.

    Returns:
        Array2d: 2-dimensional array of cumulative sums computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    return _nancumsum_nb(arr)


def _nancumprod_nb(arr):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _impl(arr):
        out = np.empty(arr.shape, dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = np.nancumprod(arr[:, col])
        return out

    if not nb_enabled:
        return _impl(arr)

    return _impl


overload(_nancumprod_nb)(_nancumprod_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def nancumprod_nb(arr: tp.Array2d) -> tp.Array2d:
    """Compute the cumulative product of array elements over axis 0, ignoring NaN values,
    using a Numba-optimized approach.

    Args:
        arr (Array2d): Input 2-dimensional array.

    Returns:
        Array2d: 2-dimensional array of cumulative products computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    return _nancumprod_nb(arr)


def _nansum_nb(arr):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _impl(arr):
        out = np.empty(arr.shape[1], dtype=dtype)
        for col in prange(arr.shape[1]):
            out[col] = np.nansum(arr[:, col])
        return out

    if not nb_enabled:
        return _impl(arr)

    return _impl


overload(_nansum_nb)(_nansum_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True)
def nansum_nb(arr: tp.Array2d) -> tp.Array1d:
    """Compute the sum of a 2D array along axis 0 while ignoring NaN values using Numba.

    Args:
        arr (Array2d): Two-dimensional array containing numerical values with possible NaNs.

    Returns:
        Array1d: One-dimensional array with the column-wise sums, ignoring NaNs.

    !!! tip
        This function is parallelizable.
    """
    return _nansum_nb(arr)


@register_jitted(cache=True)
def nancnt_1d_nb(arr: tp.Array1d) -> int:
    """Count the non-NaN elements in a 1D array without allocating extra arrays.

    Args:
        arr (Array1d): One-dimensional array containing numerical values.

    Returns:
        int: Count of non-NaN elements in the array.
    """
    cnt = 0
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            cnt += 1
    return cnt


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nancnt_nb(arr: tp.Array2d) -> tp.Array1d:
    """Count the non-NaN elements in each column of a 2D array without additional allocations.

    Args:
        arr (Array2d): Two-dimensional array containing numerical values with possible NaNs.

    Returns:
        Array1d: One-dimensional array where each element is the count of non-NaN values
            in the corresponding column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[1], dtype=int_)
    for col in prange(arr.shape[1]):
        out[col] = nancnt_1d_nb(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmin_nb(arr: tp.Array2d) -> tp.Array1d:
    """Compute the minimum value of each column in a 2D array while ignoring NaNs using Numba.

    Args:
        arr (Array2d): Two-dimensional array containing numerical values with potential NaNs.

    Returns:
        Array1d: One-dimensional array with the minimum value from each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmin(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmax_nb(arr: tp.Array2d) -> tp.Array1d:
    """Compute the maximum value of each column in a 2D array while ignoring NaNs using Numba.

    Args:
        arr (Array2d): Two-dimensional array containing numerical values with possible NaN entries.

    Returns:
        Array1d: One-dimensional array with the maximum value from each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmax(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmean_nb(arr: tp.Array2d) -> tp.Array1d:
    """Compute the mean of each column in a 2D array while ignoring NaN values using Numba.

    Args:
        arr (Array2d): Two-dimensional array containing numerical values with potential NaNs.

    Returns:
        Array1d: One-dimensional array with the column-wise means.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[1], dtype=float_)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmean(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanmedian_nb(arr: tp.Array2d) -> tp.Array1d:
    """Compute the median of each column in a 2D array while ignoring NaN values using Numba.

    Args:
        arr (Array2d): Two-dimensional array containing numerical values with possible NaNs.

    Returns:
        Array1d: One-dimensional array with the median value from each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[1], dtype=float_)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmedian(arr[:, col])
    return out


@register_jitted(cache=True)
def nanpercentile_noarr_1d_nb(arr: tp.Array1d, q: float) -> float:
    """Compute the percentile of a 1D array while ignoring NaN values without allocating additional arrays.

    Args:
        arr (Array1d): One-dimensional array containing numerical values that may include NaNs.
        q (float): Percentile to compute, where 0 represents the minimum and 100 represents the maximum.

    Returns:
        float: Computed percentile value.

    !!! note
        Has worst case time complexity of O(N^2), which makes it much slower than `np.nanpercentile`,
        but still faster if used in rolling calculations, especially for `q` near 0 and 100.
    """
    if q < 0:
        q = 0
    elif q > 100:
        q = 100
    do_min = q < 50
    if not do_min:
        q = 100 - q
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    if cnt == 0:
        return np.nan
    nth_float = q / 100 * (cnt - 1)
    if nth_float % 1 == 0:
        nth1 = nth2 = int(nth_float)
    else:
        nth1 = int(nth_float)
        nth2 = nth1 + 1
    found1 = np.nan
    found2 = np.nan
    k = 0
    if do_min:
        prev_val = -np.inf
    else:
        prev_val = np.inf
    while True:
        n_same = 0
        if do_min:
            curr_val = np.inf
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i]):
                    if arr[i] > prev_val:
                        if arr[i] < curr_val:
                            curr_val = arr[i]
                            n_same = 0
                        if arr[i] == curr_val:
                            n_same += 1
        else:
            curr_val = -np.inf
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i]):
                    if arr[i] < prev_val:
                        if arr[i] > curr_val:
                            curr_val = arr[i]
                            n_same = 0
                        if arr[i] == curr_val:
                            n_same += 1
        prev_val = curr_val
        k += n_same
        if np.isnan(found1) and k >= nth1 + 1:
            found1 = curr_val
        if np.isnan(found2) and k >= nth2 + 1:
            found2 = curr_val
            break
    if found1 == found2:
        return found1
    factor = (nth_float - nth1) / (nth2 - nth1)
    return factor * (found2 - found1) + found1


@register_jitted(cache=True)
def nanpartition_mean_noarr_1d_nb(arr: tp.Array1d, q: float) -> float:
    """Compute the average value using a partitioning algorithm while ignoring NaN values
    without additional allocations.

    Args:
        arr (Array1d): One-dimensional array containing numerical values that may include NaNs.
        q (float): Percentile threshold guiding the partitioning,
            where 0 represents the minimum and 100 represents the maximum.

    Returns:
        float: Computed average of the partitioned elements.

    !!! note
        Has worst case time complexity of O(N^2), which makes it much slower than `np.partition`,
        but still faster if used in rolling calculations, especially for `q` near 0.
    """
    if q < 0:
        q = 0
    elif q > 100:
        q = 100
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    if cnt == 0:
        return np.nan
    nth = int(q / 100 * (cnt - 1))
    prev_val = -np.inf
    partition_sum = 0.0
    partition_cnt = 0
    k = 0
    while True:
        n_same = 0
        curr_val = np.inf
        for i in range(arr.shape[0]):
            if not np.isnan(arr[i]):
                if arr[i] > prev_val:
                    if arr[i] < curr_val:
                        curr_val = arr[i]
                        n_same = 0
                    if arr[i] == curr_val:
                        n_same += 1
        if k + n_same >= nth + 1:
            partition_sum += (nth + 1 - k) * curr_val
            partition_cnt += nth + 1 - k
            break
        else:
            partition_sum += n_same * curr_val
            partition_cnt += n_same
        prev_val = curr_val
        k += n_same
    return partition_sum / partition_cnt


@register_jitted(cache=True)
def nanvar_1d_nb(arr: tp.Array1d, ddof: int = 0) -> float:
    """Compute the variance of a 1D array while ignoring NaN values using Numba without
    allocating extra arrays.

    Args:
        arr (Array1d): One-dimensional array containing numerical values that may include NaNs.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Computed variance based on the non-NaN elements.
    """
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    rcount = max(cnt - ddof, 0)
    if rcount == 0:
        return np.nan
    out = 0.0
    a_mean = np.nanmean(arr)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            out += abs(arr[i] - a_mean) ** 2
    return out / rcount


@register_jitted(cache=True)
def nanstd_1d_nb(arr: tp.Array1d, ddof: int = 0) -> float:
    """Compute the standard deviation of a 1D array while ignoring NaN values using Numba.

    Args:
        arr (Array1d): One-dimensional array containing numerical values with possible NaNs.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Computed standard deviation based on the non-NaN elements.
    """
    return np.sqrt(nanvar_1d_nb(arr, ddof=ddof))


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nanstd_nb(arr: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """Compute the nan-standard deviation for each column in a 2-dimensional array, ignoring NaN values.

    Args:
        arr (Array2d): 2-dimensional array containing numerical data.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: 1-dimensional array of standard deviations computed for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[1], dtype=float_)
    for col in prange(arr.shape[1]):
        out[col] = nanstd_1d_nb(arr[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def nancov_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, ddof: int = 0) -> float:
    """Compute the covariance for two 1-dimensional arrays while ignoring NaN values.

    Args:
        arr1 (Array1d): First 1-dimensional numerical array.
        arr2 (Array1d): Second 1-dimensional numerical array.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Covariance between `arr1` and `arr2`, or NaN if there is insufficient valid data.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length")
    arr1_sum = 0.0
    arr2_sum = 0.0
    arr12_sumprod = 0.0
    k = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            arr1_sum += arr1[i]
            arr2_sum += arr2[i]
            arr12_sumprod += arr1[i] * arr2[i]
            k += 1
    if k - ddof <= 0:
        return np.nan
    arr1_mean = arr1_sum / k
    arr2_mean = arr2_sum / k
    return (arr12_sumprod - k * arr1_mean * arr2_mean) / (k - ddof)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), ddof=None),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nancov_nb(arr1: tp.Array2d, arr2: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """Compute the covariance for corresponding columns of two 2-dimensional arrays while ignoring NaN values.

    Args:
        arr1 (Array2d): First 2-dimensional numerical array.
        arr2 (Array2d): Second 2-dimensional numerical array.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: 1-dimensional array where each element is the covariance of the
            corresponding columns in `arr1` and `arr2`.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr1.shape[1], dtype=float_)
    for col in prange(arr1.shape[1]):
        out[col] = nancov_1d_nb(arr1[:, col], arr2[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def nancorr_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d) -> float:
    """Compute the correlation coefficient for two 1-dimensional arrays while ignoring NaN values
    using a numerically stable algorithm.

    Args:
        arr1 (Array1d): First 1-dimensional numerical array.
        arr2 (Array1d): Second 1-dimensional numerical array.

    Returns:
        float: Correlation coefficient between `arr1` and `arr2`,
            or NaN if the calculation cannot be performed.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length")
    arr1_sum = 0.0
    arr2_sum = 0.0
    arr1_sumsq = 0.0
    arr2_sumsq = 0.0
    arr12_sumprod = 0.0
    k = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            arr1_sum += arr1[i]
            arr2_sum += arr2[i]
            arr1_sumsq += float(arr1[i]) ** 2
            arr2_sumsq += float(arr2[i]) ** 2
            arr12_sumprod += arr1[i] * arr2[i]
            k += 1
    if k == 0:
        return np.nan
    arr1_mean = arr1_sum / k
    arr2_mean = arr2_sum / k
    arr1_meansq = arr1_sumsq / k
    arr2_meansq = arr2_sumsq / k
    arr12_meanprod = arr12_sumprod / k
    num = arr12_meanprod - arr1_mean * arr2_mean
    denom = np.sqrt((arr1_meansq - arr1_mean**2) * (arr2_meansq - arr2_mean**2))
    if denom == 0:
        return np.nan
    return num / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def nancorr_nb(arr1: tp.Array2d, arr2: tp.Array2d) -> tp.Array1d:
    """Compute the correlation coefficients for each corresponding column of two
    2-dimensional arrays while ignoring NaN values.

    Args:
        arr1 (Array2d): First 2-dimensional numerical array.
        arr2 (Array2d): Second 2-dimensional numerical array.

    Returns:
        Array1d: 1-dimensional array containing the correlation coefficient for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr1.shape[1], dtype=float_)
    for col in prange(arr1.shape[1]):
        out[col] = nancorr_1d_nb(arr1[:, col], arr2[:, col])
    return out


@register_jitted(cache=True)
def rank_1d_nb(
    arr: tp.Array1d, argsorted: tp.Optional[tp.Array1d] = None, pct: bool = False
) -> tp.Array1d:
    """Compute the ranks for a 1-dimensional array while ignoring NaN values.

    This function mimics the behavior of `pd.Series(arr).rank(pct=pct)` by assigning
    average ranks to tied values.

    Args:
        arr (Array1d): 1-dimensional array of numerical data.
        argsorted (Optional[Array1d]): Array of indices representing the sorted order of `arr`.

            If not provided, the sorting is computed internally.
        pct (bool): If True, compute the rank as a percentile.

    Returns:
        Array1d: 1-dimensional array containing the computed ranks for each element.
    """
    if argsorted is None:
        argsorted = np.argsort(arr)
    out = np.empty_like(arr, dtype=float_)
    rank_sum = 0
    rank_cnt = 0
    nan_cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nan_cnt += 1
    if nan_cnt == arr.shape[0]:
        out[:] = np.nan
        return out
    valid_cnt = out.shape[0] - nan_cnt
    for i in range(argsorted.shape[0]):
        rank = i + 1
        if np.isnan(arr[argsorted[i]]):
            out[argsorted[i]] = np.nan
        elif i < out.shape[0] - 1 and arr[argsorted[i]] == arr[argsorted[i + 1]]:
            rank_sum += rank
            rank_cnt += 1
            if pct:
                v = rank / valid_cnt
            else:
                v = rank
            out[argsorted[i]] = v
        elif rank_sum > 0:
            rank_sum += rank
            rank_cnt += 1
            if pct:
                v = rank_sum / rank_cnt / valid_cnt
            else:
                v = rank_sum / rank_cnt
            out[argsorted[i - rank_cnt + 1 : i + 1]] = v
            rank_sum = 0
            rank_cnt = 0
        else:
            if pct:
                v = rank / valid_cnt
            else:
                v = rank
            out[argsorted[i]] = v
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), argsorted=ch.ArraySlicer(axis=1), pct=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rank_nb(
    arr: tp.Array2d, argsorted: tp.Optional[tp.Array2d] = None, pct: bool = False
) -> tp.Array2d:
    """Compute the ranks for each column in a 2-dimensional array while ignoring NaN values.

    Args:
        arr (Array2d): 2-dimensional array of numerical data.
        argsorted (Optional[Array2d]): Array of indices for each column representing the sorted order.

            If not provided, the sorting is computed internally for each column.
        pct (bool): If True, compute the rank as a percentile.

    Returns:
        Array2d: 2-dimensional array containing the computed ranks for each element.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        if argsorted is None:
            out[:, col] = rank_1d_nb(arr[:, col], argsorted=None, pct=pct)
        else:
            out[:, col] = rank_1d_nb(arr[:, col], argsorted=argsorted[:, col], pct=pct)
    return out


@register_jitted(cache=True)
def polyfit_1d_nb(x: tp.Array1d, y: tp.Array1d, deg: int, stabilize: bool = False) -> tp.Array1d:
    """Compute least squares polynomial fit.

    Args:
        x (Array1d): Independent variable values.
        y (Array1d): Dependent variable values.
        deg (int): Degree of the polynomial.
        stabilize (bool): If True, stabilize the computation by normalizing the Vandermonde matrix columns.

    Returns:
        Array1d: Polynomial coefficients in descending order.
    """
    if stabilize:
        mat_ = np.ones(shape=(x.shape[0], deg + 1))
        mat_[:, 1] = x
        if deg > 1:
            for n in range(2, deg + 1):
                mat_[:, n] = mat_[:, n - 1] * x
        scale_vect = np.empty((deg + 1,), dtype=float_)
        for n in range(0, deg + 1):
            col_norm = np.linalg.norm(mat_[:, n])
            scale_vect[n] = col_norm
            mat_[:, n] /= col_norm
        det_ = np.linalg.lstsq(mat_, y, rcond=-1)[0] / scale_vect
    else:
        mat_ = np.zeros(shape=(x.shape[0], deg + 1))
        const = np.ones_like(x)
        mat_[:, 0] = const
        mat_[:, 1] = x
        if deg > 1:
            for n in range(2, deg + 1):
                mat_[:, n] = x**n
        det_ = np.linalg.lstsq(mat_, y, rcond=-1)[0]
    return det_[::-1]


@register_jitted(cache=True)
def fir_filter_1d_nb(b: tp.Array1d, x: tp.Array1d) -> tp.Array1d:
    """Filter data along one dimension using an FIR filter.

    Args:
        b (Array1d): FIR filter coefficients.
        x (Array1d): One-dimensional signal to filter.

    Returns:
        Array1d: Filtered signal.
    """
    n = len(x)
    m = len(b)
    y = np.zeros(n)
    for i in range(n):
        for j in range(m):
            if i - j >= 0:
                y[i] += b[j] * x[i - j]
    return y


# ############# Value counts ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="codes", axis=1),
    arg_take_spec=dict(
        codes=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        n_uniques=None,
        group_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_counts_nb(codes: tp.Array2d, n_uniques: int, group_map: tp.GroupMap) -> tp.Array2d:
    """Compute value counts per group.

    Args:
        codes (Array2d): Factorized array of integer codes.
        n_uniques (int): Number of unique values.
        group_map (GroupMap): Tuple of indices and lengths for each group.

    Returns:
        Array2d: 2-dimensional array of counts with shape (n_uniques, number of groups).

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((n_uniques, group_lens.shape[0]), 0, dtype=int_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            for i in range(codes.shape[0]):
                out[codes[i, col], group] += 1
    return out


@register_jitted(cache=True)
def value_counts_1d_nb(codes: tp.Array1d, n_uniques: int) -> tp.Array1d:
    """Compute value counts.

    Args:
        codes (Array1d): Factorized array of integer codes.
        n_uniques (int): Number of unique values.

    Returns:
        Array1d: Array of counts for each unique value.
    """
    out = np.full(n_uniques, 0, dtype=int_)

    for i in range(codes.shape[0]):
        out[codes[i]] += 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="codes", axis=0),
    arg_take_spec=dict(codes=ch.ArraySlicer(axis=0), n_uniques=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_counts_per_row_nb(codes: tp.Array2d, n_uniques: int) -> tp.Array2d:
    """Compute value counts per row.

    Args:
        codes (Array2d): Factorized array of integer codes.
        n_uniques (int): Number of unique values.

    Returns:
        Array2d: 2-dimensional array of counts with shape (n_uniques, number of rows).

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((n_uniques, codes.shape[0]), dtype=int_)

    for i in prange(codes.shape[0]):
        out[:, i] = value_counts_1d_nb(codes[i, :], n_uniques)
    return out


# ############# Repartitioning ############# #


@register_jitted(cache=True)
def repartition_nb(arr: tp.Array2d, counts: tp.Array1d) -> tp.Array1d:
    """Repartition a 2-dimensional array into a 1-dimensional array by removing empty elements.

    Args:
        arr (Array2d): Input 2-dimensional array.
        counts (Array1d): Array of counts specifying the number of valid elements per column.

    Returns:
        Array1d: 1-dimensional array composed of the valid elements from `arr`.
    """
    if arr.shape[0] == 0:
        return arr.flatten()
    out = np.empty(np.sum(counts), dtype=arr.dtype)
    j = 0
    for col in range(counts.shape[0]):
        out[j : j + counts[col]] = arr[: counts[col], col]
        j += counts[col]
    return out


# ############# Crossover ############# #


@register_jitted(cache=True)
def crossed_above_1d_nb(
    arr1: tp.Array1d, arr2: tp.FlexArray1dLike, wait: int = 0, dropna: bool = False
) -> tp.Array1d:
    """Return a boolean mask indicating where the first array crosses above the second array.

    Args:
        arr1 (Array1d): One-dimensional array of values.
        arr2 (FlexArray1dLike): One-dimensional array or scalar for comparison.
        wait (int): Number of consecutive data points required above the threshold to confirm a crossover.
        dropna (bool): If True, treat indices with NaN values as missing, mimicking the removal of rows with NaN.

    Returns:
        Array1d: Boolean array with True at positions where a crossover from below to above occurs.
    """
    arr2_ = to_1d_array_nb(np.asarray(arr2))
    out = np.empty(arr1.shape, dtype=np.bool_)
    was_below = False
    confirmed = 0

    for i in range(arr1.shape[0]):
        _arr1 = arr1[i]
        _arr2 = flex_select_1d_nb(arr2_, i)
        if np.isnan(_arr1) or np.isnan(_arr2):
            if not dropna:
                was_below = False
                confirmed = 0
            out[i] = False
        elif _arr1 > _arr2:
            if was_below:
                confirmed += 1
                out[i] = confirmed == wait + 1
            else:
                out[i] = False
        elif _arr1 == _arr2:
            if confirmed > 0:
                was_below = False
            confirmed = 0
            out[i] = False
        elif _arr1 < _arr2:
            confirmed = 0
            was_below = True
            out[i] = False
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=base_ch.FlexArraySlicer(axis=1),
        wait=None,
        dropna=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def crossed_above_nb(
    arr1: tp.Array2d, arr2: tp.FlexArray2dLike, wait: int = 0, dropna: bool = False
) -> tp.Array2d:
    """Compute a 2-dimensional boolean mask indicating where elements in the first array cross
    above the corresponding elements in the second array column-wise.

    Args:
        arr1 (Array2d): 2-dimensional array of values.
        arr2 (FlexArray2dLike): 2-dimensional array or similar structure for comparison.
        wait (int): Number of consecutive data points required above the threshold to confirm a crossover.
        dropna (bool): If True, treat rows with NaN values as missing in the crossover detection.

    Returns:
        Array2d: Boolean 2-dimensional array with True at positions where the crossover condition is met.

    !!! tip
        This function is parallelizable.
    """
    arr2_ = to_2d_array_nb(np.asarray(arr2))
    out = np.empty(arr1.shape, dtype=np.bool_)
    for col in prange(arr1.shape[1]):
        _arr2 = flex_select_col_nb(arr2_, col)
        out[:, col] = crossed_above_1d_nb(arr1[:, col], _arr2, wait=wait, dropna=dropna)
    return out


@register_jitted(cache=True)
def crossed_below_1d_nb(
    arr1: tp.Array1d, arr2: tp.FlexArray1dLike, wait: int = 0, dropna: bool = False
) -> tp.Array1d:
    """Return a boolean mask indicating where the first array crosses below the second array.

    Args:
        arr1 (Array1d): One-dimensional array of values.
        arr2 (FlexArray1dLike): One-dimensional array or scalar for comparison.
        wait (int): Number of consecutive data points required below the threshold to confirm a crossover.
        dropna (bool): If True, treat indices with NaN values as missing, mimicking the removal of rows with NaN.

    Returns:
        Array1d: Boolean array with True at positions where a crossover from above to below occurs.
    """
    arr2_ = to_1d_array_nb(np.asarray(arr2))
    out = np.empty(arr1.shape, dtype=np.bool_)
    was_above = False
    confirmed = 0

    for i in range(arr1.shape[0]):
        _arr1 = arr1[i]
        _arr2 = flex_select_1d_nb(arr2_, i)
        if np.isnan(_arr1) or np.isnan(_arr2):
            if not dropna:
                was_above = False
                confirmed = 0
            out[i] = False
        elif _arr1 < _arr2:
            if was_above:
                confirmed += 1
                out[i] = confirmed == wait + 1
            else:
                out[i] = False
        elif _arr1 == _arr2:
            if confirmed > 0:
                was_above = False
            confirmed = 0
            out[i] = False
        elif _arr1 > _arr2:
            confirmed = 0
            was_above = True
            out[i] = False
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=base_ch.FlexArraySlicer(axis=1),
        wait=None,
        dropna=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def crossed_below_nb(
    arr1: tp.Array2d, arr2: tp.FlexArray2dLike, wait: int = 0, dropna: bool = False
) -> tp.Array2d:
    """Compute a boolean matrix indicating cross-below events column-wise between `arr1` and
    `arr2` using `crossed_below_1d_nb`.

    Args:
        arr1 (Array2d): Primary 2-dimensional input array.
        arr2 (FlexArray2dLike): Secondary array used for comparison.
        wait (int): Waiting period before confirming a cross-below event.
        dropna (bool): Whether to bypass NaN values during processing.

    Returns:
        Array2d: Boolean matrix with the same shape as `arr1` indicating cross-below events.

    !!! tip
        This function is parallelizable.
    """
    arr2_ = to_2d_array_nb(np.asarray(arr2))
    out = np.empty(arr1.shape, dtype=np.bool_)
    for col in prange(arr1.shape[1]):
        _arr2 = flex_select_col_nb(arr2_, col)
        out[:, col] = crossed_below_1d_nb(arr1[:, col], _arr2, wait=wait, dropna=dropna)
    return out


# ############# Transforming ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def demean_nb(arr: tp.Array2d, group_map: tp.GroupMap) -> tp.Array2d:
    """Subtract group means from each element in a 2-dimensional array.

    Args:
        arr (Array2d): Input 2-dimensional array.
        group_map (GroupMap): Tuple of indices and lengths for each group.

    Returns:
        Array2d: 2-dimensional array with each element demeaned by its group's mean.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.empty_like(arr, dtype=float_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for i in range(arr.shape[0]):
            group_sum = 0
            group_cnt = 0
            for k in range(group_len):
                col = group_idxs[start_idx + k]
                if not np.isnan(arr[i, col]):
                    group_sum += arr[i, col]
                    group_cnt += 1
            for k in range(group_len):
                col = group_idxs[start_idx + k]
                if np.isnan(arr[i, col]) or group_cnt == 0:
                    out[i, col] = np.nan
                else:
                    out[i, col] = arr[i, col] - group_sum / group_cnt
    return out


@register_jitted(cache=True)
def to_renko_1d_nb(
    arr: tp.Array1d,
    brick_size: tp.FlexArray1dLike,
    relative: tp.FlexArray1dLike = False,
    start_value: tp.Optional[float] = None,
    max_out_len: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
    """Convert a 1-dimensional array into its Renko brick representation.

    Args:
        arr (Array1d): Input array of numerical values.
        brick_size (FlexArray1dLike): Brick size for each element.

            Provided as a scalar or per row.
        relative (FlexArray1dLike): Indicator determining if relative price changes should be used.

            Provided as a scalar or per row.
        start_value (Optional[float]): Starting value for the Renko chart.
        max_out_len (Optional[int]): Maximum length of the output.

    Returns:
        Tuple[Array1d, Array1d, Array1d]: Tuple containing:

          * Array of calculated Renko brick values.
          * Indices corresponding to each brick.
          * Boolean array indicating whether each brick represents an uptrend.
    """
    brick_size_ = to_1d_array_nb(np.asarray(brick_size))
    relative_ = to_1d_array_nb(np.asarray(relative))

    if max_out_len is None:
        out_n = arr.shape[0]
    else:
        out_n = max_out_len
    arr_out = np.empty(out_n, dtype=float_)
    idx_out = np.empty(out_n, dtype=int_)
    uptrend_out = np.empty(out_n, dtype=np.bool_)
    prev_value = np.nan
    k = 0
    trend = 0

    for i in range(arr.shape[0]):
        _brick_size = abs(flex_select_1d_nb(brick_size_, i))
        _relative = flex_select_1d_nb(relative_, i)
        curr_value = arr[i]
        if np.isnan(curr_value):
            continue
        if np.isnan(prev_value):
            if start_value is None:
                if not _relative:
                    prev_value = curr_value - curr_value % _brick_size
                else:
                    prev_value = curr_value
            else:
                prev_value = start_value
            continue
        if _relative:
            diff = (curr_value - prev_value) / prev_value
        else:
            diff = curr_value - prev_value
        while abs(diff) >= _brick_size:
            prev_trend = trend
            if diff >= 0:
                if _relative:
                    prev_value *= 1 + _brick_size
                else:
                    prev_value += _brick_size
                trend = 1
            else:
                if _relative:
                    prev_value *= 1 - _brick_size
                else:
                    prev_value -= _brick_size
                trend = -1
            if _relative:
                diff = (curr_value - prev_value) / prev_value
            else:
                diff = curr_value - prev_value
            if trend == -prev_trend:
                continue
            if k >= len(arr_out):
                raise IndexError("Index out of range. Set a higher max_out_len.")
            arr_out[k] = prev_value
            idx_out[k] = i
            uptrend_out[k] = trend == 1
            k += 1

    return arr_out[:k], idx_out[:k], uptrend_out[:k]


@register_jitted(cache=True)
def to_renko_ohlc_1d_nb(
    arr: tp.Array1d,
    brick_size: tp.FlexArray1dLike,
    relative: tp.FlexArray1dLike = False,
    start_value: tp.Optional[float] = None,
    max_out_len: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array2d, tp.Array1d]:
    """Convert a 1-dimensional array into its Renko OHLC (Open-High-Low-Close) representation.

    Args:
        arr (Array1d): Input array of numerical values.
        brick_size (FlexArray1dLike): Brick size for each element.

            Provided as a scalar or per row.
        relative (FlexArray1dLike): Indicator determining if relative price changes should be used.

            Provided as a scalar or per row.
        start_value (Optional[float]): Starting value for the Renko chart.
        max_out_len (Optional[int]): Maximum length of the output.

    Returns:
        Tuple[Array2d, Array1d]: Tuple containing:

          * 2-dimensional array where each row represents an OHLC bar.
          * Array of indices corresponding to each OHLC bar.
    """
    brick_size_ = to_1d_array_nb(np.asarray(brick_size))
    relative_ = to_1d_array_nb(np.asarray(relative))

    if max_out_len is None:
        out_n = arr.shape[0]
    else:
        out_n = max_out_len
    arr_out = np.empty((out_n, 4), dtype=float_)
    idx_out = np.empty(out_n, dtype=int_)
    prev_value = np.nan
    k = 0
    trend = 0

    for i in range(arr.shape[0]):
        _brick_size = abs(flex_select_1d_nb(brick_size_, i))
        _relative = flex_select_1d_nb(relative_, i)
        curr_value = arr[i]
        if np.isnan(curr_value):
            continue
        if np.isnan(prev_value):
            if start_value is None:
                if not _relative:
                    prev_value = curr_value - curr_value % _brick_size
                else:
                    prev_value = curr_value
            else:
                prev_value = start_value
            continue
        if _relative:
            diff = (curr_value - prev_value) / prev_value
        else:
            diff = curr_value - prev_value
        while abs(diff) >= _brick_size:
            open_value = prev_value
            prev_trend = trend
            if diff >= 0:
                if _relative:
                    prev_value *= 1 + _brick_size
                else:
                    prev_value += _brick_size
                trend = 1
            else:
                if _relative:
                    prev_value *= 1 - _brick_size
                else:
                    prev_value -= _brick_size
                trend = -1
            if _relative:
                diff = (curr_value - prev_value) / prev_value
            else:
                diff = curr_value - prev_value
            if trend == -prev_trend:
                continue
            if k >= len(arr_out):
                raise IndexError("Index out of range. Set a higher max_out_len.")
            if trend == 1:
                high_value = prev_value
                low_value = open_value
            else:
                high_value = open_value
                low_value = prev_value
            close_value = prev_value
            arr_out[k, 0] = open_value
            arr_out[k, 1] = high_value
            arr_out[k, 2] = low_value
            arr_out[k, 3] = close_value
            idx_out[k] = i
            k += 1

    return arr_out[:k], idx_out[:k]


# ############# Resampling ############# #


def _realign_1d_nb(
    arr,
    source_index,
    target_index,
    source_freq,
    target_freq,
    source_rbound,
    target_rbound,
    nan_value,
    ffill,
):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    ):
        out = np.empty(target_index.shape[0], dtype=dtype)
        curr_j = -1
        last_j = curr_j
        last_valid = np.nan
        for i in range(len(target_index)):
            if i > 0 and target_index[i] < target_index[i - 1]:
                raise ValueError("Target index must be increasing")
            target_bound_inf = target_rbound and i == len(target_index) - 1 and target_freq is None

            last_valid_at_i = np.nan
            for j in range(curr_j + 1, source_index.shape[0]):
                if j > 0 and source_index[j] < source_index[j - 1]:
                    raise ValueError("Array index must be increasing")
                source_bound_inf = (
                    source_rbound and j == len(source_index) - 1 and source_freq is None
                )

                if source_bound_inf and target_bound_inf:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    break
                if source_bound_inf:
                    break
                if target_bound_inf:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    continue

                if source_rbound and target_rbound:
                    if source_freq is None:
                        source_val = source_index[j + 1]
                    else:
                        source_val = source_index[j] + source_freq
                    if target_freq is None:
                        target_val = target_index[i + 1]
                    else:
                        target_val = target_index[i] + target_freq
                    if source_val > target_val:
                        break
                elif source_rbound:
                    if source_freq is None:
                        source_val = source_index[j + 1]
                    else:
                        source_val = source_index[j] + source_freq
                    if source_val > target_index[i]:
                        break
                elif target_rbound:
                    if target_freq is None:
                        target_val = target_index[i + 1]
                    else:
                        target_val = target_index[i] + target_freq
                    if source_index[j] >= target_val:
                        break
                else:
                    if source_index[j] > target_index[i]:
                        break
                curr_j = j
                if not np.isnan(arr[curr_j]):
                    last_valid_at_i = arr[curr_j]

            if ffill and not np.isnan(last_valid_at_i):
                last_valid = last_valid_at_i
            if curr_j == -1 or (not ffill and curr_j == last_j):
                out[i] = nan_value
            else:
                if ffill:
                    if np.isnan(last_valid):
                        out[i] = nan_value
                    else:
                        out[i] = last_valid
                else:
                    if np.isnan(last_valid_at_i):
                        out[i] = nan_value
                    else:
                        out[i] = last_valid_at_i
                last_j = curr_j

        return out

    if not nb_enabled:
        return _impl(
            arr,
            source_index,
            target_index,
            source_freq,
            target_freq,
            source_rbound,
            target_rbound,
            nan_value,
            ffill,
        )

    return _impl


overload(_realign_1d_nb)(_realign_1d_nb)


@register_jitted(cache=True)
def realign_1d_nb(
    arr: tp.Array1d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
    source_rbound: bool = False,
    target_rbound: bool = False,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array1d:
    """Return the latest available value in `arr` for each index in `target_index` based on `source_index`.

    This function identifies, for each target index, the most recent corresponding value from `arr`.
    If `source_rbound` is True, each element in `source_index` is treated as a right-bound value,
    determined using the specified `source_freq` or the next index when `source_freq` is None.
    The same approach applies to `target_rbound` and `target_index`.

    Args:
        arr (Array1d): One-dimensional array of values.
        source_index (Array1d): Indices corresponding to the values in `arr`.
        target_index (Array1d): Array of target indices, which must be strictly increasing.
        source_freq (Optional[Scalar]): Frequency applied to `source_index`.
        target_freq (Optional[Scalar]): Frequency applied to `target_index`.
        source_rbound (bool): Indicates if `source_index` values represent right bounds.
        target_rbound (bool): Indicates if `target_index` values represent right bounds.
        nan_value (Scalar): Value representing missing data.
        ffill (bool): If True, forward fill missing values.

    Returns:
        Array1d: Realigned 1-dimensional array.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed.

        If `arr` contains bar data, both index arrays must represent the opening time.
    """
    return _realign_1d_nb(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    )


def _realign_nb(
    arr,
    source_index,
    target_index,
    source_freq,
    target_freq,
    source_rbound,
    target_rbound,
    nan_value,
    ffill,
):
    nb_enabled = isinstance(arr, Type)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _impl(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    ):
        out = np.empty((target_index.shape[0], arr.shape[1]), dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = realign_1d_nb(
                arr[:, col],
                source_index,
                target_index,
                source_freq=source_freq,
                target_freq=target_freq,
                source_rbound=source_rbound,
                target_rbound=target_rbound,
                nan_value=nan_value,
                ffill=ffill,
            )
        return out

    if not nb_enabled:
        return _impl(
            arr,
            source_index,
            target_index,
            source_freq,
            target_freq,
            source_rbound,
            target_rbound,
            nan_value,
            ffill,
        )

    return _impl


overload(_realign_nb)(_realign_nb)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        source_index=None,
        target_index=None,
        source_freq=None,
        target_freq=None,
        source_rbound=None,
        target_rbound=None,
        nan_value=None,
        ffill=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True)
def realign_nb(
    arr: tp.Array2d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
    source_rbound: bool = False,
    target_rbound: bool = False,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array2d:
    """Return a two-dimensional array with values realigned for each column based on `target_index`.

    This function applies `realign_1d_nb` to each column of the input array, aligning values
    according to corresponding source and target indices.

    Args:
        arr (Array2d): Two-dimensional array of values.
        source_index (Array1d): Indices corresponding to the rows of `arr`.
        target_index (Array1d): Array of target indices, which must be strictly increasing.
        source_freq (Optional[Scalar]): Frequency applied to `source_index`.
        target_freq (Optional[Scalar]): Frequency applied to `target_index`.
        source_rbound (bool): Indicates if `source_index` values represent right bounds.
        target_rbound (bool): Indicates if `target_index` values represent right bounds.
        nan_value (Scalar): Value representing missing data.
        ffill (bool): If True, forward fill missing values.

    Returns:
        Array2d: Realigned 2-dimensional array.

    !!! tip
        This function is parallelizable.
    """
    return _realign_nb(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    )
