# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for mapping, applying, and reducing."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic.nb.base import nancorr_1d_nb, nancov_1d_nb, nanstd_1d_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

# ############# Map, apply, and reduce ############# #


@register_jitted
def map_1d_nb(arr: tp.Array1d, map_func_nb: tp.MapFunc, *args) -> tp.Array1d:
    """Map each element of a 1-dimensional array using `map_func_nb`.

    Args:
        arr (Array1d): 1D array of input elements.
        map_func_nb (MapFunc): Callback function that accepts an element in `arr` and
            additional arguments, and returns a single value.
        *args: Positional arguments for `map_func_nb`.

    Returns:
        Array1d: 1D array where each element has been processed by `map_func_nb`.
    """
    i_0_out = map_func_nb(arr[0], *args)
    out = np.empty_like(arr, dtype=np.asarray(i_0_out).dtype)
    out[0] = i_0_out
    for i in range(1, arr.shape[0]):
        out[i] = map_func_nb(arr[i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        map_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def map_nb(arr: tp.Array2d, map_func_nb: tp.MapFunc, *args) -> tp.Array2d:
    """Map each element of a 2-dimensional array column-wise using `map_func_nb`.

    Args:
        arr (Array2d): 2D array of input elements where each column is processed independently.
        map_func_nb (MapFunc): Callback function that accepts an element in `arr` and
            additional arguments, and returns a single value.
        *args: Positional arguments for `map_func_nb`.

    Returns:
        Array2d: 2D array with each element mapped by `map_func_nb`, preserving the original shape.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = map_1d_nb(arr[:, 0], map_func_nb, *args)
    out = np.empty_like(arr, dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, out.shape[1]):
        out[:, col] = map_1d_nb(arr[:, col], map_func_nb, *args)
    return out


@register_jitted
def map_1d_meta_nb(n: int, col: int, map_func_nb: tp.MapMetaFunc, *args) -> tp.Array1d:
    """Map a 1-dimensional sequence using metadata with `map_func_nb`.

    Args:
        n (int): Length of the output array.
        col (int): Column index used in the mapping.
        map_func_nb (MapMetaFunc): Callback function that accepts a row index, column index,
            and additional arguments, and returns a single value.
        *args: Positional arguments for `map_func_nb`.

    Returns:
        Array1d: 1D array with each element computed by `map_func_nb` using the row and column indices.
    """
    i_0_out = map_func_nb(0, col, *args)
    out = np.empty(n, dtype=np.asarray(i_0_out).dtype)
    out[0] = i_0_out
    for i in range(1, n):
        out[i] = map_func_nb(i, col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        map_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def map_meta_nb(target_shape: tp.Shape, map_func_nb: tp.MapMetaFunc, *args) -> tp.Array2d:
    """Map a 2-dimensional array using metadata with `map_func_nb`.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        map_func_nb (MapMetaFunc): Callback function that accepts a row index, column index,
            and additional arguments, and returns a single value.
        *args: Positional arguments for `map_func_nb`.

    Returns:
        Array2d: 2D array with values computed by `map_func_nb` across columns.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = map_1d_meta_nb(target_shape[0], 0, map_func_nb, *args)
    out = np.empty(target_shape, dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, out.shape[1]):
        out[:, col] = map_1d_meta_nb(target_shape[0], col, map_func_nb, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def apply_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """Apply a Numba-compiled function to each column of a 2-dimensional array.

    Args:
        arr (Array2d): 2D array of input elements where each column is processed independently.
        apply_func_nb (ApplyFunc): Callback function that accepts a column in `arr` as a 1D array
            and additional arguments, and returns a single value or an array that broadcasts
            to the column's shape.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array2d: 2D array with the function applied column-wise.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = apply_func_nb(arr[:, 0], *args)
    out = np.empty_like(arr, dtype=np.asarray(col_0_out).dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = apply_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def apply_meta_nb(target_shape: tp.Shape, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array2d:
    """Apply a meta function to each column of a 2-dimensional array, passing the column index as the first argument.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        apply_func_nb (ApplyMetaFunc): Callback function that accepts a column index and
            additional arguments, and returns a single value or an array that broadcasts
            to the column's shape.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array2d: 2D array with the function applied column-wise using the column index.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = apply_func_nb(0, *args)
    out = np.empty(target_shape, dtype=np.asarray(col_0_out).dtype)
    out[:, 0] = col_0_out
    for col in prange(1, target_shape[1]):
        out[:, col] = apply_func_nb(col, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=0),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=0),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def row_apply_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """Apply a Numba-compiled function to each row of a 2-dimensional array.

    Args:
        arr (Array2d): 2D array of input elements where each row is processed independently.
        apply_func_nb (ApplyFunc): Callback function that accepts a row in `arr` as a 1D array
            and additional arguments, and returns a single value or an array that broadcasts
            to the row's shape.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array2d: 2D array with the function applied row-wise.

    !!! tip
        This function is parallelizable.
    """
    row_0_out = apply_func_nb(arr[0, :], *args)
    out = np.empty_like(arr, dtype=np.asarray(row_0_out).dtype)
    out[0, :] = row_0_out
    for i in prange(1, arr.shape[0]):
        out[i, :] = apply_func_nb(arr[i, :], *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=0),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="row_stack",
)
@register_jitted(tags={"can_parallel"})
def row_apply_meta_nb(target_shape: tp.Shape, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array2d:
    """Apply a meta function to each row of a 2-dimensional array, passing the row index as the first argument.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        apply_func_nb (ApplyMetaFunc): Callback function that accepts a row index and
            additional arguments, and returns a single value or an array that broadcasts
            to the row's shape.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array2d: 2D array with the function applied row-wise using the row index.

    !!! tip
        This function is parallelizable.
    """
    row_0_out = apply_func_nb(0, *args)
    out = np.empty(target_shape, dtype=np.asarray(row_0_out).dtype)
    out[0, :] = row_0_out
    for i in prange(1, target_shape[0]):
        out[i, :] = apply_func_nb(i, *args)
    return out


@register_jitted
def rolling_reduce_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Compute rolling window reduction on a 1D array.

    Args:
        arr (Array1d): Input 1D array for computation.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        reduce_func_nb (ReduceFunc): Callback function that accepts a window in `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of rolling reduction results.
    """
    if minp is None:
        minp = window
    out = np.empty_like(arr, dtype=float_)
    nancnt_arr = np.empty(arr.shape[0], dtype=int_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        nancnt_arr[i] = nancnt
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            valid_cnt = window - (nancnt - nancnt_arr[i - window])
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            arr_window = arr[from_i:to_i]
            out[i] = reduce_func_nb(arr_window, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_reduce_nb(
    arr: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """Compute rolling window reduction on each column of a 2D array.

    Args:
        arr (Array2d): Input 2D array for computation.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        reduce_func_nb (ReduceFunc): Callback function that accepts a window in `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Array containing the rolling reduction results applied column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_reduce_1d_nb(arr[:, col], window, minp, reduce_func_nb, *args)
    return out


@register_jitted
def rolling_reduce_two_1d_nb(
    arr1: tp.Array1d,
    arr2: tp.Array1d,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Compute rolling window reduction for two 1D arrays simultaneously.

    Args:
        arr1 (Array1d): First input 1D array.
        arr2 (Array1d): Second input 1D array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        reduce_func_nb (ReduceFunc): Callback function that accepts a window in `arr1` as a 1D array,
            a window in `arr2` as a 1D array, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array with rolling reduction results computed from both arrays.
    """
    if minp is None:
        minp = window
    out = np.empty_like(arr1, dtype=float_)
    nancnt_arr = np.empty(arr1.shape[0], dtype=int_)
    nancnt = 0
    for i in range(arr1.shape[0]):
        if np.isnan(arr1[i]) or np.isnan(arr2[i]):
            nancnt = nancnt + 1
        nancnt_arr[i] = nancnt
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            valid_cnt = window - (nancnt - nancnt_arr[i - window])
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            arr1_window = arr1[from_i:to_i]
            arr2_window = arr2[from_i:to_i]
            out[i] = reduce_func_nb(arr1_window, arr2_window, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_reduce_two_nb(
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """Compute rolling window reduction for two 2D arrays column-wise.

    Args:
        arr1 (Array2d): First input 2D array.
        arr2 (Array2d): Second input 2D array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        reduce_func_nb (ReduceFunc): Callback function that accepts a window in `arr1` as a 1D array,
            a window in `arr2` as a 1D array, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Array containing the column-wise rolling reduction results.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr1, dtype=float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_reduce_two_1d_nb(
            arr1[:, col], arr2[:, col], window, minp, reduce_func_nb, *args
        )
    return out


@register_jitted
def rolling_reduce_1d_meta_nb(
    n: int,
    col: int,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Compute meta rolling reduction on a 1D window using index ranges.

    Args:
        n (int): Total number of rows.
        col (int): Column index for which the reduction is computed.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        reduce_func_nb (RangeReduceMetaFunc): Callback function that accepts a start row index,
            end row index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the meta rolling reduction results.
    """
    if minp is None:
        minp = window
    out = np.empty(n, dtype=float_)
    for i in range(n):
        valid_cnt = min(i + 1, window)
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            out[i] = reduce_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        window=None,
        minp=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_reduce_meta_nb(
    target_shape: tp.Shape,
    window: int,
    minp: tp.Optional[int],
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """Compute meta rolling reduction on each column of a 2D array using index ranges.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        reduce_func_nb (RangeReduceMetaFunc): Callback function that accepts a start row index,
            end row index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Array containing the meta rolling reduction results for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(target_shape, dtype=float_)
    for col in prange(target_shape[1]):
        out[:, col] = rolling_reduce_1d_meta_nb(
            target_shape[0], col, window, minp, reduce_func_nb, *args
        )
    return out


@register_jitted
def rolling_freq_reduce_1d_nb(
    index: tp.Array1d,
    arr: tp.Array1d,
    freq: np.timedelta64,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Compute frequency-based rolling window reduction on a 1D array.

    Args:
        index (Array1d): Array of timestamps corresponding to the entries in the input array.
        arr (Array1d): Input 1D array for computation.
        freq (np.timedelta64): Frequency defining the window duration.
        reduce_func_nb (ReduceFunc): Callback function that accepts a window in `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing frequency-based rolling reduction results.
    """
    out = np.empty_like(arr, dtype=float_)
    from_i = 0
    for i in range(arr.shape[0]):
        if index[from_i] <= index[i] - freq:
            for j in range(from_i + 1, index.shape[0]):
                if index[j] > index[i] - freq:
                    from_i = j
                    break
        to_i = i + 1
        arr_window = arr[from_i:to_i]
        out[i] = reduce_func_nb(arr_window, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        index=None,
        arr=ch.ArraySlicer(axis=1),
        freq=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_freq_reduce_nb(
    index: tp.Array1d,
    arr: tp.Array2d,
    freq: np.timedelta64,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """Compute frequency-based rolling window reduction applied column-wise to a 2D array.

    Args:
        index (Array1d): Array of timestamps corresponding to the rows in the input 2D array.
        arr (Array2d): Input 2D array for computation.
        freq (np.timedelta64): Frequency defining the window duration.
        reduce_func_nb (ReduceFunc): Callback function that accepts a window in `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Array containing frequency-based rolling reduction results applied on each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_freq_reduce_1d_nb(index, arr[:, col], freq, reduce_func_nb, *args)
    return out


@register_jitted
def rolling_freq_reduce_1d_meta_nb(
    col: int,
    index: tp.Array1d,
    freq: np.timedelta64,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `rolling_freq_reduce_1d_nb`.

    Args:
        col (int): Column index to process.
        index (Array1d): One-dimensional array representing the time index.
        freq (np.timedelta64): Frequency defining the window duration.
        reduce_func_nb (RangeReduceMetaFunc): Callback function that accepts a start row index,
            end row index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of reduced values computed over the rolling window.
    """
    out = np.empty(index.shape[0], dtype=float_)
    from_i = 0
    for i in range(index.shape[0]):
        if index[from_i] <= index[i] - freq:
            for j in range(from_i + 1, index.shape[0]):
                if index[j] > index[i] - freq:
                    from_i = j
                    break
        to_i = i + 1
        out[i] = reduce_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        index=None,
        freq=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_freq_reduce_meta_nb(
    n_cols: int,
    index: tp.Array1d,
    freq: np.timedelta64,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dimensional version of `rolling_freq_reduce_1d_meta_nb`.

    Args:
        n_cols (int): Number of columns.
        index (Array1d): One-dimensional array representing the time index.
        freq (np.timedelta64): Frequency defining the window duration.
        reduce_func_nb (RangeReduceMetaFunc): Callback function that accepts a start row index,
            end row index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Two-dimensional array of reduced values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((index.shape[0], n_cols), dtype=float_)
    for col in prange(n_cols):
        out[:, col] = rolling_freq_reduce_1d_meta_nb(col, index, freq, reduce_func_nb, *args)
    return out


@register_jitted
def groupby_reduce_1d_nb(
    arr: tp.Array1d, group_map: tp.GroupMap, reduce_func_nb: tp.ReduceFunc, *args
) -> tp.Array1d:
    """Perform group-by reduction on a one-dimensional array.

    Args:
        arr (Array1d): Input array to reduce.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (ReduceFunc): Callback function that accepts a group as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced value for each group.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[group_0_idxs], *args)
    out = np.empty(group_lens.shape[0], dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in range(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(arr[idxs], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        group_map=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def groupby_reduce_nb(
    arr: tp.Array2d, group_map: tp.GroupMap, reduce_func_nb: tp.ReduceFunc, *args
) -> tp.Array2d:
    """Perform group-by reduction on a two-dimensional array.

    Args:
        arr (Array2d): Input 2D array to reduce.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (ReduceFunc): Callback function that accepts a group as a 1D array
            and additional arguments, and returns a single value.

        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Two-dimensional array with reduced values computed for each group.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = groupby_reduce_1d_nb(arr[:, 0], group_map, reduce_func_nb, *args)
    out = np.empty((col_0_out.shape[0], arr.shape[1]), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = groupby_reduce_1d_nb(arr[:, col], group_map, reduce_func_nb, *args)
    return out


@register_jitted
def groupby_reduce_1d_meta_nb(
    col: int,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.GroupByReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `groupby_reduce_1d_nb`.

    Args:
        col (int): Column index for which to compute the reduction.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (GroupByReduceMetaFunc): Callback function that accepts a group as a 1D array,
            group index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of reduced values computed for each group.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, col, *args)
    out = np.empty(group_lens.shape[0], dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in range(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(idxs, group, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        group_map=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def groupby_reduce_meta_nb(
    n_cols: int,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.GroupByReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dimensional version of `groupby_reduce_1d_meta_nb`.

    Args:
        n_cols (int): Number of columns to process.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (GroupByReduceMetaFunc): Callback function that accepts a group as a 1D array,
            group index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Two-dimensional array of reduced values.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = groupby_reduce_1d_meta_nb(0, group_map, reduce_func_nb, *args)
    out = np.empty((col_0_out.shape[0], n_cols), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, n_cols):
        out[:, col] = groupby_reduce_1d_meta_nb(col, group_map, reduce_func_nb, *args)
    return out


@register_jitted(tags={"can_parallel"})
def groupby_transform_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    transform_func_nb: tp.GroupByTransformFunc,
    *args,
) -> tp.Array2d:
    """Perform group-by transformation on a two-dimensional array.

    Args:
        arr (Array2d): Input 2D array to transform.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        transform_func_nb (GroupByTransformFunc): Callback function that accepts a group as a 2D array
            and additional arguments, and returns a single value or an array that broadcasts
            to the group's shape.
        *args: Positional arguments for `transform_func_nb`.

    Returns:
        Array2d: Two-dimensional array with transformed values.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = transform_func_nb(arr[group_0_idxs], *args)
    out = np.empty(arr.shape, dtype=np.asarray(group_0_out).dtype)
    out[group_0_idxs] = group_0_out

    for group in prange(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx : start_idx + group_len]
        out[idxs] = transform_func_nb(arr[idxs], *args)
    return out


@register_jitted(tags={"can_parallel"})
def groupby_transform_meta_nb(
    target_shape: tp.Shape,
    group_map: tp.GroupMap,
    transform_func_nb: tp.GroupByTransformMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `groupby_transform_nb`.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        group_map (GroupMap): Tuple of indices and lengths for each group.
        transform_func_nb (GroupByTransformMetaFunc): Callback function that accepts a group as a 2D array
            and additional arguments, and returns a single value or an array that broadcasts
            to the group's shape.
        *args: Positional arguments for `transform_func_nb`.

    Returns:
        Array2d: Two-dimensional array with transformed values based on group processing.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = transform_func_nb(group_0_idxs, 0, *args)
    out = np.empty(target_shape, dtype=np.asarray(group_0_out).dtype)
    out[group_0_idxs] = group_0_out

    for group in prange(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx : start_idx + group_len]
        out[idxs] = transform_func_nb(idxs, group, *args)
    return out


@register_jitted
def reduce_index_ranges_1d_nb(
    arr: tp.Array1d,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce each index range of a one-dimensional array.

    Args:
        arr (Array1d): Input one-dimensional array.
        range_starts (Array1d): Array of starting indices for each range.
        range_ends (Array1d): Array of ending indices for each range.
        reduce_func_nb (ReduceFunc): Callback function that accepts a segment of `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced value for each index range.
    """
    out = np.empty(range_starts.shape[0], dtype=float_)

    for k in range(len(range_starts)):
        from_i = range_starts[k]
        to_i = range_ends[k]
        if from_i == -1 or to_i == -1:
            out[k] = np.nan
        else:
            out[k] = reduce_func_nb(arr[from_i:to_i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        range_starts=None,
        range_ends=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_index_ranges_nb(
    arr: tp.Array2d,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """Reduce each index range of a 2-dimensional array column-wise.

    Args:
        arr (Array2d): Input two-dimensional array.
        range_starts (Array1d): Array of starting indices for each range.
        range_ends (Array1d): Array of ending indices for each range.
        reduce_func_nb (ReduceFunc): Callback function that accepts a segment of `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Two-dimensional array containing the reduced values for each index range per column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((range_starts.shape[0], arr.shape[1]), dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = reduce_index_ranges_1d_nb(
            arr[:, col], range_starts, range_ends, reduce_func_nb, *args
        )
    return out


@register_jitted
def reduce_index_ranges_1d_meta_nb(
    col: int,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Reduce index ranges on a one-dimensional segment using meta reduction.

    Args:
        col (int): Column index.
        range_starts (Array1d): Array of starting indices for each range.
        range_ends (Array1d): Array of ending indices for each range.
        reduce_func_nb (RangeReduceMetaFunc): Callback function that accepts a start row index,
            end row index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of reduced values.
    """
    out = np.empty(range_starts.shape[0], dtype=float_)

    for k in range(len(range_starts)):
        from_i = range_starts[k]
        to_i = range_ends[k]
        if from_i == -1 or to_i == -1:
            out[k] = np.nan
        else:
            out[k] = reduce_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        range_starts=None,
        range_ends=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_index_ranges_meta_nb(
    n_cols: int,
    range_starts: tp.Array1d,
    range_ends: tp.Array1d,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """Reduce index ranges of a 2-dimensional array using meta reduction column-wise.

    Args:
        n_cols (int): Number of columns in the array.
        range_starts (Array1d): Array of starting indices for each range.
        range_ends (Array1d): Array of ending indices for each range.
        reduce_func_nb (RangeReduceMetaFunc): Callback function that accepts a start row index,
            end row index, column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Two-dimensional array containing the reduced values for each index range per column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty((range_starts.shape[0], n_cols), dtype=float_)
    for col in prange(n_cols):
        out[:, col] = reduce_index_ranges_1d_meta_nb(
            col, range_starts, range_ends, reduce_func_nb, *args
        )
    return out


@register_jitted
def apply_and_reduce_1d_nb(
    arr: tp.Array1d,
    apply_func_nb: tp.ApplyFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceFunc,
    reduce_args: tuple,
) -> tp.Scalar:
    """Apply a function and reduce a one-dimensional array to a single value.

    Args:
        arr (Array1d): Input one-dimensional array.
        apply_func_nb (ApplyFunc): Callback function that accepts `arr` and additional arguments,
            and returns a 1D array.
        apply_args (tuple): Positional arguments for `apply_func_nb`.
        reduce_func_nb (ReduceFunc): Callback function that accepts the result of `apply_func_nb`
            as a 1D array and additional arguments, and returns a single value.
        reduce_args (tuple): Positional arguments for `reduce_func_nb`.

    Returns:
        Scalar: Reduced value.
    """
    temp = apply_func_nb(arr, *apply_args)
    return reduce_func_nb(temp, *reduce_args)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        apply_func_nb=None,
        apply_args=ch.ArgsTaker(),
        reduce_func_nb=None,
        reduce_args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_and_reduce_nb(
    arr: tp.Array2d,
    apply_func_nb: tp.ApplyFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceFunc,
    reduce_args: tuple,
) -> tp.Array1d:
    """Apply a function and reduce each column of a 2-dimensional array to a single value.

    Args:
        arr (Array2d): Input two-dimensional array.
        apply_func_nb (ApplyFunc): Callback function that accepts `arr` and additional arguments,
            and returns a 1D array.
        apply_args (tuple): Positional arguments for `apply_func_nb`.
        reduce_func_nb (ReduceFunc): Callback function that accepts the result of `apply_func_nb`
            as a 1D array and additional arguments, and returns a single value.
        reduce_args (tuple): Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of reduced values for each column.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = apply_and_reduce_1d_nb(
        arr[:, 0], apply_func_nb, apply_args, reduce_func_nb, reduce_args
    )
    out = np.empty(arr.shape[1], dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[col] = apply_and_reduce_1d_nb(
            arr[:, col], apply_func_nb, apply_args, reduce_func_nb, reduce_args
        )
    return out


@register_jitted
def apply_and_reduce_1d_meta_nb(
    col: int,
    apply_func_nb: tp.ApplyMetaFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceMetaFunc,
    reduce_args: tuple,
) -> tp.Scalar:
    """Apply a meta function and reduce a one-dimensional segment to a single value using meta reduction.

    Args:
        col (int): Column index.
        apply_func_nb (ApplyMetaFunc): Callback function that accepts the column index and
            additional arguments, and returns a 1D array.
        apply_args (tuple): Positional arguments for `apply_func_nb`.
        reduce_func_nb (ReduceMetaFunc): Callback function that accepts the column index,
            the result of `apply_func_nb` as a 1D array, and additional arguments, and
            returns a single value.
        reduce_args (tuple): Positional arguments for `reduce_func_nb`.

    Returns:
        Scalar: Reduced value.
    """
    temp = apply_func_nb(col, *apply_args)
    return reduce_func_nb(col, temp, *reduce_args)


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        apply_func_nb=None,
        apply_args=ch.ArgsTaker(),
        reduce_func_nb=None,
        reduce_args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_and_reduce_meta_nb(
    n_cols: int,
    apply_func_nb: tp.ApplyMetaFunc,
    apply_args: tuple,
    reduce_func_nb: tp.ReduceMetaFunc,
    reduce_args: tuple,
) -> tp.Array1d:
    """Apply a meta function and reduce each column of a 2-dimensional array
    to a single value using meta reduction.

    Args:
        n_cols (int): Number of columns.
        apply_func_nb (ApplyMetaFunc): Callback function that accepts the column index and
            additional arguments, and returns a 1D array.
        apply_args (tuple): Positional arguments for `apply_func_nb`.
        reduce_func_nb (ReduceMetaFunc): Callback function that accepts the column index,
            the result of `apply_func_nb` as a 1D array, and additional arguments, and
            returns a single value.
        reduce_args (tuple): Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of reduced values for each column.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = apply_and_reduce_1d_meta_nb(
        0, apply_func_nb, apply_args, reduce_func_nb, reduce_args
    )
    out = np.empty(n_cols, dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, n_cols):
        out[col] = apply_and_reduce_1d_meta_nb(
            col, apply_func_nb, apply_args, reduce_func_nb, reduce_args
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_nb(arr: tp.Array2d, reduce_func_nb: tp.ReduceFunc, *args) -> tp.Array1d:
    """Reduce each column of a 2-dimensional array to a single value.

    Args:
        arr (Array2d): Input two-dimensional array.
        reduce_func_nb (ReduceFunc): Callback function that accepts a column of `arr` as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced values for each column.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = reduce_func_nb(arr[:, 0], *args)
    out = np.empty(arr.shape[1], dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[col] = reduce_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_meta_nb(n_cols: int, reduce_func_nb: tp.ReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_nb`.

    Args:
        n_cols (int): Number of columns.
        reduce_func_nb (ReduceMetaFunc): Callback function that accepts a column index and
            additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced meta values for each column.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = reduce_func_nb(0, *args)
    out = np.empty(n_cols, dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, n_cols):
        out[col] = reduce_func_nb(col, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_to_array_nb(arr: tp.Array2d, reduce_func_nb: tp.ReduceToArrayFunc, *args) -> tp.Array2d:
    """Reduce each column of the array using `reduce_func_nb`, which must return an array.

    Args:
        arr (Array2d): 2-dimensional array to process.
        reduce_func_nb (ReduceToArrayFunc): Callback function that accepts a column of `arr` as a 1D array
            and additional arguments, and returns a single value or an array that broadcasts
            to the column's shape.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2-dimensional array with each column replaced by the array returned from `reduce_func_nb`.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = reduce_func_nb(arr[:, 0], *args)
    out = np.empty((col_0_out.shape[0], arr.shape[1]), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = reduce_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_to_array_meta_nb(
    n_cols: int, reduce_func_nb: tp.ReduceToArrayMetaFunc, *args
) -> tp.Array2d:
    """Meta version of `reduce_meta_nb` that returns an array for each column.

    Args:
        n_cols (int): Number of columns.
        reduce_func_nb (ReduceToArrayMetaFunc): Callback function that accepts a column index and
            additional arguments, and returns a single value or an array that broadcasts
            to the column's shape.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2-dimensional array where each column is the array produced by `reduce_func_nb`.

    !!! tip
        This function is parallelizable.
    """
    col_0_out = reduce_func_nb(0, *args)
    out = np.empty((col_0_out.shape[0], n_cols), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, n_cols):
        out[:, col] = reduce_func_nb(col, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.ReduceGroupedFunc,
    *args,
) -> tp.Array1d:
    """Reduce each group of columns into a single value using `reduce_func_nb`.

    Args:
        arr (Array2d): 2-dimensional array containing the columns to be grouped.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (ReduceGroupedFunc): Callback function that accepts a group as a 2D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced value for each group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[:, group_0_idxs], *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(arr[:, col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_meta_nb(
    group_map: tp.GroupMap, reduce_func_nb: tp.ReduceGroupedMetaFunc, *args
) -> tp.Array1d:
    """Meta version of `reduce_grouped_nb` that reduces groups based on metadata.

    Args:
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (ReduceGroupedMetaFunc): Callback function that accepts a group as a 1D array,
            group index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced meta value for each group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[group] = reduce_func_nb(col_idxs, group, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def flatten_forder_nb(arr: tp.Array2d) -> tp.Array1d:
    """Flatten the 2-dimensional array in Fortran order.

    Args:
        arr (Array2d): 2-dimensional array to be flattened.

    Returns:
        Array1d: 1-dimensional array resulting from flattening `arr` in Fortran order.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(arr.shape[0] * arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col * arr.shape[0] : (col + 1) * arr.shape[0]] = arr[:, col]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        in_c_order=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_flat_grouped_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    in_c_order: bool,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array1d:
    """Reduce each group of columns by applying `reduce_func_nb` to a flattened group array.

    Args:
        arr (Array2d): 2-dimensional array containing the data.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        in_c_order (bool): If True, flatten in C order; otherwise, in Fortran order.
        reduce_func_nb (ReduceToArrayFunc): Callback function that accepts a flattened group as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced value for each group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    if in_c_order:
        group_0_out = reduce_func_nb(arr[:, group_0_idxs].flatten(), *args)
    else:
        group_0_out = reduce_func_nb(flatten_forder_nb(arr[:, group_0_idxs]), *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        if in_c_order:
            out[group] = reduce_func_nb(arr[:, col_idxs].flatten(), *args)
        else:
            out[group] = reduce_func_nb(flatten_forder_nb(arr[:, col_idxs]), *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_to_array_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    reduce_func_nb: tp.ReduceGroupedToArrayFunc,
    *args,
) -> tp.Array2d:
    """Return an array by applying a reduction function to each grouped segment of a 2D array.

    This function behaves similarly to `reduce_grouped_nb` but requires that `reduce_func_nb`
    returns an array. It partitions the columns of the input array based on the provided group map,
    applies the reduction function to each group, and stacks the resulting arrays column-wise.

    Args:
        arr (Array2d): 2D input array.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (ReduceGroupedToArrayFunc): Callback function that accepts a group as a 2D array
            and additional arguments, and returns a 1D array.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array with each column corresponding to the reduction result of a group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[:, group_0_idxs], *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[:, group] = reduce_func_nb(arr[:, col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_grouped_to_array_meta_nb(
    group_map: tp.GroupMap,
    reduce_func_nb: tp.ReduceGroupedToArrayMetaFunc,
    *args,
) -> tp.Array2d:
    """Return an array by applying a reduction function to grouped indices.

    This function is similar to `reduce_grouped_meta_nb` but requires that `reduce_func_nb`
    returns an array. It processes the group indices and group lengths to apply the reduction
    function on each group and stacks the results column-wise.

    Args:
        group_map (GroupMap): Tuple of indices and lengths for each group.
        reduce_func_nb (ReduceGroupedToArrayMetaFunc): Callback function that accepts a group as a 2D array
            and additional arguments, and returns a 1D array.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array with each column representing the reduced result of a group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        out[:, group] = reduce_func_nb(col_idxs, group, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        in_c_order=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_flat_grouped_to_array_nb(
    arr: tp.Array2d,
    group_map: tp.GroupMap,
    in_c_order: bool,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array2d:
    """Return an array by applying a reduction function to flattened groups from a 2D array.

    This function behaves similarly to `reduce_grouped_to_array_nb` but flattens each
    group's selected data. Depending on the `in_c_order` flag, the array is flattened in
    C order or Fortran order before applying the reduction.

    Args:
        arr (Array2d): 2D input array.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        in_c_order (bool): If True, flatten in C order; otherwise, in Fortran order.
        reduce_func_nb (ReduceToArrayFunc): Callback function that accepts a flattened group as a 1D array
            and additional arguments, and returns a 1D array.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array where each column is the reduction result of a flattened group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    if in_c_order:
        group_0_out = reduce_func_nb(arr[:, group_0_idxs].flatten(), *args)
    else:
        group_0_out = reduce_func_nb(flatten_forder_nb(arr[:, group_0_idxs]), *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        if in_c_order:
            out[:, group] = reduce_func_nb(arr[:, col_idxs].flatten(), *args)
        else:
            out[:, group] = reduce_func_nb(flatten_forder_nb(arr[:, col_idxs]), *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        squeeze_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def squeeze_grouped_nb(
    arr: tp.Array2d, group_map: tp.GroupMap, squeeze_func_nb: tp.ReduceFunc, *args
) -> tp.Array2d:
    """Return a squeezed array by applying a squeeze function to each grouped segment.

    This function reduces each group of columns in the input array into a single column
    by applying `squeeze_func_nb`. It processes each row independently, applying the squeeze
    function to the corresponding group slice.

    Args:
        arr (Array2d): 2D input array.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        squeeze_func_nb (ReduceFunc): Callback function that accepts a group within a row as a 1D array
            and additional arguments, and returns a single value.
        *args: Positional arguments for `squeeze_func_nb`.

    Returns:
        Array2d: 2D array in which each column is the squeezed result of the corresponding group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_i_0_out = squeeze_func_nb(arr[0][group_0_idxs], *args)
    out = np.empty((arr.shape[0], len(group_lens)), dtype=np.asarray(group_i_0_out).dtype)
    out[0, 0] = group_i_0_out

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for i in range(arr.shape[0]):
            if group == 0 and i == 0:
                continue
            out[i, group] = squeeze_func_nb(arr[i][col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="group_map"),
    arg_take_spec=dict(
        n_rows=None,
        group_map=base_ch.GroupMapSlicer(),
        squeeze_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def squeeze_grouped_meta_nb(
    n_rows: int,
    group_map: tp.GroupMap,
    squeeze_func_nb: tp.GroupSqueezeMetaFunc,
    *args,
) -> tp.Array2d:
    """Return a meta squeezed array by applying a meta squeeze function to grouped indices.

    This meta function operates similarly to `squeeze_grouped_nb` but works with index information.
    It applies `squeeze_func_nb` using the row index, group column indices, and group index to
    produce a single value for each element.

    Args:
        n_rows (int): Number of rows in the output array.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        squeeze_func_nb (GroupSqueezeMetaFunc): Callback function that accepts a row index,
            group column indices, group index, and additional arguments, and returns a single value.
        *args: Positional arguments for `squeeze_func_nb`.

    Returns:
        Array2d: 2D array with the meta squeezed values for each row and group.

    !!! tip
        This function is parallelizable.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0] : group_start_idxs[0] + group_lens[0]]
    group_i_0_out = squeeze_func_nb(0, group_0_idxs, 0, *args)
    out = np.empty((n_rows, len(group_lens)), dtype=np.asarray(group_i_0_out).dtype)
    out[0, 0] = group_i_0_out

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for i in range(n_rows):
            if group == 0 and i == 0:
                continue
            out[i, group] = squeeze_func_nb(i, col_idxs, group, *args)
    return out


# ############# Flattening ############# #


@register_jitted(cache=True)
def flatten_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool) -> tp.Array2d:
    """Flatten groups of columns in the given 2D array.

    Args:
        arr (Array2d): Input 2D array.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        in_c_order (bool): If True, flatten in C order; otherwise, in Fortran order.

    Returns:
        Array2d: New 2D array with flattened groups.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((arr.shape[0] * np.max(group_lens), len(group_lens)), np.nan, dtype=float_)
    max_len = np.max(group_lens)

    for group in range(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            if in_c_order:
                out[k::max_len, group] = arr[:, col]
            else:
                out[k * arr.shape[0] : (k + 1) * arr.shape[0], group] = arr[:, col]
    return out


@register_jitted(cache=True)
def flatten_uniform_grouped_nb(
    arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool
) -> tp.Array2d:
    """Flatten groups of uniformly sized columns in the given 2D array.

    Args:
        arr (Array2d): Input 2D array.
        group_map (GroupMap): Tuple of indices and lengths for each group.
        in_c_order (bool): If True, flatten in C order; otherwise, in Fortran order.

    Returns:
        Array2d: New 2D array with flattened groups.
    """
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.empty((arr.shape[0] * np.max(group_lens), len(group_lens)), dtype=arr.dtype)
    max_len = np.max(group_lens)

    for group in range(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx : start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            if in_c_order:
                out[k::max_len, group] = arr[:, col]
            else:
                out[k * arr.shape[0] : (k + 1) * arr.shape[0], group] = arr[:, col]
    return out


# ############# Proximity ############# #


@register_jitted(tags={"can_parallel"})
def proximity_reduce_nb(
    arr: tp.Array2d,
    window: int,
    reduce_func_nb: tp.ProximityReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """Reduce each element by applying a reduction function to its surrounding window.

    Args:
        arr (Array2d): Input 2D array.
        window (int): Window size.
        reduce_func_nb (ProximityReduceMetaFunc): Callback function that accepts the flattened
            surrounding window as a 1D array and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array containing the reduced values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for i in prange(arr.shape[0]):
        for col in range(arr.shape[1]):
            from_i = max(0, i - window)
            to_i = min(i + window + 1, arr.shape[0])
            from_col = max(0, col - window)
            to_col = min(col + window + 1, arr.shape[1])
            stride_arr = arr[from_i:to_i, from_col:to_col]
            out[i, col] = reduce_func_nb(stride_arr.flatten(), *args)
    return out


@register_jitted(tags={"can_parallel"})
def proximity_reduce_meta_nb(
    target_shape: tp.Shape,
    window: int,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """Perform a meta reduction by applying a function over window index boundaries.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        window (int): Window size.
        reduce_func_nb (ReduceFunc): Callback function that accepts a start row index, end row index,
            start column index, end column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: Resulting array after applying the reduction.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(target_shape, dtype=float_)
    for i in prange(target_shape[0]):
        for col in range(target_shape[1]):
            from_i = max(0, i - window)
            to_i = min(i + window + 1, target_shape[0])
            from_col = max(0, col - window)
            to_col = min(col + window + 1, target_shape[1])
            out[i, col] = reduce_func_nb(from_i, to_i, from_col, to_col, *args)
    return out


# ############# Reducers ############# #


@register_jitted(cache=True)
def nth_reduce_nb(arr: tp.Array1d, n: int) -> float:
    """Return the nth element from a 1D array.

    Args:
        arr (Array1d): Input array.
        n (int): Index of the element to retrieve.

    Returns:
        float: Element at the specified index.
    """
    if (n < 0 and abs(n) > arr.shape[0]) or n >= arr.shape[0]:
        raise ValueError("index is out of bounds")
    return arr[n]


@register_jitted(cache=True)
def first_reduce_nb(arr: tp.Array1d) -> float:
    """Return the first non-NA element from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: First non-NA element, or NaN if no such element exists.
    """
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            return arr[i]
    return np.nan


@register_jitted(cache=True)
def last_reduce_nb(arr: tp.Array1d) -> float:
    """Return the last non-NA element from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Last non-NA element, or NaN if no such element exists.
    """
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr) - 1, -1, -1):
        if not np.isnan(arr[i]):
            return arr[i]
    return np.nan


@register_jitted(cache=True)
def first_index_reduce_nb(arr: tp.Array1d) -> int:
    """Return the index of the first non-NA element in a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        int: Index of the first non-NA element, or -1 if none exists.
    """
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            return i
    return -1


@register_jitted(cache=True)
def last_index_reduce_nb(arr: tp.Array1d) -> int:
    """Return the index of the last non-NA element in a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        int: Index of the last non-NA element, or -1 if none exists.
    """
    if arr.shape[0] == 0:
        raise ValueError("index is out of bounds")
    for i in range(len(arr) - 1, -1, -1):
        if not np.isnan(arr[i]):
            return i
    return -1


@register_jitted(cache=True)
def nth_index_reduce_nb(arr: tp.Array1d, n: int) -> int:
    """Return the index of the nth element in a 1D array, counting NA elements.

    Args:
        arr (Array1d): Input array.
        n (int): Index offset; negative values count from the end.

    Returns:
        int: Calculated index corresponding to the nth element.
    """
    if (n < 0 and abs(n) > arr.shape[0]) or n >= arr.shape[0]:
        raise ValueError("index is out of bounds")
    if n >= 0:
        return n
    return arr.shape[0] + n


@register_jitted(cache=True)
def any_reduce_nb(arr: tp.Array1d) -> bool:
    """Return True if any element in the 1D array is True.

    Args:
        arr (Array1d): Input array.

    Returns:
        bool: True if at least one element is True, otherwise False.
    """
    return np.any(arr)


@register_jitted(cache=True)
def all_reduce_nb(arr: tp.Array1d) -> bool:
    """Return True if all elements in the 1D array are True.

    Args:
        arr (Array1d): Input array.

    Returns:
        bool: True if every element is True, otherwise False.
    """
    return np.all(arr)


@register_jitted(cache=True)
def min_reduce_nb(arr: tp.Array1d) -> float:
    """Return the minimum non-NA value from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Smallest value among the non-NA entries.
    """
    return np.nanmin(arr)


@register_jitted(cache=True)
def max_reduce_nb(arr: tp.Array1d) -> float:
    """Return the maximum non-NA value from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Largest value among the non-NA entries.
    """
    return np.nanmax(arr)


@register_jitted(cache=True)
def mean_reduce_nb(arr: tp.Array1d) -> float:
    """Return the mean of non-NA values from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Average of the non-NA values.
    """
    return np.nanmean(arr)


@register_jitted(cache=True)
def median_reduce_nb(arr: tp.Array1d) -> float:
    """Return the median of non-NA values from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Median value among the non-NA entries.
    """
    return np.nanmedian(arr)


@register_jitted(cache=True)
def std_reduce_nb(arr: tp.Array1d, ddof) -> float:
    """Return the standard deviation of non-NA values from a 1D array.

    Args:
        arr (Array1d): Input array.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Standard deviation computed with the specified ddof.
    """
    return nanstd_1d_nb(arr, ddof=ddof)


@register_jitted(cache=True)
def sum_reduce_nb(arr: tp.Array1d) -> float:
    """Return the sum of non-NA values from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Total sum of the non-NA entries.
    """
    return np.nansum(arr)


@register_jitted(cache=True)
def prod_reduce_nb(arr: tp.Array1d) -> float:
    """Return the product of non-NA values from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Product computed from the non-NA entries.
    """
    return np.nanprod(arr)


@register_jitted(cache=True)
def nonzero_prod_reduce_nb(arr: tp.Array1d) -> float:
    """Return the product of non-zero and non-NA values from a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        float: Product of non-zero and non-NA values; returns zero if no such product is computed.
    """
    prod = 0.0
    for i in range(len(arr)):
        if not np.isnan(arr[i]) and arr[i] != 0:
            if prod == 0:
                prod = 1.0
            prod *= arr[i]
    return prod


@register_jitted(cache=True)
def count_reduce_nb(arr: tp.Array1d) -> int:
    """Return the count of non-NA values in a 1D array.

    Args:
        arr (Array1d): Input array.

    Returns:
        int: Number of non-NA entries in the array.
    """
    return np.sum(~np.isnan(arr))


@register_jitted(cache=True)
def argmin_reduce_nb(arr: tp.Array1d) -> int:
    """Return the index of the minimum value in a 1D array while ignoring NaN values.

    Args:
        arr (Array1d): Input array.

    Returns:
        int: Index of the minimum value.

    !!! note
        Raises a `ValueError` if all elements in the array are NaN.
    """
    arr = np.copy(arr)
    mask = np.isnan(arr)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    arr[mask] = np.inf
    return np.argmin(arr)


@register_jitted(cache=True)
def argmax_reduce_nb(arr: tp.Array1d) -> int:
    """Return the index of the maximum value in a 1D array while ignoring NaN values.

    Args:
        arr (Array1d): Input array.

    Returns:
        int: Index of the maximum value.

    !!! note
        Raises a `ValueError` if all elements in the array are NaN.
    """
    arr = np.copy(arr)
    mask = np.isnan(arr)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    arr[mask] = -np.inf
    return np.argmax(arr)


@register_jitted(cache=True)
def describe_reduce_nb(arr: tp.Array1d, perc: tp.Array1d, ddof: int) -> tp.Array1d:
    """Return descriptive statistics for a 1D array while ignoring NaN values.

    The returned array contains the statistics in the following order:
    count, mean, standard deviation, minimum, the specified percentiles, and maximum.

    Args:
        arr (Array1d): Input array.
        perc (Array1d): Array of percentiles as fractions (e.g., 0.25, 0.5, 0.75).
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Array of descriptive statistics.

    !!! note
        This function is a Numba equivalent to `pd.Series(arr).describe(perc)`.
    """
    arr = arr[~np.isnan(arr)]
    out = np.empty(5 + len(perc), dtype=float_)
    out[0] = len(arr)
    if len(arr) > 0:
        out[1] = np.mean(arr)
        out[2] = nanstd_1d_nb(arr, ddof=ddof)
        out[3] = np.min(arr)
        out[4:-1] = np.percentile(arr, perc * 100)
        out[4 + len(perc)] = np.max(arr)
    else:
        out[1:] = np.nan
    return out


@register_jitted(cache=True)
def cov_reduce_grouped_meta_nb(
    group_idxs: tp.GroupIdxs,
    group: int,
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    ddof: int,
) -> float:
    """Return the covariance between elements of two 2D arrays for a specified group while ignoring NaN values.

    Args:
        group_idxs (GroupIdxs): Indices used to group array elements.
        group (int): Group identifier (not used in the computation).
        arr1 (Array2d): First input array.
        arr2 (Array2d): Second input array.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Covariance of the selected group elements.
    """
    return nancov_1d_nb(arr1[:, group_idxs].flatten(), arr2[:, group_idxs].flatten(), ddof=ddof)


@register_jitted(cache=True)
def corr_reduce_grouped_meta_nb(
    group_idxs: tp.GroupIdxs, group: int, arr1: tp.Array2d, arr2: tp.Array2d
) -> float:
    """Return the Pearson correlation coefficient between elements of two 2D arrays for a specified group while ignoring NaN values.

    Args:
        group_idxs (GroupIdxs): Indices used to group array elements.
        group (int): Group identifier (not used in the computation).
        arr1 (Array2d): First input array.
        arr2 (Array2d): Second input array.

    Returns:
        float: Pearson correlation coefficient of the selected group elements.
    """
    return nancorr_1d_nb(arr1[:, group_idxs].flatten(), arr2[:, group_idxs].flatten())


@register_jitted(cache=True)
def wmean_range_reduce_meta_nb(
    from_i: int, to_i: int, col: int, arr1: tp.Array2d, arr2: tp.Array2d
) -> float:
    """Return the weighted average for a specified column over a given row range.

    Args:
        from_i (int): Start index of the range (inclusive).
        to_i (int): End index of the range (exclusive).
        col (int): Column index for which the weighted average is computed.
        arr1 (Array2d): Array containing values to be weighted.
        arr2 (Array2d): Array containing weights.

    Returns:
        float: Weighted average value, or NaN if the sum of weights is zero.
    """
    nom_cumsum = 0
    denum_cumsum = 0
    for i in range(from_i, to_i):
        nom_cumsum += arr1[i, col] * arr2[i, col]
        denum_cumsum += arr2[i, col]
    if denum_cumsum == 0:
        return np.nan
    return nom_cumsum / denum_cumsum
