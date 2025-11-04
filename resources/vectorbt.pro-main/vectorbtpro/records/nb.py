# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a suite of Numba-compiled functions for processing records and mapped arrays.

!!! note
    All functions passed as argument must be Numba-compiled.

    Records must retain the order they were created in.
"""

import numpy as np
from numba import prange
from numba.extending import overload
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


# ############# Generation ############# #


@register_jitted(cache=True)
def generate_ids_nb(col_arr: tp.Array1d, n_cols: int) -> tp.Array1d:
    """Generate monotonically increasing IDs based on a column index array.

    Args:
        col_arr (Array1d): Array of column indices.
        n_cols (int): Total number of columns.

    Returns:
        Array1d: Array of generated IDs corresponding to each entry in `col_arr`.
    """
    col_idxs = np.full(n_cols, 0, dtype=int_)
    out = np.empty_like(col_arr)
    for c in range(len(col_arr)):
        out[c] = col_idxs[col_arr[c]]
        col_idxs[col_arr[c]] += 1
    return out


# ############# Indexing ############# #


@register_jitted(cache=True)
def col_lens_nb(col_arr: tp.Array1d, n_cols: int) -> tp.GroupLens:
    """Get lengths for each column in a sorted column index array.

    Args:
        col_arr (Array1d): Array of column indices.
        n_cols (int): Total number of columns.

    Returns:
        GroupLens: Array containing the count of elements for each column.

    !!! note
        `col_arr` must be in ascending order (i.e., sorted).
    """
    col_lens = np.full(n_cols, 0, dtype=int_)
    last_col = -1

    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        if col < last_col:
            raise ValueError("col_arr must come in ascending order")
        last_col = col
        col_lens[col] += 1
    return col_lens


@register_jitted(cache=True)
def record_col_lens_select_nb(
    records: tp.RecordArray,
    col_lens: tp.GroupLens,
    new_cols: tp.Array1d,
) -> tp.Tuple[tp.Array1d, tp.RecordArray]:
    """Select and reassign records based on column lengths from a sorted records array.

    Args:
        records (RecordArray): Array of records.
        col_lens (GroupLens): Array of column lengths.
        new_cols (Array1d): Array of new column indices for selection.

    Returns:
        Tuple[Array1d, RecordArray]: Tuple containing the original indices and the updated record array.
    """
    col_end_idxs = np.cumsum(col_lens)
    col_start_idxs = col_end_idxs - col_lens
    n_values = np.sum(col_lens[new_cols])
    indices_out = np.empty(n_values, dtype=int_)
    records_arr_out = np.empty(n_values, dtype=records.dtype)
    j = 0

    for c in range(new_cols.shape[0]):
        from_r = col_start_idxs[new_cols[c]]
        to_r = col_end_idxs[new_cols[c]]
        if from_r == to_r:
            continue
        col_records = np.copy(records[from_r:to_r])
        col_records["col"][:] = c  # don't forget to assign new column indices
        rang = np.arange(from_r, to_r)
        indices_out[j : j + rang.shape[0]] = rang
        records_arr_out[j : j + rang.shape[0]] = col_records
        j += col_records.shape[0]
    return indices_out, records_arr_out


@register_jitted(cache=True)
def col_map_nb(col_arr: tp.Array1d, n_cols: int) -> tp.GroupMap:
    """Build a mapping between columns and their corresponding value indices.

    Args:
        col_arr (Array1d): Array of column indices.
        n_cols (int): Total number of columns.

    Returns:
        GroupMap: Tuple containing the indices of values segmented by column
            and the column lengths for each column.
    """
    col_lens_out = np.full(n_cols, 0, dtype=int_)
    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        col_lens_out[col] += 1

    col_start_idxs = np.cumsum(col_lens_out) - col_lens_out
    col_idxs_out = np.empty((col_arr.shape[0],), dtype=int_)
    col_i = np.full(n_cols, 0, dtype=int_)
    for c in range(col_arr.shape[0]):
        col = col_arr[c]
        col_idxs_out[col_start_idxs[col] + col_i[col]] = c
        col_i[col] += 1

    return col_idxs_out, col_lens_out


@register_jitted(cache=True)
def record_col_map_select_nb(
    records: tp.RecordArray,
    col_map: tp.GroupMap,
    new_cols: tp.Array1d,
) -> tp.Tuple[tp.Array1d, tp.RecordArray]:
    """Select and reassign records based on a provided column map.

    Args:
        records (RecordArray): Array of records.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        new_cols (Array1d): Array of new column indices for selection.

    Returns:
        Tuple[Array1d, RecordArray]: Tuple containing the original indices and
            the updated record array.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    total_count = np.sum(col_lens[new_cols])
    indices_out = np.empty(total_count, dtype=int_)
    records_arr_out = np.empty(total_count, dtype=records.dtype)
    j = 0

    for new_col_i in range(len(new_cols)):
        new_col = new_cols[new_col_i]
        col_len = col_lens[new_col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[new_col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_records = np.copy(records[idxs])
        col_records["col"][:] = new_col_i
        indices_out[j : j + col_len] = idxs
        records_arr_out[j : j + col_len] = col_records
        j += col_len
    return indices_out, records_arr_out


# ############# Sorting ############# #


@register_jitted(cache=True)
def is_col_sorted_nb(col_arr: tp.Array1d) -> bool:
    """Check whether an array of column indices is sorted in ascending order.

    Args:
        col_arr (Array1d): Array of column indices.

    Returns:
        bool: True if the column array is sorted; False otherwise.
    """
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
    return True


@register_jitted(cache=True)
def is_col_id_sorted_nb(col_arr: tp.Array1d, id_arr: tp.Array1d) -> bool:
    """Check whether both column and ID arrays are sorted.

    Args:
        col_arr (Array1d): Array of column indices.
        id_arr (Array1d): Array of IDs.

    Returns:
        bool: True if the arrays are sorted in ascending order by column and
            by ID within each column; False otherwise.
    """
    for i in range(len(col_arr) - 1):
        if col_arr[i + 1] < col_arr[i]:
            return False
        if col_arr[i + 1] == col_arr[i] and id_arr[i + 1] < id_arr[i]:
            return False
    return True


# ############# Filtering ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def first_n_nb(col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Return a boolean mask marking the first N elements in each column.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        n (int): Number of elements to mark from the beginning of each column.

    Returns:
        Array1d: Boolean mask with True for the first N elements of each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_idxs.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[:n]] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def last_n_nb(col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Return a boolean mask marking the last N elements in each column.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        n (int): Number of elements to mark from the end of each column.

    Returns:
        Array1d: Boolean mask with True for the last N elements of each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_idxs.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[-n:]] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def random_n_nb(col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Return the boolean mask selecting n random elements from each column.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        n (int): Number of elements to select randomly from each column.

    Returns:
        Array1d: Boolean mask array with True at positions of selected elements.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_idxs.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[np.random.choice(idxs, n, replace=False)] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def top_n_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Return the boolean mask selecting the top n elements from each column based on the mapped values.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        n (int): Number of top elements to select from each column.

    Returns:
        Array1d: Boolean mask array with True for indices of the top n elements in each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[np.argsort(mapped_arr[idxs])[-n:]]] = True
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        n=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bottom_n_mapped_nb(mapped_arr: tp.Array1d, col_map: tp.GroupMap, n: int) -> tp.Array1d:
    """Return the boolean mask selecting the bottom n elements from each column based on the mapped values.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        n (int): Number of bottom elements to select from each column.

    Returns:
        Array1d: Boolean mask array with True for indices of the bottom n elements in each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(mapped_arr.shape[0], False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs[np.argsort(mapped_arr[idxs])[:n]]] = True
    return out


# ############# Mapping ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="records", axis=0),
    arg_take_spec=dict(records=ch.ArraySlicer(axis=0), map_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def map_records_nb(records: tp.RecordArray, map_func_nb: tp.RecordsMapFunc, *args) -> tp.Array1d:
    """Map each record to a single value.

    Args:
        records (RecordArray): Array of records.
        map_func_nb (RecordsMapFunc): Callback function that accepts a record and
            additional arguments, and returns a single value.
        *args: Positional arguments for `map_func_nb`.

    Returns:
        Array1d: Array containing the mapped values for each record.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(records.shape[0], dtype=float_)

    for ridx in prange(records.shape[0]):
        out[ridx] = map_func_nb(records[ridx], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_values"),
    arg_take_spec=dict(n_values=ch.CountAdapter(), map_func_nb=None, args=ch.ArgsTaker()),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def map_records_meta_nb(n_values: int, map_func_nb: tp.MappedReduceMetaFunc, *args) -> tp.Array1d:
    """Map each record index to a single value using a meta mapping function.

    Args:
        n_values (int): Total number of values.
        map_func_nb (MappedReduceMetaFunc): Callback function that accepts a record index and
            additional arguments, and returns a single value.
        *args: Positional arguments for `map_func_nb`.

    Returns:
        Array1d: Array containing the meta-mapped values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(n_values, dtype=float_)

    for ridx in prange(n_values):
        out[ridx] = map_func_nb(ridx, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_nb(
    arr: tp.Array1d, col_map: tp.GroupMap, apply_func_nb: tp.ApplyFunc, *args
) -> tp.Array1d:
    """Apply a function on segments of an array corresponding to each column.

    Args:
        arr (Array1d): Array on which to apply the function.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        apply_func_nb (ApplyFunc): Callback function that accepts a column as a 1D array
            and additional arguments, and returns a single value or an array that broadcasts
            to the column's shape.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array1d: Array resulting from applying the function to each column,
            matching the shape of the input array.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(arr.shape[0], dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs] = apply_func_nb(arr[idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        n_values=ch.CountAdapter(mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        apply_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def apply_meta_nb(
    n_values: int, col_map: tp.GroupMap, apply_func_nb: tp.ApplyMetaFunc, *args
) -> tp.Array1d:
    """Apply a function to meta information for each column.

    Args:
        n_values (int): Total number of values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        apply_func_nb (ApplyMetaFunc): Callback function that accepts indices for a column as a 1D array,
            the column index, and additional arguments, and returns a single value or an array
            that broadcasts to the column's shape.
        *args: Positional arguments for `apply_func_nb`.

    Returns:
        Array1d: Array resulting from applying the function to each column's meta information.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(n_values, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[idxs] = apply_func_nb(idxs, col, *args)
    return out


# ############# Reducing ############# #


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        id_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        segment_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted
def reduce_mapped_segments_nb(
    mapped_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    id_arr: tp.Array1d,
    col_map: tp.GroupMap,
    segment_arr: tp.Array1d,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Reduce each segment of values in the mapped array.

    Uses the last column, index, and id of each segment to compute the reduced value.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        idx_arr (Array1d): Array of row indices corresponding to each element in `mapped_arr`.
        id_arr (Array1d): Array of IDs corresponding to each value in `mapped_arr`.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        segment_arr (Array1d): Array indicating segment boundaries for mapped values.
        reduce_func_nb (ReduceFunc): Callback function that accepts a segment of `mapped_arr`
            as a 1D array and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Tuple[Array1d, Array1d, Array1d, Array1d]: Tuple containing the reduced values,
            column indices, indices, and id values.

    !!! note
        Groups must be in ascending order per column, and `idx_arr` and `id_arr` must be
        in ascending order within each segment.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(len(mapped_arr), dtype=mapped_arr.dtype)
    col_arr_out = np.empty(len(mapped_arr), dtype=int_)
    idx_arr_out = np.empty(len(mapped_arr), dtype=int_)
    id_arr_out = np.empty(len(mapped_arr), dtype=int_)

    k = 0
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue

        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]

        segment_start_i = 0
        for i in range(len(idxs)):
            r = idxs[i]
            if i == 0:
                prev_r = -1
            else:
                prev_r = idxs[i - 1]
            if i < len(idxs) - 1:
                next_r = idxs[i + 1]
            else:
                next_r = -1

            if prev_r != -1:
                if segment_arr[r] < segment_arr[prev_r]:
                    raise ValueError("segment_arr must come in ascending order per column")
                elif segment_arr[r] == segment_arr[prev_r]:
                    if idx_arr[r] < idx_arr[prev_r]:
                        raise ValueError("idx_arr must come in ascending order per segment")
                    if id_arr[r] < id_arr[prev_r]:
                        raise ValueError("id_arr must come in ascending order per segment")
                else:
                    segment_start_i = i
            if next_r == -1 or segment_arr[r] != segment_arr[next_r]:
                n_values = i - segment_start_i + 1
                if n_values > 1:
                    out[k] = reduce_func_nb(mapped_arr[idxs[segment_start_i : i + 1]], *args)
                else:
                    out[k] = mapped_arr[r]
                col_arr_out[k] = col
                idx_arr_out[k] = idx_arr[r]
                id_arr_out[k] = id_arr[r]
                k += 1
    return out[:k], col_arr_out[:k], idx_arr_out[:k], id_arr_out[:k]


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce the mapped array by column to a single reduced value.

    Faster than `unstack_mapped_nb` used in combination with other vectorbtpro methods,
    and it requires less memory, though it does not benefit from caching.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        fill_value (float): Fill value used to initialize the output array.
        reduce_func_nb (ReduceFunc): Callback function that accepts a segment of `mapped_arr`
            as a 1D array and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced value for each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[col] = reduce_func_nb(mapped_arr[idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_meta_nb(
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Provide a meta reduction of the mapped array per column.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        fill_value (float): Fill value used to initialize the output array.
        reduce_func_nb (MappedReduceMetaFunc): Callback function that accepts mapped indices for a column
            as a 1D array, the column index, and additional arguments, and returns a single value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the reduced meta value for each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[col] = reduce_func_nb(idxs, col, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce the mapped array by column to determine an index for each column.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        idx_arr (Array1d): Array of row indices corresponding to each element in `mapped_arr`.
        fill_value (float): Fill value used to initialize the output array.
        reduce_func_nb (ReduceFunc): Callback function that accepts a segment of `mapped_arr`
            as a 1D array and additional arguments, and returns a single index value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array of computed indices.

    !!! note
        The reduction function must return an integer value or raise an exception.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        out[col] = idx_arr[idxs][col_out]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="concat",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_meta_nb(
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `reduce_mapped_to_idx_nb`.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        idx_arr (Array1d): Array of row indices.
        fill_value (float): Value to fill in the output for columns without data.
        reduce_func_nb (MappedReduceMetaFunc): Callback function that accepts mapped indices for a column
            as a 1D array, the column index, and additional arguments, and returns a single index value.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array1d: Array containing the meta reduced values for each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], fill_value, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        out[col] = idx_arr[idxs][col_out]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_array_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array2d:
    """Reduce mapped array by column to an array.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        fill_value (float): Value used to initialize the output array.
        reduce_func_nb (ReduceToArrayFunc): Callback function that accepts a segment of `mapped_arr`
            as a 1D array and additional arguments, and returns a 1D array.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array with reduced values across columns, where each column corresponds to a column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(mapped_arr[midxs0], *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = col_0_out[i]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        for i in range(col_out.shape[0]):
            out[i, col] = col_out[i]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_array_meta_nb(
    col_map: tp.GroupMap,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceToArrayMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `reduce_mapped_to_array_nb`.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        fill_value (float): Value used to fill the output array.
        reduce_func_nb (MappedReduceToArrayMetaFunc): Callback function that accepts mapped indices for a column
            as a 1D array, the column index, and additional arguments, and returns a 1D array.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array with meta-reduced values across columns.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(midxs0, col0, *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = col_0_out[i]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        for i in range(col_out.shape[0]):
            out[i, col] = col_out[i]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        mapped_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_array_nb(
    mapped_arr: tp.Array1d,
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.ReduceToArrayFunc,
    *args,
) -> tp.Array2d:
    """Reduce mapped array by column to an index array.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        idx_arr (Array1d): Array of row indices corresponding to each element in `mapped_arr`.
        fill_value (float): Value used to initialize the output array.
        reduce_func_nb (ReduceToArrayFunc): Callback function that accepts a segment of `mapped_arr`
            as a 1D array and additional arguments, and returns a 1D array of indices.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array with indices corresponding to reduced values for each column.

    !!! note
        Must return integers or raise an exception.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(mapped_arr[midxs0], *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = idx_arr[midxs0[col_0_out[i]]]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(mapped_arr[idxs], *args)
        for i in range(col_0_out.shape[0]):
            out[i, col] = idx_arr[idxs[col_out[i]]]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        col_map=base_ch.GroupMapSlicer(),
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        fill_value=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def reduce_mapped_to_idx_array_meta_nb(
    col_map: tp.GroupMap,
    idx_arr: tp.Array1d,
    fill_value: float,
    reduce_func_nb: tp.MappedReduceToArrayMetaFunc,
    *args,
) -> tp.Array2d:
    """Meta version of `reduce_mapped_to_idx_array_nb`.

    Uses `reduce_func_nb` similarly to `reduce_mapped_meta_nb` to compute a 2D array by reducing mapped indexes.

    Args:
        col_map (GroupMap): Tuple of indices and lengths for each column.
        idx_arr (Array1d): Array of row indices.
        fill_value (float): Fill value used to initialize the output array.
        reduce_func_nb (MappedReduceToArrayMetaFunc): Callback function that accepts mapped indices for a column
            as a 1D array, the column index, and additional arguments, and returns a 1D array of indices.
        *args: Positional arguments for `reduce_func_nb`.

    Returns:
        Array2d: 2D array with reduced values computed from the input indices.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in range(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len > 0:
            col_start_idx = col_start_idxs[col]
            col0, midxs0 = col, col_idxs[col_start_idx : col_start_idx + col_len]
            break

    col_0_out = reduce_func_nb(midxs0, col0, *args)
    out = np.full((col_0_out.shape[0], col_lens.shape[0]), fill_value, dtype=float_)
    for i in range(col_0_out.shape[0]):
        out[i, col0] = idx_arr[midxs0[col_0_out[i]]]

    for col in prange(col0 + 1, col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        col_out = reduce_func_nb(idxs, col, *args)
        for i in range(col_0_out.shape[0]):
            out[i, col] = idx_arr[idxs[col_out[i]]]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        codes=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        n_uniques=None,
        col_map=base_ch.GroupMapSlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mapped_value_counts_per_col_nb(
    codes: tp.Array1d, n_uniques: int, col_map: tp.GroupMap
) -> tp.Array2d:
    """Get value counts per column of a factorized mapped array.

    Counts the occurrences of each unique value in each column defined by the column mapping.

    Args:
        codes (Array1d): Factorized array of integer codes.
        n_uniques (int): Number of unique values.
        col_map (GroupMap): Tuple of indices and lengths for each column.

    Returns:
        Array2d: 2D integer array with counts for each unique value per column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((n_uniques, col_lens.shape[0]), 0, dtype=int_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        idxs = col_idxs[col_start_idx : col_start_idx + col_len]
        out[:, col] = generic_nb.value_counts_1d_nb(codes[idxs], n_uniques)
    return out


@register_jitted(cache=True)
def mapped_value_counts_per_row_nb(
    mapped_arr: tp.Array1d,
    n_uniques: int,
    idx_arr: tp.Array1d,
    n_rows: int,
) -> tp.Array2d:
    """Get value counts per row of a factorized mapped array.

    Counts the occurrences of each unique factor across rows using the provided index mapping.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        n_uniques (int): Number of unique values.
        idx_arr (Array1d): Array of row indices corresponding to each element in `mapped_arr`.
        n_rows (int): Total number of rows.

    Returns:
        Array2d: 2D integer array with counts for each unique value per row.
    """
    out = np.full((n_uniques, n_rows), 0, dtype=int_)

    for c in range(mapped_arr.shape[0]):
        out[mapped_arr[c], idx_arr[c]] += 1
    return out


@register_jitted(cache=True)
def mapped_value_counts_nb(mapped_arr: tp.Array1d, n_uniques: int) -> tp.Array2d:
    """Get global value counts of a factorized mapped array.

    Counts the occurrences of each unique factor in the entire array.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        n_uniques (int): Number of unique values.

    Returns:
        Array2d: 1D integer array containing the count for each unique value.
    """
    out = np.full(n_uniques, 0, dtype=int_)

    for c in range(mapped_arr.shape[0]):
        out[mapped_arr[c]] += 1
    return out


@register_jitted(cache=True)
def mapped_has_conflicts_nb(
    col_arr: tp.Array1d, idx_arr: tp.Array1d, target_shape: tp.Shape
) -> bool:
    """Check for positional conflicts in a mapped array.

    Verifies if any positions in the target array, defined by indices from `col_arr` and `idx_arr`,
    have been assigned more than once.

    Args:
        col_arr (Array1d): Array of column indices.
        idx_arr (Array1d): Array of row indices.
        target_shape (Shape): Base dimensions (rows, columns).

    Returns:
        bool: True if any positional conflict exists, False otherwise.
    """
    temp = np.zeros(target_shape)

    for i in range(len(col_arr)):
        if temp[idx_arr[i], col_arr[i]] > 0:
            return True
        temp[idx_arr[i], col_arr[i]] = 1
    return False


@register_jitted(cache=True)
def mapped_coverage_map_nb(
    col_arr: tp.Array1d, idx_arr: tp.Array1d, target_shape: tp.Shape
) -> tp.Array2d:
    """Get the coverage map of a mapped array.

    Generates a 2D map where each element indicates the number of times the corresponding
    index was referenced. A value greater than one signals a positional conflict.

    Args:
        col_arr (Array1d): Array of column indices.
        idx_arr (Array1d): Array of row indices.
        target_shape (Shape): Base dimensions (rows, columns).

    Returns:
        Array2d: 2D integer array representing the coverage map.
    """
    out = np.zeros(target_shape, dtype=int_)

    for i in range(len(col_arr)):
        out[idx_arr[i], col_arr[i]] += 1
    return out


def _unstack_mapped_nb(
    mapped_arr,
    col_arr,
    idx_arr,
    target_shape,
    fill_value,
):
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _impl(mapped_arr, col_arr, idx_arr, target_shape, fill_value):
        out = np.full(target_shape, fill_value, dtype=dtype)

        for r in range(mapped_arr.shape[0]):
            out[idx_arr[r], col_arr[r]] = mapped_arr[r]
        return out

    if not nb_enabled:
        return _impl(mapped_arr, col_arr, idx_arr, target_shape, fill_value)

    return _impl


overload(_unstack_mapped_nb)(_unstack_mapped_nb)


@register_jitted(cache=True)
def unstack_mapped_nb(
    mapped_arr: tp.Array1d,
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    target_shape: tp.Shape,
    fill_value: float,
) -> tp.Array2d:
    """Unstack mapped array using index data.

    Rearranges a mapped 1D array into a 2D array based on provided column and row indices.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_arr (Array1d): Array of column indices corresponding to each element in `mapped_arr`.
        idx_arr (Array1d): Array of row indices corresponding to each element in `mapped_arr`.
        target_shape (Shape): Base dimensions (rows, columns).
        fill_value (float): Fill value for positions with no data.

    Returns:
        Array2d: Unstacked 2D array with values placed according to `col_arr` and `idx_arr`.
    """
    return _unstack_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, fill_value)


def _ignore_unstack_mapped_nb(mapped_arr, col_map, fill_value):
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _impl(mapped_arr, col_map, fill_value):
        col_idxs, col_lens = col_map
        col_start_idxs = np.cumsum(col_lens) - col_lens
        out = np.full((np.max(col_lens), col_lens.shape[0]), fill_value, dtype=dtype)

        for col in range(col_lens.shape[0]):
            col_len = col_lens[col]
            if col_len == 0:
                continue
            col_start_idx = col_start_idxs[col]
            idxs = col_idxs[col_start_idx : col_start_idx + col_len]
            out[:col_len, col] = mapped_arr[idxs]

        return out

    if not nb_enabled:
        return _impl(mapped_arr, col_map, fill_value)

    return _impl


overload(_ignore_unstack_mapped_nb)(_ignore_unstack_mapped_nb)


@register_jitted(cache=True)
def ignore_unstack_mapped_nb(
    mapped_arr: tp.Array1d, col_map: tp.GroupMap, fill_value: float
) -> tp.Array2d:
    """Unstack mapped array by ignoring index data.

    Rearranges the mapped array into a 2D array using column information from `col_map`,
    without using explicit row indices.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        fill_value (float): Fill value for positions with no data.

    Returns:
        Array2d: 2D array where each column represents a column from the mapped array.
    """
    return _ignore_unstack_mapped_nb(mapped_arr, col_map, fill_value)


@register_jitted(cache=True)
def unstack_index_nb(repeat_cnt_arr: tp.Array1d) -> tp.Array1d:
    """Unstack index using the repeat counts.

    Generates a flattened index array where each index is repeated according to its count in `repeat_cnt_arr`.

    Args:
        repeat_cnt_arr (Array1d): Array of repetition counts for each index.

    Returns:
        Array1d: Flattened index array with indices repeated as specified.
    """
    out = np.empty(np.sum(repeat_cnt_arr), dtype=int_)

    k = 0
    for i in range(len(repeat_cnt_arr)):
        out[k : k + repeat_cnt_arr[i]] = i
        k += repeat_cnt_arr[i]
    return out


def _repeat_unstack_mapped_nb(
    mapped_arr,
    col_arr,
    idx_arr,
    repeat_cnt_arr,
    n_cols,
    fill_value,
):
    nb_enabled = not isinstance(mapped_arr, np.ndarray)
    if nb_enabled:
        mapped_arr_dtype = as_dtype(mapped_arr.dtype)
        fill_value_dtype = as_dtype(fill_value)
    else:
        mapped_arr_dtype = mapped_arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(mapped_arr_dtype, fill_value_dtype)

    def _impl(mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value):
        index_start_arr = np.cumsum(repeat_cnt_arr) - repeat_cnt_arr
        out = np.full((np.sum(repeat_cnt_arr), n_cols), fill_value, dtype=dtype)
        temp = np.zeros((len(repeat_cnt_arr), n_cols), dtype=int_)

        for i in range(len(col_arr)):
            out[index_start_arr[idx_arr[i]] + temp[idx_arr[i], col_arr[i]], col_arr[i]] = (
                mapped_arr[i]
            )
            temp[idx_arr[i], col_arr[i]] += 1
        return out

    if not nb_enabled:
        return _impl(mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value)

    return _impl


overload(_repeat_unstack_mapped_nb)(_repeat_unstack_mapped_nb)


@register_jitted(cache=True)
def repeat_unstack_mapped_nb(
    mapped_arr: tp.Array1d,
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    repeat_cnt_arr: tp.Array1d,
    n_cols: int,
    fill_value: float,
) -> tp.Array2d:
    """Return an unstacked 2D array from a mapped array using repeated index data.

    Args:
        mapped_arr (Array1d): Array of mapped values.
        col_arr (Array1d): Array of column indices corresponding to each element in `mapped_arr`.
        idx_arr (Array1d): Array of row indices corresponding to each element in `mapped_arr`.
        repeat_cnt_arr (Array1d): Array of repetition counts for each index.
        n_cols (int): Total number of columns in the output array.
        fill_value (float): Value used to initialize the output array.

    Returns:
        Array2d: Unstacked 2D array, where values are arranged based on index and repetition count.
    """
    return _repeat_unstack_mapped_nb(
        mapped_arr, col_arr, idx_arr, repeat_cnt_arr, n_cols, fill_value
    )
