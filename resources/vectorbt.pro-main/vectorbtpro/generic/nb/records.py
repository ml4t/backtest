# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for records."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb, flex_select_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.nb.base import repartition_nb
from vectorbtpro.generic.nb.patterns import pattern_similarity_nb
from vectorbtpro.generic.nb.sim_range import prepare_sim_range_nb
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.template import Rep

# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), gap_value=None),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_ranges_nb(arr: tp.Array2d, gap_value: tp.Scalar) -> tp.RecordArray:
    """Fill range records in a 2D array by detecting gaps.

    Args:
        arr (Array2d): 2D NumPy array where each column represents a time series.
        gap_value (Scalar): Value that signifies a gap in the data.

    Returns:
        RecordArray: Record array containing range details with fields
            `id`, `col`, `start_idx`, `end_idx`, and `status`.

            Has the `vectorbtpro.generic.enums.range_dt` dtype.

    !!! tip
        This function is parallelizable.

    Examples:
        Find ranges in time series:

        ```pycon
        >>> from vectorbtpro import *

        >>> a = np.array([
        ...     [np.nan, np.nan, np.nan, np.nan],
        ...     [     2, np.nan, np.nan, np.nan],
        ...     [     3,      3, np.nan, np.nan],
        ...     [np.nan,      4,      4, np.nan],
        ...     [     5, np.nan,      5,      5],
        ...     [     6,      6, np.nan,      6]
        ... ])
        >>> records = vbt.nb.get_ranges_nb(a, np.nan)

        >>> pd.DataFrame.from_records(records)
           id  col  start_idx  end_idx  status
        0   0    0          1        3       1
        1   1    0          4        5       0
        2   0    1          2        4       1
        3   1    1          5        5       0
        4   0    2          3        5       1
        5   0    3          4        5       0
        ```
    """
    new_records = np.empty(arr.shape, dtype=range_dt)
    counts = np.full(arr.shape[1], 0, dtype=int_)

    for col in prange(arr.shape[1]):
        range_started = False
        start_idx = -1
        end_idx = -1
        store_record = False
        status = -1

        for i in range(arr.shape[0]):
            cur_val = arr[i, col]

            if cur_val == gap_value or np.isnan(cur_val) and np.isnan(gap_value):
                if range_started:
                    # If stopped, save the current range
                    end_idx = i
                    range_started = False
                    store_record = True
                    status = RangeStatus.Closed
            else:
                if not range_started:
                    # If started, register a new range
                    start_idx = i
                    range_started = True

            if i == arr.shape[0] - 1 and range_started:
                # If still running, mark for save
                end_idx = arr.shape[0] - 1
                range_started = False
                store_record = True
                status = RangeStatus.Open

            if store_record:
                # Save range to the records
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = start_idx
                new_records["end_idx"][r, col] = end_idx
                new_records["status"][r, col] = status
                counts[col] += 1

                # Reset running vars for a new range
                store_record = False

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        n_rows=None,
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        id_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index=None,
        delta=None,
        delta_use_index=None,
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_ranges_from_delta_nb(
    n_rows: int,
    idx_arr: tp.Array1d,
    id_arr: tp.Array1d,
    col_map: tp.GroupMap,
    index: tp.Optional[tp.Array1d] = None,
    delta: int = 0,
    delta_use_index: bool = False,
    shift: int = 0,
) -> tp.RecordArray:
    """Build delta-based ranges for record indices.

    Args:
        n_rows (int): Total number of rows in the dataset.
        idx_arr (Array1d): Array of row indices.
        id_arr (Array1d): Array of IDs.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        index (Optional[Array1d]): Index array in nanosecond format.
        delta (int): Delta offset to compute range boundaries.
        delta_use_index (bool): Flag indicating whether to use the index values for delta calculation.
        shift (int): Shift applied to the starting index.

    Returns:
        RecordArray: Record array with fields `id`, `col`, `start_idx`, `end_idx`,
            and `status` indicating the computed ranges.

            Has the `vectorbtpro.generic.enums.range_dt` dtype.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(idx_arr.shape[0], dtype=range_dt)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]

        for r in ridxs:
            r_idx = idx_arr[r] + shift
            if r_idx < 0:
                r_idx = 0
            if r_idx > n_rows - 1:
                r_idx = n_rows - 1
            if delta >= 0:
                start_idx = r_idx
                if delta_use_index:
                    if index is None:
                        raise ValueError("Index is required")
                    end_idx = len(index) - 1
                    status = RangeStatus.Open
                    for i in range(start_idx, index.shape[0]):
                        if index[i] >= index[start_idx] + delta:
                            end_idx = i
                            status = RangeStatus.Closed
                            break
                else:
                    if start_idx + delta < n_rows:
                        end_idx = start_idx + delta
                        status = RangeStatus.Closed
                    else:
                        end_idx = n_rows - 1
                        status = RangeStatus.Open
            else:
                end_idx = r_idx
                status = RangeStatus.Closed
                if delta_use_index:
                    if index is None:
                        raise ValueError("Index is required")
                    start_idx = 0
                    for i in range(end_idx, -1, -1):
                        if index[i] <= index[end_idx] + delta:
                            start_idx = i
                            break
                else:
                    if end_idx + delta >= 0:
                        start_idx = end_idx + delta
                    else:
                        start_idx = 0

            out["id"][r] = id_arr[r]
            out["col"][r] = col
            out["start_idx"][r] = start_idx
            out["end_idx"][r] = end_idx
            out["status"][r] = status

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
        status_arr=ch.ArraySlicer(axis=0),
        freq=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def range_duration_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    freq: int = 1,
) -> tp.Array1d:
    """Calculate the duration of each range record.

    Args:
        start_idx_arr (Array1d): Array of starting row indices for each range record.
        end_idx_arr (Array1d): Array of ending row indices for each range record.
        status_arr (Array2d): Array indicating the status of each range record.

            If a record is open, the duration is adjusted.
        freq (int): Frequency increment added to the duration for open range records.

    Returns:
        Array1d: Array containing the computed durations of the range records.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(start_idx_arr.shape[0], dtype=int_)
    for r in prange(start_idx_arr.shape[0]):
        if status_arr[r] == RangeStatus.Open:
            out[r] = end_idx_arr[r] - start_idx_arr[r] + freq
        else:
            out[r] = end_idx_arr[r] - start_idx_arr[r]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_lens=ch.ArraySlicer(axis=0),
        overlapping=None,
        normalize=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def range_coverage_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.GroupMap,
    index_lens: tp.Array1d,
    overlapping: bool = False,
    normalize: bool = False,
) -> tp.Array1d:
    """Calculate the coverage of range records for each column.

    Args:
        start_idx_arr (Array1d): Array of starting row indices for each range record.
        end_idx_arr (Array1d): Array of ending row indices for each range record.
        status_arr (Array2d): Array indicating the status of each range record.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        index_lens (Array1d): Array of index lengths for each column.
        overlapping (bool): If True, compute the count of overlapping steps.
        normalize (bool): If True, return the coverage as a normalized ratio relative
            to the total number of steps (if overlapping is False) or the number of
            covered steps (if overlapping is True).

    Returns:
        Array1d: Array containing the computed coverage values for each column.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], np.nan, dtype=float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]
        temp = np.full(index_lens[col], 0, dtype=int_)
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                temp[start_idx_arr[r] : end_idx_arr[r] + 1] += 1
            else:
                temp[start_idx_arr[r] : end_idx_arr[r]] += 1
        if overlapping:
            if normalize:
                pos_temp_sum = np.sum(temp > 0)
                if pos_temp_sum == 0:
                    out[col] = np.nan
                else:
                    out[col] = np.sum(temp > 1) / pos_temp_sum
            else:
                out[col] = np.sum(temp > 1)
        else:
            if normalize:
                if index_lens[col] == 0:
                    out[col] = np.nan
                else:
                    out[col] = np.sum(temp > 0) / index_lens[col]
            else:
                out[col] = np.sum(temp > 0)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_len=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ranges_to_mask_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.GroupMap,
    index_len: int,
) -> tp.Array2d:
    """Convert range records into a 2-dimensional mask.

    Args:
        start_idx_arr (Array1d): Array of starting row indices for each range record.
        end_idx_arr (Array1d): Array of ending row indices for each range record.
        status_arr (Array2d): Array indicating the status of each range record.
        col_map (GroupMap): Tuple of indices and lengths for each column.
        index_len (int): Length of the index for the resulting mask.

    Returns:
        Array2d: 2-dimensional boolean mask where True indicates that the position is within a range.

    !!! tip
        This function is parallelizable.
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((index_len, col_lens.shape[0]), False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                out[start_idx_arr[r] : end_idx_arr[r] + 1, col] = True
            else:
                out[start_idx_arr[r] : end_idx_arr[r], col] = True

    return out


@register_jitted(cache=True)
def map_ranges_to_projections_nb(
    close: tp.Array2d,
    col_arr: tp.Array1d,
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array1d,
    index: tp.Optional[tp.Array1d] = None,
    proj_start: int = 0,
    proj_start_use_index: bool = False,
    proj_period: tp.Optional[int] = None,
    proj_period_use_index: bool = False,
    incl_end_idx: bool = True,
    extend: bool = False,
    rebase: bool = True,
    start_value: tp.FlexArray1dLike = 1.0,
    ffill: bool = False,
    remove_empty: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array2d]:
    """Map each specified range to a projection.

    Args:
        close (Array2d): 2D array of close prices.
        col_arr (Array1d): 1D array of column indices for selecting data from `close`.
        start_idx_arr (Array1d): 1D array of starting row indices for each range.
        end_idx_arr (Array1d): 1D array of ending row indices for each range.
        status_arr (Array1d): 1D array indicating the status of each range.
        index (Optional[Array1d]): Index array in nanosecond format.
        proj_start (int): Offset from the start index to begin the projection.
        proj_start_use_index (bool): Whether to compute the projection start using the `index` array.
        proj_period (Optional[int]): Period or duration for the projection.
        proj_period_use_index (bool): Whether to determine the projection period based on the `index` array.
        incl_end_idx (bool): Whether the end index of a range is inclusive.
        extend (bool): Whether to extend the projection to a fixed length (beyond the end).

            The extension period is taken from the longest range duration if `proj_period`
            is None, and from the longest `proj_period` if not None.
        rebase (bool): Rebase projection values so that each projection starts with `start_value`.

            If False, the projection retains the original close values.
        start_value (ArrayLike): Initial value for rebasing.

            Can be a scalar or an array per column. If set to -1, uses the latest close value.
        ffill (bool): Forward fill NaN values in the projection, even if they are NaN in `close`.
        remove_empty (bool): Remove projections that are NaN or contain only a single element.

    Returns:
        Tuple[Array1d, Array2d]: Tuple where the first element is a 1D array of
            record indices and the second element is a 2D array of projections,
            with each row corresponding to a range.
    """
    start_value_ = to_1d_array_nb(np.asarray(start_value))

    index_ranges_temp = np.empty((start_idx_arr.shape[0], 2), dtype=int_)

    max_duration = 0
    for r in range(start_idx_arr.shape[0]):
        if proj_start_use_index:
            if index is None:
                raise ValueError("Index is required")
            r_proj_start = len(index) - start_idx_arr[r]
            for i in range(start_idx_arr[r], index.shape[0]):
                if index[i] >= index[start_idx_arr[r]] + proj_start:
                    r_proj_start = i - start_idx_arr[r]
                    break
            r_start_idx = start_idx_arr[r] + r_proj_start
        else:
            r_start_idx = start_idx_arr[r] + proj_start
        if status_arr[r] == RangeStatus.Open:
            if incl_end_idx:
                r_duration = end_idx_arr[r] - start_idx_arr[r] + 1
            else:
                r_duration = end_idx_arr[r] - start_idx_arr[r]
        else:
            if incl_end_idx:
                r_duration = end_idx_arr[r] - start_idx_arr[r]
            else:
                r_duration = end_idx_arr[r] - start_idx_arr[r] - 1
        if proj_period is None:
            r_end_idx = start_idx_arr[r] + r_duration
        else:
            if proj_period_use_index:
                if index is None:
                    raise ValueError("Index is required")
                r_proj_period = -1
                for i in range(r_start_idx, index.shape[0]):
                    if index[i] <= index[r_start_idx] + proj_period:
                        r_proj_period = i - r_start_idx
                    else:
                        break
            else:
                r_proj_period = proj_period
            if extend:
                r_end_idx = r_start_idx + r_proj_period
            else:
                r_end_idx = min(start_idx_arr[r] + r_duration, r_start_idx + r_proj_period)
        r_end_idx = r_end_idx + 1
        if r_end_idx > close.shape[0]:
            r_end_idx = close.shape[0]
        if r_start_idx > r_end_idx:
            r_start_idx = r_end_idx
        if r_end_idx - r_start_idx > max_duration:
            max_duration = r_end_idx - r_start_idx
        index_ranges_temp[r, 0] = r_start_idx
        index_ranges_temp[r, 1] = r_end_idx

    ridx_out = np.empty((start_idx_arr.shape[0],), dtype=int_)
    proj_out = np.empty((start_idx_arr.shape[0], max_duration), dtype=float_)

    k = 0
    for r in range(start_idx_arr.shape[0]):
        if extend:
            r_start_idx = index_ranges_temp[r, 0]
            r_end_idx = index_ranges_temp[r, 0] + proj_out.shape[1]
        else:
            r_start_idx = index_ranges_temp[r, 0]
            r_end_idx = index_ranges_temp[r, 1]
        r_close = close[r_start_idx:r_end_idx, col_arr[r]]
        any_set = False
        for i in range(proj_out.shape[1]):
            if i >= r_close.shape[0]:
                proj_out[k, i] = np.nan
            else:
                if rebase:
                    if i == 0:
                        _start_value = flex_select_1d_pc_nb(start_value_, col_arr[r])
                        if _start_value == -1:
                            proj_out[k, i] = close[-1, col_arr[r]]
                        else:
                            proj_out[k, i] = _start_value
                    else:
                        if r_close[i - 1] == 0:
                            proj_out[k, i] = np.nan
                        else:
                            proj_out[k, i] = proj_out[k, i - 1] * r_close[i] / r_close[i - 1]
                else:
                    proj_out[k, i] = r_close[i]
            if not np.isnan(proj_out[k, i]) and i > 0:
                any_set = True
            if ffill and np.isnan(proj_out[k, i]) and i > 0:
                proj_out[k, i] = proj_out[k, i - 1]
        if any_set or not remove_empty:
            ridx_out[k] = r
            k += 1
    if remove_empty:
        return ridx_out[:k], proj_out[:k]
    return ridx_out, proj_out


@register_jitted(cache=True)
def find_pattern_1d_nb(
    arr: tp.Array1d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
    roll_forward: bool = False,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = 0.85,
    minp: tp.Optional[int] = None,
    overlap_mode: int = OverlapMode.Disallow,
    max_records: tp.Optional[int] = None,
    col: int = 0,
) -> tp.RecordArray:
    """Find all occurrences of a pattern in a 1D array.

    Iterate through the input array to identify segments that closely match the provided pattern.
    For each valid window length between `window` and `max_window`, compute a similarity score
    using `vectorbtpro.generic.nb.patterns.pattern_similarity_nb`. If the similarity meets the
    `min_similarity` threshold, record the segment in a record array.

    If `roll_forward` is True, windows are processed forward ensuring sorted `start_idx`;
    otherwise, windows are processed backward with sorted `end_idx`. The function selectively
    processes rows and windows based on `row_select_prob` and `window_select_prob`.
    The length of the output record array is limited by `max_records` when specified.

    Args:
        arr (Array1d): Input one-dimensional array in which to search for the pattern.
        pattern (Array1d): 1D array representing the pattern to locate.

            Can be smaller or larger than the source array. In such cases,
            the smaller array is stretched using the interpolation mode specified by `interp_mode`.
        window (Optional[int]): Window size.

            If None, defaults to the length of `pattern`.
        max_window (Optional[int]): Maximum length of the rolling window for matching.

            If None, defaults to `window`.
        row_select_prob (float): Probability of selecting a row.
        window_select_prob (float): Probability of selecting a window size.
        roll_forward (bool): Process windows in forward direction if True; otherwise, in backward direction.
        interp_mode (int): Interpolation mode.

            See `vectorbtpro.generic.enums.InterpMode`.
        rescale_mode (int): Rescaling mode for adjusting the ranges of `arr` and `pattern`.

            See `vectorbtpro.generic.enums.RescaleMode`.
        vmin (float): Minimum value used for rescaling `arr`.

            Use only when the array has fixed bounds. Used in rescaling with `RescaleMode.MinMax`
            and for verifying `min_pct_change` and `max_pct_change`.

            If set to NaN, it is calculated dynamically.
        vmax (float): Maximum value used for rescaling `arr`.

            Use only when the array has fixed bounds. Used in rescaling with `RescaleMode.MinMax`
            and for verifying `min_pct_change` and `max_pct_change`.

            If set to NaN, it is calculated dynamically.
        pmin (float): Minimum value used for rescaling `pattern`.

            Used in rescaling with `RescaleMode.MinMax` and for computing the maximum distance
            at each point when `max_error_as_maxdist` is disabled.

            If set to NaN, it is calculated dynamically.
        pmax (float): Maximum value used for rescaling `pattern`.

            Used in rescaling with `RescaleMode.MinMax` and for computing the maximum distance
            at each point when `max_error_as_maxdist` is disabled.

            If set to NaN, it is calculated dynamically.
        invert (bool): Invert the pattern by reflecting its values.
        error_type (int): Error computation mode.

            See `vectorbtpro.generic.enums.ErrorType`.
        distance_measure (int): Method for measuring distance (e.g., MAE, MSE, RMSE).

            See `vectorbtpro.generic.enums.DistanceMeasure`.
        max_error (FlexArray1dLike): Maximum error threshold for normalization.

            Provided as a scalar or per element in the pattern.
        max_error_interp_mode (Optional[int]): Interpolation mode for `max_error`.

            If None, defaults to `interp_mode`.

            See `vectorbtpro.generic.enums.InterpMode`.
        max_error_as_maxdist (bool): Indicates whether `max_error` represents the maximum distance at each point.

            If False, exceeding `max_error` sets the distance to the maximum derived from
            `pmin`, `pmax`, and the pattern value at that point. If True and any point
            in a window is NaN, that point is skipped.
        max_error_strict (bool): If True, any instance of exceeding `max_error` results in a similarity of NaN.
        min_pct_change (float): Minimum percentage change required for a window to remain a search candidate.

            Window similarity is set to NaN if this threshold is not met.
        max_pct_change (float): Maximum percentage change allowed for a window to remain a search candidate.

            Window similarity is set to NaN if this threshold is exceeded.
        min_similarity (float): Minimum similarity threshold.

            If the computed similarity falls below this, returns NaN.
        minp (Optional[int]): Minimum number of observations required.
        overlap_mode (int): Mode for handling overlapping matches.

            See `vectorbtpro.generic.enums.OverlapMode`.
        max_records (Optional[int]): Maximum number of records to be filled.

            If None, defaults to the number of rows in `arr`.
        col (int): Column index assigned to the result records.

    Returns:
        RecordArray: Array of matching segment records with fields such as
            `id`, `col`, `start_idx`, `end_idx`, `status`, and `similarity`.

            Has the `vectorbtpro.generic.enums.pattern_range_dt` dtype.
    """
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    if max_records is None:
        records_out = np.empty(arr.shape[0], dtype=pattern_range_dt)
    else:
        records_out = np.empty(max_records, dtype=pattern_range_dt)
    r = 0
    min_max_required = False
    if rescale_mode == RescaleMode.MinMax:
        min_max_required = True
    if not np.isnan(min_pct_change):
        min_max_required = True
    if not np.isnan(max_pct_change):
        min_max_required = True
    if not max_error_as_maxdist:
        min_max_required = True
    if min_max_required:
        if np.isnan(pmin):
            pmin = np.nanmin(pattern)
        if np.isnan(pmax):
            pmax = np.nanmax(pattern)

    for i in range(arr.shape[0]):
        if roll_forward:
            from_i = i
            to_i = i + window
            if to_i > arr.shape[0]:
                break
        else:
            from_i = i - window + 1
            to_i = i + 1
            if from_i < 0:
                continue

        if np.random.uniform(0, 1) < row_select_prob:
            _vmin = vmin
            _vmax = vmax
            if min_max_required:
                if np.isnan(_vmin) or np.isnan(_vmax):
                    for j in range(from_i, to_i):
                        if np.isnan(_vmin) or arr[j] < _vmin:
                            _vmin = arr[j]
                        if np.isnan(_vmax) or arr[j] > _vmax:
                            _vmax = arr[j]

            for w in range(window, max_window + 1):
                if roll_forward:
                    from_i = i
                    to_i = i + w
                    if to_i > arr.shape[0]:
                        break
                    if min_max_required:
                        if w > window:
                            if arr[to_i - 1] < _vmin:
                                _vmin = arr[to_i - 1]
                            if arr[to_i - 1] > _vmax:
                                _vmax = arr[to_i - 1]
                else:
                    from_i = i - w + 1
                    to_i = i + 1
                    if from_i < 0:
                        continue
                    if min_max_required:
                        if w > window:
                            if arr[from_i] < _vmin:
                                _vmin = arr[from_i]
                            if arr[from_i] > _vmax:
                                _vmax = arr[from_i]

                if np.random.uniform(0, 1) < window_select_prob:
                    arr_window = arr[from_i:to_i]
                    similarity = pattern_similarity_nb(
                        arr_window,
                        pattern,
                        interp_mode=interp_mode,
                        rescale_mode=rescale_mode,
                        vmin=_vmin,
                        vmax=_vmax,
                        pmin=pmin,
                        pmax=pmax,
                        invert=invert,
                        error_type=error_type,
                        distance_measure=distance_measure,
                        max_error=max_error_,
                        max_error_interp_mode=max_error_interp_mode,
                        max_error_as_maxdist=max_error_as_maxdist,
                        max_error_strict=max_error_strict,
                        min_pct_change=min_pct_change,
                        max_pct_change=max_pct_change,
                        min_similarity=min_similarity,
                        minp=minp,
                    )
                    if not np.isnan(similarity):
                        skip = False
                        while True:
                            if r > 0:
                                if roll_forward:
                                    prev_same_row = records_out["start_idx"][r - 1] == from_i
                                else:
                                    prev_same_row = records_out["end_idx"][r - 1] == to_i
                                if overlap_mode != OverlapMode.AllowAll and prev_same_row:
                                    if similarity > records_out["similarity"][r - 1]:
                                        r -= 1
                                        continue
                                    else:
                                        skip = True
                                        break
                                elif overlap_mode >= 0:
                                    overlap = records_out["end_idx"][r - 1] - from_i
                                    if overlap > overlap_mode:
                                        if similarity > records_out["similarity"][r - 1]:
                                            r -= 1
                                            continue
                                        else:
                                            skip = True
                                            break
                            break
                        if skip:
                            continue
                        if r >= records_out.shape[0]:
                            raise IndexError(
                                "Records index out of range. Set a higher max_records."
                            )
                        records_out["id"][r] = r
                        records_out["col"][r] = col
                        records_out["start_idx"][r] = from_i
                        if to_i <= arr.shape[0] - 1:
                            records_out["end_idx"][r] = to_i
                            records_out["status"][r] = RangeStatus.Closed
                        else:
                            records_out["end_idx"][r] = arr.shape[0] - 1
                            records_out["status"][r] = RangeStatus.Open
                        records_out["similarity"][r] = similarity
                        r += 1

    return records_out[:r]


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        pattern=None,
        window=None,
        max_window=None,
        row_select_prob=None,
        window_select_prob=None,
        roll_forward=None,
        interp_mode=None,
        rescale_mode=None,
        vmin=None,
        vmax=None,
        pmin=None,
        pmax=None,
        invert=None,
        error_type=None,
        distance_measure=None,
        max_error=None,
        max_error_interp_mode=None,
        max_error_as_maxdist=None,
        max_error_strict=None,
        min_pct_change=None,
        max_pct_change=None,
        min_similarity=None,
        minp=None,
        overlap_mode=None,
        max_records=None,
    ),
    merge_func=records_ch.merge_records,
)
@register_jitted(cache=True, tags={"can_parallel"})
def find_pattern_nb(
    arr: tp.Array2d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
    roll_forward: bool = False,
    interp_mode: int = InterpMode.Mixed,
    rescale_mode: int = RescaleMode.MinMax,
    vmin: float = np.nan,
    vmax: float = np.nan,
    pmin: float = np.nan,
    pmax: float = np.nan,
    invert: bool = False,
    error_type: int = ErrorType.Absolute,
    distance_measure: int = DistanceMeasure.MAE,
    max_error: tp.FlexArray1dLike = np.nan,
    max_error_interp_mode: tp.Optional[int] = None,
    max_error_as_maxdist: bool = False,
    max_error_strict: bool = False,
    min_pct_change: float = np.nan,
    max_pct_change: float = np.nan,
    min_similarity: float = 0.85,
    minp: tp.Optional[int] = None,
    overlap_mode: int = OverlapMode.Disallow,
    max_records: tp.Optional[int] = None,
) -> tp.RecordArray:
    """Find pattern records in a 2D array column-wise.

    Args:
        arr (Array2d): Input 2D array in which pattern matching is performed per column.
        pattern (Array1d): 1D array representing the pattern to locate.

            Can be smaller or larger than the source array. In such cases,
            the smaller array is stretched using the interpolation mode specified by `interp_mode`.
        window (Optional[int]): Window size.

            If None, defaults to the length of `pattern`.
        max_window (Optional[int]): Maximum length of the rolling window for matching.

            If None, defaults to `window`.
        row_select_prob (float): Probability of selecting a row.
        window_select_prob (float): Probability of selecting a window size.
        roll_forward (bool): Process windows in forward direction if True; otherwise, in backward direction.
        interp_mode (int): Interpolation mode.

            See `vectorbtpro.generic.enums.InterpMode`.
        rescale_mode (int): Rescaling mode for adjusting the ranges of `arr` and `pattern`.

            See `vectorbtpro.generic.enums.RescaleMode`.
        vmin (float): Minimum value used for rescaling `arr`.

            Use only when the array has fixed bounds. Used in rescaling with `RescaleMode.MinMax`
            and for verifying `min_pct_change` and `max_pct_change`.

            If set to NaN, it is calculated dynamically.
        vmax (float): Maximum value used for rescaling `arr`.

            Use only when the array has fixed bounds. Used in rescaling with `RescaleMode.MinMax`
            and for verifying `min_pct_change` and `max_pct_change`.

            If set to NaN, it is calculated dynamically.
        pmin (float): Minimum value used for rescaling `pattern`.

            Used in rescaling with `RescaleMode.MinMax` and for computing the maximum distance
            at each point when `max_error_as_maxdist` is disabled.

            If set to NaN, it is calculated dynamically.
        pmax (float): Maximum value used for rescaling `pattern`.

            Used in rescaling with `RescaleMode.MinMax` and for computing the maximum distance
            at each point when `max_error_as_maxdist` is disabled.

            If set to NaN, it is calculated dynamically.
        invert (bool): Invert the pattern by reflecting its values.
        error_type (int): Error computation mode.

            See `vectorbtpro.generic.enums.ErrorType`.
        distance_measure (int): Method for measuring distance (e.g., MAE, MSE, RMSE).

            See `vectorbtpro.generic.enums.DistanceMeasure`.
        max_error (FlexArray1dLike): Maximum error threshold for normalization.

            Provided as a scalar or per element in the pattern.
        max_error_interp_mode (Optional[int]): Interpolation mode for `max_error`.

            If None, defaults to `interp_mode`.

            See `vectorbtpro.generic.enums.InterpMode`.
        max_error_as_maxdist (bool): Indicates whether `max_error` represents the maximum distance at each point.

            If False, exceeding `max_error` sets the distance to the maximum derived from
            `pmin`, `pmax`, and the pattern value at that point. If True and any point
            in a window is NaN, that point is skipped.
        max_error_strict (bool): If True, any instance of exceeding `max_error` results in a similarity of NaN.
        min_pct_change (float): Minimum percentage change required for a window to remain a search candidate.

            Window similarity is set to NaN if this threshold is not met.
        max_pct_change (float): Maximum percentage change allowed for a window to remain a search candidate.

            Window similarity is set to NaN if this threshold is exceeded.
        min_similarity (float): Minimum similarity threshold.

            If the computed similarity falls below this, returns NaN.
        minp (Optional[int]): Minimum number of observations required.
        overlap_mode (int): Mode for handling overlapping matches.

            See `vectorbtpro.generic.enums.OverlapMode`.
        max_records (Optional[int]): Maximum number of records to be filled.

            If None, defaults to the number of rows in `arr`.

    Returns:
        RecordArray: Array of records detailing the located pattern matches.

            Has the `vectorbtpro.generic.enums.pattern_range_dt` dtype.

    !!! tip
        This function is parallelizable.
    """
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    if max_records is None:
        records_out = np.empty((arr.shape[0], arr.shape[1]), dtype=pattern_range_dt)
    else:
        records_out = np.empty((max_records, arr.shape[1]), dtype=pattern_range_dt)
    record_counts = np.full(arr.shape[1], 0, dtype=int_)
    for col in prange(arr.shape[1]):
        records = find_pattern_1d_nb(
            arr[:, col],
            pattern,
            window=window,
            max_window=max_window,
            row_select_prob=row_select_prob,
            window_select_prob=window_select_prob,
            roll_forward=roll_forward,
            interp_mode=interp_mode,
            rescale_mode=rescale_mode,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            invert=invert,
            error_type=error_type,
            distance_measure=distance_measure,
            max_error=max_error_,
            max_error_interp_mode=max_error_interp_mode,
            max_error_as_maxdist=max_error_as_maxdist,
            max_error_strict=max_error_strict,
            min_pct_change=min_pct_change,
            max_pct_change=max_pct_change,
            min_similarity=min_similarity,
            minp=minp,
            overlap_mode=overlap_mode,
            max_records=max_records,
            col=col,
        )
        record_counts[col] = records.shape[0]
        records_out[: records.shape[0], col] = records
    return repartition_nb(records_out, record_counts)


# ############# Drawdowns ############# #


@register_jitted(cache=True)
def drawdown_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Compute the drawdown for a 1D array.

    Args:
        arr (Array1d): Input array of numerical values.

    Returns:
        Array1d: Array representing the drawdown, calculated as the relative decline
            from the maximum value encountered up to each index.
    """
    out = np.empty_like(arr, dtype=float_)
    max_val = np.nan
    for i in range(arr.shape[0]):
        if np.isnan(max_val) or arr[i] > max_val:
            max_val = arr[i]
        if np.isnan(max_val) or max_val == 0:
            out[i] = np.nan
        else:
            out[i] = arr[i] / max_val - 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def drawdown_nb(arr: tp.Array2d) -> tp.Array2d:
    """Compute the drawdown for a 2D array column-wise.

    Args:
        arr (Array2d): 2D input array where the drawdown is computed separately for each column.

    Returns:
        Array2d: Array of drawdown values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = drawdown_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def fill_drawdown_record_nb(
    new_records: tp.RecordArray2d,
    counts: tp.Array2d,
    i: int,
    col: int,
    start_idx: int,
    valley_idx: int,
    start_val: float,
    valley_val: float,
    end_val: float,
    status: int,
) -> None:
    """Populate a drawdown record with indices and values.

    Args:
        new_records (RecordArray2d): Array structure to store the drawdown record fields.

            Must adhere to the `vectorbtpro.generic.enums.drawdown_dt` dtype.
        counts (Array2d): Array tracking the current count of records per column.
        i (int): Current index indicating the end of the drawdown.
        col (int): Column index being processed.
        start_idx (int): Index at which the drawdown begins.
        valley_idx (int): Index corresponding to the trough of the drawdown.
        start_val (float): Value at the start of the drawdown.
        valley_val (float): Value at the valley (lowest point) of the drawdown.
        end_val (float): Value at the end of the drawdown.
        status (int): Code representing the status of the drawdown.

            See `vectorbtpro.generic.enums.RangeStatus`.

    Returns:
        None: This function modifies `new_records` in place.
    """
    r = counts[col]
    new_records["id"][r, col] = r
    new_records["col"][r, col] = col
    new_records["start_idx"][r, col] = start_idx
    new_records["valley_idx"][r, col] = valley_idx
    new_records["end_idx"][r, col] = i
    new_records["start_val"][r, col] = start_val
    new_records["valley_val"][r, col] = valley_val
    new_records["end_val"][r, col] = end_val
    new_records["status"][r, col] = status
    counts[col] += 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        open=ch.ArraySlicer(axis=1),
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_drawdowns_nb(
    open: tp.Optional[tp.Array2d],
    high: tp.Optional[tp.Array2d],
    low: tp.Optional[tp.Array2d],
    close: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.RecordArray:
    """Fill drawdown records by analyzing a time series.

    This function computes drawdown records using the provided price series arrays.
    Only `close` is mandatory; other arrays supply supplementary price information.

    Args:
        open (Optional[Array2d]): Open price time series for each asset column.
        high (Optional[Array2d]): High price time series for each asset column.
        low (Optional[Array2d]): Low price time series for each asset column.
        close (Array2d): Close price time series for each asset column.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        RecordArray: Array of computed drawdown records.

            Has the `vectorbtpro.generic.enums.drawdown_dt` dtype.

    !!! tip
        This function is parallelizable.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> close = np.array([
        ...     [1, 5, 1, 3],
        ...     [2, 4, 2, 2],
        ...     [3, 3, 3, 1],
        ...     [4, 2, 2, 2],
        ...     [5, 1, 1, 3]
        ... ])
        >>> records = vbt.nb.get_drawdowns_nb(None, None, None, close)

        >>> pd.DataFrame.from_records(records)
           id  col  start_idx  valley_idx  end_idx  start_val  valley_val  end_val  \\
        0   0    1          0           4        4        5.0         1.0      1.0
        1   0    2          2           4        4        3.0         1.0      1.0
        2   0    3          0           2        4        3.0         1.0      3.0

           status
        0       0
        1       0
        2       1
        ```
    """
    new_records = np.empty(close.shape, dtype=drawdown_dt)
    counts = np.full(close.shape[1], 0, dtype=int_)

    sim_start_, sim_end_ = prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )

    for col in prange(close.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        drawdown_started = False
        _close = close[0, col]
        if open is None:
            _open = np.nan
        else:
            _open = open[0, col]
        start_idx = 0
        valley_idx = 0
        start_val = _open
        valley_val = _open

        for i in range(_sim_start, _sim_end):
            _close = close[i, col]
            if open is None:
                _open = np.nan
            else:
                _open = open[i, col]
            if high is None:
                _high = np.nan
            else:
                _high = high[i, col]
            if low is None:
                _low = np.nan
            else:
                _low = low[i, col]
            if np.isnan(_high):
                if np.isnan(_open):
                    _high = _close
                elif np.isnan(_close):
                    _high = _open
                else:
                    _high = max(_open, _close)
            if np.isnan(_low):
                if np.isnan(_open):
                    _low = _close
                elif np.isnan(_close):
                    _low = _open
                else:
                    _low = min(_open, _close)

            if drawdown_started:
                if _open >= start_val:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        start_idx=start_idx,
                        valley_idx=valley_idx,
                        start_val=start_val,
                        valley_val=valley_val,
                        end_val=_open,
                        status=DrawdownStatus.Recovered,
                    )
                    start_idx = i
                    valley_idx = i
                    start_val = _open
                    valley_val = _open

            if drawdown_started:
                if _low < valley_val:
                    valley_idx = i
                    valley_val = _low
                if _high >= start_val:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        start_idx=start_idx,
                        valley_idx=valley_idx,
                        start_val=start_val,
                        valley_val=valley_val,
                        end_val=_high,
                        status=DrawdownStatus.Recovered,
                    )
                    start_idx = i
                    valley_idx = i
                    start_val = _high
                    valley_val = _high
            else:
                if np.isnan(start_val) or _high >= start_val:
                    start_idx = i
                    valley_idx = i
                    start_val = _high
                    valley_val = _high
                elif _low < valley_val:
                    if not np.isnan(valley_val):
                        drawdown_started = True
                    valley_idx = i
                    valley_val = _low

            if drawdown_started:
                if i == _sim_end - 1:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        start_idx=start_idx,
                        valley_idx=valley_idx,
                        start_val=start_val,
                        valley_val=valley_val,
                        end_val=_close,
                        status=DrawdownStatus.Active,
                    )

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_val_arr", axis=0),
    arg_take_spec=dict(start_val_arr=ch.ArraySlicer(axis=0), valley_val_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_drawdown_nb(start_val_arr: tp.Array1d, valley_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute drawdown metrics for each record.

    Args:
        start_val_arr (Array1d): Array of starting values from each drawdown record.
        valley_val_arr (Array1d): Array of valley values from each drawdown record.

    Returns:
        Array1d: Array of drawdown percentages for each record.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(valley_val_arr.shape[0], dtype=float_)
    for r in prange(valley_val_arr.shape[0]):
        if start_val_arr[r] == 0:
            out[r] = np.nan
        else:
            out[r] = (valley_val_arr[r] - start_val_arr[r]) / start_val_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(start_idx_arr=ch.ArraySlicer(axis=0), valley_idx_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_decline_duration_nb(start_idx_arr: tp.Array1d, valley_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the drawdown decline phase.

    Args:
        start_idx_arr (Array1d): Array of start row indices marking the peak of each drawdown.
        valley_idx_arr (Array1d): Array of valley row indices where each drawdown reaches its minimum.

    Returns:
        Array1d: Array of durations representing the time from peak to valley in each record.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(valley_idx_arr.shape[0], dtype=float_)
    for r in prange(valley_idx_arr.shape[0]):
        out[r] = valley_idx_arr[r] - start_idx_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="valley_idx_arr", axis=0),
    arg_take_spec=dict(valley_idx_arr=ch.ArraySlicer(axis=0), end_idx_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_duration_nb(valley_idx_arr: tp.Array1d, end_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the drawdown recovery phase.

    Args:
        valley_idx_arr (Array1d): Array of valley row indices for each drawdown record.
        end_idx_arr (Array1d): Array of recovery row indices marking the end of each drawdown.

    Returns:
        Array1d: Array of durations representing the time from the drawdown valley to recovery.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(end_idx_arr.shape[0], dtype=float_)
    for r in prange(end_idx_arr.shape[0]):
        out[r] = end_idx_arr[r] - valley_idx_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        valley_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_duration_ratio_nb(
    start_idx_arr: tp.Array1d,
    valley_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
) -> tp.Array1d:
    """Compute the recovery duration ratio for each drawdown record.

    This function computes the ratio between the recovery duration and the decline duration.

    Args:
        start_idx_arr (Array1d): Array of starting row indices for the decline period.
        valley_idx_arr (Array1d): Array of row indices where the drawdown reaches its lowest point.
        end_idx_arr (Array1d): Array of ending row indices for the recovery period.

    Returns:
        Array1d: Array of recovery duration ratios calculated as (recovery duration)
            divided by (decline duration).

            NaN is returned when the decline duration is zero.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(start_idx_arr.shape[0], dtype=float_)
    for r in prange(start_idx_arr.shape[0]):
        if valley_idx_arr[r] - start_idx_arr[r] == 0:
            out[r] = np.nan
        else:
            out[r] = (end_idx_arr[r] - valley_idx_arr[r]) / (valley_idx_arr[r] - start_idx_arr[r])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="valley_val_arr", axis=0),
    arg_take_spec=dict(valley_val_arr=ch.ArraySlicer(axis=0), end_val_arr=ch.ArraySlicer(axis=0)),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_return_nb(valley_val_arr: tp.Array1d, end_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the recovery return for each drawdown record.

    This function calculates the recovery return as the relative change from
    the valley value to the end value.

    Args:
        valley_val_arr (Array1d): Array of drawdown valley values.
        end_val_arr (Array1d): Array of values at the end of the recovery period.

    Returns:
        Array1d: Array of recovery returns computed as (end value - valley value)
            divided by the valley value.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty(end_val_arr.shape[0], dtype=float_)
    for r in prange(end_val_arr.shape[0]):
        if valley_val_arr[r] == 0:
            out[r] = np.nan
        else:
            out[r] = (end_val_arr[r] - valley_val_arr[r]) / valley_val_arr[r]
    return out


@register_jitted(cache=True)
def bar_price_nb(records: tp.RecordArray, price: tp.Optional[tp.FlexArray2d]) -> tp.Array1d:
    """Return the bar prices corresponding to each record.

    This function retrieves the bar price for each record using a flex selection from the price array.
    If the price array is not provided, NaN is returned for the corresponding record.

    Args:
        records (RecordArray): Array of records.
        price (Optional[FlexArray2d]): Two-dimensional array of bar prices to select
            from using record indices and columns.

    Returns:
        Array1d: Array of bar prices for each record, with NaN for records
            when the price array is None.
    """
    out = np.empty(len(records), dtype=float_)
    for i in range(len(records)):
        record = records[i]
        if price is not None:
            out[i] = float(flex_select_nb(price, record["idx"], record["col"]))
        else:
            out[i] = np.nan
    return out
