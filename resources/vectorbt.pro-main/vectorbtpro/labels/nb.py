# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for generating labels.

!!! note
    Set `wait` to 1 to exclude the current value from future value calculations.

!!! warning
    Do not use these functions for building predictor variables as they may introduce look-ahead bias.
    Use them only for constructing target variables.
"""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_nb, flex_select_col_nb
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.indicators.enums import Pivot
from vectorbtpro.labels.enums import TrendLabelMode
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


# ############# FMEAN ############# #


@register_jitted(cache=True)
def future_mean_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    wait: int = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Calculate rolling average over future values.

    Args:
        close (Array1d): 1-D array of input values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        wait (int): Number of periods to delay the result to exclude the current value.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array containing the computed rolling averages over future values.
    """
    future_mean = generic_nb.ma_1d_nb(close[::-1], window, wtype=wtype, minp=minp, adjust=adjust)[
        ::-1
    ]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_mean, wait)
    return future_mean


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_mean_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """Calculate rolling average over future values for each column.

    Args:
        close (Array2d): 2-D array of input values.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        wait (FlexArray1dLike): Waiting period.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: 2-D array of computed rolling averages over future values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_mean = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        future_mean[:, col] = future_mean_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
            adjust=adjust,
        )
    return future_mean


# ############# FSTD ############# #


@register_jitted(cache=True)
def future_std_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    wait: int = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array1d:
    """Calculate rolling standard deviation over future values.

    Args:
        close (Array1d): 1-D array of input values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        wait (int): Number of periods to delay the result to exclude the current value.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Array containing the computed rolling standard deviations over future values.
    """
    future_std = generic_nb.msd_1d_nb(
        close[::-1], window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof
    )[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_std, wait)
    return future_std


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
        ddof=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_std_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array2d:
    """Calculate rolling standard deviation over future values for each column.

    Args:
        close (Array2d): 2-D array of input values.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        wait (FlexArray1dLike): Waiting period.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array2d: 2-D array containing the computed rolling standard deviations
            over future values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_std = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        future_std[:, col] = future_std_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
            adjust=adjust,
            ddof=ddof,
        )
    return future_std


# ############# FMIN ############# #


@register_jitted(cache=True)
def future_min_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wait: int = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array1d:
    """Calculate rolling minimum over future values.

    Args:
        close (Array1d): 1-D array of input values.
        window (int): Window size.
        wait (int): Number of periods to delay the result to exclude the current value.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array with the computed rolling minimum values over future values.
    """
    future_min = generic_nb.rolling_min_1d_nb(close[::-1], window, minp=minp)[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_min, wait)
    return future_min


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_min_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array2d:
    """Calculate rolling minimum over future values for each column.

    Args:
        close (Array2d): 2-D array of input values.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wait (FlexArray1dLike): Waiting period.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: 2-D array containing the computed rolling minimum values over future values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_min = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        future_min[:, col] = future_min_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
        )
    return future_min


# ############# FMAX ############# #


@register_jitted(cache=True)
def future_max_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wait: int = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array1d:
    """Calculate rolling maximum over future values.

    Args:
        close (Array1d): 1-D array of input values.
        window (int): Window size.
        wait (int): Number of periods to delay the result to exclude the current value.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array with the computed rolling maximum values over future values.
    """
    future_max = generic_nb.rolling_max_1d_nb(close[::-1], window, minp=minp)[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_max, wait)
    return future_max


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_max_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array2d:
    """Return a 2-dim array of future maximum values computed for each column of close prices.

    Args:
        close (Array2d): Two-dimensional array containing close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wait (FlexArray1dLike): Number of periods to wait before starting computation.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Two-dimensional array where each column contains the computed future maximum values.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_max = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        future_max[:, col] = future_max_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
        )
    return future_max


# ############# FIXLB ############# #


@register_jitted(cache=True)
def fixed_labels_1d_nb(
    close: tp.Array1d,
    n: int = 1,
) -> tp.Array1d:
    """Return the percentage change from the current value to the future value shifted by n periods.

    Args:
        close (Array1d): One-dimensional array of close prices.
        n (int): Period offset used for shifting to compute the future value.

    Returns:
        Array1d: Array of percentage changes.
    """
    return (generic_nb.bshift_1d_nb(close, n) - close) / close


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        n=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def fixed_labels_nb(
    close: tp.Array2d,
    n: tp.FlexArray1dLike = 1,
) -> tp.Array2d:
    """Return a 2-dim array of percentage changes computed for each column from the current to the future value.

    Args:
        close (Array2d): Two-dimensional array of close prices.
        n (FlexArray1dLike): Period offset for computing future values.

            Provided as a scalar or per column.

    Returns:
        Array2d: Two-dimensional array where each column contains the computed percentage changes.

    !!! tip
        This function is parallelizable.
    """
    n_ = to_1d_array_nb(np.asarray(n))

    fixed_labels = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        fixed_labels[:, col] = fixed_labels_1d_nb(
            close=close[:, col],
            n=flex_select_1d_nb(n_, col),
        )
    return fixed_labels


# ############# MEANLB ############# #


@register_jitted(cache=True)
def mean_labels_1d_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Return the percentage change of the current value relative to the average of future periods.

    Args:
        close (Array2d): Array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        wait (FlexArray1dLike): Number of periods to wait before applying the window.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array of computed percentage changes.
    """
    future_mean = future_mean_1d_nb(
        close, window=window, wtype=wtype, wait=wait, minp=minp, adjust=adjust
    )
    return (future_mean - close) / close


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mean_labels_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """Return a 2-dim array of percentage changes computed for each column,
    where each value is derived by comparing the current price to the average of future periods.

    Args:
        close (Array2d): Two-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        wait (FlexArray1dLike): Number of periods to wait before applying the window.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: Two-dimensional array where each column contains the computed percentage changes.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    wait_ = to_1d_array_nb(np.asarray(wait))

    mean_labels = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        mean_labels[:, col] = mean_labels_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
            adjust=adjust,
        )
    return mean_labels


# ############# PIVOTLB ############# #


@register_jitted(cache=True)
def iter_symmetric_up_th_nb(down_th: float) -> float:
    """Return the symmetric positive threshold corresponding to a given negative
    threshold for a single iteration.

    Args:
        down_th (float): Negative threshold value.

    Returns:
        float: Calculated positive threshold.

            For example, a 50% drop requires a 100% increase to return to the initial level.
    """
    return down_th / (1 - down_th)


@register_jitted(cache=True)
def iter_symmetric_down_th_nb(up_th: float) -> float:
    """Return the symmetric negative threshold corresponding to a given positive
    threshold for a single iteration.

    Args:
        up_th (float): Positive threshold value.

    Returns:
        float: Calculated negative threshold.
    """
    return up_th / (1 + up_th)


@register_jitted(cache=True)
def pivots_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
) -> tp.Array1d:
    """Return an array indicating pivot points in a price series,
    where 1 represents a peak, -1 represents a valley, and 0 denotes no pivot.

    Args:
        high (Array1d): One-dimensional array of high prices.
        low (Array1d): One-dimensional array of low prices.
        up_th (FlexArray1dLike): Upper threshold for detecting peaks.

            Provided as a scalar or per row.
        down_th (FlexArray1dLike): Lower threshold for detecting valleys.

            Provided as a scalar or per row.

    Returns:
        Array1d: Array of integer values indicating the detected pivot points.

            See `vectorbtpro.indicators.enums.Pivot`.

    !!! note
        Two adjacent peak and valley points must surpass the respective threshold values.
        If thresholds are specified element-wise, each new or updated pivot applies its corresponding threshold.
    """
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    pivots = np.full(high.shape, 0, dtype=int_)

    last_pivot = 0
    last_i = -1
    last_value = np.nan
    first_valid_i = -1
    for i in range(high.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            if first_valid_i == -1:
                first_valid_i = 0
            if last_i == -1:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, first_valid_i))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, first_valid_i))
                if not np.isnan(_up_th) and high[i] >= low[first_valid_i] * _up_th:
                    if not np.isnan(_down_th) and low[i] <= high[first_valid_i] * _down_th:
                        pass  # wait
                    else:
                        pivots[first_valid_i] = Pivot.Valley
                        last_i = i
                        last_value = high[i]
                        last_pivot = Pivot.Peak
                if not np.isnan(_down_th) and low[i] <= high[first_valid_i] * _down_th:
                    if not np.isnan(_up_th) and high[i] >= low[first_valid_i] * _up_th:
                        pass  # wait
                    else:
                        pivots[first_valid_i] = Pivot.Peak
                        last_i = i
                        last_value = low[i]
                        last_pivot = Pivot.Valley
            else:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, last_i))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, last_i))
                if last_pivot == Pivot.Valley:
                    if (
                        not np.isnan(last_value)
                        and not np.isnan(_up_th)
                        and high[i] >= last_value * _up_th
                    ):
                        pivots[last_i] = last_pivot
                        last_i = i
                        last_value = high[i]
                        last_pivot = Pivot.Peak
                    elif np.isnan(last_value) or low[i] < last_value:
                        last_i = i
                        last_value = low[i]
                elif last_pivot == Pivot.Peak:
                    if (
                        not np.isnan(last_value)
                        and not np.isnan(_down_th)
                        and low[i] <= last_value * _down_th
                    ):
                        pivots[last_i] = last_pivot
                        last_i = i
                        last_value = low[i]
                        last_pivot = Pivot.Valley
                    elif np.isnan(last_value) or high[i] > last_value:
                        last_i = i
                        last_value = high[i]

        if last_i != -1 and i == high.shape[0] - 1:
            pivots[last_i] = last_pivot

    return pivots


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivots_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
) -> tp.Array2d:
    """Generate pivot labels column-wise for two-dimensional input arrays.

    Args:
        high (Array2d): Two-dimensional array of high values.
        low (Array2d): Two-dimensional array of low values.
        up_th (FlexArray2dLike): Upper threshold for detecting peaks.

            Provided as a scalar, or per row, column, or element.
        down_th (FlexArray2dLike): Lower threshold for detecting valleys.

            Provided as a scalar, or per row, column, or element.

    Returns:
        Array2d: Two-dimensional integer array containing pivot labels.

    !!! tip
        This function is parallelizable.
    """
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))

    pivots = np.empty(high.shape, dtype=int_)
    for col in prange(high.shape[1]):
        pivots[:, col] = pivots_1d_nb(
            high[:, col],
            low[:, col],
            flex_select_col_nb(up_th_, col),
            flex_select_col_nb(down_th_, col),
        )
    return pivots


# ############# TRENDLB ############# #


@register_jitted(cache=True)
def bin_trend_labels_1d_nb(pivots: tp.Array1d) -> tp.Array1d:
    """Classify trend labels for a one-dimensional array of pivots.

    Args:
        pivots (Array1d): One-dimensional array of pivot indicators.

    Returns:
        Array1d: One-dimensional array with trend labels where 0 indicates downtrend and 1 indicates uptrend.
    """
    bin_trend_labels = np.full(pivots.shape, np.nan, dtype=float_)
    idxs = np.flatnonzero(pivots)
    if idxs.shape[0] == 0:
        return bin_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]

        for i in range(prev_i, next_i):
            if pivots[next_i] == Pivot.Peak:
                bin_trend_labels[i] = 1
            else:
                bin_trend_labels[i] = 0

    return bin_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        pivots=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bin_trend_labels_nb(pivots: tp.Array2d) -> tp.Array2d:
    """Generate trend labels for two-dimensional pivot arrays by applying
    one-dimensional classification column‐wise.

    Args:
        pivots (Array2d): Two-dimensional array of pivot indicators.

    Returns:
        Array2d: Two-dimensional array where each column contains trend labels
            (0 for downtrend and 1 for uptrend).

    !!! tip
        This function is parallelizable.
    """
    bin_trend_labels = np.empty(pivots.shape, dtype=float_)
    for col in prange(pivots.shape[1]):
        bin_trend_labels[:, col] = bin_trend_labels_1d_nb(pivots[:, col])
    return bin_trend_labels


@register_jitted(cache=True)
def binc_trend_labels_1d_nb(high: tp.Array1d, low: tp.Array1d, pivots: tp.Array1d) -> tp.Array1d:
    """Compute normalized trend labels for one-dimensional arrays based on median values.

    Args:
        high (Array1d): One-dimensional array of high values.
        low (Array1d): One-dimensional array of low values.
        pivots (Array1d): One-dimensional array of pivot indicators.

    Returns:
        Array1d: One-dimensional array of trend labels normalized between 0 (downtrend) and 1 (uptrend).
    """
    binc_trend_labels = np.full(pivots.shape, np.nan, dtype=float_)
    idxs = np.flatnonzero(pivots[:])
    if idxs.shape[0] == 0:
        return binc_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]
        _min = np.nanmin(low[prev_i : next_i + 1])
        _max = np.nanmax(high[prev_i : next_i + 1])

        for i in range(prev_i, next_i):
            _med = (high[i] + low[i]) / 2
            binc_trend_labels[i] = 1 - (_med - _min) / (_max - _min)

    return binc_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        pivots=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def binc_trend_labels_nb(high: tp.Array2d, low: tp.Array2d, pivots: tp.Array2d) -> tp.Array2d:
    """Apply one-dimensional median-based normalization for trend labels to two-dimensional arrays column‐wise.

    Args:
        high (Array2d): Two-dimensional array of high values.
        low (Array2d): Two-dimensional array of low values.
        pivots (Array2d): Two-dimensional array of pivot indicators.

    Returns:
        Array2d: Two-dimensional array of normalized trend labels.

    !!! tip
        This function is parallelizable.
    """
    binc_trend_labels = np.empty(pivots.shape, dtype=float_)
    for col in prange(pivots.shape[1]):
        binc_trend_labels[:, col] = binc_trend_labels_1d_nb(
            high[:, col], low[:, col], pivots[:, col]
        )
    return binc_trend_labels


@register_jitted(cache=True)
def bincs_trend_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    pivots: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
) -> tp.Array1d:
    """Compute capped trend labels for one-dimensional arrays by normalizing
    median values with applied thresholds.

    Args:
        high (Array1d): One-dimensional array of high values.
        low (Array1d): One-dimensional array of low values.
        pivots (Array1d): One-dimensional array of pivot indicators.
        up_th (FlexArray1dLike): Upper threshold for detecting peaks.

            Provided as a scalar or per row.
        down_th (FlexArray1dLike): Lower threshold for detecting valleys.

            Provided as a scalar or per row.

    Returns:
        Array1d: One-dimensional array of capped trend labels normalized
            between 0 (downtrend) and 1 (uptrend).
    """
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    bincs_trend_labels = np.full(pivots.shape, np.nan, dtype=float_)
    idxs = np.flatnonzero(pivots)
    if idxs.shape[0] == 0:
        return bincs_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]
        _up_th = 1 + abs(flex_select_1d_nb(up_th_, prev_i))
        _down_th = 1 - abs(flex_select_1d_nb(down_th_, prev_i))
        _min = np.min(low[prev_i : next_i + 1])
        _max = np.max(high[prev_i : next_i + 1])

        for i in range(prev_i, next_i):
            if not np.isnan(high[i]) and not np.isnan(low[i]):
                _med = (high[i] + low[i]) / 2
                if pivots[next_i] == Pivot.Peak:
                    if not np.isnan(_up_th):
                        _start = _max / _up_th
                        _end = _min * _up_th
                        if _max >= _end and _med <= _start:
                            bincs_trend_labels[i] = 1
                        else:
                            bincs_trend_labels[i] = 1 - (_med - _start) / (_max - _start)
                else:
                    if not np.isnan(_down_th):
                        _start = _min / _down_th
                        _end = _max * _down_th
                        if _min <= _end and _med >= _start:
                            bincs_trend_labels[i] = 0
                        else:
                            bincs_trend_labels[i] = 1 - (_med - _min) / (_start - _min)

    return bincs_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        pivots=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bincs_trend_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    pivots: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
) -> tp.Array2d:
    """Compute capped trend labels for two-dimensional arrays by applying
    one-dimensional capped normalization column‐wise.

    Args:
        high (Array2d): Two-dimensional array of high values.
        low (Array2d): Two-dimensional array of low values.
        pivots (Array2d): Two-dimensional array of pivot signals.
        up_th (FlexArray2dLike): Upper threshold for detecting peaks.

            Provided as a scalar, or per row, column, or element.
        down_th (FlexArray2dLike): Lower threshold for detecting valleys.

            Provided as a scalar, or per row, column, or element.

    Returns:
        Array2d: Two-dimensional array of capped trend labels normalized between
            0 (downtrend) and 1 (uptrend).

    !!! tip
        This function is parallelizable.
    """
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))

    bincs_trend_labels = np.empty(pivots.shape, dtype=float_)
    for col in prange(pivots.shape[1]):
        bincs_trend_labels[:, col] = bincs_trend_labels_1d_nb(
            high[:, col],
            low[:, col],
            pivots[:, col],
            flex_select_col_nb(up_th_, col),
            flex_select_col_nb(down_th_, col),
        )
    return bincs_trend_labels


@register_jitted(cache=True)
def pct_trend_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    pivots: tp.Array1d,
    normalize: bool = False,
) -> tp.Array1d:
    """Compute percentage change of median prices relative to the subsequent pivot.

    Args:
        high (Array1d): 1D array of high prices.
        low (Array1d): 1D array of low prices.
        pivots (Array1d): 1D array marking pivot events.
        normalize (bool): Flag to determine if the percentage change is normalized.

    Returns:
        Array1d: Array of percentage changes with non-pivot positions set to NaN.
    """
    pct_trend_labels = np.full(pivots.shape, np.nan, dtype=float_)
    idxs = np.flatnonzero(pivots)
    if idxs.shape[0] == 0:
        return pct_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]

        for i in range(prev_i, next_i):
            _med = (high[i] + low[i]) / 2
            if pivots[next_i] == Pivot.Peak:
                if normalize:
                    pct_trend_labels[i] = (high[next_i] - _med) / high[next_i]
                else:
                    pct_trend_labels[i] = (high[next_i] - _med) / _med
            else:
                if normalize:
                    pct_trend_labels[i] = (low[next_i] - _med) / _med
                else:
                    pct_trend_labels[i] = (low[next_i] - _med) / low[next_i]

    return pct_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        pivots=ch.ArraySlicer(axis=1),
        normalize=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pct_trend_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    pivots: tp.Array2d,
    normalize: bool = False,
) -> tp.Array2d:
    """Compute 2D percentage change of median prices relative to the subsequent pivot for each column.

    Args:
        high (Array2d): 2D array of high prices.
        low (Array2d): 2D array of low prices.
        pivots (Array2d): 2D array indicating pivot events.
        normalize (bool): Flag to determine if the percentage change is normalized.

    Returns:
        Array2d: 2D array of percentage changes.

    !!! tip
        This function is parallelizable.
    """
    pct_trend_labels = np.empty(pivots.shape, dtype=float_)
    for col in prange(pivots.shape[1]):
        pct_trend_labels[:, col] = pct_trend_labels_1d_nb(
            high[:, col],
            low[:, col],
            pivots[:, col],
            normalize=normalize,
        )
    return pct_trend_labels


@register_jitted(cache=True)
def trend_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
    mode: int = TrendLabelMode.Binary,
) -> tp.Array2d:
    """Generate trend labels for a 1D price series based on specified thresholds and mode.

    Args:
        high (Array1d): 1D array of high prices.
        low (Array1d): 1D array of low prices.
        up_th (FlexArray1dLike): Upper threshold for detecting peaks.

            Provided as a scalar or per row.
        down_th (FlexArray1dLike): Lower threshold for detecting valleys.

            Provided as a scalar or per row.
        mode (int): Trend mode.

            See `vectorbtpro.labels.enums.TrendLabelMode`.

    Returns:
        Array2d: 2D array of trend labels determined by the specified mode.
    """
    pivots = pivots_1d_nb(high, low, up_th, down_th)
    if mode == TrendLabelMode.Binary:
        return bin_trend_labels_1d_nb(pivots)
    if mode == TrendLabelMode.BinaryCont:
        return binc_trend_labels_1d_nb(high, low, pivots)
    if mode == TrendLabelMode.BinaryContSat:
        return bincs_trend_labels_1d_nb(high, low, pivots, up_th, down_th)
    if mode == TrendLabelMode.PctChange:
        return pct_trend_labels_1d_nb(high, low, pivots, normalize=False)
    if mode == TrendLabelMode.PctChangeNorm:
        return pct_trend_labels_1d_nb(high, low, pivots, normalize=True)
    raise ValueError("Invalid trend mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
        mode=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def trend_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
    mode: tp.FlexArray1dLike = TrendLabelMode.Binary,
) -> tp.Array2d:
    """Compute 2D trend labels for each column from high and low prices using specified thresholds and mode.

    Args:
        high (Array2d): 2D array of high prices.
        low (Array2d): 2D array of low prices.
        up_th (FlexArray2dLike): Upper threshold for detecting peaks.

            Provided as a scalar, or per row, column, or element.
        down_th (FlexArray2dLike): Lower threshold for detecting valleys.

            Provided as a scalar, or per row, column, or element.
        mode (FlexArray1dLike): Trend mode.

            Provided as a scalar or per column.

            See `vectorbtpro.labels.enums.TrendLabelMode`.

    Returns:
        Array2d: 2D array of computed trend labels.

    !!! tip
        This function is parallelizable.
    """
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))
    mode_ = to_1d_array_nb(np.asarray(mode))

    trend_labels = np.empty(high.shape, dtype=float_)
    for col in prange(high.shape[1]):
        trend_labels[:, col] = trend_labels_1d_nb(
            high[:, col],
            low[:, col],
            flex_select_col_nb(up_th_, col),
            flex_select_col_nb(down_th_, col),
            mode=flex_select_1d_nb(mode_, col),
        )
    return trend_labels


# ############# BOLB ############# #


@register_jitted(cache=True)
def breakout_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    window: int = 14,
    up_th: tp.FlexArray1dLike = np.inf,
    down_th: tp.FlexArray1dLike = np.inf,
    wait: int = 1,
) -> tp.Array1d:
    """Compute breakout labels for a 1D price series based on future price comparisons.

    For each index in the input series, the function searches forward over a specified window:

    * Returns 1 if a future high price exceeds the calculated positive breakout threshold.
    * Returns -1 if a future low price falls below the calculated negative breakout threshold.
    * Returns 0 if no breakout is identified or if both thresholds are met simultaneously.

    Args:
        high (Array1d): 1D array of high prices.
        low (Array1d): 1D array of low prices.
        window (int): Window size.
        up_th (FlexArray1dLike): Upper threshold for detecting peaks.

            Provided as a scalar or per row.
        down_th (FlexArray1dLike): Lower threshold for detecting valleys.

            Provided as a scalar or per row.
        wait (int): Number of periods to delay before starting the breakout search.

    Returns:
        Array1d: Array of breakout labels with values 1, -1, or 0.
    """
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    breakout_labels = np.full(high.shape, 0, dtype=float_)
    for i in range(high.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            _up_th = 1 + abs(flex_select_1d_nb(up_th_, i))
            _down_th = 1 - abs(flex_select_1d_nb(down_th_, i))

            for j in range(i + wait, min(i + window + wait, high.shape[0])):
                if not np.isnan(high[j]) and not np.isnan(low[j]):
                    if not np.isnan(_up_th) and high[j] >= low[i] * _up_th:
                        breakout_labels[i] = 1
                        break
                    if not np.isnan(_down_th) and low[j] <= high[i] * _down_th:
                        if breakout_labels[i] == 1:
                            breakout_labels[i] = 0
                            continue
                        breakout_labels[i] = -1
                        break

    return breakout_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
        wait=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def breakout_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    up_th: tp.FlexArray2dLike = np.inf,
    down_th: tp.FlexArray2dLike = np.inf,
    wait: tp.FlexArray1dLike = 1,
) -> tp.Array2d:
    """Compute 2D breakout labels for each column based on future price comparisons.

    Args:
        high (Array2d): 2D array of high prices.
        low (Array2d): 2D array of low prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        up_th (FlexArray2dLike): Upper threshold for detecting peaks.

            Provided as a scalar, or per row, column, or element.
        down_th (FlexArray2dLike): Lower threshold for detecting valleys.

            Provided as a scalar, or per row, column, or element.
        wait (FlexArray1dLike): Number of periods to delay before starting the breakout search.

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array of breakout labels.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))
    wait_ = to_1d_array_nb(np.asarray(wait))

    breakout_labels = np.empty(high.shape, dtype=float_)
    for col in prange(high.shape[1]):
        breakout_labels[:, col] = breakout_labels_1d_nb(
            high[:, col],
            low[:, col],
            window=flex_select_1d_nb(window_, col),
            up_th=flex_select_col_nb(up_th_, col),
            down_th=flex_select_col_nb(down_th_, col),
            wait=flex_select_1d_nb(wait_, col),
        )
    return breakout_labels
