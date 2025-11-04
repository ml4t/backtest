# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for rolling and expanding windows."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.nb.base import rank_1d_nb
from vectorbtpro.generic.nb.patterns import pattern_similarity_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

# ############# Rolling functions ############# #


@register_jitted(cache=True)
def rolling_sum_acc_nb(in_state: RollSumAIS) -> RollSumAOS:
    """Accumulate the rolling sum state for one iteration of `rolling_sum_1d_nb`.

    Args:
        in_state (RollSumAIS): Current rolling sum accumulator input state
            of type `vectorbtpro.generic.enums.RollSumAOS`.

    Returns:
        RollSumAOS: Updated rolling sum accumulator output state
            of type `vectorbtpro.generic.enums.RollSumAOS`.
    """
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumsum = cumsum
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
        window_cumsum = cumsum
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumsum

    return RollSumAOS(cumsum=cumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_sum_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling sum for a one-dimensional array.

    Uses `rolling_sum_acc_nb` to update the accumulation state for each iteration,
    emulating the behavior of `pd.Series(arr).rolling(window, min_periods=minp).sum()`.

    Args:
        arr (Array1d): Array of numerical values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array containing the rolling sum values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollSumAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_sum_acc_nb(in_state)
        cumsum = out_state.cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_sum_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """Apply a rolling sum computation column-wise to a two-dimensional array.

    Uses `rolling_sum_1d_nb` to compute the rolling sum for each column.

    Args:
        arr (Array2d): Two-dimensional array of numerical values, with each column processed independently.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Two-dimensional array with the rolling sum computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_sum_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_prod_acc_nb(in_state: RollProdAIS) -> RollProdAOS:
    """Accumulate the rolling product state for one iteration of `rolling_prod_1d_nb`.

    Args:
        in_state (RollProdAIS): Current rolling product accumulator input state
            of type `vectorbtpro.generic.enums.RollProdAIS`.

    Returns:
        RollProdAOS: Updated rolling product accumulator output state
            of type `vectorbtpro.generic.enums.RollProdAOS`.
    """
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumprod = in_state.cumprod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumprod = cumprod * value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumprod = cumprod
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumprod = cumprod / pre_window_value
        window_len = window - nancnt
        window_cumprod = cumprod
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumprod

    return RollProdAOS(cumprod=cumprod, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_prod_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling product for a one-dimensional array.

    Uses `rolling_prod_acc_nb` to update the accumulation state for each iteration,
    replicating the behavior of `pd.Series(arr).rolling(window, min_periods=minp).apply(np.prod)`.

    Args:
        arr (Array1d): Array of numerical values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array containing the rolling product values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    cumprod = 1.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollProdAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumprod=cumprod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_prod_acc_nb(in_state)
        cumprod = out_state.cumprod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_prod_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """Apply a rolling product computation column-wise to a two-dimensional array.

    Uses `rolling_prod_1d_nb` to compute the rolling product for each column independently.

    Args:
        arr (Array2d): Two-dimensional array of numerical values, with each column processed separately.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Two-dimensional array with the rolling product computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_prod_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_mean_acc_nb(in_state: RollMeanAIS) -> RollMeanAOS:
    """Accumulate the rolling mean state for one iteration of `rolling_mean_1d_nb`.

    Args:
        in_state (RollMeanAIS): Current rolling mean accumulator input state
            of type `vectorbtpro.generic.enums.RollMeanAIS`.

    Returns:
        RollMeanAOS: Updated rolling mean accumulator output state
            of type `vectorbtpro.generic.enums.RollMeanAOS`.
    """
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumsum = cumsum
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
        window_cumsum = cumsum
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumsum / window_len

    return RollMeanAOS(cumsum=cumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute the rolling mean of a one-dimensional array.

    Uses `rolling_mean_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).mean()`.

    Args:
        arr (Array1d): One-dimensional array of numeric data.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array containing the rolling means.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollMeanAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_mean_acc_nb(in_state)
        cumsum = out_state.cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """Compute the rolling mean for a two-dimensional array column-wise.

    Applies `rolling_mean_1d_nb` to each column.

    Args:
        arr (Array2d): Two-dimensional array of numeric data.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Array containing column-wise rolling mean values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_std_acc_nb(in_state: RollStdAIS) -> RollStdAOS:
    """Accumulate values for computing the rolling standard deviation.

    Processes a state of type `vectorbtpro.generic.enums.RollStdAIS` and returns
    an updated state of type `vectorbtpro.generic.enums.RollStdAOS`.

    Args:
        in_state (RollStdAIS): Input state for the rolling standard deviation calculation.

    Returns:
        RollStdAOS: Updated state after processing the current value.
    """
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    cumsum_sq = in_state.cumsum_sq
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    ddof = in_state.ddof

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
        cumsum_sq = cumsum_sq + value**2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
            cumsum_sq = cumsum_sq - pre_window_value**2
        window_len = window - nancnt
    if window_len < minp or window_len == ddof:
        value = np.nan
    else:
        mean = cumsum / window_len
        value = np.sqrt(
            np.abs(cumsum_sq - 2 * cumsum * mean + window_len * mean**2) / (window_len - ddof)
        )

    return RollStdAOS(
        cumsum=cumsum, cumsum_sq=cumsum_sq, nancnt=nancnt, window_len=window_len, value=value
    )


@register_jitted(cache=True)
def rolling_std_1d_nb(
    arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, ddof: int = 0
) -> tp.Array1d:
    """Compute the rolling standard deviation of a one-dimensional array.

    Uses `rolling_std_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).std(ddof=ddof)`.

    Args:
        arr (Array1d): One-dimensional array of numeric data.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Array containing rolling standard deviation values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    cumsum = 0.0
    cumsum_sq = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollStdAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            cumsum_sq=cumsum_sq,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_std_acc_nb(in_state)
        cumsum = out_state.cumsum
        cumsum_sq = out_state.cumsum_sq
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_std_nb(
    arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, ddof: int = 0
) -> tp.Array2d:
    """Compute the rolling standard deviation for a two-dimensional array column-wise.

    Applies `rolling_std_1d_nb` to each column.

    Args:
        arr (Array2d): Two-dimensional array of numeric data.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array2d: Array containing column-wise rolling standard deviation values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_std_1d_nb(arr[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_zscore_acc_nb(in_state: RollZScoreAIS) -> RollZScoreAOS:
    """Accumulate values for computing a rolling z-score.

    Processes a state of type `vectorbtpro.generic.enums.RollZScoreAIS` and
    returns an updated state of type `vectorbtpro.generic.enums.RollZScoreAOS`.

    Args:
        in_state (RollZScoreAIS): Input state for the rolling z-score computation.

    Returns:
        RollZScoreAOS: Updated state after processing the current value.
    """
    mean_in_state = RollMeanAIS(
        i=in_state.i,
        value=in_state.value,
        pre_window_value=in_state.pre_window_value,
        cumsum=in_state.cumsum,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
    )
    std_in_state = RollStdAIS(
        i=in_state.i,
        value=in_state.value,
        pre_window_value=in_state.pre_window_value,
        cumsum=in_state.cumsum,
        cumsum_sq=in_state.cumsum_sq,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
        ddof=in_state.ddof,
    )
    mean_out_state = rolling_mean_acc_nb(mean_in_state)
    std_out_state = rolling_std_acc_nb(std_in_state)
    if std_out_state.value == 0:
        value = np.nan
    else:
        value = (in_state.value - mean_out_state.value) / std_out_state.value

    return RollZScoreAOS(
        cumsum=std_out_state.cumsum,
        cumsum_sq=std_out_state.cumsum_sq,
        nancnt=std_out_state.nancnt,
        window_len=std_out_state.window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_zscore_1d_nb(
    arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, ddof: int = 0
) -> tp.Array1d:
    """Compute the rolling z-score of a one-dimensional array.

    Iteratively applies `rolling_zscore_acc_nb` to compute z-scores.

    Args:
        arr (Array1d): One-dimensional array of numeric data.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Array containing rolling z-score values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    cumsum = 0.0
    cumsum_sq = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollZScoreAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            cumsum_sq=cumsum_sq,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_zscore_acc_nb(in_state)
        cumsum = out_state.cumsum
        cumsum_sq = out_state.cumsum_sq
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_zscore_nb(
    arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, ddof: int = 0
) -> tp.Array2d:
    """Compute rolling z-score for each column of a 2D array using `rolling_zscore_1d_nb`.

    Args:
        arr (Array2d): Input 2D array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array2d: 2D array of computed z-scores.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_zscore_1d_nb(arr[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def wm_mean_acc_nb(in_state: WMMeanAIS) -> WMMeanAOS:
    """Update the accumulator state for the weighted moving average computation.

    Args:
        in_state (WMMeanAIS): State object from `vectorbtpro.generic.enums.WMMeanAIS`
            representing the current accumulator state.

    Returns:
        WMMeanAOS: Updated accumulator state from `vectorbtpro.generic.enums.WMMeanAOS`.
    """
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    wcumsum = in_state.wcumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if i >= window and not np.isnan(pre_window_value):
        wcumsum = wcumsum - cumsum
    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
    if not np.isnan(value):
        wcumsum = wcumsum + value * window_len
    if window_len < minp:
        value = np.nan
    else:
        value = wcumsum * 2 / (window_len + 1) / window_len

    return WMMeanAOS(
        cumsum=cumsum, wcumsum=wcumsum, nancnt=nancnt, window_len=window_len, value=value
    )


@register_jitted(cache=True)
def wm_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute the weighted moving average for a 1D array using `wm_mean_acc_nb`.

    Args:
        arr (Array1d): Input 1D array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array of weighted moving averages.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    cumsum = 0.0
    wcumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = WMMeanAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            wcumsum=wcumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = wm_mean_acc_nb(in_state)
        cumsum = out_state.cumsum
        wcumsum = out_state.wcumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def wm_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """Compute the weighted moving average for each column of a 2D array using `wm_mean_1d_nb`.

    Args:
        arr (Array2d): Input 2D array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: 2D array of weighted moving averages computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wm_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def alpha_from_com_nb(com: float) -> float:
    """Compute the smoothing factor `alpha` from the center of mass.

    Args:
        com (float): Center of mass value.

    Returns:
        float: Computed smoothing factor `alpha`.
    """
    return 1.0 / (1.0 + com)


@register_jitted(cache=True)
def alpha_from_span_nb(span: float) -> float:
    """Compute the smoothing factor `alpha` from a span value.

    Args:
        span (float): Span value.

    Returns:
        float: Computed smoothing factor `alpha`.
    """
    com = (span - 1) / 2.0
    return alpha_from_com_nb(com)


@register_jitted(cache=True)
def alpha_from_halflife_nb(halflife: float) -> float:
    """Compute the smoothing factor `alpha` from a half-life value.

    Args:
        halflife (float): Half-life value.

    Returns:
        float: Computed smoothing factor `alpha`.
    """
    return 1 - np.exp(-np.log(2) / halflife)


@register_jitted(cache=True)
def alpha_from_wilder_nb(period: int) -> float:
    """Compute the smoothing factor `alpha` using Wilder's period.

    Args:
        period (int): Wilder's period.

    Returns:
        float: Computed smoothing factor `alpha`.
    """
    return 1 / period


@register_jitted(cache=True)
def ewm_mean_acc_nb(in_state: EWMMeanAIS) -> EWMMeanAOS:
    """Update the accumulator state for the exponential weighted moving average computation.

    Args:
        in_state (EWMMeanAIS): State object from `vectorbtpro.generic.enums.EWMMeanAIS`
            representing the current accumulator state.

    Returns:
        EWMMeanAOS: Updated accumulator state from `vectorbtpro.generic.enums.EWMMeanAOS`.
    """
    i = in_state.i
    value = in_state.value
    old_wt = in_state.old_wt
    weighted_avg = in_state.weighted_avg
    nobs = in_state.nobs
    alpha = in_state.alpha
    minp = in_state.minp
    adjust = in_state.adjust

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha

    if i > 0:
        is_observation = not np.isnan(value)
        nobs += is_observation
        if not np.isnan(weighted_avg):
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != value:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * value)) / (old_wt + new_wt)
                if adjust:
                    old_wt += new_wt
                else:
                    old_wt = 1.0
        elif is_observation:
            weighted_avg = value
    else:
        is_observation = not np.isnan(weighted_avg)
        nobs += int(is_observation)
    value = weighted_avg if (nobs >= minp) else np.nan

    return EWMMeanAOS(old_wt=old_wt, weighted_avg=weighted_avg, nobs=nobs, value=value)


@register_jitted(cache=True)
def ewm_mean_1d_nb(
    arr: tp.Array1d, span: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array1d:
    """Compute the exponential weighted moving average for a 1D array using `ewm_mean_acc_nb`.

    This function serves as a Numba equivalent to:
    `pd.Series(arr).ewm(span=span, min_periods=minp, adjust=adjust).mean()`, and is adapted from
    `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments.

    Args:
        arr (Array1d): Input 1D array.
        span (int): Window span for the exponential weighting.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array of computed exponential weighted moving averages.

    !!! note
        In contrast to the Pandas implementation, `minp` is applied within `span`.
    """
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    out = np.empty(len(arr), dtype=float_)
    if len(arr) == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    weighted_avg = float(arr[0]) + 0.0  # cast to float_
    nobs = 0
    n_obs_lagged = 0
    old_wt = 1.0

    for i in range(len(arr)):
        if i >= span:
            if not np.isnan(arr[i - span]):
                n_obs_lagged += 1
        in_state = EWMMeanAIS(
            i=i,
            value=arr[i],
            old_wt=old_wt,
            weighted_avg=weighted_avg,
            nobs=nobs - n_obs_lagged,
            alpha=alpha,
            minp=minp,
            adjust=adjust,
        )
        out_state = ewm_mean_acc_nb(in_state)
        old_wt = out_state.old_wt
        weighted_avg = out_state.weighted_avg
        nobs = out_state.nobs + n_obs_lagged
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), span=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ewm_mean_nb(
    arr: tp.Array2d, span: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array2d:
    """Compute the 2-dimensional exponential weighted moving average for each column
    independently using `ewm_mean_1d_nb`.

    Args:
        arr (Array2d): Input 2-dimensional data array.
        span (int): Window span for the exponential weighting.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: Computed 2-dimensional exponential weighted moving average.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_mean_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def ewm_std_acc_nb(in_state: EWMStdAIS) -> EWMStdAOS:
    """Accumulate and update the state for computing the 1-dimensional exponential
    weighted moving standard deviation.

    Args:
        in_state (EWMStdAIS): Current accumulator state of type `vectorbtpro.generic.enums.EWMStdAIS`
            containing the observation index, value, means, covariance, and weights.

    Returns:
        EWMStdAOS: Updated accumulator state of type `vectorbtpro.generic.enums.EWMStdAOS` with
            recalculated means, covariance, weights, and computed value.
    """
    i = in_state.i
    value = in_state.value
    mean_x = in_state.mean_x
    mean_y = in_state.mean_y
    cov = in_state.cov
    sum_wt = in_state.sum_wt
    sum_wt2 = in_state.sum_wt2
    old_wt = in_state.old_wt
    nobs = in_state.nobs
    alpha = in_state.alpha
    minp = in_state.minp
    adjust = in_state.adjust

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha

    cur_x = value
    cur_y = value
    is_observation = not np.isnan(cur_x) and not np.isnan(cur_y)
    nobs += is_observation
    if i > 0:
        if not np.isnan(mean_x):
            sum_wt *= old_wt_factor
            sum_wt2 *= old_wt_factor * old_wt_factor
            old_wt *= old_wt_factor
            if is_observation:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series
                if mean_x != cur_x:
                    mean_x = ((old_wt * old_mean_x) + (new_wt * cur_x)) / (old_wt + new_wt)

                # avoid numerical errors on constant series
                if mean_y != cur_y:
                    mean_y = ((old_wt * old_mean_y) + (new_wt * cur_y)) / (old_wt + new_wt)
                cov = (
                    (old_wt * (cov + ((old_mean_x - mean_x) * (old_mean_y - mean_y))))
                    + (new_wt * ((cur_x - mean_x) * (cur_y - mean_y)))
                ) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += new_wt * new_wt
                old_wt += new_wt
                if not adjust:
                    sum_wt /= old_wt
                    sum_wt2 /= old_wt * old_wt
                    old_wt = 1.0
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y
    else:
        if not is_observation:
            mean_x = np.nan
            mean_y = np.nan

    if nobs >= minp:
        numerator = sum_wt * sum_wt
        denominator = numerator - sum_wt2
        if denominator > 0.0:
            value = (numerator / denominator) * cov
        else:
            value = np.nan
    else:
        value = np.nan

    return EWMStdAOS(
        mean_x=mean_x,
        mean_y=mean_y,
        cov=cov,
        sum_wt=sum_wt,
        sum_wt2=sum_wt2,
        old_wt=old_wt,
        nobs=nobs,
        value=value,
    )


@register_jitted(cache=True)
def ewm_std_1d_nb(
    arr: tp.Array1d, span: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array1d:
    """Compute the exponential weighted moving standard deviation for a 1-dimensional array.

    Updates the accumulator state with `ewm_std_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp).std()`
    and adapted from `pd._libs.window.aggregations.window_aggregations.ewmcov`.

    Args:
        arr (Array1d): Input data array.
        span (int): Window span for the exponential weighting.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Computed exponential weighted moving standard deviation.

    !!! note
        In contrast to Pandas, the parameter `minp` is applied within the span.
    """
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    out = np.empty(len(arr), dtype=float_)
    if len(arr) == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    mean_x = float(arr[0]) + 0.0  # cast to float_
    mean_y = float(arr[0]) + 0.0  # cast to float_
    nobs = 0
    n_obs_lagged = 0
    cov = 0.0
    sum_wt = 1.0
    sum_wt2 = 1.0
    old_wt = 1.0

    for i in range(len(arr)):
        if i >= span:
            if not np.isnan(arr[i - span]):
                n_obs_lagged += 1
        in_state = EWMStdAIS(
            i=i,
            value=arr[i],
            mean_x=mean_x,
            mean_y=mean_y,
            cov=cov,
            sum_wt=sum_wt,
            sum_wt2=sum_wt2,
            old_wt=old_wt,
            nobs=nobs - n_obs_lagged,
            alpha=alpha,
            minp=minp,
            adjust=adjust,
        )
        out_state = ewm_std_acc_nb(in_state)
        mean_x = out_state.mean_x
        mean_y = out_state.mean_y
        cov = out_state.cov
        sum_wt = out_state.sum_wt
        sum_wt2 = out_state.sum_wt2
        old_wt = out_state.old_wt
        nobs = out_state.nobs + n_obs_lagged
        out[i] = out_state.value

    return np.sqrt(out)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), span=None, minp=None, adjust=None, ddof=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ewm_std_nb(
    arr: tp.Array2d, span: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array2d:
    """Compute the 2-dimensional exponential weighted moving standard deviation for each
    column independently using `ewm_std_1d_nb`.

    Args:
        arr (Array2d): Input 2-dimensional data array.
        span (int): Window span for the exponential weighting.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: Computed 2-dimensional exponential weighted moving standard deviation.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_std_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def wwm_mean_1d_nb(
    arr: tp.Array1d, period: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array1d:
    """Compute Wilder's exponential weighted moving average for a 1-dimensional array.

    Applies `ewm_mean_1d_nb` internally with a span calculated as `2 * period - 1`.

    Args:
        arr (Array1d): Input data array.
        period (int): Period used for the moving average computation.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Wilder's exponential weighted moving average.
    """
    if minp is None:
        minp = period
    return ewm_mean_1d_nb(arr, 2 * period - 1, minp=minp, adjust=adjust)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), period=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def wwm_mean_nb(
    arr: tp.Array2d, period: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array2d:
    """Compute the 2-dimensional Wilder's exponential weighted moving average for each column
    independently using `wwm_mean_1d_nb`.

    Args:
        arr (Array2d): Input 2-dimensional data array.
        period (int): Period used for the moving average computation.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: Computed 2-dimensional Wilder's exponential weighted moving average.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wwm_mean_1d_nb(arr[:, col], period, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def wwm_std_1d_nb(
    arr: tp.Array1d, period: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array1d:
    """Compute Wilder's exponential weighted moving standard deviation.

    Args:
        arr (Array1d): 1-dimensional input array.
        period (int): Period used for computing the standard deviation.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array containing the computed standard deviations.
    """
    if minp is None:
        minp = period
    return ewm_std_1d_nb(arr, 2 * period - 1, minp=minp, adjust=adjust)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), period=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def wwm_std_nb(
    arr: tp.Array2d, period: int, minp: tp.Optional[int] = None, adjust: bool = False
) -> tp.Array2d:
    """Compute a 2-dimensional version of Wilder's exponential weighted moving standard deviation.

    Args:
        arr (Array2d): 2-dimensional input array.
        period (int): Period used for computing the standard deviation.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: 2-dimensional array containing the computed standard deviations.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wwm_std_1d_nb(arr[:, col], period, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def vidya_acc_nb(in_state: VidyaAIS) -> VidyaAOS:
    """Accumulate and update the VIDYA state from a given input state.

    Args:
        in_state (VidyaAIS): Input state of type `vectorbtpro.generic.enums.VidyaAIS`
            containing parameters for VIDYA calculation.

    Returns:
        VidyaAOS: Updated state of type `vectorbtpro.generic.enums.VidyaAOS` with computed VIDYA values.
    """
    i = in_state.i
    prev_value = in_state.prev_value
    value = in_state.value
    pre_window_prev_value = in_state.pre_window_prev_value
    pre_window_value = in_state.pre_window_value
    pos_cumsum = in_state.pos_cumsum
    neg_cumsum = in_state.neg_cumsum
    prev_vidya = in_state.prev_vidya
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    alpha = 2 / (window + 1)

    diff = value - prev_value
    if np.isnan(diff):
        nancnt = nancnt + 1
    else:
        if diff > 0:
            pos_value = diff
            neg_value = 0.0
        else:
            pos_value = 0.0
            neg_value = abs(diff)
        pos_cumsum = pos_cumsum + pos_value
        neg_cumsum = neg_cumsum + neg_value
    if i < window:
        window_len = i + 1 - nancnt
    else:
        pre_window_diff = pre_window_value - pre_window_prev_value
        if np.isnan(pre_window_diff):
            nancnt = nancnt - 1
        else:
            if pre_window_diff > 0:
                pre_window_pos_value = pre_window_diff
                pre_window_neg_value = 0.0
            else:
                pre_window_pos_value = 0.0
                pre_window_neg_value = abs(pre_window_diff)
            pos_cumsum = pos_cumsum - pre_window_pos_value
            neg_cumsum = neg_cumsum - pre_window_neg_value
        window_len = window - nancnt
    window_pos_cumsum = pos_cumsum
    window_neg_cumsum = neg_cumsum
    if window_len < minp:
        cmo = np.nan
        vidya = np.nan
    else:
        sh = window_pos_cumsum
        sl = window_neg_cumsum
        if sh + sl == 0:
            cmo = 0.0
        else:
            cmo = np.abs((sh - sl) / (sh + sl))
        if np.isnan(prev_vidya):
            prev_vidya = 0.0
        vidya = alpha * cmo * value + prev_vidya * (1 - alpha * cmo)

    return VidyaAOS(
        pos_cumsum=pos_cumsum,
        neg_cumsum=neg_cumsum,
        nancnt=nancnt,
        window_len=window_len,
        cmo=cmo,
        vidya=vidya,
    )


@register_jitted(cache=True)
def vidya_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute the Variable Index Dynamic Average (VIDYA) for a 1-dimensional array.

    Args:
        arr (Array1d): 1-dimensional input array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array containing the computed VIDYA values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    pos_cumsum = 0.0
    neg_cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = VidyaAIS(
            i=i,
            prev_value=arr[i - 1] if i - 1 >= 0 else np.nan,
            value=arr[i],
            pre_window_prev_value=arr[i - window - 1] if i - window - 1 >= 0 else np.nan,
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            pos_cumsum=pos_cumsum,
            neg_cumsum=neg_cumsum,
            prev_vidya=out[i - 1] if i - 1 >= 0 else np.nan,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = vidya_acc_nb(in_state)
        pos_cumsum = out_state.pos_cumsum
        neg_cumsum = out_state.neg_cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.vidya

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def vidya_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """Compute a 2-dimensional VIDYA by applying the 1-dimensional VIDYA calculation to each column.

    Args:
        arr (Array2d): 2-dimensional input array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: 2-dimensional array containing the computed VIDYA values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = vidya_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def ma_1d_nb(
    arr: tp.Array1d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Compute a moving average for a 1-dimensional array based on the specified mode.

    Args:
        arr (Array1d): 1-dimensional input array.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array containing the computed moving average values.
    """
    if wtype == WType.Simple:
        return rolling_mean_1d_nb(arr, window, minp=minp)
    if wtype == WType.Weighted:
        return wm_mean_1d_nb(arr, window, minp=minp)
    if wtype == WType.Exp:
        return ewm_mean_1d_nb(arr, window, minp=minp, adjust=adjust)
    if wtype == WType.Wilder:
        return wwm_mean_1d_nb(arr, window, minp=minp, adjust=adjust)
    if wtype == WType.Vidya:
        return vidya_1d_nb(arr, window, minp=minp)
    raise ValueError("Invalid rolling mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, wtype=None, minp=None, adjust=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ma_nb(
    arr: tp.Array2d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """Compute a 2-dimensional moving average by applying the 1-dimensional calculation column-wise.

    Args:
        arr (Array2d): 2-dimensional input array.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: 2-dimensional array containing the computed moving average values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ma_1d_nb(arr[:, col], window, wtype=wtype, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def msd_1d_nb(
    arr: tp.Array1d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array1d:
    """Compute a moving standard deviation on a 1-dimensional array using a specified weighting type.

    Args:
        arr (Array1d): Input array of numerical values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Array containing the computed moving standard deviation.
    """
    if wtype == WType.Simple:
        return rolling_std_1d_nb(arr, window, minp=minp, ddof=ddof)
    if wtype == WType.Weighted:
        raise ValueError("Weighted mode is not supported for standard deviations")
    if wtype == WType.Exp:
        return ewm_std_1d_nb(arr, window, minp=minp, adjust=adjust)
    if wtype == WType.Wilder:
        return wwm_std_1d_nb(arr, window, minp=minp, adjust=adjust)
    raise ValueError("Invalid rolling mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1), window=None, wtype=None, minp=None, adjust=None, ddof=None
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def msd_nb(
    arr: tp.Array2d,
    window: int,
    wtype: int = WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array2d:
    """Compute moving standard deviation for each column in a 2-dimensional array.

    Args:
        arr (Array2d): 2-dimensional input array of numerical values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array2d: Array with computed moving standard deviation for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = msd_1d_nb(
            arr[:, col], window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof
        )
    return out


@register_jitted(cache=True)
def rolling_cov_acc_nb(in_state: RollCovAIS) -> RollCovAOS:
    """Accumulate rolling covariance statistics.

    Args:
        in_state (RollCovAIS): Current state of type `vectorbtpro.generic.enums.RollCovAIS`
            containing intermediate sums, counts, and covariance parameters.

    Returns:
        RollCovAOS: Updated state of type `vectorbtpro.generic.enums.RollCovAOS`
            with computed rolling covariance and intermediate values.
    """
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    ddof = in_state.ddof

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp or window_len == ddof:
        value = np.nan
    else:
        window_prod_mean = cumsum_prod / (window_len - ddof)
        window_mean1 = cumsum1 / window_len
        window_mean2 = cumsum2 / window_len
        window_mean_prod = window_mean1 * window_mean2 * window_len / (window_len - ddof)
        value = window_prod_mean - window_mean_prod

    return RollCovAOS(
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_cov_1d_nb(
    arr1: tp.Array1d,
    arr2: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
) -> tp.Array1d:
    """Compute rolling covariance between two 1-dimensional arrays.

    Equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).cov(arr2)`.

    Args:
        arr1 (Array1d): First input array.
        arr2 (Array1d): Second input array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Array containing the computed rolling covariance values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=float_)
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollCovAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_cov_acc_nb(in_state)
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_cov_nb(
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
) -> tp.Array2d:
    """Compute rolling covariance column-wise between two 2-dimensional arrays.

    Args:
        arr1 (Array2d): First 2-dimensional input array.
        arr2 (Array2d): Second 2-dimensional input array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array2d: Array with computed rolling covariance for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr1, dtype=float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_cov_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_corr_acc_nb(in_state: RollCorrAIS) -> RollCorrAOS:
    """Accumulator for rolling correlation computation.

    Args:
        in_state (RollCorrAIS): Input state of type `vectorbtpro.generic.enums.RollCorrAIS`
            containing rolling correlation accumulation fields.

    Returns:
        RollCorrAOS: Updated state of type `vectorbtpro.generic.enums.RollCorrAOS`
            with accumulated rolling correlation and related metrics.
    """
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_sq1 = in_state.cumsum_sq1
    cumsum_sq2 = in_state.cumsum_sq2
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_sq1 = cumsum_sq1 + value1**2
        cumsum_sq2 = cumsum_sq2 + value2**2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_sq1 = cumsum_sq1 - pre_window_value1**2
            cumsum_sq2 = cumsum_sq2 - pre_window_value2**2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp:
        value = np.nan
    else:
        nom = window_len * cumsum_prod - cumsum1 * cumsum2
        denom1 = np.sqrt(window_len * cumsum_sq1 - cumsum1**2)
        denom2 = np.sqrt(window_len * cumsum_sq2 - cumsum2**2)
        denom = denom1 * denom2
        if denom == 0:
            value = np.nan
        else:
            value = nom / denom

    return RollCorrAOS(
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_sq1=cumsum_sq1,
        cumsum_sq2=cumsum_sq2,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_corr_1d_nb(
    arr1: tp.Array1d, arr2: tp.Array1d, window: int, minp: tp.Optional[int] = None
) -> tp.Array1d:
    """Compute rolling correlation coefficient for one-dimensional arrays.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).corr(arr2)`.

    Args:
        arr1 (Array1d): First input array.
        arr2 (Array1d): Second input array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array of rolling correlation coefficients.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=float_)
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_sq1 = 0.0
    cumsum_sq2 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollCorrAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_sq1=cumsum_sq1,
            cumsum_sq2=cumsum_sq2,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_corr_acc_nb(in_state)
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_sq1 = out_state.cumsum_sq1
        cumsum_sq2 = out_state.cumsum_sq2
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_corr_nb(
    arr1: tp.Array2d, arr2: tp.Array2d, window: int, minp: tp.Optional[int] = None
) -> tp.Array2d:
    """Compute rolling correlation coefficient for each column in two-dimensional arrays.

    Args:
        arr1 (Array2d): First input two-dimensional array.
        arr2 (Array2d): Second input two-dimensional array.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Two-dimensional array of rolling correlation coefficients computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr1, dtype=float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_corr_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_ols_acc_nb(in_state: RollOLSAIS) -> RollOLSAOS:
    """Accumulator for rolling ordinary least squares (OLS) regression.

    Args:
        in_state (RollOLSAIS): Input state containing rolling OLS accumulation fields.

    Returns:
        RollOLSAOS: Updated state with accumulated values and computed OLS slope and intercept.
    """
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    validcnt = in_state.validcnt
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_sq1 = in_state.cumsum_sq1
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        validcnt = validcnt + 1
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_sq1 = cumsum_sq1 + value1**2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            validcnt = validcnt - 1
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_sq1 = cumsum_sq1 - pre_window_value1**2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp:
        slope_value = np.nan
        intercept_value = np.nan
    else:
        slope_num = validcnt * cumsum_prod - cumsum1 * cumsum2
        slope_denom = validcnt * cumsum_sq1 - cumsum1**2
        if slope_denom != 0:
            slope_value = slope_num / slope_denom
        else:
            slope_value = np.nan
        intercept_num = cumsum2 - slope_value * cumsum1
        intercept_denom = validcnt
        if intercept_denom != 0:
            intercept_value = intercept_num / intercept_denom
        else:
            intercept_value = np.nan

    return RollOLSAOS(
        validcnt=validcnt,
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_sq1=cumsum_sq1,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        slope_value=slope_value,
        intercept_value=intercept_value,
    )


@register_jitted(cache=True)
def rolling_ols_1d_nb(
    arr1: tp.Array1d,
    arr2: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Compute rolling linear regression.

    Args:
        arr1 (Array1d): 1-dimensional array of independent variable values.
        arr2 (Array1d): 1-dimensional array of dependent variable values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Tuple[Array1d, Array1d]: Tuple containing the slope and intercept arrays.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    slope_out = np.empty_like(arr1, dtype=float_)
    intercept_out = np.empty_like(arr1, dtype=float_)
    validcnt = 0
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_sq1 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollOLSAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            validcnt=validcnt,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_sq1=cumsum_sq1,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_ols_acc_nb(in_state)
        validcnt = out_state.validcnt
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_sq1 = out_state.cumsum_sq1
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        slope_out[i] = out_state.slope_value
        intercept_out[i] = out_state.intercept_value

    return slope_out, intercept_out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_ols_nb(
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """2-dim version of `rolling_ols_1d_nb`.

    Args:
        arr1 (Array2d): 2-dimensional array of independent variable values.
        arr2 (Array2d): 2-dimensional array of dependent variable values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Tuple[Array2d, Array2d]: Tuple containing the slope and intercept arrays.

    !!! tip
        This function is parallelizable.
    """
    slope_out = np.empty_like(arr1, dtype=float_)
    intercept_out = np.empty_like(arr1, dtype=float_)
    for col in prange(arr1.shape[1]):
        slope_out[:, col], intercept_out[:, col] = rolling_ols_1d_nb(
            arr1[:, col], arr2[:, col], window, minp=minp
        )
    return slope_out, intercept_out


@register_jitted(cache=True)
def rolling_rank_1d_nb(
    arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, pct: bool = False
) -> tp.Array1d:
    """Rolling version of `rank_1d_nb`.

    Args:
        arr (Array1d): 1-dimensional array of values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        pct (bool): If True, compute the rank as a percentile.

    Returns:
        Array1d: Array of rolling ranks.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            valid_cnt = window - nancnt
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            arr_window = arr[from_i:to_i]
            out[i] = rank_1d_nb(arr_window, pct=pct)[-1]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, pct=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_rank_nb(
    arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, pct: bool = False
) -> tp.Array2d:
    """2-dim version of `rolling_rank_1d_nb`.

    Args:
        arr (Array2d): 2-dimensional array of values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        pct (bool): If True, compute the rank as a percentile.

    Returns:
        Array2d: Array of rolling ranks.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_rank_1d_nb(arr[:, col], window, minp=minp, pct=pct)
    return out


@register_jitted(cache=True)
def rolling_min_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling minimum value.

    Equivalent to `pd.Series(arr).rolling(window, min_periods=minp).min()` in pandas.

    Args:
        arr (Array1d): 1-dimensional array of values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array of rolling minimum values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_min_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_min_1d_nb`.

    Args:
        arr (Array2d): 2-dimensional array of values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Array of rolling minimum values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_min_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_max_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling maximum value.

    Equivalent to `pd.Series(arr).rolling(window, min_periods=minp).max()` in pandas.

    Args:
        arr (Array1d): 1-dimensional array of values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Array of rolling maximum values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_max_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_max_1d_nb`.

    Args:
        arr (Array2d): 2-dimensional array of values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Array of rolling maximum values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_max_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_argmin_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    local: bool = False,
) -> tp.Array1d:
    """Return the index of the minimum value within each rolling window of a 1D array.

    Args:
        arr (Array1d): 1D array of numerical values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        local (bool): If True, return the index relative to the window;
            otherwise, return the index in the original array.

    Returns:
        Array1d: Array of indices where each position corresponds to the location of
            the minimum value in the rolling window.

            Returns -1 if the count of non-NaN values is less than minp.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        if local:
            mini = 0
        else:
            mini = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
                if local:
                    mini = k
                else:
                    mini = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = mini
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, local=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_argmin_nb(
    arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, local: bool = False
) -> tp.Array2d:
    """Return a 2D array of rolling minimum indices computed column-wise.

    Args:
        arr (Array2d): 2D array of numerical values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        local (bool): If True, return the index relative to the window;
            otherwise, return the index in the original array.

    Returns:
        Array2d: 2D array where each column contains indices corresponding to
            the minimum value in its rolling window.

            Positions with fewer than the required non-NaN values are set to -1.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmin_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_argmax_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    local: bool = False,
) -> tp.Array1d:
    """Return the index of the maximum value within each rolling window of a 1D array.

    Args:
        arr (Array1d): 1D array of numerical values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        local (bool): If True, return the index relative to the window;
            otherwise, return the index in the original array.

    Returns:
        Array1d: Array of indices where each position corresponds to the location of
            the maximum value in the rolling window.

            Returns -1 if the count of non-NaN values is less than minp.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        if local:
            maxi = 0
        else:
            maxi = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
                if local:
                    maxi = k
                else:
                    maxi = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = maxi
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, local=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_argmax_nb(
    arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, local: bool = False
) -> tp.Array2d:
    """Return a 2D array of rolling maximum indices computed column-wise.

    Args:
        arr (Array2d): 2D array of numerical values.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        local (bool): If True, return the index relative to the window;
            otherwise, return the index in the original array.

    Returns:
        Array2d: 2D array where each column contains indices corresponding to
            the maximum value in its rolling window.

            Positions with fewer than the required non-NaN values are set to -1.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmax_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_any_1d_nb(arr: tp.Array1d, window: int) -> tp.Array1d:
    """Return a boolean array indicating whether any element in each rolling window is True.

    Args:
        arr (Array1d): 1D array of boolean or numerical values.
        window (int): Window size.

    Returns:
        Array1d: Boolean array where each element is True if at least one True value is
            present in the corresponding rolling window, otherwise False.
    """
    out = np.empty_like(arr, dtype=np.bool_)
    last_true_i = -1
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]) and arr[i]:
            last_true_i = i
        from_i = max(0, i + 1 - window)
        if last_true_i >= from_i:
            out[i] = True
        else:
            out[i] = False
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_any_nb(arr: tp.Array2d, window: int) -> tp.Array2d:
    """Return a 2D boolean array computed column-wise, indicating the presence
    of any True value in each rolling window.

    Args:
        arr (Array2d): 2D array of boolean or numerical values.
        window (int): Window size.

    Returns:
        Array2d: 2D boolean array where each column contains True if any True value is
            detected in the corresponding rolling window, otherwise False.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=np.bool_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_any_1d_nb(arr[:, col], window)
    return out


@register_jitted(cache=True)
def rolling_all_1d_nb(arr: tp.Array1d, window: int) -> tp.Array1d:
    """Return a boolean array indicating whether all elements in each rolling window are True.

    Args:
        arr (Array1d): 1D array of boolean or numerical values.
        window (int): Window size.

    Returns:
        Array1d: Boolean array where each element is True if all values in
            the corresponding rolling window are True, otherwise False.
    """
    out = np.empty_like(arr, dtype=np.bool_)
    last_false_i = -1
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]) or not arr[i]:
            last_false_i = i
        from_i = max(0, i + 1 - window)
        if last_false_i >= from_i:
            out[i] = False
        else:
            out[i] = True
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_all_nb(arr: tp.Array2d, window: int) -> tp.Array2d:
    """Return a 2D boolean array computed column-wise, indicating whether
    all elements in each rolling window are True.

    Args:
        arr (Array2d): 2D array of boolean or numerical values.
        window (int): Window size.

    Returns:
        Array2d: 2D boolean array where each column contains True if all values in
            the corresponding rolling window are True, otherwise False.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=np.bool_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_all_1d_nb(arr[:, col], window)
    return out


@register_jitted(cache=True)
def rolling_pattern_similarity_1d_nb(
    arr: tp.Array1d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
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
) -> tp.Array1d:
    """Compute rolling pattern similarity on a 1D array.

    Uses `vectorbtpro.generic.nb.patterns.pattern_similarity_nb`.

    Args:
        arr (Array1d): 1-dimensional array to compare with the pattern.
        pattern (Array1d): 1D array representing the pattern to locate.

            Can be smaller or larger than the source array. In such cases,
            the smaller array is stretched using the interpolation mode specified by `interp_mode`.
        window (Optional[int]): Window size.

            If None, defaults to the length of `pattern`.
        max_window (Optional[int]): Maximum length of the rolling window for matching.

            If None, defaults to `window`.
        row_select_prob (float): Probability of selecting a row.
        window_select_prob (float): Probability of selecting a window size.
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

    Returns:
        Array1d: Array containing the computed pattern similarity values.
    """
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    out = np.full(arr.shape, np.nan, dtype=float_)
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
                        if not np.isnan(out[i]):
                            if similarity > out[i]:
                                out[i] = similarity
                        else:
                            out[i] = similarity

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        pattern=None,
        window=None,
        max_window=None,
        row_select_prob=None,
        window_select_prob=None,
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
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_pattern_similarity_nb(
    arr: tp.Array2d,
    pattern: tp.Array1d,
    window: tp.Optional[int] = None,
    max_window: tp.Optional[int] = None,
    row_select_prob: float = 1.0,
    window_select_prob: float = 1.0,
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
) -> tp.Array2d:
    """Compute rolling pattern similarity on a 2D array column-wise.

    Each column of the input array is processed using `rolling_pattern_similarity_1d_nb`.

    Args:
        arr (Array2d): 2D input array where similarity is computed for each column.
        pattern (Array1d): 1D array representing the pattern to locate.

            Can be smaller or larger than the source array. In such cases,
            the smaller array is stretched using the interpolation mode specified by `interp_mode`.
        window (Optional[int]): Window size.

            If None, defaults to the length of `pattern`.
        max_window (Optional[int]): Maximum length of the rolling window for matching.

            If None, defaults to `window`.
        row_select_prob (float): Probability of selecting a row.
        window_select_prob (float): Probability of selecting a window size.
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

    Returns:
        Array2d: 2D array where each column contains the computed similarity values.

    !!! tip
        This function is parallelizable.
    """
    max_error_ = to_1d_array_nb(np.asarray(max_error))

    if window is None:
        window = pattern.shape[0]
    if max_window is None:
        max_window = window
    out = np.full(arr.shape, np.nan, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_pattern_similarity_1d_nb(
            arr[:, col],
            pattern,
            window=window,
            max_window=max_window,
            row_select_prob=row_select_prob,
            window_select_prob=window_select_prob,
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
        )
    return out


# ############# Expanding functions ############# #


@register_jitted(cache=True)
def expanding_min_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute the expanding minimum of a 1D array.

    Equivalent to using `pd.Series(arr).expanding(min_periods=minp).min()`.

    Args:
        arr (Array1d): Input array for which the expanding minimum is calculated.
        minp (int): Minimum number of observations required.

    Returns:
        Array1d: Array containing the expanding minimum values.
    """
    out = np.empty_like(arr, dtype=float_)
    minv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(minv) or arr[i] < minv:
            minv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_min_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """Compute the expanding minimum for each column of a 2D array.

    Applies `expanding_min_1d_nb` column-wise to compute the expanding minimum.

    Args:
        arr (Array2d): 2D input array.
        minp (int): Minimum number of observations required.

    Returns:
        Array2d: 2D array where each column contains the expanding minimum values.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_min_1d_nb(arr[:, col], minp=minp)
    return out


@register_jitted(cache=True)
def expanding_max_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute the expanding maximum of a one-dimensional array.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).max()`.

    Args:
        arr (Array1d): Input one-dimensional array.
        minp (int): Minimum number of observations required.

    Returns:
        Array1d: Array containing the expanding maximum values computed from `arr`.
    """
    out = np.empty_like(arr, dtype=float_)
    maxv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(maxv) or arr[i] > maxv:
            maxv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), minp=None),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_max_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """Compute the expanding maximum for each column in a two-dimensional array.

    Column-wise computation is parallelized using `prange` and leverages `expanding_max_1d_nb` for each column.

    Args:
        arr (Array2d): Input two-dimensional array.
        minp (int): Minimum number of observations required.

    Returns:
        Array2d: Two-dimensional array containing the expanding maximum computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(arr, dtype=float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_max_1d_nb(arr[:, col], minp=minp)
    return out
