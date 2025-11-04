# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for custom indicators.

These only accept NumPy arrays and other Numba-compatible types.
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
from vectorbtpro.indicators.enums import HurstMethod, Pivot, SuperTrendAIS, SuperTrendAOS
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


# ############# MA ############# #


@register_jitted(cache=True)
def ma_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Moving average.

    Computes a moving average for a 1-dimensional array.

    Args:
        close (Array1d): 1D array of numerical values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Calculated moving average.
    """
    return generic_nb.ma_1d_nb(close, window, wtype=wtype, minp=minp, adjust=adjust)


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ma_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """2-dim version of `ma_1d_nb`.

    Computes the moving average for each column of a 2-dimensional array.

    Args:
        close (Array2d): 2D array where each column represents a data series.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: 2D array of moving average values.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    ma = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        ma[:, col] = ma_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
        )
    return ma


# ############# MSD ############# #


@register_jitted(cache=True)
def msd_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array1d:
    """Moving standard deviation.

    Computes the moving standard deviation for a 1-dimensional array.

    Args:
        close (Array1d): 1D array of numerical values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array1d: Moving standard deviation.
    """
    return generic_nb.msd_1d_nb(close, window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof)


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
        ddof=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def msd_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array2d:
    """2-dim version of `msd_1d_nb`.

    Computes the moving standard deviation for each column of a 2-dimensional array.

    Args:
        close (Array2d): 2D array where each column represents a data series.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Array2d: 2D array of moving standard deviation values.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    msd = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        msd[:, col] = msd_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
            ddof=ddof,
        )
    return msd


# ############# BBANDS ############# #


@register_jitted(cache=True)
def bbands_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    alpha: float = 2.0,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
    """Bollinger Bands.

    Computes Bollinger Bands for a 1-dimensional array, returning the upper, middle, and lower bands.

    Args:
        close (Array1d): 1D array of numerical values.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        alpha (float): Multiplier for the moving standard deviation to determine the band width.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Tuple[Array1d, Array1d, Array1d]: Upper band, middle band, and lower band.
    """
    ma = ma_1d_nb(close, window=window, wtype=wtype, minp=minp, adjust=adjust)
    msd = msd_1d_nb(close, window=window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof)
    upper = ma + alpha * msd
    middle = ma
    lower = ma - alpha * msd
    return upper, middle, lower


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        alpha=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
        ddof=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bbands_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    alpha: tp.FlexArray1dLike = 2.0,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """2-dim version of `bbands_1d_nb`.

    Computes Bollinger Bands for each column of a 2-dimensional array.

    Args:
        close (Array2d): 2D array where each column is a data series.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        alpha (FlexArray1dLike): Multiplier for the standard deviation.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.
        ddof (int): Delta degrees of freedom.

    Returns:
        Tuple[Array2d, Array2d, Array2d]: Upper, middle, and lower Bollinger Bands for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    alpha_ = to_1d_array_nb(np.asarray(alpha))

    upper = np.empty(close.shape, dtype=float_)
    middle = np.empty(close.shape, dtype=float_)
    lower = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        upper[:, col], middle[:, col], lower[:, col] = bbands_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            alpha=flex_select_1d_nb(alpha_, col),
            minp=minp,
            adjust=adjust,
            ddof=ddof,
        )
    return upper, middle, lower


@register_jitted(cache=True)
def bbands_percent_b_1d_nb(close: tp.Array1d, upper: tp.Array1d, lower: tp.Array1d) -> tp.Array1d:
    """Bollinger Bands %B.

    Computes the %B indicator, which represents the position of the close price
    relative to the lower and upper Bollinger Bands.

    Args:
        close (Array1d): 1D array of close prices.
        upper (Array1d): 1D array representing the upper band.
        lower (Array1d): 1D array representing the lower band.

    Returns:
        Array1d: %B values computed as `(close - lower) / (upper - lower)`.
    """
    return (close - lower) / (upper - lower)


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        upper=ch.ArraySlicer(axis=1),
        lower=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bbands_percent_b_nb(close: tp.Array2d, upper: tp.Array2d, lower: tp.Array2d) -> tp.Array2d:
    """Calculate percent b values for Bollinger Bands using 2-dimensional inputs.

    Applies `bbands_percent_b_1d_nb` column-wise on 2D arrays of close prices and Bollinger Bands.

    Args:
        close (Array2d): 2-dimensional array of close prices.
        upper (Array2d): 2-dimensional array of upper band values.
        lower (Array2d): 2-dimensional array of lower band values.

    Returns:
        Array2d: 2-dimensional array of percent b values.

    !!! tip
        This function is parallelizable.
    """
    percent_b = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        percent_b[:, col] = bbands_percent_b_1d_nb(close[:, col], upper[:, col], lower[:, col])
    return percent_b


@register_jitted(cache=True)
def bbands_bandwidth_1d_nb(upper: tp.Array1d, middle: tp.Array1d, lower: tp.Array1d) -> tp.Array1d:
    """Calculate the Bollinger Bands bandwidth.

    Computes the bandwidth as the difference between the upper and lower bands divided by the middle band.

    Args:
        upper (Array1d): 1D array representing the upper band.
        middle (Array1d): 1D array representing the middle band.
        lower (Array1d): 1D array representing the lower band.

    Returns:
        Array1d: Array of computed Bollinger Bands bandwidth values.
    """
    return (upper - lower) / middle


@register_chunkable(
    size=ch.ArraySizer(arg_query="upper", axis=1),
    arg_take_spec=dict(
        upper=ch.ArraySlicer(axis=1),
        middle=ch.ArraySlicer(axis=1),
        lower=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bbands_bandwidth_nb(upper: tp.Array2d, middle: tp.Array2d, lower: tp.Array2d) -> tp.Array2d:
    """Calculate Bollinger Bands bandwidth for 2-dimensional arrays.

    Applies `bbands_bandwidth_1d_nb` column-wise on 2D arrays of Bollinger Bands values.

    Args:
        upper (Array2d): 2-dimensional array of upper band values.
        middle (Array2d): 2-dimensional array of middle band values.
        lower (Array2d): 2-dimensional array of lower band values.

    Returns:
        Array2d: 2-dimensional array of Bollinger Bands bandwidth values.

    !!! tip
        This function is parallelizable.
    """
    bandwidth = np.empty(upper.shape, dtype=float_)
    for col in prange(upper.shape[1]):
        bandwidth[:, col] = bbands_bandwidth_1d_nb(upper[:, col], middle[:, col], lower[:, col])
    return bandwidth


@register_jitted(cache=True)
def avg_gain_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Calculate average gain over a specified window.

    Computes the average gain from a 1-dimensional array of close prices by calculating
    the positive differences between consecutive values and applying a moving average via `ma_1d_nb`.

    Args:
        close (Array1d): 1-dimensional array of close prices.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array of average gain values.
    """
    up_change = np.empty(close.shape, dtype=float_)
    for i in range(close.shape[0]):
        if i == 0:
            up_change[i] = np.nan
        else:
            change = close[i] - close[i - 1]
            if change < 0:
                up_change[i] = 0.0
            else:
                up_change[i] = change
    avg_gain = ma_1d_nb(up_change, window=window, wtype=wtype, minp=minp, adjust=adjust)
    return avg_gain


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def avg_gain_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """Calculate average gain for 2-dimensional arrays.

    Computes average gain for each column by applying `avg_gain_1d_nb` on slices
    of a 2D array of close prices.

    Args:
        close (Array2d): 2-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: 2-dimensional array of average gain values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    avg_gain = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        avg_gain[:, col] = avg_gain_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
        )
    return avg_gain


@register_jitted(cache=True)
def avg_loss_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Calculate average loss over a specified window.

    Computes the average loss from a 1-dimensional array of close prices by measuring
    the absolute negative changes between consecutive values and applying a moving average via `ma_1d_nb`.

    Args:
        close (Array1d): 1-dimensional array of close prices.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array of average loss values.
    """
    down_change = np.empty(close.shape, dtype=float_)
    for i in range(close.shape[0]):
        if i == 0:
            down_change[i] = np.nan
        else:
            change = close[i] - close[i - 1]
            if change < 0:
                down_change[i] = abs(change)
            else:
                down_change[i] = 0.0
    avg_loss = ma_1d_nb(down_change, window=window, wtype=wtype, minp=minp, adjust=adjust)
    return avg_loss


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def avg_loss_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """Calculate average loss for 2-dimensional arrays.

    Computes average loss for each column by applying `avg_loss_1d_nb` on slices
    of a 2D array of close prices.

    Args:
        close (Array2d): 2-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: 2-dimensional array of average loss values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    avg_loss = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        avg_loss[:, col] = avg_loss_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
        )
    return avg_loss


@register_jitted(cache=True)
def rsi_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Calculate the Relative Strength Index (RSI) for a 1-dimensional array.

    Computes the RSI by calculating the average gain and loss, then using these values to quantify the relative strength.

    Args:
        close (Array1d): 1-dimensional array of close prices.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array1d: Array of RSI values.
    """
    avg_gain = avg_gain_1d_nb(close, window=window, wtype=wtype, minp=minp, adjust=adjust)
    avg_loss = avg_loss_1d_nb(close, window=window, wtype=wtype, minp=minp, adjust=adjust)
    return 100 * avg_gain / (avg_gain + avg_loss)


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rsi_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """Calculate the Relative Strength Index (RSI) for 2-dimensional arrays.

    Computes the RSI for each column by applying `rsi_1d_nb` on slices of a 2D array of close prices.

    Args:
        close (Array2d): 2-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Array2d: Array of computed RSI values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    rsi = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        rsi[:, col] = rsi_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
        )
    return rsi


# ############# STOCH ############# #


@register_jitted(cache=True)
def stoch_k_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    window: int = 14,
    minp: tp.Optional[int] = None,
) -> tp.Array1d:
    """Calculate the Stochastic Oscillator %K for one-dimensional price arrays.

    Args:
        high (Array1d): Array of high prices.
        low (Array1d): Array of low prices.
        close (Array1d): Array of close prices.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array1d: Computed %K values.
    """
    lowest_low = generic_nb.rolling_min_1d_nb(low, window, minp=minp)
    highest_high = generic_nb.rolling_max_1d_nb(high, window, minp=minp)
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return stoch_k


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        minp=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def stoch_k_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    minp: tp.Optional[int] = None,
) -> tp.Array2d:
    """Calculate the Stochastic Oscillator %K for two-dimensional price arrays column-wise.

    Args:
        high (Array2d): Two-dimensional array of high prices.
        low (Array2d): Two-dimensional array of low prices.
        close (Array2d): Two-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        minp (Optional[int]): Minimum number of observations required.

    Returns:
        Array2d: Computed %K values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))

    stoch_k = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        stoch_k[:, col] = stoch_k_1d_nb(
            high=high[:, col],
            low=low[:, col],
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            minp=minp,
        )
    return stoch_k


@register_jitted(cache=True)
def stoch_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    fast_k_window: int = 14,
    slow_k_window: int = 3,
    slow_d_window: int = 3,
    wtype: int = generic_enums.WType.Simple,
    slow_k_wtype: tp.Optional[int] = None,
    slow_d_wtype: tp.Optional[int] = None,
    minp: tp.Optional[int] = None,
    fast_k_minp: tp.Optional[int] = None,
    slow_k_minp: tp.Optional[int] = None,
    slow_d_minp: tp.Optional[int] = None,
    adjust: bool = False,
    slow_k_adjust: tp.Optional[bool] = None,
    slow_d_adjust: tp.Optional[bool] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
    """Calculate the Stochastic Oscillator for one-dimensional price arrays.

    This function computes the fast %K, slow %K, and slow %D values using moving averages.

    Args:
        high (Array1d): Array of high prices.
        low (Array1d): Array of low prices.
        close (Array1d): Array of close prices.
        fast_k_window (int): Window size for fast %K calculation.
        slow_k_window (int): Window size for the slow %K moving average.
        slow_d_window (int): Window size for the slow %D moving average.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        slow_k_wtype (Optional[int]): Weighting type for the slow %K moving average.

            See `vectorbtpro.generic.enums.WType`. Defaults to `wtype` if not provided.
        slow_d_wtype (Optional[int]): Weighting type for the slow %D moving average.

            See `vectorbtpro.generic.enums.WType`. Defaults to `wtype` if not provided.
        minp (Optional[int]): Minimum number of observations required.
        fast_k_minp (Optional[int]): Minimum periods for fast %K calculation.

            Falls back to `minp` if not specified.
        slow_k_minp (Optional[int]): Minimum periods for slow %K calculation.

            Falls back to `minp` if not specified.
        slow_d_minp (Optional[int]): Minimum periods for slow %D calculation.

            Falls back to `minp` if not specified.
        adjust (bool): Flag indicating whether to adjust weights.
        slow_k_adjust (Optional[bool]): Flag for adjusting the slow %K moving average.

            Defaults to `adjust` if not provided.
        slow_d_adjust (Optional[bool]): Flag for adjusting the slow %D moving average.

            Defaults to `adjust` if not provided.

    Returns:
        Tuple[Array1d, Array1d, Array1d]: Tuple containing fast %K, slow %K, and slow %D values.
    """
    if slow_k_wtype is not None:
        slow_k_wtype_ = slow_k_wtype
    else:
        slow_k_wtype_ = wtype
    if slow_d_wtype is not None:
        slow_d_wtype_ = slow_d_wtype
    else:
        slow_d_wtype_ = wtype
    if fast_k_minp is not None:
        fast_k_minp_ = fast_k_minp
    else:
        fast_k_minp_ = minp
    if slow_k_minp is not None:
        slow_k_minp_ = slow_k_minp
    else:
        slow_k_minp_ = minp
    if slow_d_minp is not None:
        slow_d_minp_ = slow_d_minp
    else:
        slow_d_minp_ = minp
    if slow_k_adjust is not None:
        slow_k_adjust_ = slow_k_adjust
    else:
        slow_k_adjust_ = adjust
    if slow_d_adjust is not None:
        slow_d_adjust_ = slow_d_adjust
    else:
        slow_d_adjust_ = adjust
    fast_k = stoch_k_1d_nb(high, low, close, window=fast_k_window, minp=fast_k_minp_)
    slow_k = ma_1d_nb(
        fast_k, window=slow_k_window, wtype=slow_k_wtype_, minp=slow_k_minp_, adjust=slow_k_adjust_
    )
    slow_d = ma_1d_nb(
        slow_k, window=slow_d_window, wtype=slow_d_wtype_, minp=slow_d_minp_, adjust=slow_d_adjust_
    )
    return fast_k, slow_k, slow_d


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        fast_k_window=base_ch.FlexArraySlicer(),
        slow_k_window=base_ch.FlexArraySlicer(),
        slow_d_window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        slow_k_wtype=base_ch.FlexArraySlicer(),
        slow_d_wtype=base_ch.FlexArraySlicer(),
        minp=None,
        fast_k_minp=None,
        slow_k_minp=None,
        slow_d_minp=None,
        adjust=None,
        slow_k_adjust=None,
        slow_d_adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def stoch_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    fast_k_window: tp.FlexArray1dLike = 14,
    slow_k_window: tp.FlexArray1dLike = 3,
    slow_d_window: tp.FlexArray1dLike = 3,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    slow_k_wtype: tp.Optional[tp.FlexArray1dLike] = None,
    slow_d_wtype: tp.Optional[tp.FlexArray1dLike] = None,
    minp: tp.Optional[int] = None,
    fast_k_minp: tp.Optional[int] = None,
    slow_k_minp: tp.Optional[int] = None,
    slow_d_minp: tp.Optional[int] = None,
    adjust: bool = False,
    slow_k_adjust: tp.Optional[bool] = None,
    slow_d_adjust: tp.Optional[bool] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Calculate the Stochastic Oscillator for two-dimensional price arrays column-wise.

    This function computes the fast %K, slow %K, and slow %D values for each column using `stoch_1d_nb`.

    Args:
        high (Array2d): Two-dimensional array of high prices.
        low (Array2d): Two-dimensional array of low prices.
        close (Array2d): Two-dimensional array of close prices.
        fast_k_window (FlexArray1dLike): Window size for fast %K calculation.

            Provided as a scalar or per column.
        slow_k_window (FlexArray1dLike): Window size for the slow %K moving average.

            Provided as a scalar or per column.
        slow_d_window (FlexArray1dLike): Window size for the slow %D moving average.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        slow_k_wtype (Optional[FlexArray1dLike]): Weighting type for the slow %K moving average.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`. Uses `wtype` if not provided.
        slow_d_wtype (Optional[FlexArray1dLike]): Weighting type for the slow %D moving average.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`. Uses `wtype` if not provided.
        minp (Optional[int]): Minimum number of observations required.
        fast_k_minp (Optional[int]): Minimum periods for fast %K calculation.

            Falls back to `minp` if not specified.
        slow_k_minp (Optional[int]): Minimum periods for slow %K calculation.

            Falls back to `minp` if not specified.
        slow_d_minp (Optional[int]): Minimum periods for slow %D calculation.

            Falls back to `minp` if not specified.
        adjust (bool): Flag indicating whether to adjust weights.
        slow_k_adjust (Optional[bool]): Flag for adjusting the slow %K moving average.

            Defaults to `adjust` if not provided.
        slow_d_adjust (Optional[bool]): Flag for adjusting the slow %D moving average.

            Defaults to `adjust` if not provided.

    Returns:
        Tuple[Array2d, Array2d, Array2d]: Tuple containing fast %K, slow %K, and slow %D values for each column.

    !!! tip
        This function is parallelizable.
    """
    fast_k_window_ = to_1d_array_nb(np.asarray(fast_k_window))
    slow_k_window_ = to_1d_array_nb(np.asarray(slow_k_window))
    slow_d_window_ = to_1d_array_nb(np.asarray(slow_d_window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    if slow_k_wtype is not None:
        slow_k_wtype_ = to_1d_array_nb(np.asarray(slow_k_wtype))
    else:
        slow_k_wtype_ = wtype_
    if slow_d_wtype is not None:
        slow_d_wtype_ = to_1d_array_nb(np.asarray(slow_d_wtype))
    else:
        slow_d_wtype_ = wtype_

    fast_k = np.empty(close.shape, dtype=float_)
    slow_k = np.empty(close.shape, dtype=float_)
    slow_d = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        fast_k[:, col], slow_k[:, col], slow_d[:, col] = stoch_1d_nb(
            high=high[:, col],
            low=low[:, col],
            close=close[:, col],
            fast_k_window=flex_select_1d_nb(fast_k_window_, col),
            slow_k_window=flex_select_1d_nb(slow_k_window_, col),
            slow_d_window=flex_select_1d_nb(slow_d_window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            slow_k_wtype=flex_select_1d_nb(slow_k_wtype_, col),
            slow_d_wtype=flex_select_1d_nb(slow_d_wtype_, col),
            minp=minp,
            fast_k_minp=fast_k_minp,
            slow_k_minp=slow_k_minp,
            slow_d_minp=slow_d_minp,
            adjust=adjust,
            slow_k_adjust=slow_k_adjust,
            slow_d_adjust=slow_d_adjust,
        )
    return fast_k, slow_k, slow_d


# ############# MACD ############# #


@register_jitted(cache=True)
def macd_1d_nb(
    close: tp.Array1d,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    wtype: int = generic_enums.WType.Exp,
    macd_wtype: tp.Optional[int] = None,
    signal_wtype: tp.Optional[int] = None,
    minp: tp.Optional[int] = None,
    macd_minp: tp.Optional[int] = None,
    signal_minp: tp.Optional[int] = None,
    adjust: bool = False,
    macd_adjust: tp.Optional[bool] = None,
    signal_adjust: tp.Optional[bool] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Calculate MACD and signal line for a 1-D series of close prices.

    Compute the fast and slow moving averages, derive the MACD as their difference,
    and calculate the signal line as the moving average of the MACD.

    Args:
        close (Array1d): 1-D array of close prices.
        fast_window (int): Window size for computing the fast moving average.
        slow_window (int): Window size for computing the slow moving average.
        signal_window (int): Window size for computing the signal moving average.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        macd_wtype (Optional[int]): Weighting type for MACD computation.

            See `vectorbtpro.generic.enums.WType`. Uses `wtype` if not provided.
        signal_wtype (Optional[int]): Weighting type for signal computation.

            See `vectorbtpro.generic.enums.WType`. Uses `wtype` if not provided.
        minp (Optional[int]): Minimum number of observations required.
        macd_minp (Optional[int]): Minimum period for the MACD moving average.

            Uses `minp` if not provided.
        signal_minp (Optional[int]): Minimum period for the signal moving average.

            Uses `minp` if not provided.
        adjust (bool): Flag indicating whether to adjust weights.
        macd_adjust (Optional[bool]): Alternative flag for MACD moving average adjustment.

            Uses `adjust` if not provided.
        signal_adjust (Optional[bool]): Alternative flag for signal moving average adjustment.

            Uses `adjust` if not provided.

    Returns:
        Tuple[Array1d, Array1d]: Tuple where the first element is the MACD and the second is the signal line.
    """
    if macd_wtype is not None:
        macd_wtype_ = macd_wtype
    else:
        macd_wtype_ = wtype
    if signal_wtype is not None:
        signal_wtype_ = signal_wtype
    else:
        signal_wtype_ = wtype
    if macd_minp is not None:
        macd_minp_ = macd_minp
    else:
        macd_minp_ = minp
    if signal_minp is not None:
        signal_minp_ = signal_minp
    else:
        signal_minp_ = minp
    if macd_adjust is not None:
        macd_adjust_ = macd_adjust
    else:
        macd_adjust_ = adjust
    if signal_adjust is not None:
        signal_adjust_ = signal_adjust
    else:
        signal_adjust_ = adjust

    fast_ma = ma_1d_nb(
        close, window=fast_window, wtype=macd_wtype_, minp=macd_minp_, adjust=macd_adjust_
    )
    slow_ma = ma_1d_nb(
        close, window=slow_window, wtype=macd_wtype_, minp=macd_minp_, adjust=macd_adjust_
    )
    macd = fast_ma - slow_ma
    signal = ma_1d_nb(
        macd, window=signal_window, wtype=signal_wtype_, minp=signal_minp_, adjust=signal_adjust_
    )
    return macd, signal


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        fast_window=base_ch.FlexArraySlicer(),
        slow_window=base_ch.FlexArraySlicer(),
        signal_window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        macd_wtype=base_ch.FlexArraySlicer(),
        signal_wtype=base_ch.FlexArraySlicer(),
        minp=None,
        macd_minp=None,
        signal_minp=None,
        adjust=None,
        macd_adjust=None,
        signal_adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def macd_nb(
    close: tp.Array2d,
    fast_window: tp.FlexArray1dLike = 12,
    slow_window: tp.FlexArray1dLike = 26,
    signal_window: tp.FlexArray1dLike = 9,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Exp,
    macd_wtype: tp.Optional[tp.FlexArray1dLike] = None,
    signal_wtype: tp.Optional[tp.FlexArray1dLike] = None,
    minp: tp.Optional[int] = None,
    macd_minp: tp.Optional[int] = None,
    signal_minp: tp.Optional[int] = None,
    adjust: bool = False,
    macd_adjust: tp.Optional[bool] = None,
    signal_adjust: tp.Optional[bool] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Calculate 2-D MACD and signal lines for multiple price series.

    Compute the fast and slow moving averages along each column, derive the MACD as their difference,
    and calculate the signal line as the moving average of the MACD for each column.

    Args:
        close (Array2d): 2-D array of close prices.
        fast_window (FlexArray1dLike): Window size for computing the fast moving average.

            Provided as a scalar or per column.
        slow_window (FlexArray1dLike): Window size for computing the slow moving average.

            Provided as a scalar or per column.
        signal_window (FlexArray1dLike): Window size for computing the signal moving average.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        macd_wtype (Optional[FlexArray1dLike]): Weighting type for MACD computation.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`. Uses `wtype` if not provided.
        signal_wtype (Optional[FlexArray1dLike]): Weighting type for signal computation.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`. Uses `wtype` if not provided.
        minp (Optional[int]): Minimum number of observations required.
        macd_minp (Optional[int]): Minimum period for the MACD moving average.

            Uses `minp` if not provided.
        signal_minp (Optional[int]): Minimum period for the signal moving average.

            Uses `minp` if not provided.
        adjust (bool): Flag indicating whether to adjust weights.
        macd_adjust (Optional[bool]): Alternative flag for MACD moving average adjustment.

            Uses `adjust` if not provided.
        signal_adjust (Optional[bool]): Alternative flag for signal moving average adjustment.

            Uses `adjust` if not provided.

    Returns:
        Tuple[Array2d, Array2d]: Tuple where the first element contains MACD values and
            the second contains the signal line.

    !!! tip
        This function is parallelizable.
    """
    fast_window_ = to_1d_array_nb(np.asarray(fast_window))
    slow_window_ = to_1d_array_nb(np.asarray(slow_window))
    signal_window_ = to_1d_array_nb(np.asarray(signal_window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    if macd_wtype is not None:
        macd_wtype_ = to_1d_array_nb(np.asarray(macd_wtype))
    else:
        macd_wtype_ = wtype_
    if signal_wtype is not None:
        signal_wtype_ = to_1d_array_nb(np.asarray(signal_wtype))
    else:
        signal_wtype_ = wtype_

    macd = np.empty(close.shape, dtype=float_)
    signal = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        macd[:, col], signal[:, col] = macd_1d_nb(
            close=close[:, col],
            fast_window=flex_select_1d_nb(fast_window_, col),
            slow_window=flex_select_1d_nb(slow_window_, col),
            signal_window=flex_select_1d_nb(signal_window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            macd_wtype=flex_select_1d_nb(macd_wtype_, col),
            signal_wtype=flex_select_1d_nb(signal_wtype_, col),
            minp=minp,
            macd_minp=macd_minp,
            signal_minp=signal_minp,
            adjust=adjust,
            macd_adjust=macd_adjust,
            signal_adjust=signal_adjust,
        )
    return macd, signal


@register_jitted(cache=True)
def macd_hist_1d_nb(macd: tp.Array1d, signal: tp.Array1d) -> tp.Array1d:
    """Calculate MACD histogram for a 1-D series.

    Compute the difference between the MACD and the signal line for each element.

    Args:
        macd (Array1d): 1-D array of MACD values.
        signal (Array1d): 1-D array of signal line values.

    Returns:
        Array1d: 1-D array representing the MACD histogram.
    """
    return macd - signal


@register_chunkable(
    size=ch.ArraySizer(arg_query="macd", axis=1),
    arg_take_spec=dict(
        macd=ch.ArraySlicer(axis=1),
        signal=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def macd_hist_nb(macd: tp.Array2d, signal: tp.Array2d) -> tp.Array2d:
    """Calculate 2-D MACD histogram for multiple price series.

    Compute the difference between the MACD and signal line for each element along each column.

    Args:
        macd (Array2d): 2-D array of MACD values.
        signal (Array2d): 2-D array of signal line values.

    Returns:
        Array2d: 2-D array representing the MACD histogram values.

    !!! tip
        This function is parallelizable.
    """
    macd_hist = np.empty(macd.shape, dtype=float_)
    for col in prange(macd.shape[1]):
        macd_hist[:, col] = macd_hist_1d_nb(macd[:, col], signal[:, col])
    return macd_hist


@register_jitted(cache=True)
def iter_tr_nb(high: float, low: float, prev_close: float) -> float:
    """Calculate the True Range (TR) for a single iteration.

    Compute the True Range based on the current high, low, and previous close values.
    Returns NaN if any computed difference is NaN.

    Args:
        high (float): Current high price.
        low (float): Current low price.
        prev_close (float): Previous close price.

    Returns:
        float: True Range value.
    """
    tr0 = abs(high - low)
    tr1 = abs(high - prev_close)
    tr2 = abs(low - prev_close)
    if np.isnan(tr0) or np.isnan(tr1) or np.isnan(tr2):
        tr = np.nan
    else:
        tr = max(tr0, tr1, tr2)
    return tr


@register_jitted(cache=True)
def tr_1d_nb(high: tp.Array1d, low: tp.Array1d, close: tp.Array1d) -> tp.Array1d:
    """Calculate the True Range (TR) for a 1-D series of prices.

    Iterate over the close prices to compute the True Range using the corresponding high, low,
    and previous close values.

    Args:
        high (Array1d): 1-D array of high prices.
        low (Array1d): 1-D array of low prices.
        close (Array1d): 1-D array of close prices.

    Returns:
        Array1d: 1-D array of True Range values.
    """
    tr = np.empty(close.shape, dtype=float_)
    for i in range(close.shape[0]):
        tr[i] = iter_tr_nb(high[i], low[i], close[i - 1] if i > 0 else np.nan)
    return tr


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def tr_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """Calculate the True Range (TR) for each column in 2-D price arrays.

    Process each column independently to compute the True Range using high, low, and close prices.

    Args:
        high (Array2d): 2-D array of high prices.
        low (Array2d): 2-D array of low prices.
        close (Array2d): 2-D array of close prices.

    Returns:
        Array2d: 2-D array of True Range values computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    tr = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        tr[:, col] = tr_1d_nb(high[:, col], low[:, col], close[:, col])
    return tr


@register_jitted(cache=True)
def atr_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Average True Range (ATR).

    Calculates the True Range and Average True Range from high, low, and close price arrays.

    Args:
        high (Array1d): Array of high prices.
        low (Array1d): Array of low prices.
        close (Array1d): Array of close prices.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Tuple[Array1d, Array1d]: True Range and Average True Range.
    """
    tr = tr_1d_nb(high, low, close)
    atr = ma_1d_nb(tr, window, wtype=wtype, minp=minp, adjust=adjust)
    return tr, atr


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def atr_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """2-dim version of `atr_1d_nb`.

    Computes the True Range and Average True Range for 2-dimensional input arrays column-wise.

    Args:
        high (Array2d): 2-dimensional array of high prices.
        low (Array2d): 2-dimensional array of low prices.
        close (Array2d): 2-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Tuple[Array2d, Array2d]: True Range and Average True Range for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    tr = np.empty(close.shape, dtype=float_)
    atr = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        tr[:, col], atr[:, col] = atr_1d_nb(
            high[:, col],
            low[:, col],
            close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
        )
    return tr, atr


# ############# ADX ############# #


@register_jitted(cache=True)
def adx_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Average Directional Movement Index (ADX).

    Calculates the positive and negative directional indicators (+DI, -DI),
    the directional movement index (DX), and the ADX using high, low, and close price arrays.

    Args:
        high (Array1d): Array of high prices.
        low (Array1d): Array of low prices.
        close (Array1d): Array of close prices.
        window (int): Window size.
        wtype (int): Weighting type.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Tuple[Array1d, Array1d, Array1d, Array1d]: +DI, -DI, DX, and ADX values.
    """
    _, atr = atr_1d_nb(
        high,
        low,
        close,
        window=window,
        wtype=wtype,
        minp=minp,
        adjust=adjust,
    )
    dm_plus = np.empty(close.shape, dtype=float_)
    dm_minus = np.empty(close.shape, dtype=float_)
    for i in range(close.shape[0]):
        up_change = np.nan if i == 0 else high[i] - high[i - 1]
        down_change = np.nan if i == 0 else low[i - 1] - low[i]
        if up_change > down_change and up_change > 0:
            dm_plus[i] = up_change
        else:
            dm_plus[i] = 0.0
        if down_change > up_change and down_change > 0:
            dm_minus[i] = down_change
        else:
            dm_minus[i] = 0.0
    dm_plus_smoothed = ma_1d_nb(dm_plus, window, wtype=wtype, minp=minp, adjust=adjust)
    dm_minus_smoothed = ma_1d_nb(dm_minus, window, wtype=wtype, minp=minp, adjust=adjust)
    plus_di = 100 * dm_plus_smoothed / atr
    minus_di = 100 * dm_minus_smoothed / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = ma_1d_nb(dx, window, wtype=wtype, minp=minp, adjust=adjust)
    return plus_di, minus_di, dx, adx


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def adx_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Wilder,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """2-dim version of `adx_1d_nb`.

    Computes the average directional movement index components (+DI, -DI, DX, and ADX)
    for 2-dimensional input arrays column-wise.

    Args:
        high (Array2d): 2-dimensional array of high prices.
        low (Array2d): 2-dimensional array of low prices.
        close (Array2d): 2-dimensional array of close prices.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        wtype (FlexArray1dLike): Weighting type.

            Provided as a scalar or per column.

            See `vectorbtpro.generic.enums.WType`.
        minp (Optional[int]): Minimum number of observations required.
        adjust (bool): Flag indicating whether to adjust weights.

    Returns:
        Tuple[Array2d, Array2d, Array2d, Array2d]: +DI, -DI, DX, and ADX values for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))

    plus_di = np.empty(close.shape, dtype=float_)
    minus_di = np.empty(close.shape, dtype=float_)
    dx = np.empty(close.shape, dtype=float_)
    adx = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        plus_di[:, col], minus_di[:, col], dx[:, col], adx[:, col] = adx_1d_nb(
            high[:, col],
            low[:, col],
            close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            minp=minp,
            adjust=adjust,
        )
    return plus_di, minus_di, dx, adx


# ############# OBV ############# #


@register_jitted(cache=True)
def obv_1d_nb(close: tp.Array1d, volume: tp.Array1d) -> tp.Array1d:
    """On-Balance Volume (OBV).

    Calculates the on-balance volume (OBV) from close prices and trading volumes.

    Args:
        close (Array1d): Array of close prices.
        volume (Array1d): Array of trading volumes.

    Returns:
        Array1d: Computed on-balance volume.
    """
    obv = np.empty(close.shape, dtype=float_)
    cumsum = 0.0
    for i in range(close.shape[0]):
        prev_close = close[i - 1] if i > 0 else np.nan
        if close[i] < prev_close:
            value = -volume[i]
        else:
            value = volume[i]
        if not np.isnan(value):
            cumsum += value
        obv[i] = cumsum
    return obv


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        volume=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def obv_nb(close: tp.Array2d, volume: tp.Array2d) -> tp.Array2d:
    """2-dim version of `obv_1d_nb`.

    Calculates the on-balance volume (OBV) for 2-dimensional input arrays column-wise.

    Args:
        close (Array2d): 2-dimensional array of close prices.
        volume (Array2d): 2-dimensional array of trading volumes.

    Returns:
        Array2d: Computed on-balance volume for each column.

    !!! tip
        This function is parallelizable.
    """
    obv = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        obv[:, col] = obv_1d_nb(close[:, col], volume[:, col])
    return obv


# ############# OLS ############# #


@register_jitted(cache=True)
def ols_1d_nb(
    x: tp.Array1d,
    y: tp.Array1d,
    window: int = 14,
    norm_window: tp.Optional[int] = None,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
    with_zscore: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
    """Compute rolling ordinary least squares (OLS) regression between 1-dimensional arrays.

    Args:
        x (Array1d): 1-dimensional array of independent variable values.
        y (Array1d): 1-dimensional array of dependent variable values.
        window (int): Window size.
        norm_window (Optional[int]): Window size for error normalization.

            Defaults to `window` if None.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.
        with_zscore (bool): Whether to compute and return the z-score of regression errors.

    Returns:
        Tuple[Array1d, Array1d, Array1d]: Arrays containing slopes, intercepts, and z-scores.
    """
    if norm_window is not None:
        norm_window_ = norm_window
    else:
        norm_window_ = window
    slope, intercept = generic_nb.rolling_ols_1d_nb(x, y, window, minp=minp)
    if with_zscore:
        pred = intercept + slope * x
        error = y - pred
        error_mean = generic_nb.rolling_mean_1d_nb(error, norm_window_, minp=minp)
        error_std = generic_nb.rolling_std_1d_nb(error, norm_window_, minp=minp, ddof=ddof)
        zscore = (error - error_mean) / error_std
    else:
        zscore = np.full(x.shape, np.nan, dtype=float_)
    return slope, intercept, zscore


@register_chunkable(
    size=ch.ArraySizer(arg_query="x", axis=1),
    arg_take_spec=dict(
        x=ch.ArraySlicer(axis=1),
        y=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        norm_window=base_ch.FlexArraySlicer(),
        minp=None,
        ddof=None,
        with_zscore=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ols_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    norm_window: tp.Optional[tp.FlexArray1dLike] = None,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
    with_zscore: bool = True,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Compute rolling ordinary least squares (OLS) regression for 2-dimensional arrays.

    This function applies a 1-dimensional OLS regression on each column.

    Args:
        x (Array2d): 2-dimensional array of independent variable values.
        y (Array2d): 2-dimensional array of dependent variable values.
        window (FlexArray1dLike): Window size.

            Provided as a scalar or per column.
        norm_window (Optional[FlexArray1dLike]): Window size for error normalization.

            Provided as a scalar or per column.

            Defaults to `window` if None.
        minp (Optional[int]): Minimum number of observations required.
        ddof (int): Delta degrees of freedom.
        with_zscore (bool): Whether to compute and return the z-score of regression errors.

    Returns:
        Tuple[Array2d, Array2d, Array2d]: Arrays of slopes, intercepts, and z-scores for each column.

    !!! tip
        This function is parallelizable.
    """
    window_ = to_1d_array_nb(np.asarray(window))
    if norm_window is not None:
        norm_window_ = to_1d_array_nb(np.asarray(norm_window))
    else:
        norm_window_ = window_

    slope = np.empty(x.shape, dtype=float_)
    intercept = np.empty(x.shape, dtype=float_)
    zscore = np.empty(x.shape, dtype=float_)
    for col in prange(x.shape[1]):
        slope[:, col], intercept[:, col], zscore[:, col] = ols_1d_nb(
            x[:, col],
            y[:, col],
            window=flex_select_1d_nb(window_, col),
            norm_window=flex_select_1d_nb(norm_window_, col),
            minp=minp,
            ddof=ddof,
            with_zscore=with_zscore,
        )
    return slope, intercept, zscore


@register_jitted(cache=True)
def ols_pred_1d_nb(x: tp.Array1d, slope: tp.Array1d, intercept: tp.Array1d) -> tp.Array1d:
    """Compute OLS prediction for 1-dimensional arrays.

    Args:
        x (Array1d): 1-dimensional array of independent variable values.
        slope (Array1d): 1-dimensional array of slope values.
        intercept (Array1d): 1-dimensional array of intercept values.

    Returns:
        Array1d: Predicted values computed as `intercept + slope * x`.
    """
    return intercept + slope * x


@register_chunkable(
    size=ch.ArraySizer(arg_query="x", axis=1),
    arg_take_spec=dict(
        x=ch.ArraySlicer(axis=1),
        slope=ch.ArraySlicer(axis=1),
        intercept=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ols_pred_nb(x: tp.Array2d, slope: tp.Array2d, intercept: tp.Array2d) -> tp.Array2d:
    """Compute OLS predictions for 2-dimensional arrays.

    This function applies `ols_pred_1d_nb` to compute predictions column-wise.

    Args:
        x (Array2d): 2-dimensional array of independent variable values.
        slope (Array2d): 2-dimensional array of slope values.
        intercept (Array2d): 2-dimensional array of intercept values.

    Returns:
        Array2d: 2-dimensional array of predicted values.

    !!! tip
        This function is parallelizable.
    """
    pred = np.empty(x.shape, dtype=float_)
    for col in prange(x.shape[1]):
        pred[:, col] = ols_pred_1d_nb(x[:, col], slope[:, col], intercept[:, col])
    return pred


@register_jitted(cache=True)
def ols_error_1d_nb(y: tp.Array1d, pred: tp.Array1d) -> tp.Array1d:
    """Compute the error between observed and predicted values for OLS regression.

    Args:
        y (Array1d): 1-dimensional array of observed values.
        pred (Array1d): 1-dimensional array of predicted values.

    Returns:
        Array1d: Errors computed as the difference between y and pred.
    """
    return y - pred


@register_chunkable(
    size=ch.ArraySizer(arg_query="y", axis=1),
    arg_take_spec=dict(
        y=ch.ArraySlicer(axis=1),
        pred=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ols_error_nb(y: tp.Array2d, pred: tp.Array2d) -> tp.Array2d:
    """Compute OLS regression errors for 2-dimensional arrays.

    This function applies error computation column-wise using `ols_error_1d_nb`.

    Args:
        y (Array2d): 2-dimensional array of observed values.
        pred (Array2d): 2-dimensional array of predicted values.

    Returns:
        Array2d: 2-dimensional array of errors for each column.

    !!! tip
        This function is parallelizable.
    """
    error = np.empty(y.shape, dtype=float_)
    for col in prange(y.shape[1]):
        error[:, col] = ols_error_1d_nb(y[:, col], pred[:, col])
    return error


@register_jitted(cache=True)
def ols_angle_1d_nb(slope: tp.Array1d) -> tp.Array1d:
    """Compute the angle in degrees from OLS regression slopes for 1-dimensional arrays.

    Args:
        slope (Array1d): 1-dimensional array of slope values.

    Returns:
        Array1d: Angles in degrees computed from the slope values.
    """
    return np.arctan(slope) * 180 / np.pi


@register_chunkable(
    size=ch.ArraySizer(arg_query="slope", axis=1),
    arg_take_spec=dict(
        slope=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def ols_angle_nb(slope: tp.Array2d) -> tp.Array2d:
    """Compute the angles in degrees from OLS regression slopes for 2-dimensional arrays.

    This function applies `ols_angle_1d_nb` column-wise to compute the angles.

    Args:
        slope (Array2d): 2-dimensional array of slope values.

    Returns:
        Array2d: 2-dimensional array of angles in degrees for each column.

    !!! tip
        This function is parallelizable.
    """
    angle = np.empty(slope.shape, dtype=float_)
    for col in prange(slope.shape[1]):
        angle[:, col] = ols_angle_1d_nb(slope[:, col])
    return angle


@register_jitted(cache=True)
def typical_price_1d_nb(high: tp.Array1d, low: tp.Array1d, close: tp.Array1d) -> tp.Array1d:
    """Compute the typical price from high, low, and close prices for 1-dimensional arrays.

    Args:
        high (Array1d): 1-dimensional array of high prices.
        low (Array1d): 1-dimensional array of low prices.
        close (Array1d): 1-dimensional array of close prices.

    Returns:
        Array1d: Typical price calculated as (high + low + close) / 3.
    """
    return (high + low + close) / 3


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def typical_price_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """Compute the typical price for 2-dimensional arrays of high, low, and close prices.

    This function applies `typical_price_1d_nb` column-wise to compute the typical price.

    Args:
        high (Array2d): 2-dimensional array of high prices.
        low (Array2d): 2-dimensional array of low prices.
        close (Array2d): 2-dimensional array of close prices.

    Returns:
        Array2d: 2-dimensional array of typical prices for each column.

    !!! tip
        This function is parallelizable.
    """
    typical_price = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        typical_price[:, col] = typical_price_1d_nb(high[:, col], low[:, col], close[:, col])
    return typical_price


@register_jitted(cache=True)
def vwap_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    volume: tp.Array1d,
    group_lens: tp.GroupLens,
) -> tp.Array1d:
    """Compute the Volume-Weighted Average Price (VWAP) for 1-dimensional arrays.

    The calculation uses typical prices weighted by volume, aggregated over groups defined by `group_lens`.

    Args:
        high (Array1d): 1-dimensional array of high prices.
        low (Array1d): 1-dimensional array of low prices.
        close (Array1d): 1-dimensional array of close prices.
        volume (Array1d): 1-dimensional array of trading volumes.
        group_lens (GroupLens): Array defining the number of columns in each group.

    Returns:
        Array1d: Computed VWAP values for each element.
    """
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    out = np.full(volume.shape, np.nan, dtype=float_)

    typical_price = typical_price_1d_nb(high, low, close)
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        to_i = group_end_idxs[group]
        nom_cumsum = 0
        denum_cumsum = 0
        for i in range(from_i, to_i):
            nom_cumsum += volume[i] * typical_price[i]
            denum_cumsum += volume[i]
            if denum_cumsum == 0:
                out[i] = np.nan
            else:
                out[i] = nom_cumsum / denum_cumsum
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        volume=ch.ArraySlicer(axis=1),
        group_lens=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def vwap_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    volume: tp.Array2d,
    group_lens: tp.GroupLens,
) -> tp.Array2d:
    """Compute the volume weighted average price (VWAP) for 2D arrays.

    Apply the 1D computation from `vwap_1d_nb` to each column independently.

    Args:
        high (Array2d): Array of high prices.
        low (Array2d): Array of low prices.
        close (Array2d): Array of close prices.
        volume (Array2d): Array of traded volumes.
        group_lens (GroupLens): Array defining the number of columns in each group.

    Returns:
        Array2d: 2D array containing the calculated VWAP values.

    !!! tip
        This function is parallelizable.
    """
    vwap = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        vwap[:, col] = vwap_1d_nb(
            high[:, col],
            low[:, col],
            close[:, col],
            volume[:, col],
            group_lens,
        )
    return vwap


# ############# PIVOTINFO ############# #


@register_jitted(cache=True)
def pivot_info_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Compute pivot information for 1D arrays.

    Determine pivot types and their corresponding indices based on threshold values.

    Args:
        high (Array1d): Array of high prices.
        low (Array1d): Array of low prices.
        up_th (FlexArray1dLike): Upper threshold for detecting peaks.

            Provided as a scalar or per row.
        down_th (FlexArray1dLike): Lower threshold for detecting valleys.

            Provided as a scalar or per row.

    Returns:
        Tuple[Array1d, Array1d, Array1d, Array1d]: Tuple containing:

            * Confirmed pivot types.
            * Indices of confirmed pivots.
            * Last pivot types.
            * Indices of the last pivots.
    """
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    conf_pivot = np.empty(high.shape, dtype=int_)
    conf_idx = np.empty(high.shape, dtype=int_)
    last_pivot = np.empty(high.shape, dtype=int_)
    last_idx = np.empty(high.shape, dtype=int_)

    _conf_pivot = 0
    _conf_idx = -1
    _conf_value = np.nan
    _last_pivot = 0
    _last_idx = -1
    _last_value = np.nan
    first_valid_idx = -1
    for i in range(high.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            if first_valid_idx == -1:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, i))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, i))
                if np.isnan(_up_th) or np.isnan(_down_th):
                    conf_pivot[i] = _conf_pivot
                    conf_idx[i] = _conf_idx
                    last_pivot[i] = _last_pivot
                    last_idx[i] = _last_idx
                    continue

                first_valid_idx = i
            if _last_idx == -1:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, first_valid_idx))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, first_valid_idx))
                if not np.isnan(_up_th) and high[i] >= low[first_valid_idx] * _up_th:
                    if not np.isnan(_down_th) and low[i] <= high[first_valid_idx] * _down_th:
                        pass  # wait
                    else:
                        _conf_pivot = Pivot.Valley
                        _conf_idx = first_valid_idx
                        _conf_value = low[first_valid_idx]
                        _last_pivot = Pivot.Peak
                        _last_idx = i
                        _last_value = high[i]
                if not np.isnan(_down_th) and low[i] <= high[first_valid_idx] * _down_th:
                    if not np.isnan(_up_th) and high[i] >= low[first_valid_idx] * _up_th:
                        pass  # wait
                    else:
                        _conf_pivot = Pivot.Peak
                        _conf_idx = first_valid_idx
                        _conf_value = high[first_valid_idx]
                        _last_pivot = Pivot.Valley
                        _last_idx = i
                        _last_value = low[i]
            else:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, _last_idx))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, _last_idx))
                if _last_pivot == Pivot.Valley:
                    if (
                        not np.isnan(_last_value)
                        and not np.isnan(_up_th)
                        and high[i] >= _last_value * _up_th
                    ):
                        _conf_pivot = _last_pivot
                        _conf_idx = _last_idx
                        _conf_value = _last_value
                        _last_pivot = Pivot.Peak
                        _last_idx = i
                        _last_value = high[i]
                    elif np.isnan(_last_value) or low[i] < _last_value:
                        _last_idx = i
                        _last_value = low[i]
                elif _last_pivot == Pivot.Peak:
                    if (
                        not np.isnan(_last_value)
                        and not np.isnan(_down_th)
                        and low[i] <= _last_value * _down_th
                    ):
                        _conf_pivot = _last_pivot
                        _conf_idx = _last_idx
                        _conf_value = _last_value
                        _last_pivot = Pivot.Valley
                        _last_idx = i
                        _last_value = low[i]
                    elif np.isnan(_last_value) or high[i] > _last_value:
                        _last_idx = i
                        _last_value = high[i]

        conf_pivot[i] = _conf_pivot
        conf_idx[i] = _conf_idx
        last_pivot[i] = _last_pivot
        last_idx[i] = _last_idx

    return conf_pivot, conf_idx, last_pivot, last_idx


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
def pivot_info_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Compute pivot information for 2D arrays.

    Apply the 1D pivot information computation from `pivot_info_1d_nb` to each column independently.

    Args:
        high (Array2d): Array of high prices.
        low (Array2d): Array of low prices.
        up_th (FlexArray2dLike): Upper threshold for detecting peaks.

            Provided as a scalar, or per row, column, or element.
        down_th (FlexArray2dLike): Lower threshold for detecting valleys.

            Provided as a scalar, or per row, column, or element.

    Returns:
        Tuple[Array2d, Array2d, Array2d, Array2d]: Tuple containing:

            * Confirmed pivot types.
            * Indices of confirmed pivots.
            * Last pivot types.
            * Indices of the last pivots.

    !!! tip
        This function is parallelizable.
    """
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))

    conf_pivot = np.empty(high.shape, dtype=int_)
    conf_idx = np.empty(high.shape, dtype=int_)
    last_pivot = np.empty(high.shape, dtype=int_)
    last_idx = np.empty(high.shape, dtype=int_)
    for col in prange(high.shape[1]):
        conf_pivot[:, col], conf_idx[:, col], last_pivot[:, col], last_idx[:, col] = (
            pivot_info_1d_nb(
                high[:, col],
                low[:, col],
                flex_select_col_nb(up_th_, col),
                flex_select_col_nb(down_th_, col),
            )
        )
    return conf_pivot, conf_idx, last_pivot, last_idx


@register_jitted(cache=True)
def pivot_value_1d_nb(
    high: tp.Array1d, low: tp.Array1d, last_pivot: tp.Array1d, last_idx: tp.Array1d
) -> tp.Array1d:
    """Compute pivot values for 1D arrays.

    Determine the pivot price based on the last pivot type and its corresponding index.

    Args:
        high (Array1d): Array of high prices.
        low (Array1d): Array of low prices.
        last_pivot (Array1d): Array indicating the type of the last pivot.
        last_idx (Array1d): Array of indices corresponding to the last pivot positions.

    Returns:
        Array1d: 1D array of pivot prices calculated from `high` or `low` based on the last pivot type.
    """
    pivot_value = np.empty(high.shape, dtype=float_)
    for i in range(high.shape[0]):
        if last_pivot[i] == Pivot.Peak:
            pivot_value[i] = high[last_idx[i]]
        elif last_pivot[i] == Pivot.Valley:
            pivot_value[i] = low[last_idx[i]]
        else:
            pivot_value[i] = np.nan
    return pivot_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        last_pivot=ch.ArraySlicer(axis=1),
        last_idx=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivot_value_nb(
    high: tp.Array2d, low: tp.Array2d, last_pivot: tp.Array2d, last_idx: tp.Array2d
) -> tp.Array2d:
    """Compute pivot values for 2D arrays.

    Apply the 1D pivot value computation from `pivot_value_1d_nb` to each column independently.

    Args:
        high (Array2d): Array of high prices.
        low (Array2d): Array of low prices.
        last_pivot (Array2d): Array indicating the type of the last pivot for each element.
        last_idx (Array2d): Array of indices corresponding to the last pivot positions.

    Returns:
        Array2d: 2D array containing the calculated pivot values for each column.

    !!! tip
        This function is parallelizable.
    """
    pivot_value = np.empty(high.shape, dtype=float_)
    for col in prange(high.shape[1]):
        pivot_value[:, col] = pivot_value_1d_nb(
            high[:, col], low[:, col], last_pivot[:, col], last_idx[:, col]
        )
    return pivot_value


@register_jitted(cache=True)
def pivots_1d_nb(
    conf_pivot: tp.Array1d, conf_idx: tp.Array1d, last_pivot: tp.Array1d
) -> tp.Array1d:
    """Return pivot values based on input configuration arrays.

    Args:
        conf_pivot (Array1d): Array of pivot configuration values.
        conf_idx (Array1d): Array of indices where pivot values should be assigned.
        last_pivot (Array1d): Array indicating the type of the last pivot.

    Returns:
        Array1d: Array of computed pivot values.

    !!! warning
        To be used in plotting only. Do not use it as an indicator!
    """
    pivots = np.zeros(conf_pivot.shape, dtype=int_)
    for i in range(conf_pivot.shape[0] - 1):
        pivots[conf_idx[i]] = conf_pivot[i]
    pivots[-1] = last_pivot[-1]
    return pivots


@register_chunkable(
    size=ch.ArraySizer(arg_query="conf_pivot", axis=1),
    arg_take_spec=dict(
        conf_pivot=ch.ArraySlicer(axis=1),
        conf_idx=ch.ArraySlicer(axis=1),
        last_pivot=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivots_nb(conf_pivot: tp.Array2d, conf_idx: tp.Array2d, last_pivot: tp.Array2d) -> tp.Array2d:
    """Return pivot values for a 2-dimensional array by applying `pivots_1d_nb` per column.

    Args:
        conf_pivot (Array2d): Array of pivot configuration values for each column.
        conf_idx (Array2d): Array of indices corresponding to pivot positions for each column.
        last_pivot (Array2d): Array indicating the type of the last pivot for each element.

    Returns:
        Array2d: Array containing the computed pivot values for each column.

    !!! tip
        This function is parallelizable.
    """
    pivots = np.empty(conf_pivot.shape, dtype=int_)
    for col in prange(conf_pivot.shape[1]):
        pivots[:, col] = pivots_1d_nb(conf_pivot[:, col], conf_idx[:, col], last_pivot[:, col])
    return pivots


@register_jitted(cache=True)
def modes_1d_nb(pivots: tp.Array1d) -> tp.Array1d:
    """Return mode values computed from pivot signals.

    Args:
        pivots (Array1d): Array of pivot values.

    Returns:
        Array1d: Array containing mode values corresponding to each pivot entry.

    !!! warning
        To be used in plotting only. Do not use it as an indicator!
    """
    modes = np.empty(pivots.shape, dtype=int_)
    mode = 0
    for i in range(pivots.shape[0]):
        if pivots[i] != 0:
            mode = -pivots[i]
        modes[i] = mode
    return modes


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        pivots=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def modes_nb(pivots: tp.Array2d) -> tp.Array2d:
    """Return 2-dimensional mode values by applying `modes_1d_nb` to each column.

    Args:
        pivots (Array2d): Array of pivot values arranged by columns.

    Returns:
        Array2d: Array of computed mode values for each column.

    !!! tip
        This function is parallelizable.
    """
    modes = np.empty(pivots.shape, dtype=int_)
    for col in prange(pivots.shape[1]):
        modes[:, col] = modes_1d_nb(pivots[:, col])
    return modes


# ############# SUPERTREND ############# #


@register_jitted(cache=True)
def iter_med_price_nb(high: float, low: float) -> float:
    """Return the median price for one iteration.

    Args:
        high (float): High price value.
        low (float): Low price value.

    Returns:
        float: Computed median price.
    """
    return (high + low) / 2


@register_jitted(cache=True)
def iter_basic_bands_nb(
    high: float, low: float, atr: float, multiplier: float
) -> tp.Tuple[float, float]:
    """Return the upper and lower basic band values for one iteration.

    Args:
        high (float): High price value.
        low (float): Low price value.
        atr (float): Average true range.
        multiplier (float): Multiplier applied to the ATR.

    Returns:
        Tuple[float, float]: Computed upper and lower band values.
    """
    med_price = iter_med_price_nb(high, low)
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower


@register_jitted(cache=True)
def final_basic_bands_nb(
    close: float,
    upper_basic: float,
    lower_basic: float,
    prev_close: float,
    prev_upper: float,
    prev_lower: float,
    prev_direction: int,
) -> tp.Tuple[float, float, float, int, float, float]:
    """Return final basic band values along with trend details for one iteration.

    Args:
        close (float): Current close price.
        upper_basic (float): Current basic upper band value.
        lower_basic (float): Current basic lower band value.
        prev_close (float): Previous close price.
        prev_upper (float): Previous upper band value.
        prev_lower (float): Previous lower band value.
        prev_direction (int): Previous direction.

    Returns:
        Tuple[float, float, float, int, float, float]: Tuple containing the adjusted
            upper band, lower band, trend, direction, long band, and short band values.
    """
    if np.isnan(prev_upper) or (upper_basic < prev_upper) or (prev_close > prev_upper):
        upper = upper_basic
    else:
        upper = prev_upper
    if np.isnan(prev_lower) or (lower_basic > prev_lower) or (prev_close < prev_lower):
        lower = lower_basic
    else:
        lower = prev_lower
    if close > upper:
        direction = 1
    elif close < lower:
        direction = -1
    else:
        direction = prev_direction
    if direction > 0:
        trend = long = lower
        short = np.nan
    else:
        trend = short = upper
        long = np.nan
    return upper, lower, trend, direction, long, short


@register_jitted(cache=True)
def supertrend_acc_nb(in_state: SuperTrendAIS) -> SuperTrendAOS:
    """Return an updated state for the supertrend indicator based on the provided input state.

    Args:
        in_state (SuperTrendAIS): Input state of type `vectorbtpro.indicators.enums.SuperTrendAIS`
            containing price, band, and trend information.

    Returns:
        SuperTrendAOS: Output state of type `vectorbtpro.indicators.enums.SuperTrendAOS`
            with updated band and trend values.
    """
    i = in_state.i
    high = in_state.high
    low = in_state.low
    close = in_state.close
    prev_close = in_state.prev_close
    prev_upper = in_state.prev_upper
    prev_lower = in_state.prev_lower
    prev_direction = in_state.prev_direction
    nobs = in_state.nobs
    weighted_avg = in_state.weighted_avg
    old_wt = in_state.old_wt
    period = in_state.period
    multiplier = in_state.multiplier

    tr = iter_tr_nb(high, low, prev_close)
    alpha = generic_nb.alpha_from_wilder_nb(period)
    ewm_mean_in_state = generic_enums.EWMMeanAIS(
        i=i,
        value=tr,
        old_wt=old_wt,
        weighted_avg=weighted_avg,
        nobs=nobs,
        alpha=alpha,
        minp=period,
        adjust=False,
    )
    ewm_mean_out_state = generic_nb.ewm_mean_acc_nb(ewm_mean_in_state)
    atr = ewm_mean_out_state.value
    upper, lower = iter_basic_bands_nb(high, low, atr, multiplier)
    if i == 0:
        trend, direction, long, short = np.nan, 1, np.nan, np.nan
    else:
        upper, lower, trend, direction, long, short = final_basic_bands_nb(
            close,
            upper,
            lower,
            prev_close,
            prev_upper,
            prev_lower,
            prev_direction,
        )
    return SuperTrendAOS(
        nobs=ewm_mean_out_state.nobs,
        weighted_avg=ewm_mean_out_state.weighted_avg,
        old_wt=ewm_mean_out_state.old_wt,
        upper=upper,
        lower=lower,
        trend=trend,
        direction=direction,
        long=long,
        short=short,
    )


@register_jitted(cache=True)
def supertrend_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    period: int = 7,
    multiplier: float = 3.0,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Return supertrend indicator arrays for a one-dimensional price series.

    Args:
        high (Array1d): Array of high price values.
        low (Array1d): Array of low price values.
        close (Array1d): Array of close price values.
        period (int): Period for the supertrend calculation.
        multiplier (float): Multiplier applied to the ATR.

    Returns:
        Tuple[Array1d, Array1d, Array1d, Array1d]: Tuple containing
            the trend, direction, long, and short arrays.
    """
    trend = np.empty(close.shape, dtype=float_)
    direction = np.empty(close.shape, dtype=int_)
    long = np.empty(close.shape, dtype=float_)
    short = np.empty(close.shape, dtype=float_)

    if close.shape[0] == 0:
        return trend, direction, long, short

    nobs = 0
    old_wt = 1.0
    weighted_avg = np.nan
    prev_upper = np.nan
    prev_lower = np.nan

    for i in range(close.shape[0]):
        in_state = SuperTrendAIS(
            i=i,
            high=high[i],
            low=low[i],
            close=close[i],
            prev_close=close[i - 1] if i > 0 else np.nan,
            prev_upper=prev_upper,
            prev_lower=prev_lower,
            prev_direction=direction[i - 1] if i > 0 else 1,
            nobs=nobs,
            weighted_avg=weighted_avg,
            old_wt=old_wt,
            period=period,
            multiplier=multiplier,
        )

        out_state = supertrend_acc_nb(in_state)

        nobs = out_state.nobs
        weighted_avg = out_state.weighted_avg
        old_wt = out_state.old_wt
        prev_upper = out_state.upper
        prev_lower = out_state.lower
        trend[i] = out_state.trend
        direction[i] = out_state.direction
        long[i] = out_state.long
        short[i] = out_state.short

    return trend, direction, long, short


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        period=base_ch.FlexArraySlicer(),
        multiplier=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def supertrend_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    period: tp.FlexArray1dLike = 7,
    multiplier: tp.FlexArray1dLike = 3.0,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Compute the 2-dim Supertrend indicator.

    Applies the one-dimensional `supertrend_1d_nb` to each column of the input arrays to compute
    the Supertrend indicator for multi-dimensional data.

    Args:
        high (Array2d): Array of high prices.
        low (Array2d): Array of low prices.
        close (Array2d): Array of close prices.
        period (FlexArray1dLike): Period for the indicator calculation.

            Provided as a scalar or per column.
        multiplier (FlexArray1dLike): Multiplier used to determine the volatility threshold.

            Provided as a scalar or per column.

    Returns:
        Tuple[Array2d, Array2d, Array2d, Array2d]: Tuple containing
            the trend, direction, long, and short arrays.

    !!! tip
        This function is parallelizable.
    """
    period_ = to_1d_array_nb(np.asarray(period))
    multiplier_ = to_1d_array_nb(np.asarray(multiplier))

    trend = np.empty(close.shape, dtype=float_)
    direction = np.empty(close.shape, dtype=int_)
    long = np.empty(close.shape, dtype=float_)
    short = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        trend[:, col], direction[:, col], long[:, col], short[:, col] = supertrend_1d_nb(
            high[:, col],
            low[:, col],
            close[:, col],
            period=flex_select_1d_nb(period_, col),
            multiplier=flex_select_1d_nb(multiplier_, col),
        )
    return trend, direction, long, short


# ############# SIGDET ############# #


@register_jitted(cache=True)
def signal_detection_1d_nb(
    close: tp.Array1d,
    lag: int = 14,
    factor: tp.FlexArray1dLike = 1.0,
    influence: tp.FlexArray1dLike = 1.0,
    up_factor: tp.Optional[tp.FlexArray1dLike] = None,
    down_factor: tp.Optional[tp.FlexArray1dLike] = None,
    mean_influence: tp.Optional[tp.FlexArray1dLike] = None,
    std_influence: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
    """Detect signals from a one-dimensional close price series.

    Analyzes the input array using a moving window defined by `lag` to detect upward or downward
    signals based on dynamic thresholding with influence factors. Computed signals and corresponding
    upper/lower bands are returned.

    Args:
        close (Array1d): Array of close prices.

        lag (int): Window size for computing moving averages and standard deviations.
        factor (FlexArray1dLike): Factor to determine the threshold for signal detection.

            Provided as a scalar or per row.
        influence (FlexArray1dLike): Influence factor for updating the filtering process.

            Provided as a scalar or per row.
        up_factor (Optional[FlexArray1dLike]): Factor for the upward threshold.

            Provided as a scalar or per row.

            If None, uses `factor`.
        down_factor (Optional[FlexArray1dLike]): Factor for the downward threshold.

            Provided as a scalar or per row.

            If None, uses `factor`.
        mean_influence (Optional[FlexArray1dLike]): Influence factor for updating the moving average filter.

            Provided as a scalar or per column.

            If None, uses `influence`.
        std_influence (Optional[FlexArray1dLike]): Influence factor for updating the standard deviation filter.

            Provided as a scalar or per column.

            If None, uses `influence`.

    Returns:
        Tuple[Array1d, Array1d, Array1d]: Tuple containing
            the signal array, upper band, and lower band.
    """
    factor_ = to_1d_array_nb(np.asarray(factor))
    influence_ = to_1d_array_nb(np.asarray(influence))
    if up_factor is not None:
        up_factor_ = to_1d_array_nb(np.asarray(up_factor))
    else:
        up_factor_ = factor_
    if down_factor is not None:
        down_factor_ = to_1d_array_nb(np.asarray(down_factor))
    else:
        down_factor_ = factor_
    if mean_influence is not None:
        mean_influence_ = to_1d_array_nb(np.asarray(mean_influence))
    else:
        mean_influence_ = influence_
    if std_influence is not None:
        std_influence_ = to_1d_array_nb(np.asarray(std_influence))
    else:
        std_influence_ = influence_

    signal = np.full(close.shape, 0, dtype=int_)
    close_mean_filter = close.astype(float_)
    close_std_filter = close.astype(float_)
    mean_filter = np.full(close.shape, np.nan, dtype=float_)
    std_filter = np.full(close.shape, np.nan, dtype=float_)
    upper_band = np.full(close.shape, np.nan, dtype=float_)
    lower_band = np.full(close.shape, np.nan, dtype=float_)
    if lag == 0:
        raise ValueError("Lag cannot be zero")
    if lag - 1 >= close.shape[0]:
        raise ValueError("Lag must be smaller than close")

    mean_filter[lag - 1] = np.nanmean(close[:lag])
    std_filter[lag - 1] = np.nanstd(close[:lag])

    for i in range(lag, close.shape[0]):
        _up_factor = abs(flex_select_1d_nb(up_factor_, i))
        _down_factor = abs(flex_select_1d_nb(down_factor_, i))
        _mean_influence = abs(flex_select_1d_nb(mean_influence_, i))
        _std_influence = abs(flex_select_1d_nb(std_influence_, i))

        up_crossed = close[i] - mean_filter[i - 1] >= _up_factor * std_filter[i - 1]
        down_crossed = close[i] - mean_filter[i - 1] <= -_down_factor * std_filter[i - 1]
        if up_crossed or down_crossed:
            if up_crossed:
                signal[i] = 1
            else:
                signal[i] = -1

            close_mean_filter[i] = (
                _mean_influence * close[i] + (1 - _mean_influence) * close_mean_filter[i - 1]
            )
            close_std_filter[i] = (
                _std_influence * close[i] + (1 - _std_influence) * close_std_filter[i - 1]
            )
        else:
            signal[i] = 0
            close_mean_filter[i] = close[i]
            close_std_filter[i] = close[i]

        mean_filter[i] = np.nanmean(close_mean_filter[(i - lag + 1) : i + 1])
        std_filter[i] = np.nanstd(close_std_filter[(i - lag + 1) : i + 1])
        upper_band[i] = mean_filter[i] + _up_factor * std_filter[i - 1]
        lower_band[i] = mean_filter[i] - _down_factor * std_filter[i - 1]

    return signal, upper_band, lower_band


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        lag=base_ch.FlexArraySlicer(),
        factor=base_ch.FlexArraySlicer(axis=1),
        influence=base_ch.FlexArraySlicer(axis=1),
        up_factor=base_ch.FlexArraySlicer(axis=1),
        down_factor=base_ch.FlexArraySlicer(axis=1),
        mean_influence=base_ch.FlexArraySlicer(axis=1),
        std_influence=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def signal_detection_nb(
    close: tp.Array2d,
    lag: tp.FlexArray1dLike = 14,
    factor: tp.FlexArray2dLike = 1.0,
    influence: tp.FlexArray2dLike = 1.0,
    up_factor: tp.Optional[tp.FlexArray2dLike] = None,
    down_factor: tp.Optional[tp.FlexArray2dLike] = None,
    mean_influence: tp.Optional[tp.FlexArray2dLike] = None,
    std_influence: tp.Optional[tp.FlexArray2dLike] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Compute the 2-dim signal detection.

    Applies the one-dimensional `signal_detection_1d_nb` function to each column of the input array
    to detect signals and compute corresponding upper and lower bands for multi-dimensional data.

    Args:
        close (Array2d): Array of close prices.
        lag (FlexArray1dLike): Window size for computing moving averages and standard deviations.

            Provided as a scalar or per column.
        factor (FlexArray2dLike): Factor to determine the threshold for signal detection.

            Provided as a scalar, or per row, column, or element.
        influence (FlexArray2dLike): Influence factor for updating the filtering process.

            Provided as a scalar, or per row, column, or element.
        up_factor (Optional[FlexArray2dLike]): Factor for the upward threshold.

            Provided as a scalar, or per row, column, or element.

            If None, uses `factor`.
        down_factor (Optional[FlexArray2dLike]): Factor for the downward threshold.

            Provided as a scalar, or per row, column, or element.

            If None, uses `factor`.
        mean_influence (Optional[FlexArray2dLike]): Influence factor for updating the moving average filter.

            Provided as a scalar, or per row, column, or element.

            If None, uses `influence`.
        std_influence (Optional[FlexArray2dLike]): Influence factor for updating the standard deviation filter.

            Provided as a scalar, or per row, column, or element.

            If None, uses `influence`.

    Returns:
        Tuple[Array2d, Array2d, Array2d]: Tuple containing
            the signal array, upper band, and lower band arrays.

    !!! tip
        This function is parallelizable.
    """
    lag_ = to_1d_array_nb(np.asarray(lag))
    factor_ = to_2d_array_nb(np.asarray(factor))
    influence_ = to_2d_array_nb(np.asarray(influence))
    if up_factor is not None:
        up_factor_ = to_2d_array_nb(np.asarray(up_factor))
    else:
        up_factor_ = factor_
    if down_factor is not None:
        down_factor_ = to_2d_array_nb(np.asarray(down_factor))
    else:
        down_factor_ = factor_
    if mean_influence is not None:
        mean_influence_ = to_2d_array_nb(np.asarray(mean_influence))
    else:
        mean_influence_ = influence_
    if std_influence is not None:
        std_influence_ = to_2d_array_nb(np.asarray(std_influence))
    else:
        std_influence_ = influence_

    signal = np.empty(close.shape, dtype=int_)
    upper_band = np.empty(close.shape, dtype=float_)
    lower_band = np.empty(close.shape, dtype=float_)
    for col in prange(close.shape[1]):
        signal[:, col], upper_band[:, col], lower_band[:, col] = signal_detection_1d_nb(
            close[:, col],
            lag=flex_select_1d_nb(lag_, col),
            factor=flex_select_col_nb(factor_, col),
            influence=flex_select_col_nb(influence_, col),
            up_factor=flex_select_col_nb(up_factor_, col),
            down_factor=flex_select_col_nb(down_factor_, col),
            mean_influence=flex_select_col_nb(mean_influence_, col),
            std_influence=flex_select_col_nb(std_influence_, col),
        )
    return signal, upper_band, lower_band


# ############# HURST ############# #


@register_jitted(cache=True)
def get_standard_hurst_nb(
    close: tp.Array1d,
    max_lag: int = 20,
    stabilize: bool = False,
) -> float:
    """Estimate the Hurst exponent using the standard method.

    Args:
        close (Array1d): Array of close prices.
        max_lag (int): Maximum lag parameter for the standard computation.
        stabilize (bool): Flag to enable stabilization in the polynomial fit.

    Returns:
        float: Estimated Hurst exponent.
    """
    if max_lag is None:
        lags = np.arange(2, len(close) - 1)
    else:
        lags = np.arange(2, min(max_lag, len(close) - 1))
    tau = np.empty(len(lags), dtype=float_)
    for i, lag in enumerate(lags):
        tau[i] = np.var(np.subtract(close[lag:], close[:-lag]))
    coef = generic_nb.polyfit_1d_nb(np.log(lags), np.log(tau), 1, stabilize=stabilize)
    return coef[0] / 2


@register_jitted(cache=True)
def get_rs_nb(close: tp.Array1d) -> float:
    """Compute the rescaled range (R/S) used in Hurst exponent estimation.

    Args:
        close (Array1d): Array of close prices.

    Returns:
        float: Computed R/S ratio, or 0 if the range or standard deviation is zero.
    """
    incs = close[1:] / close[:-1] - 1.0
    mean_inc = np.sum(incs) / len(incs)
    deviations = incs - mean_inc
    Z = np.cumsum(deviations)
    R = np.max(Z) - np.min(Z)
    S = generic_nb.nanstd_1d_nb(incs, ddof=1)
    if R == 0 or S == 0:
        return 0
    return R / S


@register_jitted(cache=True)
def get_log_rs_hurst_nb(
    close: tp.Array1d,
    min_log: int = 1,
    max_log: int = 2,
    log_step: int = 0.25,
) -> float:
    """Estimate the Hurst exponent using the R/S method with logarithmically distributed window sizes.

    Args:
        close (Array1d): Array of close prices.
        min_log (int): Minimum logarithmic window size.
        max_log (int): Maximum logarithmic window size.
        log_step (float): Increment for logarithmic window size.

    Returns:
        float: Estimated Hurst exponent.
    """
    max_log = min(max_log, np.log10(len(close) - 1))
    log_range = np.arange(min_log, max_log, log_step)
    windows = np.empty(len(log_range) + 1, dtype=int_)
    windows[: len(log_range)] = 10**log_range
    windows[-1] = len(close)
    RS = np.empty(len(windows), dtype=float_)
    W = np.empty(len(windows), dtype=int_)
    k = 0

    for i, w in enumerate(windows):
        rs_sum = 0.0
        rs_count = 0
        for start in range(0, len(close), w):
            if (start + w) > len(close):
                break
            rs = get_rs_nb(close[start : start + w])
            if rs != 0:
                rs_sum += rs
                rs_count += 1
        if rs_count != 0:
            RS[k] = rs_sum / rs_count
            W[k] = w
            k += 1

    if k == 0:
        return np.nan
    A = np.vstack((np.log10(W[:k]), np.ones(len(RS[:k])))).T
    H, c = np.linalg.lstsq(A, np.log10(RS[:k]), rcond=-1)[0]
    return H


@register_jitted(cache=True)
def get_rs_hurst_nb(
    close: tp.Array1d,
    min_chunk: int = 8,
    max_chunk: int = 100,
    num_chunks: int = 5,
) -> float:
    """Estimate the Hurst exponent using the R/S method with linearly distributed window sizes.

    Args:
        close (Array1d): Array of close prices.
        min_chunk (int): Minimum chunk size for splitting the series.
        max_chunk (int): Maximum chunk size for splitting the series.
        num_chunks (int): Number of chunk sizes to use in the estimation.

    Returns:
        float: Estimated Hurst exponent.
    """
    diff = close[1:] - close[:-1]
    N = len(diff)
    max_chunk += 1
    max_chunk = min(max_chunk, len(diff) - 1)
    rs_tmp = np.empty(N, dtype=float_)
    chunk_size_range = np.linspace(min_chunk, max_chunk, num_chunks).astype(int_)
    chunk_size_list = np.empty(len(chunk_size_range), dtype=int_)
    rs_values_list = np.empty(len(chunk_size_range), dtype=float_)
    k = 0

    for chunk_size in chunk_size_range:
        number_of_chunks = int(len(diff) / chunk_size)

        for idx in range(number_of_chunks):
            ini = idx * chunk_size
            end = ini + chunk_size
            chunk = diff[ini:end]
            z = np.cumsum(chunk - np.mean(chunk))
            rs_tmp[idx] = np.divide(np.max(z) - np.min(z), np.nanstd(chunk))

        rs = np.nanmean(rs_tmp[: idx + 1])
        if not np.isnan(rs) and rs != 0:
            chunk_size_list[k] = chunk_size
            rs_values_list[k] = rs
            k += 1

    H, c = np.linalg.lstsq(
        a=np.vstack((np.log(chunk_size_list[:k]), np.ones(len(chunk_size_list[:k])))).T,
        b=np.log(rs_values_list[:k]),
        rcond=-1,
    )[0]
    return H


@register_jitted(cache=True)
def get_dma_hurst_nb(
    close: tp.Array1d,
    min_chunk: int = 8,
    max_chunk: int = 100,
    num_chunks: int = 5,
) -> float:
    """Estimate the Hurst exponent using the DMA method with linearly distributed window sizes.

    Args:
        close (Array1d): Array of close prices.
        min_chunk (int): Minimum window size.
        max_chunk (int): Maximum window size.
        num_chunks (int): Number of windows to use in the regression.

    Returns:
        float: Estimated Hurst exponent.
    """
    max_chunk += 1
    max_chunk = min(max_chunk, len(close) - 1)
    N = len(close)
    n_range = np.linspace(min_chunk, max_chunk, num_chunks).astype(int_)
    n_list = np.empty(len(n_range), dtype=int_)
    dma_list = np.empty(len(n_range), dtype=float_)
    k = 0
    factor = 1 / (N - max_chunk)

    for i, n in enumerate(n_range):
        x1 = np.full(n, -1, int_)
        x1[0] = n - 1
        b = np.divide(x1, n)  # do the same as:  y - y_ma_n
        noise = np.power(generic_nb.fir_filter_1d_nb(b, close)[max_chunk:], 2)
        dma = np.sqrt(factor * np.sum(noise))
        if not np.isnan(dma) and dma != 0:
            n_list[k] = n
            dma_list[k] = dma
            k += 1

    if k == 0:
        return np.nan
    H, const = np.linalg.lstsq(
        a=np.vstack((np.log10(n_list[:k]), np.ones(len(n_list[:k])))).T,
        b=np.log10(dma_list[:k]),
        rcond=-1,
    )[0]
    return H


@register_jitted(cache=True)
def get_dsod_hurst_nb(close: tp.Array1d) -> float:
    """Estimate the Hurst exponent using the discrete second order derivative method.

    Args:
        close (Array1d): Array of close prices.

    Returns:
        float: Estimated Hurst exponent based on the ratio of variances from second order differences.
    """
    diff = close[1:] - close[:-1]
    y = np.cumsum(diff)

    b1 = [1, -2, 1]
    y1 = generic_nb.fir_filter_1d_nb(b1, y)
    y1 = y1[len(b1) - 1 :]

    b2 = [1, 0, -2, 0, 1]
    y2 = generic_nb.fir_filter_1d_nb(b2, y)
    y2 = y2[len(b2) - 1 :]

    s1 = np.mean(y1**2)
    s2 = np.mean(y2**2)

    return 0.5 * np.log2(s2 / s1)


@register_jitted(cache=True)
def get_hurst_nb(
    close: tp.Array1d,
    method: int = HurstMethod.Standard,
    max_lag: int = 20,
    min_log: int = 1,
    max_log: int = 2,
    log_step: int = 0.25,
    min_chunk: int = 8,
    max_chunk: int = 100,
    num_chunks: int = 5,
    stabilize: bool = False,
) -> float:
    """Estimate the Hurst exponent using one of several computation methods.

    Args:
        close (Array1d): One-dimensional array of price data.
        method (int): Hurst exponent computation method.

            See `vectorbtpro.indicators.enums.HurstMethod`.
        max_lag (int): Maximum lag parameter for the standard computation.
        min_log (int): Minimum logarithmic scale for the LogRS method.
        max_log (int): Maximum logarithmic scale for the LogRS method.
        log_step (float): Increment on the logarithmic scale for the LogRS method.
        min_chunk (int): Minimum chunk size for RS and DMA methods.
        max_chunk (int): Maximum chunk size for RS and DMA methods.
        num_chunks (int): Number of chunks for RS and DMA methods.
        stabilize (bool): Flag to enable stabilization in the polynomial fit.

    Returns:
        float: Estimated Hurst exponent.
    """
    if method == HurstMethod.Standard:
        return get_standard_hurst_nb(close, max_lag=max_lag, stabilize=stabilize)
    if method == HurstMethod.LogRS:
        return get_log_rs_hurst_nb(close, min_log=min_log, max_log=max_log, log_step=log_step)
    if method == HurstMethod.RS:
        return get_rs_hurst_nb(
            close, min_chunk=min_chunk, max_chunk=max_chunk, num_chunks=num_chunks
        )
    if method == HurstMethod.DMA:
        return get_dma_hurst_nb(
            close, min_chunk=min_chunk, max_chunk=max_chunk, num_chunks=num_chunks
        )
    if method == HurstMethod.DSOD:
        return get_dsod_hurst_nb(close)
    raise ValueError("Invalid HurstMethod option")


@register_jitted(cache=True)
def rolling_hurst_1d_nb(
    close: tp.Array1d,
    window: int,
    method: int = HurstMethod.Standard,
    max_lag: int = 20,
    min_log: int = 1,
    max_log: int = 2,
    log_step: int = 0.25,
    min_chunk: int = 8,
    max_chunk: int = 100,
    num_chunks: int = 5,
    minp: tp.Optional[int] = None,
    stabilize: bool = False,
) -> tp.Array1d:
    """Compute the rolling Hurst exponent over a one-dimensional array.

    Args:
        close (Array1d): One-dimensional array of price data.
        window (int): Window size.
        method (int): Hurst exponent computation method.

            See `vectorbtpro.indicators.enums.HurstMethod`.
        max_lag (int): Maximum lag parameter for the standard computation.
        min_log (int): Minimum logarithmic scale for the LogRS method.
        max_log (int): Maximum logarithmic scale for the LogRS method.
        log_step (float): Increment on the logarithmic scale for the LogRS method.
        min_chunk (int): Minimum chunk size for RS and DMA methods.
        max_chunk (int): Maximum chunk size for RS and DMA methods.
        num_chunks (int): Number of chunks for RS and DMA methods.
        minp (Optional[int]): Minimum number of observations required.
        stabilize (bool): Flag to enable stabilization in the polynomial fit.

    Returns:
        Array1d: Array of rolling Hurst exponent values.
    """
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(close, dtype=float_)
    nancnt = 0
    for i in range(close.shape[0]):
        if np.isnan(close[i]):
            nancnt = nancnt + 1
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            if np.isnan(close[i - window]):
                nancnt = nancnt - 1
            valid_cnt = window - nancnt
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            close_window = close[from_i:to_i]
            out[i] = get_hurst_nb(
                close_window,
                method=method,
                max_lag=max_lag,
                min_log=min_log,
                max_log=max_log,
                log_step=log_step,
                min_chunk=min_chunk,
                max_chunk=max_chunk,
                num_chunks=num_chunks,
                stabilize=stabilize,
            )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=None,
        method=None,
        max_lag=None,
        min_log=None,
        max_log=None,
        log_step=None,
        min_chunk=None,
        max_chunk=None,
        num_chunks=None,
        minp=None,
        stabilize=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_hurst_nb(
    close: tp.Array2d,
    window: int,
    method: int = HurstMethod.Standard,
    max_lag: int = 20,
    min_log: int = 1,
    max_log: int = 2,
    log_step: int = 0.25,
    min_chunk: int = 8,
    max_chunk: int = 100,
    num_chunks: int = 5,
    minp: tp.Optional[int] = None,
    stabilize: bool = False,
) -> tp.Array2d:
    """Compute the rolling Hurst exponent for each column in a two-dimensional array.

    Args:
        close (Array2d): Two-dimensional array of price data where each column represents a series.
        window (int): Window size.
        method (int): Hurst exponent computation method.

            See `vectorbtpro.indicators.enums.HurstMethod`.
        max_lag (int): Maximum lag parameter for the standard computation.
        min_log (int): Minimum logarithmic scale for the LogRS method.
        max_log (int): Maximum logarithmic scale for the LogRS method.
        log_step (float): Increment on the logarithmic scale for the LogRS method.
        min_chunk (int): Minimum chunk size for RS and DMA methods.
        max_chunk (int): Maximum chunk size for RS and DMA methods.
        num_chunks (int): Number of chunks for RS and DMA methods.
        minp (Optional[int]): Minimum number of observations required.
        stabilize (bool): Flag to enable stabilization in the polynomial fit.

    Returns:
        Array2d: Two-dimensional array of rolling Hurst exponent values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.empty_like(close, dtype=float_)
    for col in prange(close.shape[1]):
        out[:, col] = rolling_hurst_1d_nb(
            close[:, col],
            window,
            method=method,
            max_lag=max_lag,
            min_log=min_log,
            max_log=max_log,
            log_step=log_step,
            min_chunk=min_chunk,
            max_chunk=max_chunk,
            num_chunks=num_chunks,
            minp=minp,
            stabilize=stabilize,
        )
    return out
