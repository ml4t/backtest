# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for processing OHLCV data.

!!! note
    Vectorbtpro treats matrices as first-class citizens and expects input arrays to be
    two-dimensional unless a function has a `_1d` suffix or is used as input to another function.
    Data is processed along the index (axis 0).
"""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_nb, flex_select_1d_pr_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.ohlcv.enums import PriceFeature
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


@register_jitted(cache=True)
def ohlc_every_1d_nb(price: tp.Array1d, n: tp.FlexArray1dLike) -> tp.Array2d:
    """Aggregate every n price points into an OHLC bar.

    Args:
        price (Array1d): One-dimensional array of price data.
        n (FlexArray1dLike): Number of consecutive price points to aggregate.

            Provided as a scalar or per bar.

    Returns:
        Array2d: Two-dimensional array where each row represents an OHLC bar.
    """
    n_ = to_1d_array_nb(np.asarray(n))
    out = np.empty((price.shape[0], 4), dtype=float_)
    vmin = np.inf
    vmax = -np.inf
    k = 0
    start_i = 0
    for i in range(price.shape[0]):
        _n = flex_select_1d_pr_nb(n_, k)
        if _n <= 0:
            out[k, 0] = np.nan
            out[k, 1] = np.nan
            out[k, 2] = np.nan
            out[k, 3] = np.nan
            vmin = np.inf
            vmax = -np.inf
            if i < price.shape[0] - 1:
                k = k + 1
            continue
        if price[i] < vmin:
            vmin = price[i]
        if price[i] > vmax:
            vmax = price[i]
        if i == start_i:
            out[k, 0] = price[i]
        if i == start_i + _n - 1 or i == price.shape[0] - 1:
            out[k, 1] = vmax
            out[k, 2] = vmin
            out[k, 3] = price[i]
            vmin = np.inf
            vmax = -np.inf
            if i < price.shape[0] - 1:
                k = k + 1
                start_i = start_i + _n
    return out[: k + 1]


@register_jitted(cache=True)
def mirror_ohlc_1d_nb(
    n_rows: int,
    open: tp.Optional[tp.Array1d] = None,
    high: tp.Optional[tp.Array1d] = None,
    low: tp.Optional[tp.Array1d] = None,
    close: tp.Optional[tp.Array1d] = None,
    start_value: float = np.nan,
    ref_feature: int = -1,
) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
    """Mirror OHLC arrays by applying a cumulative scaling based on logarithmic returns.

    Args:
        n_rows (int): Number of rows for the output arrays.
        open (Optional[Array1d]): 1D array of open prices.
        high (Optional[Array1d]): 1D array of high prices.
        low (Optional[Array1d]): 1D array of low prices.
        close (Optional[Array1d]): 1D array of close prices.
        start_value (float): Initial value for the reference price.

            If provided and not NaN, it replaces the first valid reference price.
        ref_feature (int): Indicator for the reference price feature.

            See `vectorbtpro.ohlcv.enums.PriceFeature`.
            A value of -1 enables auto-detection of the first non-NaN price.

    Returns:
        Tuple[Array1d, Array1d, Array1d, Array1d]: Mirrored open, high, low, and close arrays.
    """
    new_open = np.empty(n_rows, dtype=float_)
    new_high = np.empty(n_rows, dtype=float_)
    new_low = np.empty(n_rows, dtype=float_)
    new_close = np.empty(n_rows, dtype=float_)

    cumsum = 0.0
    first_idx = -1
    factor = 1.0
    for i in range(n_rows):
        _open = open[i] if open is not None else np.nan
        _high = high[i] if high is not None else np.nan
        _low = low[i] if low is not None else np.nan
        _close = close[i] if close is not None else np.nan

        if ref_feature == PriceFeature.Open or (ref_feature == -1 and not np.isnan(_open)):
            if first_idx == -1:
                first_idx = i
                if not np.isnan(start_value):
                    new_open[i] = start_value
                else:
                    new_open[i] = _open
                factor = new_open[i] / _open
                new_high[i] = _high * factor if not np.isnan(_high) else np.nan
                new_low[i] = _low * factor if not np.isnan(_low) else np.nan
                new_close[i] = _close * factor if not np.isnan(_close) else np.nan
            else:
                prev_open = open[i - 1] if open is not None else np.nan
                cumsum += -np.log(_open / prev_open)
                new_open[i] = open[first_idx] * np.exp(cumsum) * factor
                new_high[i] = (_open / _low) * new_open[i] if not np.isnan(_low) else np.nan
                new_low[i] = (_open / _high) * new_open[i] if not np.isnan(_high) else np.nan
                new_close[i] = (_open / _close) * new_open[i] if not np.isnan(_close) else np.nan
        elif ref_feature == PriceFeature.Close or (ref_feature == -1 and not np.isnan(_close)):
            if first_idx == -1:
                first_idx = i
                if not np.isnan(start_value):
                    new_close[i] = start_value
                else:
                    new_close[i] = _close
                factor = new_close[i] / _close
                new_open[i] = _open * factor if not np.isnan(_open) else np.nan
                new_high[i] = _high * factor if not np.isnan(_high) else np.nan
                new_low[i] = _low * factor if not np.isnan(_low) else np.nan
            else:
                prev_close = close[i - 1] if close is not None else np.nan
                cumsum += -np.log(_close / prev_close)
                new_close[i] = close[first_idx] * np.exp(cumsum) * factor
                new_open[i] = (_close / _open) * new_close[i] if not np.isnan(_open) else np.nan
                new_high[i] = (_close / _low) * new_close[i] if not np.isnan(_low) else np.nan
                new_low[i] = (_close / _high) * new_close[i] if not np.isnan(_high) else np.nan
        elif ref_feature == PriceFeature.High or (ref_feature == -1 and not np.isnan(_high)):
            if first_idx == -1:
                first_idx = i
                if not np.isnan(start_value):
                    new_high[i] = start_value
                else:
                    new_high[i] = _high
                factor = new_high[i] / _high
                new_open[i] = _open * factor if not np.isnan(_open) else np.nan
                new_low[i] = _low * factor * new_high[i] if not np.isnan(_low) else np.nan
                new_close[i] = _close * factor * new_high[i] if not np.isnan(_close) else np.nan
            else:
                prev_high = high[i - 1] if high is not None else np.nan
                cumsum += -np.log(_high / prev_high)
                new_high[i] = high[first_idx] * np.exp(cumsum) * factor
                new_open[i] = (_high / _open) * new_high[i] if not np.isnan(_open) else np.nan
                new_high[i] = (_high / _low) * new_high[i] if not np.isnan(_low) else np.nan
                new_close[i] = (_high / _close) * new_high[i] if not np.isnan(_close) else np.nan
        elif ref_feature == PriceFeature.Low or (ref_feature == -1 and not np.isnan(_low)):
            if first_idx == -1:
                first_idx = i
                if not np.isnan(start_value):
                    new_low[i] = start_value
                else:
                    new_low[i] = _low
                factor = new_low[i] / _low
                new_open[i] = _open * factor if not np.isnan(_open) else np.nan
                new_high[i] = _high * factor if not np.isnan(_high) else np.nan
                new_close[i] = _close * factor if not np.isnan(_close) else np.nan
            else:
                prev_low = low[i - 1] if low is not None else np.nan
                cumsum += -np.log(_low / prev_low)
                new_low[i] = low[first_idx] * np.exp(cumsum) * factor
                new_open[i] = (_low / _open) * new_low[i] if not np.isnan(_open) else np.nan
                new_high[i] = (_low / _high) * new_low[i] if not np.isnan(_high) else np.nan
                new_close[i] = (_low / _close) * new_low[i] if not np.isnan(_close) else np.nan
        else:
            new_open[i] = np.nan
            new_high[i] = np.nan
            new_low[i] = np.nan
            new_close[i] = np.nan

    return new_open, new_high, new_low, new_close


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        open=base_ch.ArraySlicer(axis=1),
        high=base_ch.ArraySlicer(axis=1),
        low=base_ch.ArraySlicer(axis=1),
        close=base_ch.ArraySlicer(axis=1),
        start_value=base_ch.FlexArraySlicer(),
        ref_feature=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mirror_ohlc_nb(
    target_shape: tp.Shape,
    open: tp.Optional[tp.Array2d] = None,
    high: tp.Optional[tp.Array2d] = None,
    low: tp.Optional[tp.Array2d] = None,
    close: tp.Optional[tp.Array2d] = None,
    start_value: tp.FlexArray1dLike = np.nan,
    ref_feature: tp.FlexArray1dLike = -1,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Generate mirrored OHLC arrays for two-dimensional inputs by applying the 1D operation column-wise.

    Args:
        target_shape (Shape): Base dimensions (rows, columns).
        open (Optional[Array2d]): 2D array of open prices.
        high (Optional[Array2d]): 2D array of high prices.
        low (Optional[Array2d]): 2D array of low prices.
        close (Optional[Array2d]): 2D array of close prices.
        start_value (FlexArray1dLike): Starting value.

            Provided as a scalar or per column.

            If provided and not NaN, it replaces the first valid reference price.
        ref_feature (FlexArray1dLike): Reference price feature.

            Provided as a scalar or per column.

            See `vectorbtpro.ohlcv.enums.PriceFeature`.
            A value of -1 enables auto-detection of the first non-NaN price.

    Returns:
        Tuple[Array2d, Array2d, Array2d, Array2d]: Mirrored open, high, low, and
            close arrays with the specified shape.

    !!! tip
        This function is parallelizable.
    """
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    ref_feature_ = to_1d_array_nb(np.asarray(ref_feature))

    new_open = np.empty(target_shape, dtype=float_)
    new_high = np.empty(target_shape, dtype=float_)
    new_low = np.empty(target_shape, dtype=float_)
    new_close = np.empty(target_shape, dtype=float_)

    for col in prange(target_shape[1]):
        new_open[:, col], new_high[:, col], new_low[:, col], new_close[:, col] = mirror_ohlc_1d_nb(
            target_shape[0],
            open[:, col] if open is not None else None,
            high[:, col] if high is not None else None,
            low[:, col] if low is not None else None,
            close[:, col] if close is not None else None,
            start_value=flex_select_1d_nb(start_value_, col),
            ref_feature=flex_select_1d_nb(ref_feature_, col),
        )

    return new_open, new_high, new_low, new_close
