# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for calculating returns and portfolio performance metrics.

!!! note
    Vectorbtpro treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless the function has a `_1d` suffix or is intended as input to another function.
    Data is processed along index (axis 0).

    All functions passed as arguments must be Numba-compiled.

!!! info
    For default settings, see `vectorbtpro._settings.returns`.
"""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro._settings import settings
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.returns.enums import RollSharpeAIS, RollSharpeAOS
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import add_nb

__all__ = []

_inf_to_nan = settings["returns"]["inf_to_nan"]
_nan_to_zero = settings["returns"]["nan_to_zero"]


# ############# Metrics ############# #


@register_jitted(cache=True)
def get_return_nb(
    input_value: float,
    output_value: float,
    log_returns: bool = False,
    inf_to_nan: bool = _inf_to_nan,
    nan_to_zero: bool = _nan_to_zero,
) -> float:
    """Calculate return from given input and output values.

    Args:
        input_value (float): Initial value used for the return calculation.
        output_value (float): Final value used for the return calculation.
        log_returns (bool): Flag to compute logarithmic returns.
        inf_to_nan (bool): If True, convert an infinite return to NaN.
        nan_to_zero (bool): If True, convert a NaN return to 0.

    Returns:
        float: Calculated return value.
    """
    if input_value == 0:
        if output_value == 0:
            r = 0.0
        else:
            r = np.inf * np.sign(output_value)
    else:
        return_value = add_nb(output_value, -input_value) / input_value
        if log_returns:
            r = np.log1p(return_value)
        else:
            r = return_value
    if inf_to_nan and np.isinf(r):
        r = np.nan
    if nan_to_zero and np.isnan(r):
        r = 0.0
    return r


@register_jitted(cache=True)
def returns_1d_nb(
    arr: tp.Array1d,
    init_value: float = np.nan,
    log_returns: bool = False,
) -> tp.Array1d:
    """Calculate returns from a 1-dimensional array.

    Args:
        arr (Array1d): 1-dimensional array of asset prices or values.
        init_value (float): Initial value to use.

            If NaN, the first element of `arr` is used.
        log_returns (bool): Flag to compute logarithmic returns.

    Returns:
        Array1d: Array of calculated return values.
    """
    out = np.empty(arr.shape, dtype=float_)
    if np.isnan(init_value) and arr.shape[0] > 0:
        input_value = arr[0]
    else:
        input_value = init_value
    for i in range(arr.shape[0]):
        output_value = arr[i]
        out[i] = get_return_nb(input_value, output_value, log_returns=log_returns)
        input_value = output_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def returns_nb(
    arr: tp.Array2d,
    init_value: tp.FlexArray1dLike = np.nan,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate returns for a 2-dimensional array.

    Args:
        arr (Array2d): 2-dimensional array of asset prices or values.
        init_value (FlexArray1dLike): Initial value.

            Provided as a scalar or per column.

            If NaN, the first element of each column is used.
        log_returns (bool): Flag to compute logarithmic returns.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2-dimensional array containing calculated return values for each column.

    !!! tip
        This function is parallelizable.
    """
    init_value_ = to_1d_array_nb(np.asarray(init_value))

    out = np.full(arr.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=arr.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(arr.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _init_value = flex_select_1d_pc_nb(init_value_, col)

        out[_sim_start:_sim_end, col] = returns_1d_nb(
            arr[_sim_start:_sim_end, col],
            init_value=_init_value,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def mirror_returns_1d_nb(returns: tp.Array1d, log_returns: bool = False) -> tp.Array1d:
    """Calculate mirrored returns for a 1-dimensional array.

    A mirrored return is the inverse of a return value. For logarithmic returns, each return is negated.
    For simple returns, the mirrored return is calculated as `(1 / (1 + R_t)) - 1`.

    Args:
        returns (Array1d): Array of return values.
        log_returns (bool): Flag indicating whether returns are logarithmic.

    Returns:
        Array1d: Array of mirrored return values.
    """
    out = np.empty(returns.shape, dtype=float_)
    for i in range(returns.shape[0]):
        if log_returns:
            out[i] = -returns[i]
        else:
            if returns[i] <= -1:
                out[i] = np.inf
            else:
                out[i] = (1 / (1 + returns[i])) - 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mirror_returns_nb(
    returns: tp.Array2d,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate mirrored returns for a 2-dimensional array.

    Args:
        returns (Array2d): 2-dimensional array of return values.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2-dimensional array of mirrored return values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = mirror_returns_1d_nb(
            returns[_sim_start:_sim_end, col],
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def cumulative_returns_1d_nb(
    returns: tp.Array1d,
    start_value: float = 1.0,
    log_returns: bool = False,
) -> tp.Array1d:
    """Calculate cumulative returns from a 1-dimensional array of returns.

    Args:
        returns (Array1d): Array of individual return values.
        start_value (float): Initial value used to scale the cumulative returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.

    Returns:
        Array1d: Array of cumulative return values.
    """
    out = np.empty_like(returns, dtype=float_)
    if log_returns:
        cumsum = 0
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumsum += returns[i]
            if start_value == 0:
                out[i] = cumsum
            else:
                out[i] = np.exp(cumsum) * start_value
    else:
        cumprod = 1
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumprod *= 1 + returns[i]
            if start_value == 0:
                out[i] = cumprod - 1
            else:
                out[i] = cumprod * start_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        start_value=None,
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cumulative_returns_nb(
    returns: tp.Array2d,
    start_value: float = 1.0,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute cumulative returns from a 2D array of returns.

    Args:
        returns (Array2d): 2D array of periodic returns.
        start_value (float): Initial value used to scale the cumulative returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array of cumulative returns computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = cumulative_returns_1d_nb(
            returns[_sim_start:_sim_end, col],
            start_value=start_value,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def final_value_1d_nb(
    returns: tp.Array1d,
    start_value: float = 1.0,
    log_returns: bool = False,
) -> float:
    """Compute the final portfolio value from a 1D array of returns.

    Args:
        returns (Array1d): 1D array of periodic returns.
        start_value (float): Initial value used to scale the cumulative returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.

    Returns:
        float: Final value computed from the returns.
    """
    if log_returns:
        cumsum = 0
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumsum += returns[i]
        if start_value == 0:
            return cumsum
        return np.exp(cumsum) * start_value
    else:
        cumprod = 1
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumprod *= 1 + returns[i]
        if start_value == 0:
            return cumprod - 1
        return cumprod * start_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        start_value=None,
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def final_value_nb(
    returns: tp.Array2d,
    start_value: float = 1.0,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute final portfolio values for each column from a 2D array of returns.

    Args:
        returns (Array2d): 2D array of periodic returns.
        start_value (float): Initial value used to scale the cumulative returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array of final portfolio values computed for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = final_value_1d_nb(
            returns[_sim_start:_sim_end, col],
            start_value=start_value,
            log_returns=log_returns,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        start_value=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_final_value_nb(
    returns: tp.Array2d,
    window: int,
    start_value: float = 1.0,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute rolling final portfolio values from a 2D array of returns.

    Args:
        returns (Array2d): 2D array of periodic returns.
        window (int): Window size.
        start_value (float): Initial value used to scale the cumulative returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array containing rolling final portfolio values computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            final_value_1d_nb,
            start_value,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def total_return_1d_nb(returns: tp.Array1d, log_returns: bool = False) -> float:
    """Compute the total return from a 1D array of returns.

    Args:
        returns (Array1d): 1D array of periodic returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.

    Returns:
        float: Total return computed by setting the initial value to zero.
    """
    return final_value_1d_nb(returns, start_value=0.0, log_returns=log_returns)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def total_return_nb(
    returns: tp.Array2d,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute total returns for each column from a 2D array of returns.

    Args:
        returns (Array2d): 2D array of periodic returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array of total returns computed for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = total_return_1d_nb(returns[_sim_start:_sim_end, col], log_returns=log_returns)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_total_return_nb(
    returns: tp.Array2d,
    window: int,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute rolling total returns from a 2D array of returns.

    Args:
        returns (Array2d): 2D array of periodic returns.
        window (int): Window size.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array containing rolling total returns computed for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            total_return_1d_nb,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def annualized_return_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[float] = None,
) -> float:
    """Annualized total return.

    Equivalent to the compound annual growth rate (CAGR).

    Args:
        returns (Array1d): Array of daily returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[float]): Number of periods for annualization.

            Defaults to the length of the returns array.

    Returns:
        float: Computed annualized return.
    """
    if periods is None:
        periods = returns.shape[0]
    final_value = final_value_1d_nb(returns, log_returns=log_returns)
    if periods == 0:
        return np.nan
    return final_value ** (ann_factor / periods) - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        periods=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def annualized_return_nb(
    returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `annualized_return_1d_nb`.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[FlexArray1dLike]): Number of periods for annualization.

            Provided as a scalar or per column.

            Defaults to the length of the simulation range.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Annualized returns for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if periods is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(periods).astype(int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = annualized_return_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            periods=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_annualized_return_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `annualized_return_1d_nb`.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Rolling annualized returns computed over the specified window.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            annualized_return_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def annualized_volatility_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 0,
) -> float:
    """Annualized volatility of a strategy.

    Args:
        returns (Array1d): Array of returns.
        ann_factor (float): Annualization factor.
        levy_alpha (float): Levy alpha parameter.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Computed annualized volatility.
    """
    return generic_nb.nanstd_1d_nb(returns, ddof=ddof) * ann_factor ** (1.0 / levy_alpha)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        levy_alpha=None,
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def annualized_volatility_nb(
    returns: tp.Array2d,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `annualized_volatility_1d_nb`.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        ann_factor (float): Annualization factor.
        levy_alpha (float): Levy alpha parameter.
        ddof (int): Delta degrees of freedom.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Annualized volatility for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = annualized_volatility_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            levy_alpha=levy_alpha,
            ddof=ddof,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        levy_alpha=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_annualized_volatility_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `annualized_volatility_1d_nb`.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        levy_alpha (float): Levy alpha parameter.
        ddof (int): Delta degrees of freedom.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Rolling annualized volatility computed over the window.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            annualized_volatility_1d_nb,
            ann_factor,
            levy_alpha,
            ddof,
        )
    return out


@register_jitted(cache=True)
def max_drawdown_1d_nb(returns: tp.Array1d, log_returns: bool = False) -> float:
    """Total maximum drawdown (MDD).

    Args:
        returns (Array1d): Array of returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.

    Returns:
        float: Maximum drawdown value; returns nan if no valid returns are present.
    """
    cum_ret = np.nan
    value_max = 1.0
    out = 0.0
    for i in range(returns.shape[0]):
        if not np.isnan(returns[i]):
            if np.isnan(cum_ret):
                cum_ret = 1.0
            if log_returns:
                ret = np.exp(returns[i]) - 1
            else:
                ret = returns[i]
            cum_ret *= ret + 1.0
        if cum_ret > value_max:
            value_max = cum_ret
        elif cum_ret < value_max:
            dd = cum_ret / value_max - 1
            if dd < out:
                out = dd
    if np.isnan(cum_ret):
        return np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def max_drawdown_nb(
    returns: tp.Array2d,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Return the maximum drawdown values for each column using a 1-dimensional drawdown computation.

    Args:
        returns (Array2d): 2-dimensional array of returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array containing the maximum drawdown for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = max_drawdown_1d_nb(returns[_sim_start:_sim_end, col], log_returns=log_returns)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_max_drawdown_nb(
    returns: tp.Array2d,
    window: int,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Return the rolling maximum drawdown values over a specified window for each column.

    Args:
        returns (Array2d): 2-dimensional array of returns.
        window (int): Window size.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2-dimensional array of rolling maximum drawdown values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            max_drawdown_1d_nb,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def calmar_ratio_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[float] = None,
) -> float:
    """Return the Calmar ratio (drawdown ratio) of a strategy based on 1-dimensional returns.

    Args:
        returns (Array1d): 1-dimensional array of returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[float]): Number of periods for annualization.

            Defaults to the length of the returns array.

    Returns:
        float: Calmar ratio of the strategy.

            Returns NaN if the maximum drawdown is zero, or infinity when appropriate.
    """
    max_drawdown = max_drawdown_1d_nb(returns, log_returns=log_returns)
    if max_drawdown == 0:
        return np.nan
    annualized_return = annualized_return_1d_nb(
        returns,
        ann_factor,
        log_returns=log_returns,
        periods=periods,
    )
    if max_drawdown == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / np.abs(max_drawdown)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        periods=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def calmar_ratio_nb(
    returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Return the Calmar ratio for each column of a 2-dimensional returns array.

    Args:
        returns (Array2d): 2-dimensional array of returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[FlexArray1dLike]): Number of periods for annualization.

            Provided as a scalar or per column.

            Defaults to the length of the simulation range.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array containing the Calmar ratio for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if periods is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(periods).astype(int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = calmar_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            periods=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_calmar_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Return the rolling Calmar ratio values over a specified window for each column.

    Args:
        returns (Array2d): 2-dimensional array of returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2-dimensional array containing the rolling Calmar ratios for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            calmar_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def deannualized_return_nb(ret: float, ann_factor: float) -> float:
    """Return the deannualized return based on the annualized return and annualization factor.

    Args:
        ret (float): Annualized return.
        ann_factor (float): Annualization factor.

    Returns:
        float: Deannualized return.
    """
    if ann_factor == 1:
        return ret
    if ann_factor <= -1:
        return np.nan
    return (1 + ret) ** (1.0 / ann_factor) - 1


@register_jitted(cache=True)
def omega_ratio_1d_nb(returns: tp.Array1d) -> float:
    """Return the Omega ratio of a strategy based on 1-dimensional returns.

    Args:
        returns (Array1d): 1-dimensional array of returns.

    Returns:
        float: Omega ratio, calculated as the sum of positive returns divided by
            the absolute sum of negative returns.

            Returns NaN if both sums are zero, or infinity if there are positive returns
            with no negative returns.
    """
    numer = 0.0
    denom = 0.0
    for i in range(returns.shape[0]):
        ret = returns[i]
        if ret > 0:
            numer += ret
        elif ret < 0:
            denom -= ret
    if denom == 0:
        if numer == 0:
            return np.nan
        return np.inf
    return numer / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def omega_ratio_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `omega_ratio_1d_nb`.

    Computes the omega ratio for each column in a 2D returns array using a specified simulation range.

    Args:
        returns (Array2d): 2D array of returns.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Computed omega ratio for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = omega_ratio_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_omega_ratio_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `omega_ratio_1d_nb`.

    Computes a rolling omega ratio for each column in a 2D returns array based on a given window.

    Args:
        returns (Array2d): 2D array of returns.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array of rolling omega ratio values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            omega_ratio_1d_nb,
        )
    return out


@register_jitted(cache=True)
def sharpe_ratio_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    ddof: int = 0,
) -> float:
    """Calculate the Sharpe ratio of a strategy from a 1D returns array.

    Args:
        returns (Array1d): 1D array of strategy returns.
        ann_factor (float): Annualization factor.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Computed Sharpe ratio.

            Returns nan if the standard deviation is zero and mean is zero,
            or inf if mean is non-zero.
    """
    mean = np.nanmean(returns)
    std = generic_nb.nanstd_1d_nb(returns, ddof=ddof)
    if std == 0:
        if mean == 0:
            return np.nan
        return np.inf
    return mean / std * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sharpe_ratio_nb(
    returns: tp.Array2d,
    ann_factor: float,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `sharpe_ratio_1d_nb`.

    Calculates the Sharpe ratio for each column in a 2D returns array.

    Args:
        returns (Array2d): 2D array of returns.
        ann_factor (float): Annualization factor.
        ddof (int): Delta degrees of freedom.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Sharpe ratio for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = sharpe_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            ddof=ddof,
        )
    return out


@register_jitted(cache=True)
def rolling_sharpe_ratio_acc_nb(in_state: RollSharpeAIS) -> RollSharpeAOS:
    """Accumulate rolling Sharpe ratio state.

    Updates the current state by calculating the rolling Sharpe ratio using the provided returns and metrics.

    Args:
        in_state (RollSharpeAIS): Input state containing returns and accumulated metrics.

    Returns:
        RollSharpeAOS: Updated state with the computed Sharpe ratio.
    """
    mean_in_state = generic_enums.RollMeanAIS(
        i=in_state.i,
        value=in_state.ret,
        pre_window_value=in_state.pre_window_ret,
        cumsum=in_state.cumsum,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
    )
    mean_out_state = generic_nb.rolling_mean_acc_nb(mean_in_state)
    std_in_state = generic_enums.RollStdAIS(
        i=in_state.i,
        value=in_state.ret,
        pre_window_value=in_state.pre_window_ret,
        cumsum=in_state.cumsum,
        cumsum_sq=in_state.cumsum_sq,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
        ddof=in_state.ddof,
    )
    std_out_state = generic_nb.rolling_std_acc_nb(std_in_state)
    mean = mean_out_state.value
    std = std_out_state.value
    if std == 0:
        sharpe = np.nan
    else:
        sharpe = mean / std * np.sqrt(in_state.ann_factor)

    return RollSharpeAOS(
        cumsum=std_out_state.cumsum,
        cumsum_sq=std_out_state.cumsum_sq,
        nancnt=std_out_state.nancnt,
        value=sharpe,
    )


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_sharpe_ratio_stream_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate rolling Sharpe ratio in a streaming fashion.

    Computes the Sharpe ratio over a rolling window for each column of a 2D returns array,
    updating the state iteratively via `rolling_sharpe_ratio_acc_nb`.

    Args:
        returns (Array2d): 2D array of strategy returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        ddof (int): Delta degrees of freedom.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array of rolling Sharpe ratio values for each asset.

    !!! tip
        This function is parallelizable.
    """
    if window is None:
        window = returns.shape[0]
    if minp is None:
        minp = window

    out = np.full(returns.shape, np.nan, dtype=float_)
    if returns.shape[0] == 0:
        return out

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        cumsum = 0.0
        cumsum_sq = 0.0
        nancnt = 0

        for i in range(_sim_start, _sim_end):
            in_state = RollSharpeAIS(
                i=i - _sim_start,
                ret=returns[i, col],
                pre_window_ret=returns[i - window, col] if i - window >= 0 else np.nan,
                cumsum=cumsum,
                cumsum_sq=cumsum_sq,
                nancnt=nancnt,
                window=window,
                minp=minp,
                ddof=ddof,
                ann_factor=ann_factor,
            )
            out_state = rolling_sharpe_ratio_acc_nb(in_state)
            cumsum = out_state.cumsum
            cumsum_sq = out_state.cumsum_sq
            nancnt = out_state.nancnt
            out[i, col] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        stream_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_sharpe_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    stream_mode: bool = True,
) -> tp.Array2d:
    """Compute rolling Sharpe ratio for each column of a 2D returns array.

    Applies a rolling window to compute the Sharpe ratio using `sharpe_ratio_1d_nb`.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        ddof (int): Delta degrees of freedom.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        stream_mode (bool): Flag indicating whether to use stream mode.

    Returns:
        Array2d: Array containing the rolling Sharpe ratio for each column.

    !!! tip
        This function is parallelizable.
    """
    if stream_mode:
        return rolling_sharpe_ratio_stream_nb(
            returns,
            window,
            ann_factor,
            minp=minp,
            ddof=ddof,
            sim_start=sim_start,
            sim_end=sim_end,
        )

    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            sharpe_ratio_1d_nb,
            ann_factor,
            ddof,
        )
    return out


@register_jitted(cache=True)
def downside_risk_1d_nb(returns: tp.Array1d, ann_factor: float) -> float:
    """Calculate downside deviation for a 1D returns array.

    Computes the square root of the mean squared negative returns multiplied by the
    square root of the annualization factor, representing the downside risk.

    Args:
        returns (Array1d): One-dimensional array of returns.
        ann_factor (float): Annualization factor.

    Returns:
        float: Downside deviation value.
    """
    cnt = 0
    adj_ret_sqrd_sum = np.nan
    for i in range(returns.shape[0]):
        if not np.isnan(returns[i]):
            cnt += 1
            if np.isnan(adj_ret_sqrd_sum):
                adj_ret_sqrd_sum = 0.0
            if returns[i] <= 0:
                adj_ret_sqrd_sum += returns[i] ** 2
    if cnt == 0:
        return np.nan
    adj_ret_sqrd_mean = adj_ret_sqrd_sum / cnt
    return np.sqrt(adj_ret_sqrd_mean) * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def downside_risk_nb(
    returns: tp.Array2d,
    ann_factor: float,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute downside deviation for each column of a 2D returns array.

    Applies `downside_risk_1d_nb` column-wise within a specified simulation range.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        ann_factor (float): Annualization factor.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: One-dimensional array of downside risk values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = downside_risk_1d_nb(returns[_sim_start:_sim_end, col], ann_factor)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_downside_risk_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute rolling downside deviation for each column of a 2D returns array.

    Uses a rolling window to apply `downside_risk_1d_nb` along each column within a simulation range.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array of rolling downside risk values computed column-wise.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            downside_risk_1d_nb,
            ann_factor,
        )
    return out


@register_jitted(cache=True)
def sortino_ratio_1d_nb(returns: tp.Array1d, ann_factor: float) -> float:
    """Compute the Sortino ratio for a 1D returns array.

    Calculates the ratio of the mean annualized return to the downside risk.
    Returns `np.nan` if insufficient data or `np.inf` if the downside risk is zero but
    the average return is non-zero.

    Args:
        returns (Array1d): One-dimensional array of returns.
        ann_factor (float): Annualization factor.

    Returns:
        float: Sortino ratio for the given strategy.
    """
    avg_annualized_return = np.nanmean(returns) * ann_factor
    downside_risk = downside_risk_1d_nb(returns, ann_factor)
    if downside_risk == 0:
        if avg_annualized_return == 0:
            return np.nan
        return np.inf
    return avg_annualized_return / downside_risk


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sortino_ratio_nb(
    returns: tp.Array2d,
    ann_factor: float,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute the Sortino ratio for each column of a 2D returns array.

    Applies `sortino_ratio_1d_nb` column-wise within a specified simulation range.

    Args:
        returns (Array2d): Two-dimensional array of returns.
        ann_factor (float): Annualization factor.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: One-dimensional array of Sortino ratios for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = sortino_ratio_1d_nb(returns[_sim_start:_sim_end, col], ann_factor)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_sortino_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate the rolling sortino ratio over a specified window.

    Args:
        returns (Array2d): Array of returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array with rolling sortino ratios.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            sortino_ratio_1d_nb,
            ann_factor,
        )
    return out


@register_jitted(cache=True)
def information_ratio_1d_nb(returns: tp.Array1d, ddof: int = 0) -> float:
    """Calculate the information ratio for a strategy.

    Args:
        returns (Array1d): Array of strategy returns.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Information ratio, computed as the ratio of the mean to the standard deviation.
    """
    mean = np.nanmean(returns)
    std = generic_nb.nanstd_1d_nb(returns, ddof=ddof)
    if std == 0:
        if mean == 0:
            return np.nan
        return np.inf
    return mean / std


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def information_ratio_nb(
    returns: tp.Array2d,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Calculate the 2-dim information ratio for each column.

    Args:
        returns (Array2d): 2D array of strategy returns.
        ddof (int): Delta degrees of freedom.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array of information ratios for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = information_ratio_1d_nb(returns[_sim_start:_sim_end, col], ddof=ddof)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_information_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate the rolling information ratio over a specified window.

    Args:
        returns (Array2d): 2D array of strategy returns.
        window (int): Window size.
        ddof (int): Delta degrees of freedom.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array with rolling information ratios.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            information_ratio_1d_nb,
            ddof,
        )
    return out


@register_jitted(cache=True)
def beta_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ddof: int = 0,
) -> float:
    """Calculate the beta coefficient for a single dimension.

    Args:
        returns (Array1d): Array of strategy returns.
        bm_returns (Array1d): Array of benchmark returns.
        ddof (int): Delta degrees of freedom.

    Returns:
        float: Beta coefficient computed as covariance divided by variance.
    """
    cov = generic_nb.nancov_1d_nb(returns, bm_returns, ddof=ddof)
    var = generic_nb.nanvar_1d_nb(bm_returns, ddof=ddof)
    if var == 0:
        if cov == 0:
            return np.nan
        return np.inf
    return cov / var


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def beta_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Calculate the 2-dim beta for each column.

    Args:
        returns (Array2d): 2D array of strategy returns.
        bm_returns (Array2d): 2D array of benchmark returns.
        ddof (int): Delta degrees of freedom.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array of beta coefficients for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = beta_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ddof=ddof,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_beta_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate the rolling beta over a specified window.

    Args:
        returns (Array2d): 2D array of strategy returns.
        bm_returns (Array2d): 2D array of benchmark returns.
        window (int): Window size.
        ddof (int): Delta degrees of freedom.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array with rolling beta coefficients.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            beta_1d_nb,
            ddof,
        )
    return out


@register_jitted(cache=True)
def alpha_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
) -> float:
    """Calculate the annualized alpha.

    Args:
        returns (Array1d): Array of returns.
        bm_returns (Array1d): Array of benchmark returns.
        ann_factor (float): Annualization factor.

    Returns:
        float: Annualized alpha.
    """
    beta = beta_1d_nb(returns, bm_returns)
    return (np.nanmean(returns) - beta * np.nanmean(bm_returns) + 1) ** ann_factor - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def alpha_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Calculate the annualized alpha for a 2-dimensional array of returns.

    Args:
        returns (Array2d): 2D array of returns.
        bm_returns (Array2d): 2D array of benchmark returns.
        ann_factor (float): Annualization factor.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array containing annualized alpha values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = alpha_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_alpha_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Calculate the rolling annualized alpha for a 2-dimensional array of returns.

    Args:
        returns (Array2d): 2D array of returns.
        bm_returns (Array2d): 2D array of benchmark returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Array with the rolling annualized alpha values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            alpha_1d_nb,
            ann_factor,
        )
    return out


@register_jitted(cache=True)
def tail_ratio_1d_nb(returns: tp.Array1d) -> float:
    """Calculate the tail ratio by dividing the absolute 95th percentile by the absolute 5th percentile.

    Args:
        returns (Array1d): Array of returns.

    Returns:
        float: Tail ratio value.
    """
    perc_95 = np.abs(np.nanpercentile(returns, 95))
    perc_5 = np.abs(np.nanpercentile(returns, 5))
    if perc_5 == 0:
        if perc_95 == 0:
            return np.nan
        return np.inf
    return perc_95 / perc_5


@register_jitted(cache=True)
def tail_ratio_noarr_1d_nb(returns: tp.Array1d) -> float:
    """Calculate the tail ratio without allocating additional arrays.

    Args:
        returns (Array1d): Array of returns.

    Returns:
        float: Tail ratio value.
    """
    perc_95 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(returns, 95))
    perc_5 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(returns, 5))
    if perc_5 == 0:
        if perc_95 == 0:
            return np.nan
        return np.inf
    return perc_95 / perc_5


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def tail_ratio_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array1d:
    """Calculate the tail ratio for each column of a 2-dimensional returns array using
    either an array-allocating or no-array approach.

    Args:
        returns (Array2d): 2D array of returns.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.

    Returns:
        Array1d: Array of tail ratio values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[col] = tail_ratio_noarr_1d_nb(returns[_sim_start:_sim_end, col])
        else:
            out[col] = tail_ratio_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_tail_ratio_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array2d:
    """Calculate the rolling tail ratio for a 2-dimensional returns array.

    Args:
        returns (Array2d): 2D array of returns.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.

    Returns:
        Array2d: 2D array with rolling tail ratio values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                tail_ratio_noarr_1d_nb,
            )
        else:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                tail_ratio_1d_nb,
            )
    return out


@register_jitted(cache=True)
def profit_factor_1d_nb(returns: tp.Array1d) -> float:
    """Calculate the profit factor as the ratio of the sum of positive returns
    to the sum of absolute negative returns.

    Args:
        returns (Array1d): Array of returns.

    Returns:
        float: Profit factor.
    """
    numer = 0
    denom = 0
    for i in range(returns.shape[0]):
        if not np.isnan(returns[i]):
            if returns[i] > 0:
                numer += returns[i]
            elif returns[i] < 0:
                denom += abs(returns[i])
    if denom == 0:
        if numer == 0:
            return np.nan
        return np.inf
    return numer / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def profit_factor_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute the profit factor for each column in a 2D returns array.

    Args:
        returns (Array2d): 2D array of returns where each column represents a distinct series.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: 1D array containing the profit factor for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = profit_factor_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_profit_factor_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute the rolling profit factor over a specified window for each column of a 2D returns array.

    Args:
        returns (Array2d): 2D array of returns where each column represents a distinct series.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array containing the rolling profit factor values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            profit_factor_1d_nb,
        )
    return out


@register_jitted(cache=True)
def common_sense_ratio_1d_nb(returns: tp.Array1d) -> float:
    """Compute the common sense ratio for a 1D returns array.

    Combines the tail ratio and profit factor computed for the returns.

    Args:
        returns (Array1d): 1D array of returns.

    Returns:
        float: Common sense ratio calculated as the product of the tail ratio and profit factor.
    """
    tail_ratio = tail_ratio_1d_nb(returns)
    profit_factor = profit_factor_1d_nb(returns)
    return tail_ratio * profit_factor


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def common_sense_ratio_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute the common sense ratio for each column in a 2D returns array.

    Args:
        returns (Array2d): 2D array of returns where each column represents a distinct series.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: 1D array containing the common sense ratio for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = common_sense_ratio_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_common_sense_ratio_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute the rolling common sense ratio over a specified window for each column in a 2D returns array.

    Args:
        returns (Array2d): 2D array of returns where each column represents a distinct series.
        window (int): Window size.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2D array containing the rolling common sense ratio for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            common_sense_ratio_1d_nb,
        )
    return out


@register_jitted(cache=True)
def value_at_risk_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """Compute the value at risk (VaR) for a 1D returns series.

    Args:
        returns (Array1d): 1D array of returns.
        cutoff (float): Fractional cutoff level.

    Returns:
        float: VaR computed as the 100 * cutoff percentile of the returns.
    """
    return np.nanpercentile(returns, 100 * cutoff)


@register_jitted(cache=True)
def value_at_risk_noarr_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """Compute the value at risk (VaR) for a 1D returns series without allocating additional arrays.

    Args:
        returns (Array1d): 1D array of returns.
        cutoff (float): Fractional cutoff level.

    Returns:
        float: VaR computed as the 100 * cutoff percentile of the returns.
    """
    return generic_nb.nanpercentile_noarr_1d_nb(returns, 100 * cutoff)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        cutoff=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_at_risk_nb(
    returns: tp.Array2d,
    cutoff: float = 0.05,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array1d:
    """Compute the value at risk (VaR) for each column in a 2D returns array.

    Depending on the `noarr_mode` flag, this function uses either a no-allocation method or
    the standard approach to compute VaR.

    Args:
        returns (Array2d): 2D array of returns where each column represents a distinct series.
        cutoff (float): Fractional cutoff level.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.

            If True, uses `value_at_risk_noarr_1d_nb`; otherwise, uses `value_at_risk_1d_nb`.

    Returns:
        Array1d: 1D array containing the VaR values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[col] = value_at_risk_noarr_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
        else:
            out[col] = value_at_risk_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        cutoff=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_value_at_risk_nb(
    returns: tp.Array2d,
    window: int,
    cutoff: float = 0.05,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array2d:
    """Compute rolling value at risk.

    Computes the rolling value at risk for each column in a 2D returns array using the specified window.

    Args:
        returns (Array2d): 2D array of returns.
        window (int): Window size.
        cutoff (float): Fractional cutoff level.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.

    Returns:
        Array2d: 2D array containing the computed rolling value at risk values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                value_at_risk_noarr_1d_nb,
                cutoff,
            )
        else:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                value_at_risk_1d_nb,
                cutoff,
            )
    return out


@register_jitted(cache=True)
def cond_value_at_risk_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """Compute conditional value at risk (CVaR) for a 1D returns stream.

    Args:
        returns (Array1d): 1D array of returns.
        cutoff (float): Fractional cutoff level.

    Returns:
        float: Computed conditional value at risk.
    """
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[: cutoff_index + 1])


@register_jitted(cache=True)
def cond_value_at_risk_noarr_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """Compute conditional value at risk (CVaR) for a 1D returns stream without additional array allocation.

    Args:
        returns (Array1d): 1D array of returns.
        cutoff (float): Fractional cutoff level.

    Returns:
        float: Computed conditional value at risk.
    """
    return generic_nb.nanpartition_mean_noarr_1d_nb(returns, cutoff * 100)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        cutoff=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cond_value_at_risk_nb(
    returns: tp.Array2d,
    cutoff: float = 0.05,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array1d:
    """Compute conditional value at risk (CVaR) across a 2D returns array.

    Computes the CVaR for each column in a 2D returns array based on the provided cutoff level.

    Args:
        returns (Array2d): 2D array of returns.
        cutoff (float): Fractional cutoff level.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.

    Returns:
        Array1d: Array of computed CVaR values for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[col] = cond_value_at_risk_noarr_1d_nb(
                returns[_sim_start:_sim_end, col], cutoff=cutoff
            )
        else:
            out[col] = cond_value_at_risk_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        cutoff=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_cond_value_at_risk_nb(
    returns: tp.Array2d,
    window: int,
    cutoff: float = 0.05,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array2d:
    """Compute rolling conditional value at risk (CVaR).

    Computes the rolling CVaR for each column in a 2D returns array using the specified rolling window.

    Args:
        returns (Array2d): 2D array of returns.
        window (int): Window size.
        cutoff (float): Fractional cutoff level.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.
        noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.

    Returns:
        Array2d: 2D array containing the computed rolling CVaR values.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                cond_value_at_risk_noarr_1d_nb,
                cutoff,
            )
        else:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                cond_value_at_risk_1d_nb,
                cutoff,
            )
    return out


@register_jitted(cache=True)
def capture_ratio_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[float] = None,
) -> float:
    """Compute the capture ratio between asset returns and benchmark returns.

    Calculates the capture ratio as the ratio of the asset's annualized return
    to the benchmark's annualized return. If the benchmark's annualized return is zero,
    returns infinity if the asset's return is non-zero, or NaN if both are zero.

    Args:
        returns (Array1d): 1D array of asset returns.
        bm_returns (Array1d): 1D array of benchmark returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[float]): Number of periods for annualization.

            Defaults to the length of the returns array.

    Returns:
        float: Computed capture ratio.
    """
    annualized_return1 = annualized_return_1d_nb(
        returns,
        ann_factor,
        log_returns=log_returns,
        periods=periods,
    )
    annualized_return2 = annualized_return_1d_nb(
        bm_returns,
        ann_factor,
        log_returns=log_returns,
        periods=periods,
    )
    if annualized_return2 == 0:
        if annualized_return1 == 0:
            return np.nan
        return np.inf
    return annualized_return1 / annualized_return2


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        periods=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Return the 2-d version of `capture_ratio_1d_nb`.

    Args:
        returns (Array2d): Returns array with shape (n_periods, n_assets).
        bm_returns (Array2d): Benchmark returns array with matching dimensions.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[FlexArray1dLike]): Number of periods for annualization.

            Provided as a scalar or per column.

            Defaults to the length of the simulation range.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Capture ratios computed for each column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if periods is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(periods).astype(int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = capture_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            periods=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Return the rolling version of `capture_ratio_1d_nb`.

    Args:
        returns (Array2d): Array of returns with shape (n_periods, n_assets).
        bm_returns (Array2d): Array of benchmark returns with matching dimensions.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Rolling capture ratios with the same shape as `returns`.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            capture_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def up_capture_ratio_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[float] = None,
) -> float:
    """Return the capture ratio for periods with positive benchmark returns.

    Args:
        returns (Array1d): 1-d array of returns.
        bm_returns (Array1d): 1-d array of benchmark returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[float]): Number of periods for annualization.

            Defaults to the length of the returns array.

    Returns:
        float: Calculated up capture ratio.
    """
    if periods is None:
        periods = returns.shape[0]

    def _annualized_pos_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if log_returns:
                    _a = np.exp(a[i]) - 1
                else:
                    _a = a[i]
                if np.isnan(ann_ret):
                    ann_ret = 1.0
                if _a > 0:
                    ann_ret *= _a + 1.0
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        if periods == 0:
            return np.nan
        return ann_ret ** (ann_factor / periods) - 1

    annualized_return = _annualized_pos_return(returns)
    annualized_bm_return = _annualized_pos_return(bm_returns)
    if annualized_bm_return == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / annualized_bm_return


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        periods=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def up_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Return the 2-d version of `up_capture_ratio_1d_nb`.

    Args:
        returns (Array2d): 2-d array of returns.
        bm_returns (Array2d): 2-d array of benchmark returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[FlexArray1dLike]): Number of periods for annualization.

            Provided as a scalar or per column.

            Defaults to the length of the simulation range.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Up capture ratios computed for each series.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if periods is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(periods).astype(int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = up_capture_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            periods=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_up_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Return the rolling version of `up_capture_ratio_1d_nb`.

    Args:
        returns (Array2d): 2-d array of returns.
        bm_returns (Array2d): 2-d array of benchmark returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: Rolling up capture ratios with the same shape as `returns`.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            up_capture_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def down_capture_ratio_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[float] = None,
) -> float:
    """Compute the down capture ratio for periods with negative benchmark returns.

    This function computes the ratio of the asset's annualized negative return
    to the benchmark's annualized negative return. Only periods where the benchmark
    return is negative are considered.

    Args:
        returns (Array1d): 1-dimensional array of asset returns.
        bm_returns (Array1d): 1-dimensional array of benchmark returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[float]): Number of periods for annualization.

            Defaults to the length of the returns array.

    Returns:
        float: Computed down capture ratio.

            Returns `np.nan` if no negative returns are found, or `np.inf` if the benchmark's
            annualized negative return is zero while the asset's is non-zero.
    """
    if periods is None:
        periods = returns.shape[0]

    def _annualized_neg_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if log_returns:
                    _a = np.exp(a[i]) - 1
                else:
                    _a = a[i]
                if np.isnan(ann_ret):
                    ann_ret = 1.0
                if _a < 0:
                    ann_ret *= _a + 1.0
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        if periods == 0:
            return np.nan
        return ann_ret ** (ann_factor / periods) - 1

    annualized_return = _annualized_neg_return(returns)
    annualized_bm_return = _annualized_neg_return(bm_returns)
    if annualized_bm_return == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / annualized_bm_return


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        periods=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def down_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    periods: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Compute the down capture ratio for each column using 2-dimensional arrays.

    This function calculates the down capture ratio for each column by computing the ratio of the
    asset's annualized negative return to the benchmark's annualized negative return.
    Only periods with negative benchmark returns are considered.

    Args:
        returns (Array2d): 2-dimensional array of asset returns.
        bm_returns (Array2d): 2-dimensional array of benchmark returns.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        periods (Optional[FlexArray1dLike]): Number of periods for annualization.

            Provided as a scalar or per column.

            Defaults to the length of the simulation range.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array1d: Array of down capture ratios for each column.

            Each element is `np.nan` or `np.inf` if the computation is invalid for that column.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape[1], np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if periods is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(periods).astype(int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = down_capture_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            periods=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_down_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Compute the rolling down capture ratio over a moving window for each column.

    This function calculates the down capture ratio over a rolling window by applying the computation
    to segments of the asset and benchmark returns. It returns a 2-dimensional array where each column
    contains the rolling down capture ratio.

    Args:
        returns (Array2d): 2-dimensional array of asset returns.
        bm_returns (Array2d): 2-dimensional array of benchmark returns.
        window (int): Window size.
        ann_factor (float): Annualization factor.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        minp (Optional[int]): Minimum number of observations required.
        sim_start (Optional[FlexArray1dLike]): Start position of the simulation range (inclusive).

            Provided as a scalar or per column.
        sim_end (Optional[FlexArray1dLike]): End position of the simulation range (exclusive).

            Provided as a scalar or per column.

    Returns:
        Array2d: 2-dimensional array of rolling down capture ratios for each column,
            with invalid computations represented by `np.nan`.

    !!! tip
        This function is parallelizable.
    """
    out = np.full(returns.shape, np.nan, dtype=float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            down_capture_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out
