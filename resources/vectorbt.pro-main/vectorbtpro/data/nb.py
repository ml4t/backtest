# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing Numba-compiled functions for generating data.

This module offers a collection of functions compiled with Numba to efficiently
simulate data using statistical models such as cumulative return processes and
Geometric Brownian Motion. All functions are designed to work with NumPy arrays
and other Numba-compatible types.
"""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = []


@register_jitted(cache=True)
def generate_random_data_1d_nb(
    n_rows: int,
    start_value: float = 100.0,
    mean: float = 0.0,
    std: float = 0.01,
    symmetric: bool = False,
) -> tp.Array1d:
    """Generate a one-dimensional array of data using cumulative product of returns drawn
    from a Gaussian distribution.

    Args:
        n_rows (int): Number of data points to generate.
        start_value (float): Initial value at the start of the series.
        mean (float): Mean of the Gaussian distribution for sampling returns.
        std (float): Standard deviation of the Gaussian distribution for sampling returns.
        symmetric (bool): If True, converts negative returns to be symmetric to positive ones,
            reducing their negative impact.

    Returns:
        Array1d: Array of generated data.
    """
    out = np.empty(n_rows, dtype=float_)

    for i in range(n_rows):
        if i == 0:
            prev_value = start_value
        else:
            prev_value = out[i - 1]
        return_ = np.random.normal(mean, std)
        if symmetric and return_ < 0:
            return_ = -abs(return_) / (1 + abs(return_))
        out[i] = prev_value * (1 + return_)

    return out


@register_jitted(cache=True, tags={"can_parallel"})
def generate_random_data_nb(
    shape: tp.Shape,
    start_value: tp.FlexArray1dLike = 100.0,
    mean: tp.FlexArray1dLike = 0.0,
    std: tp.FlexArray1dLike = 0.01,
    symmetric: tp.FlexArray1dLike = False,
) -> tp.Array2d:
    """Generate a two-dimensional array of data using cumulative product of returns drawn
    from a Gaussian distribution, applied column-wise.

    Args:
        shape (Shape): Tuple specifying the dimensions of the output array.
        start_value (FlexArray1dLike): Initial value.

            Provided as a scalar or per column.
        mean (FlexArray1dLike): Mean value of the Gaussian distribution.

            Provided as a scalar or per column.
        std (FlexArray1dLike): Standard deviation of the Gaussian distribution.

            Provided as a scalar or per column.
        symmetric (FlexArray1dLike): Boolean flag indicating whether to adjust negative returns to be symmetric
            to positive returns.

            Provided as a scalar or per column.

    Returns:
        Array2d: Two-dimensional array of generated data.
    """
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    mean_ = to_1d_array_nb(np.asarray(mean))
    std_ = to_1d_array_nb(np.asarray(std))
    symmetric_ = to_1d_array_nb(np.asarray(symmetric))

    out = np.empty(shape, dtype=float_)

    for col in prange(shape[1]):
        out[:, col] = generate_random_data_1d_nb(
            shape[0],
            start_value=flex_select_1d_pc_nb(start_value_, col),
            mean=flex_select_1d_pc_nb(mean_, col),
            std=flex_select_1d_pc_nb(std_, col),
            symmetric=flex_select_1d_pc_nb(symmetric_, col),
        )

    return out


@register_jitted(cache=True)
def generate_gbm_data_1d_nb(
    n_rows: int,
    start_value: float = 100.0,
    mean: float = 0.0,
    std: float = 0.01,
    dt: float = 1.0,
) -> tp.Array2d:
    """Generate a one-dimensional array of data using Geometric Brownian Motion (GBM).

    Args:
        n_rows (int): Number of data points to generate.
        start_value (float): Initial value of the series.
        mean (float): Drift factor in the GBM process.
        std (float): Volatility factor in the GBM process.
        dt (float): Time increment per period.

    Returns:
        Array2d: Array containing the simulated GBM data.
    """
    out = np.empty(n_rows, dtype=float_)

    for i in range(n_rows):
        if i == 0:
            prev_value = start_value
        else:
            prev_value = out[i - 1]
        rand = np.random.standard_normal()
        out[i] = prev_value * np.exp((mean - 0.5 * std**2) * dt + std * np.sqrt(dt) * rand)

    return out


@register_jitted(cache=True, tags={"can_parallel"})
def generate_gbm_data_nb(
    shape: tp.Shape,
    start_value: tp.FlexArray1dLike = 100.0,
    mean: tp.FlexArray1dLike = 0.0,
    std: tp.FlexArray1dLike = 0.01,
    dt: tp.FlexArray1dLike = 1.0,
) -> tp.Array2d:
    """Generate a two-dimensional array of data using Geometric Brownian Motion (GBM), computed column-wise.

    Args:
        shape (Shape): Tuple specifying the dimensions of the output array.
        start_value (FlexArray1dLike): Initial value.

            Provided as a scalar or per column.
        mean (FlexArray1dLike): Drift factor for the GBM process.

            Provided as a scalar or per column.
        std (FlexArray1dLike): Volatility factor for the GBM process.

            Provided as a scalar or per column.
        dt (FlexArray1dLike): Time increment used for the simulation.

            Provided as a scalar or per column.

    Returns:
        Array2d: Two-dimensional array containing the simulated GBM data.
    """
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    mean_ = to_1d_array_nb(np.asarray(mean))
    std_ = to_1d_array_nb(np.asarray(std))
    dt_ = to_1d_array_nb(np.asarray(dt))

    out = np.empty(shape, dtype=float_)

    for col in prange(shape[1]):
        out[:, col] = generate_gbm_data_1d_nb(
            shape[0],
            start_value=flex_select_1d_pc_nb(start_value_, col),
            mean=flex_select_1d_pc_nb(mean_, col),
            std=flex_select_1d_pc_nb(std_, col),
            dt=flex_select_1d_pc_nb(dt_, col),
        )

    return out
