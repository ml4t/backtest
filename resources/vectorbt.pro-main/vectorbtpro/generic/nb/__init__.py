# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing Numba-compiled functions for generic data.

This package offers a collection of functions compiled with Numba that are used by accessors
and various parts of a backtesting pipeline, including technical indicators. All functions
accept only NumPy arrays and other Numba-compatible types.

!!! note
    Vectorbtpro treats matrices as first-class citizens and expects input arrays to be 2-dim,
    unless a function has the `_1d` suffix or is intended for use as input to another function.
    Data is processed along the index (axis 0).

    Rolling functions with `minp=None` set `min_periods` to the window size.

    All functions passed as arguments must be Numba-compiled.

    Records retain the order in which they are created.

!!! warning
    Use `parallel=True` only if your columns are independent.
"""

from vectorbtpro.generic.nb.apply_reduce import *
from vectorbtpro.generic.nb.base import *
from vectorbtpro.generic.nb.iter_ import *
from vectorbtpro.generic.nb.patterns import *
from vectorbtpro.generic.nb.records import *
from vectorbtpro.generic.nb.rolling import *
from vectorbtpro.generic.nb.sim_range import *

__all__ = []
