# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing Numba-compiled functions used in portfolio simulation.

Provides a suite of Numba-compiled functions for portfolio simulation, including generating and
filling orders. All functions accept only NumPy arrays and other Numba-compatible types.

!!! note
    Vectorbtpro treats matrices as first-class citizens and expects input arrays to be 2-dimensional,
    unless a function has the `_1d` suffix or is used as input to another function.

    All functions supplied as arguments must be Numba-compiled.

    Records preserve their original creation order.

!!! warning
    Round-off error accumulation is possible.
    See [here](https://en.wikipedia.org/wiki/Round-off_error#Accumulation_of_roundoff_error) for an explanation.

    Rounding errors can cause trades and positions to not close properly:

    ```pycon
    >>> print('%.50f' % 0.1)  # has positive error
    0.10000000000000000555111512312578270211815834045410

    >>> # many buy transactions with positive error -> cannot close position
    >>> sum([0.1 for _ in range(1000000)]) - 100000
    1.3328826753422618e-06

    >>> print('%.50f' % 0.3)  # has negative error
    0.29999999999999998889776975374843459576368331909180

    >>> # many sell transactions with negative error -> cannot close position
    >>> 300000 - sum([0.3 for _ in range(1000000)])
    5.657668225467205e-06
    ```

    While vectorbtpro implements tolerance checks when comparing floats for equality, repeatedly
    adding or subtracting small values may introduce a noticeable error that cannot be corrected afterwards.

    To mitigate this issue, avoid performing numerous micro-transactions of the same sign.
    For instance, use reduction by `np.inf` or `position_now` to close a long/short position.

    See `vectorbtpro.utils.math_` for the current tolerance values.

!!! warning
    Use `parallel=True` only if your columns are independent.
"""

from vectorbtpro.portfolio.nb.analysis import *
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb.ctx_helpers import *
from vectorbtpro.portfolio.nb.from_order_func import *
from vectorbtpro.portfolio.nb.from_orders import *
from vectorbtpro.portfolio.nb.from_signals import *
from vectorbtpro.portfolio.nb.iter_ import *
from vectorbtpro.portfolio.nb.records import *

__all__ = []
