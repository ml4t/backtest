# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining named tuples and enumerated types for financial indicators."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = [
    "Pivot",
    "TrendMode",
    "HurstMethod",
    "SuperTrendAIS",
    "SuperTrendAOS",
]

__pdoc__ = {}


# ############# Enums ############# #


class PivotT(tp.NamedTuple):
    Valley: int = -1
    Peak: int = 1


Pivot = PivotT()
"""_"""

__pdoc__["Pivot"] = f"""Pivot enumeration.

Fields:
    Valley: Valley point.
    Peak: Peak point.

```python
{prettify_doc(Pivot)}
```
"""


class TrendModeT(tp.NamedTuple):
    Downtrend: int = -1
    Uptrend: int = 1


TrendMode = TrendModeT()
"""_"""

__pdoc__["TrendMode"] = f"""Trend mode enumeration.

Fields:
    Downtrend: Downtrend direction.
    Uptrend: Uptrend direction.

```python
{prettify_doc(TrendMode)}
```
"""


class HurstMethodT(tp.NamedTuple):
    Standard: int = 0
    LogRS: int = 1
    RS: int = 2
    DMA: int = 3
    DSOD: int = 4


HurstMethod = HurstMethodT()
"""_"""

__pdoc__["HurstMethod"] = f"""Hurst method enumeration.

Fields:
    Standard: Standard method.
    LogRS: Rescaled range (R/S) method with logarithmically distributed window sizes.
    RS: Rescaled range (R/S) method with linearly distributed window sizes.
    DMA: Detrending moving average method with linearly distributed window sizes.
    DSOD: Discrete second order derivative method.

```python
{prettify_doc(HurstMethod)}
```
"""


# ############# States ############# #


class SuperTrendAIS(tp.NamedTuple):
    i: int
    high: float
    low: float
    close: float
    prev_close: float
    prev_upper: float
    prev_lower: float
    prev_direction: int
    nobs: int
    weighted_avg: float
    old_wt: float
    period: int
    multiplier: float


__pdoc__[
    "SuperTrendAIS"
] = """Named tuple representing the input state for the Super Trend indicator calculation in
`vectorbtpro.indicators.nb.supertrend_acc_nb`."""


class SuperTrendAOS(tp.NamedTuple):
    nobs: int
    weighted_avg: float
    old_wt: float
    upper: float
    lower: float
    trend: float
    direction: int
    long: float
    short: float


__pdoc__[
    "SuperTrendAOS"
] = """Named tuple representing the output state for the Super Trend indicator calculation in
`vectorbtpro.indicators.nb.supertrend_acc_nb`."""
