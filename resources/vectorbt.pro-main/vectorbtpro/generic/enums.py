# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing named tuples and enumerations for generic data."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = [
    "BarZone",
    "WType",
    "RangeStatus",
    "InterpMode",
    "RescaleMode",
    "ErrorType",
    "DistanceMeasure",
    "OverlapMode",
    "DrawdownStatus",
    "range_dt",
    "pattern_range_dt",
    "drawdown_dt",
    "RollSumAIS",
    "RollSumAOS",
    "RollProdAIS",
    "RollProdAOS",
    "RollMeanAIS",
    "RollMeanAOS",
    "RollStdAIS",
    "RollStdAOS",
    "RollZScoreAIS",
    "RollZScoreAOS",
    "WMMeanAIS",
    "WMMeanAOS",
    "EWMMeanAIS",
    "EWMMeanAOS",
    "EWMStdAIS",
    "EWMStdAOS",
    "VidyaAIS",
    "VidyaAOS",
    "RollCovAIS",
    "RollCovAOS",
    "RollCorrAIS",
    "RollCorrAOS",
    "RollOLSAIS",
    "RollOLSAOS",
]

__pdoc__ = {}


# ############# Enums ############# #


class BarZoneT(tp.NamedTuple):
    Open: int = 0
    Middle: int = 1
    Close: int = 2


BarZone = BarZoneT()
"""_"""

__pdoc__["BarZone"] = f"""Bar zone enumeration.

```python
{prettify_doc(BarZone)}
```
"""


class WTypeT(tp.NamedTuple):
    Simple: int = 0
    Weighted: int = 1
    Exp: int = 2
    Wilder: int = 3
    Vidya: int = 4


WType = WTypeT()
"""_"""

__pdoc__["WType"] = f"""Rolling weighting type enumeration.

```python
{prettify_doc(WType)}
```
"""


class RangeStatusT(tp.NamedTuple):
    Open: int = 0
    Closed: int = 1


RangeStatus = RangeStatusT()
"""_"""

__pdoc__["RangeStatus"] = f"""Range status enumeration.

```python
{prettify_doc(RangeStatus)}
```
"""


class InterpModeT(tp.NamedTuple):
    Linear: int = 0
    Nearest: int = 1
    Discrete: int = 2
    Mixed: int = 3


InterpMode = InterpModeT()
"""_"""

__pdoc__["InterpMode"] = f"""Interpolation mode enumeration.

```python
{prettify_doc(InterpMode)}
```

Fields:
    Linear: Linear interpolation.

        For example: `[1.0, 2.0, 3.0]` to 5 yields `[1.0, 1.5, 2.0, 2.5, 3.0]`
    Nearest: Nearest-neighbor interpolation.

        For example: `[1.0, 2.0, 3.0]` to 5 yields `[1.0, 1.0, 2.0, 3.0, 3.0]`
    Discrete: Discrete interpolation.

        For example: `[1.0, 2.0, 3.0]` to 5 yields `[1.0, np.nan, 2.0, np.nan, 3.0]`
    Mixed: Mixed interpolation.

        For example: `[1.0, 2.0, 3.0]` to 5 yields `[1.0, 1.5, 2.0, 2.5, 3.0]`
"""


class RescaleModeT(tp.NamedTuple):
    MinMax: int = 0
    Rebase: int = 1
    Disable: int = 2


RescaleMode = RescaleModeT()
"""_"""

__pdoc__["RescaleMode"] = f"""Rescaling mode enumeration.

```python
{prettify_doc(RescaleMode)}
```

Fields:
    MinMax: Rescales an array from its min-max range to match the min-max range of another array.

        For example: `[3.0, 2.0, 1.0]` to `[10, 11, 12]` yields `[12.0, 11.0, 10.0]`

        Use this to search for patterns irrespective of their vertical scale.
    Rebase: Rebases an array to the first value of another array.

        For example: `[3.0, 2.0, 1.0]` to `[10, 11, 12]` yields `[10.0, 6.6, 3.3]`

        Use this to search for percentage changes.
    Disable: Disables rescaling, leaving the array unchanged.

        For example: `[3.0, 2.0, 1.0]` to `[10, 11, 12]` yields `[3.0, 2.0, 1.0]`

        Use this to search for particular numbers.
"""


class ErrorTypeT(tp.NamedTuple):
    Absolute: int = 0
    Relative: int = 1


ErrorType = ErrorTypeT()
"""_"""

__pdoc__["ErrorType"] = f"""Error type enumeration.

```python
{prettify_doc(ErrorType)}
```

Fields:
    Absolute: Represents the absolute error `(x1 - x0)`.
    Relative: Represents the relative error `((x1 - x0) / x0)`.
"""


class DistanceMeasureT(tp.NamedTuple):
    MAE: int = 0
    MSE: int = 1
    RMSE: int = 2


DistanceMeasure = DistanceMeasureT()
"""_"""

__pdoc__["DistanceMeasure"] = f"""Distance measure enumeration.

```python
{prettify_doc(DistanceMeasure)}
```

Fields:
    MAE: Mean absolute error.
    MSE: Mean squared error.
    RMSE: Root mean squared error.
"""


class OverlapModeT(tp.NamedTuple):
    AllowAll: int = -2
    Allow: int = -1
    Disallow: int = 0


OverlapMode = OverlapModeT()
"""_"""

__pdoc__["OverlapMode"] = f"""Overlap mode enumeration.

```python
{prettify_doc(OverlapMode)}
```

Fields:
    AllowAll: Allows any overlapping ranges, including those starting at the same row.
    Allow: Allows overlapping ranges only if they do not share the same starting row.
    Disallow: Disallows any overlapping ranges.

Any other positive number specifies a minimum number of overlapping rows required to
select the range with the highest similarity.
"""


class DrawdownStatusT(tp.NamedTuple):
    Active: int = 0
    Recovered: int = 1


DrawdownStatus = DrawdownStatusT()
"""_"""

__pdoc__["DrawdownStatus"] = f"""Drawdown status enumeration.

```python
{prettify_doc(DrawdownStatus)}
```
"""

# ############# Records ############# #

range_dt = np.dtype(
    [
        ("id", int_),
        ("col", int_),
        ("start_idx", int_),
        ("end_idx", int_),
        ("status", int_),
    ],
    align=True,
)
"""_"""

__pdoc__["range_dt"] = f"""NumPy dtype of range records.

```python
{prettify_doc(range_dt)}
```
"""

pattern_range_dt = np.dtype(
    [
        ("id", int_),
        ("col", int_),
        ("start_idx", int_),
        ("end_idx", int_),
        ("status", int_),
        ("similarity", float_),
    ],
    align=True,
)
"""_"""

__pdoc__["pattern_range_dt"] = f"""NumPy dtype of pattern range records.

```python
{prettify_doc(pattern_range_dt)}
```
"""

drawdown_dt = np.dtype(
    [
        ("id", int_),
        ("col", int_),
        ("start_idx", int_),
        ("valley_idx", int_),
        ("end_idx", int_),
        ("start_val", float_),
        ("valley_val", float_),
        ("end_val", float_),
        ("status", int_),
    ],
    align=True,
)
"""_"""

__pdoc__["drawdown_dt"] = f"""NumPy dtype for drawdown records.

```python
{prettify_doc(drawdown_dt)}
```
"""


# ############# States ############# #


class RollSumAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["RollSumAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_sum_acc_nb`."""
)


class RollSumAOS(tp.NamedTuple):
    cumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollSumAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_sum_acc_nb`."""
)


class RollProdAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumprod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["RollProdAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_prod_acc_nb`."""
)


class RollProdAOS(tp.NamedTuple):
    cumprod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollProdAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_prod_acc_nb`."""
)


class RollMeanAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["RollMeanAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_mean_acc_nb`."""
)


class RollMeanAOS(tp.NamedTuple):
    cumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollMeanAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_mean_acc_nb`."""
)


class RollStdAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__["RollStdAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_std_acc_nb`."""
)


class RollStdAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollStdAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_std_acc_nb`."""
)


class RollZScoreAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__["RollZScoreAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_zscore_acc_nb`."""
)


class RollZScoreAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollZScoreAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_zscore_acc_nb`."""
)


class WMMeanAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    wcumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["WMMeanAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.wm_mean_acc_nb`."""
)


class WMMeanAOS(tp.NamedTuple):
    cumsum: float
    wcumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["WMMeanAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.wm_mean_acc_nb`."""
)


class EWMMeanAIS(tp.NamedTuple):
    i: int
    value: float
    old_wt: float
    weighted_avg: float
    nobs: int
    alpha: float
    minp: tp.Optional[int]
    adjust: bool


__pdoc__[
    "EWMMeanAIS"
] = """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.ewm_mean_acc_nb`.

To obtain `alpha`, use one of the following:

* `vectorbtpro.generic.nb.rolling.alpha_from_com_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_span_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_halflife_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_wilder_nb`"""


class EWMMeanAOS(tp.NamedTuple):
    old_wt: float
    weighted_avg: float
    nobs: int
    value: float


__pdoc__["EWMMeanAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.ewm_mean_acc_nb`."""
)


class EWMStdAIS(tp.NamedTuple):
    i: int
    value: float
    mean_x: float
    mean_y: float
    cov: float
    sum_wt: float
    sum_wt2: float
    old_wt: float
    nobs: int
    alpha: float
    minp: tp.Optional[int]
    adjust: bool


__pdoc__[
    "EWMStdAIS"
] = """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.ewm_std_acc_nb`.

For tips on `alpha`, see `EWMMeanAIS`.
"""


class EWMStdAOS(tp.NamedTuple):
    mean_x: float
    mean_y: float
    cov: float
    sum_wt: float
    sum_wt2: float
    old_wt: float
    nobs: int
    value: float


__pdoc__["EWMStdAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.ewm_std_acc_nb`."""
)


class VidyaAIS(tp.NamedTuple):
    i: int
    prev_value: float
    value: float
    pre_window_prev_value: float
    pre_window_value: float
    pos_cumsum: float
    neg_cumsum: float
    prev_vidya: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["VidyaAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.vidya_acc_nb`."""
)


class VidyaAOS(tp.NamedTuple):
    pos_cumsum: float
    neg_cumsum: float
    nancnt: int
    window_len: int
    cmo: float
    vidya: float


__pdoc__["VidyaAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.vidya_acc_nb`."""
)


class RollCovAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    cumsum1: float
    cumsum2: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__["RollCovAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_cov_acc_nb`."""
)


class RollCovAOS(tp.NamedTuple):
    cumsum1: float
    cumsum2: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollCovAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_cov_acc_nb`."""
)


class RollCorrAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_sq2: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["RollCorrAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_corr_acc_nb`."""
)


class RollCorrAOS(tp.NamedTuple):
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_sq2: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__["RollCorrAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_corr_acc_nb`."""
)


class RollOLSAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    validcnt: int
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__["RollOLSAIS"] = (
    """Named tuple representing the input state for `vectorbtpro.generic.nb.rolling.rolling_ols_acc_nb`."""
)


class RollOLSAOS(tp.NamedTuple):
    validcnt: int
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    slope_value: float
    intercept_value: float


__pdoc__["RollOLSAOS"] = (
    """Named tuple representing the output state for `vectorbtpro.generic.nb.rolling.rolling_ols_acc_nb`."""
)
