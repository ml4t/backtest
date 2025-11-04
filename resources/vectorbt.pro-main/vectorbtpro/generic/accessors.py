# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom Pandas accessors for generic data.

Custom accessors extend Pandas objects:

* `GenericSRAccessor` -> `pd.Series.vbt.*`
* `GenericDFAccessor` -> `pd.DataFrame.vbt.*`

```pycon
>>> from vectorbtpro import *

>>> # vectorbtpro.generic.accessors.GenericAccessor.rolling_mean
>>> pd.Series([1, 2, 3, 4]).vbt.rolling_mean(2)
0    NaN
1    1.5
2    2.5
3    3.5
dtype: float64
```

The accessors inherit from `vectorbtpro.base.accessors` and serve as the base for more specialized
accessors such as `vectorbtpro.signals.accessors` and `vectorbtpro.returns.accessors`.

!!! note
    Grouping is only supported by methods accepting the `group_by` argument.

    Accessors do not utilize caching.

Run for the examples below:

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1],
...     'c': [1, 2, 3, 2, 1]
... }, index=pd.Index(pd.date_range("2020", periods=5)))
>>> df
            a  b  c
2020-01-01  1  5  1
2020-01-02  2  4  2
2020-01-03  3  3  3
2020-01-04  4  2  2
2020-01-05  5  1  1

>>> sr = pd.Series(np.arange(10), index=pd.date_range("2020", periods=10))
>>> sr
2020-01-01    0
2020-01-02    1
2020-01-03    2
2020-01-04    3
2020-01-05    4
2020-01-06    5
2020-01-07    6
2020-01-08    7
2020-01-09    8
2020-01-10    9
dtype: int64
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `GenericAccessor.metrics`.

```pycon
>>> df2 = pd.DataFrame({
...     'a': [np.nan, 2, 3],
...     'b': [4, np.nan, 5],
...     'c': [6, 7, np.nan]
... }, index=['x', 'y', 'z'])

>>> df2.vbt(freq='d').stats(column='a')
Start                      x
End                        z
Period       3 days 00:00:00
Count                      2
Mean                     2.5
Std                 0.707107
Min                      2.0
Median                   2.5
Max                      3.0
Min Index                  y
Max Index                  z
Name: a, dtype: object
```

### Mapping

Mapping can be configured both in `GenericAccessor` and via `GenericAccessor.stats`:

```pycon
>>> mapping = {x: 'test_' + str(x) for x in pd.unique(df2.values.flatten())}
>>> df2.vbt(freq='d', mapping=mapping).stats(column='a')
Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_4.0                  0
Value Counts: test_5.0                  0
Value Counts: test_6.0                  0
Value Counts: test_7.0                  0
Value Counts: test_nan                  1
Name: a, dtype: object

>>> df2.vbt(freq='d').stats(column='a', settings=dict(mapping=mapping))
UserWarning: Changing the mapping will create a copy of this object.
Consider setting it upon object creation to re-use existing cache.

Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_4.0                  0
Value Counts: test_5.0                  0
Value Counts: test_6.0                  0
Value Counts: test_7.0                  0
Value Counts: test_nan                  1
Name: a, dtype: object
```

Selecting a column before calling `stats` applies mapping to uniques in that column:

```pycon
>>> df2['a'].vbt(freq='d', mapping=mapping).stats()
Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_nan                  1
Name: a, dtype: object
```

To include all keys from the mapping, pass `incl_all_keys=True`:

```pycon
>>> df2['a'].vbt(freq='d', mapping=mapping).stats(settings=dict(incl_all_keys=True))
Start                                   x
End                                     z
Period                    3 days 00:00:00
Count                                   2
Value Counts: test_2.0                  1
Value Counts: test_3.0                  1
Value Counts: test_4.0                  0
Value Counts: test_5.0                  0
Value Counts: test_6.0                  0
Value Counts: test_7.0                  0
Value Counts: test_nan                  1
Name: a, dtype: object
```

`GenericAccessor.stats` also supports regrouping:

```pycon
>>> df2.vbt(freq='d').stats(column=0, group_by=[0, 0, 1])
Start                      x
End                        z
Period       3 days 00:00:00
Count                      4
Mean                     3.5
Std                 1.290994
Min                      2.0
Median                   3.5
Max                      5.0
Min Index                  y
Max Index                  z
Name: 0, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `GenericAccessor.subplots`.

`GenericAccessor` provides a single subplot via `GenericAccessor.plot`:

```pycon
>>> df2.vbt.plots().show()
```

![](/assets/images/api/generic_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/generic_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

from functools import partial

import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as PandasResampler

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro._settings import settings
from vectorbtpro.base import indexes, reshaping
from vectorbtpro.base.accessors import BaseAccessor, BaseDFAccessor, BaseSRAccessor
from vectorbtpro.base.indexes import repeat_index
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic import nb
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.decorators import attach_nb_methods, attach_transform_methods
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.enums import DistanceMeasure, ErrorType, InterpMode, RescaleMode, WType
from vectorbtpro.generic.plots_builder import PlotsBuilderMixin
from vectorbtpro.generic.ranges import PatternRanges, Ranges
from vectorbtpro.generic.stats_builder import StatsBuilderMixin
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.colors import adjust_opacity, map_value_to_cmap
from vectorbtpro.utils.config import Config, HybridConfig, ReadonlyConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import hybrid_method, hybrid_property
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.mapping import apply_mapping, to_value_mapping
from vectorbtpro.utils.template import substitute_templates
from vectorbtpro.utils.warnings_ import warn

try:
    import bottleneck as bn

    nanmean = bn.nanmean
    nanstd = bn.nanstd
    nansum = bn.nansum
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanmedian = bn.nanmedian
    nanargmax = bn.nanargmax
    nanargmin = bn.nanargmin
except ImportError:
    # slower NumPy
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanmedian = np.nanmedian
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin

__all__ = [
    "GenericAccessor",
    "GenericSRAccessor",
    "GenericDFAccessor",
]

__pdoc__ = {}

GenericAccessorT = tp.TypeVar("GenericAccessorT", bound="GenericAccessor")
SplitOutputT = tp.Union[tp.MaybeTuple[tp.Tuple[tp.Frame, tp.Index]], tp.BaseFigure]


nb_config = ReadonlyConfig(
    {
        "shuffle": dict(func=nb.shuffle_nb, disable_chunked=True),
        "fillna": dict(func=nb.fillna_nb),
        "bshift": dict(func=nb.bshift_nb),
        "fshift": dict(func=nb.fshift_nb),
        "diff": dict(func=nb.diff_nb),
        "pct_change": dict(func=nb.pct_change_nb),
        "ffill": dict(func=nb.ffill_nb),
        "bfill": dict(func=nb.bfill_nb),
        "fbfill": dict(func=nb.fbfill_nb),
        "cumsum": dict(func=nb.nancumsum_nb),
        "cumprod": dict(func=nb.nancumprod_nb),
        "rolling_sum": dict(func=nb.rolling_sum_nb),
        "rolling_prod": dict(func=nb.rolling_prod_nb),
        "rolling_min": dict(func=nb.rolling_min_nb),
        "rolling_max": dict(func=nb.rolling_max_nb),
        "expanding_min": dict(func=nb.expanding_min_nb),
        "expanding_max": dict(func=nb.expanding_max_nb),
        "rolling_any": dict(func=nb.rolling_any_nb),
        "rolling_all": dict(func=nb.rolling_all_nb),
        "product": dict(func=nb.nanprod_nb, is_reducing=True),
    }
)
"""_"""

__pdoc__["nb_config"] = f"""Configuration for Numba functions attached to `GenericAccessor`.

```python
{nb_config.prettify_doc()}
```
"""


@attach_nb_methods(nb_config)
class GenericAccessor(BaseAccessor, Analyzable):
    """Class representing an accessor for Series and DataFrames of any data type.

    Accessible via `pd.Series.vbt` and `pd.DataFrame.vbt`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Object containing the data.
        mapping (Optional[MappingLike]): Mapping configuration.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseAccessor`.

    !!! info
        For default settings, see `vectorbtpro._settings.generic`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        mapping: tp.Optional[tp.MappingLike] = None,
        **kwargs,
    ) -> None:
        BaseAccessor.__init__(self, wrapper, obj=obj, mapping=mapping, **kwargs)
        StatsBuilderMixin.__init__(self)
        PlotsBuilderMixin.__init__(self)

        self._mapping = mapping

    @hybrid_property
    def sr_accessor_cls(cls_or_self) -> tp.Type["GenericSRAccessor"]:
        return GenericSRAccessor

    @hybrid_property
    def df_accessor_cls(cls_or_self) -> tp.Type["GenericDFAccessor"]:
        return GenericDFAccessor

    # ############# Mapping ############# #

    @property
    def mapping(self) -> tp.Optional[tp.MappingLike]:
        """Mapping configuration.

        Returns:
            Optional[MappingLike]: Mapping configuration.
        """
        return self._mapping

    def resolve_mapping(
        self, mapping: tp.Union[None, bool, tp.MappingLike] = None
    ) -> tp.Optional[tp.Mapping]:
        """Resolve and return the mapping configuration based on the input.

        If `mapping` is set to False, mapping is disabled and the function returns None.

        Args:
            mapping (Union[None, bool, MappingLike]): Mapping configuration to resolve.

        Returns:
            Optional[Mapping]: Resolved mapping configuration.
        """
        if mapping is None:
            mapping = self.mapping
        if isinstance(mapping, bool):
            if not mapping:
                return None
            raise ValueError("Mapping cannot be True")
        if isinstance(mapping, str):
            if mapping.lower() == "index":
                mapping = self.wrapper.index
            elif mapping.lower() == "columns":
                mapping = self.wrapper.columns
            elif mapping.lower() == "groups":
                mapping = self.wrapper.get_columns()
            mapping = to_value_mapping(mapping)
        return mapping

    def apply_mapping(
        self, mapping: tp.Union[None, bool, tp.MappingLike] = None, **kwargs
    ) -> tp.SeriesFrame:
        """Apply the resolved mapping configuration to the data.

        Args:
            mapping (Union[None, bool, MappingLike]): Mapping configuration to apply.
            **kwargs: Keyword arguments for `GenericAccessor.apply_mapping`.

        Returns:
            SeriesFrame: Data with the applied mapping.
        """
        mapping = self.resolve_mapping(mapping)
        return apply_mapping(self.obj, mapping, **kwargs)

    # ############# Shifting ############# #

    def ago(
        self,
        n: tp.Union[int, tp.FrequencyLike],
        fill_value: tp.Scalar = np.nan,
        get_indexer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return a copy of the data with each value replaced by the value from `n` periods ago.

        Args:
            n (Union[int, FrequencyLike]): Number of periods to shift the data.
            fill_value (Scalar): Value to fill in for missing data.
            get_indexer_kwargs (KwargsLike): Keyword arguments for `pd.Index.get_indexer`.
            **kwargs: Keyword arguments for `GenericAccessor.fshift`.

        Returns:
            SeriesFrame: Data with the shifted values.
        """
        if checks.is_int(n):
            return self.fshift(n, fill_value=fill_value, **kwargs)
        if get_indexer_kwargs is None:
            get_indexer_kwargs = {}
        n = dt.to_timedelta(n)
        indices = self.wrapper.index.get_indexer(self.wrapper.index - n, **get_indexer_kwargs)
        new_obj = self.wrapper.fill(fill_value=fill_value)
        found_mask = indices != -1
        new_obj.iloc[np.flatnonzero(found_mask)] = self.obj.iloc[indices[found_mask]]
        return new_obj

    def any_ago(self, n: tp.Union[int, tp.FrequencyLike], **kwargs) -> tp.SeriesFrame:
        """Return a boolean Series/DataFrame indicating if any value in a rolling window of
        `n` previous periods is True.

        Args:
            n (Union[int, FrequencyLike]): Number of periods to check for True values.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_any`.

        Returns:
            SeriesFrame: Boolean Series/DataFrame indicating if any value is True.
        """
        wrap_kwargs = kwargs.pop("wrap_kwargs", {})
        wrap_kwargs = merge_dicts(dict(fillna=False, dtype=bool), wrap_kwargs)
        if checks.is_int(n):
            return self.rolling_any(n, wrap_kwargs=wrap_kwargs, **kwargs)
        return self.rolling_apply(n, "any", wrap_kwargs=wrap_kwargs, **kwargs)

    def all_ago(self, n: tp.Union[int, tp.FrequencyLike], **kwargs) -> tp.SeriesFrame:
        """Return a boolean Series/DataFrame indicating if all values in a rolling window of
        `n` previous periods are True.

        Args:
            n (Union[int, FrequencyLike]): Number of periods to check for True values.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_all`.

        Returns:
            SeriesFrame: Boolean Series/DataFrame indicating if all values are True.
        """
        wrap_kwargs = kwargs.pop("wrap_kwargs", {})
        wrap_kwargs = merge_dicts(dict(fillna=False, dtype=bool), wrap_kwargs)
        if checks.is_int(n):
            return self.rolling_all(n, wrap_kwargs=wrap_kwargs, **kwargs)
        return self.rolling_apply(n, "all", wrap_kwargs=wrap_kwargs, **kwargs)

    # ############# Rolling ############# #

    def rolling_idxmin(
        self,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        local: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the index of the minimum value within each rolling window.

        Args:
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            local (bool): If True, computes the index of the minimum value within the local window.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Series/DataFrame containing the indices of the minimum values
                within the rolling windows.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_argmin_nb`
        """
        if window is None:
            window = self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.rolling_argmin_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp, local=local)
        if not local:
            wrap_kwargs = merge_dicts(dict(to_index=True), wrap_kwargs)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def expanding_idxmin(self, minp: tp.Optional[int] = 1, **kwargs) -> tp.SeriesFrame:
        """Return the expanding index of the minimum value, equivalent to using the full-length
        window as in `GenericAccessor.rolling_idxmin`.

        Args:
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_idxmin`.

        Returns:
            SeriesFrame: Series/DataFrame containing the indices of the minimum values
                within the expanding window.
        """
        return self.rolling_idxmin(None, minp=minp, **kwargs)

    def rolling_idxmax(
        self,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        local: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the index of the maximum value within each rolling window.

        Args:
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            local (bool): If True, computes the index of the maximum value within the local window.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Series/DataFrame containing the indices of the maximum values
                within the rolling windows.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_argmax_nb`
        """
        if window is None:
            window = self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.rolling_argmax_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp, local=local)
        if not local:
            wrap_kwargs = merge_dicts(dict(to_index=True), wrap_kwargs)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def expanding_idxmax(self, minp: tp.Optional[int] = 1, **kwargs) -> tp.SeriesFrame:
        """Return the expanding index of the maximum value, equivalent to using the full-length
        window as in `GenericAccessor.rolling_idxmax`.

        Args:
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_idxmax`.

        Returns:
            SeriesFrame: Series/DataFrame containing the indices of the maximum values
                within the expanding window.
        """
        return self.rolling_idxmax(None, minp=minp, **kwargs)

    def rolling_mean(
        self,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the rolling mean over a moving window.

        Args:
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated rolling mean values.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_mean_nb`
        """
        if window is None:
            window = self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.rolling_mean_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def expanding_mean(self, minp: tp.Optional[int] = 1, **kwargs) -> tp.SeriesFrame:
        """Return the expanding mean computed over an increasing window, equivalent to using
        the full-length window as in `GenericAccessor.rolling_mean`.

        Args:
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_mean`.

        Returns:
            SeriesFrame: Calculated expanding mean values.
        """
        return self.rolling_mean(None, minp=minp, **kwargs)

    def rolling_std(
        self,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        ddof: int = 1,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the rolling standard deviation over a moving window.

        Args:
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            ddof (int): Delta degrees of freedom.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated rolling standard deviation values.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_std_nb`
        """
        if window is None:
            window = self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.rolling_std_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp, ddof=ddof)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def expanding_std(self, minp: tp.Optional[int] = 1, **kwargs) -> tp.SeriesFrame:
        """Return the expanding standard deviation computed over an increasing window,
        equivalent to using the full-length window as in `GenericAccessor.rolling_std`.

        Args:
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_std`.

        Returns:
            SeriesFrame: Calculated expanding standard deviation values.
        """
        return self.rolling_std(None, minp=minp, **kwargs)

    def rolling_zscore(
        self,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        ddof: int = 1,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the rolling z-score for the instance's data using a specified window.

        Args:
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            ddof (int): Delta degrees of freedom.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated rolling z-score values.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_zscore_nb`
        """
        if window is None:
            window = self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.rolling_zscore_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp, ddof=ddof)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def expanding_zscore(self, minp: tp.Optional[int] = 1, **kwargs) -> tp.SeriesFrame:
        """Compute the expanding z-score for the instance's data over its entire length.

        Args:
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_zscore`.

        Returns:
            SeriesFrame: Calculated expanding z-score values.
        """
        return self.rolling_zscore(None, minp=minp, **kwargs)

    def wm_mean(
        self,
        span: int,
        minp: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the weighted moving mean for the instance's data using a specified span.

        Args:
            span (int): Window span for the exponential weighting.
            minp (Optional[int]): Minimum number of observations required.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated weighted moving mean values.

        See:
            `vectorbtpro.generic.nb.rolling.wm_mean_nb`
        """
        func = jit_reg.resolve_option(nb.wm_mean_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), span, minp=minp)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def ewm_mean(
        self,
        span: int,
        minp: tp.Optional[int] = 0,
        adjust: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the exponentially weighted moving mean for the instance's data using a specified span.

        Args:
            span (int): Window span for the exponential weighting.
            minp (Optional[int]): Minimum number of observations required.
            adjust (bool): Flag indicating whether to adjust weights.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated exponentially weighted moving mean values.

        See:
            `vectorbtpro.generic.nb.rolling.ewm_mean_nb`
        """
        func = jit_reg.resolve_option(nb.ewm_mean_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), span, minp=minp, adjust=adjust)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def ewm_std(
        self,
        span: int,
        minp: tp.Optional[int] = 0,
        adjust: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the exponentially weighted moving standard deviation for the instance's
        data using a specified span.

        Args:
            span (int): Window span for the exponential weighting.
            minp (Optional[int]): Minimum number of observations required.
            adjust (bool): Flag indicating whether to adjust weights.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated exponentially weighted moving standard deviation values.

        See:
            `vectorbtpro.generic.nb.rolling.ewm_std_nb`
        """
        func = jit_reg.resolve_option(nb.ewm_std_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), span, minp=minp, adjust=adjust)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def wwm_mean(
        self,
        period: int,
        minp: tp.Optional[int] = 0,
        adjust: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the weighted window moving mean for the instance's data using a specified period.

        Args:
            period (int): Window period for computing the moving mean.
            minp (Optional[int]): Minimum number of observations required.
            adjust (bool): Flag indicating whether to adjust weights.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated weighted window moving mean values.

        See:
            `vectorbtpro.generic.nb.rolling.wwm_mean_nb`
        """
        func = jit_reg.resolve_option(nb.wwm_mean_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), period, minp=minp, adjust=adjust)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def wwm_std(
        self,
        period: int,
        minp: tp.Optional[int] = 0,
        adjust: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the weighted window moving standard deviation for the instance's data
        using a specified period.

        Args:
            period (int): Window period for computing the moving standard deviation.
            minp (Optional[int]): Minimum number of observations required.
            adjust (bool): Flag indicating whether to adjust weights.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated weighted window moving standard deviation values.

        See:
            `vectorbtpro.generic.nb.rolling.wwm_std_nb`
        """
        func = jit_reg.resolve_option(nb.wwm_std_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), period, minp=minp, adjust=adjust)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def vidya(
        self,
        window: int,
        minp: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the Variable Index Dynamic Average (VIDYA) for the instance's data
        using a specified window.

        Args:
            window (int): Window size.
            minp (Optional[int]): Minimum number of observations required.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated VIDYA values.

        See:
            `vectorbtpro.generic.nb.rolling.vidya_nb`
        """
        func = jit_reg.resolve_option(nb.vidya_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def ma(
        self,
        window: int,
        wtype: tp.Union[int, str] = "simple",
        minp: tp.Optional[int] = 0,
        adjust: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the moving average for the instance's data using a specified window and weighting type.

        Args:
            window (int): Window size.
            wtype (Union[int, str]): Weighting type.

                Mapped using `vectorbtpro.generic.enums.WType` if provided as a string.
            minp (Optional[int]): Minimum number of observations required.
            adjust (bool): Flag indicating whether to adjust weights.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated moving average values.

        See:
            `vectorbtpro.generic.nb.rolling.ma_nb`
        """
        if isinstance(wtype, str):
            wtype = map_enum_fields(wtype, WType)
        func = jit_reg.resolve_option(nb.ma_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, wtype=wtype, minp=minp, adjust=adjust)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def msd(
        self,
        window: int,
        wtype: tp.Union[int, str] = "simple",
        minp: tp.Optional[int] = 0,
        adjust: bool = True,
        ddof: int = 1,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the moving standard deviation for the instance's data using a specified
        window and weighting type.

        Args:
            window (int): Window size.
            wtype (Union[int, str]): Weighting type.

                Mapped using `vectorbtpro.generic.enums.WType` if provided as a string.
            minp (Optional[int]): Minimum number of observations required.
            adjust (bool): Flag indicating whether to adjust weights.
            ddof (int): Delta degrees of freedom.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated moving standard deviation values.

        See:
            `vectorbtpro.generic.nb.rolling.msd_nb`
        """
        if isinstance(wtype, str):
            wtype = map_enum_fields(wtype, WType)
        func = jit_reg.resolve_option(nb.msd_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def rolling_cov(
        self,
        other: tp.SeriesFrame,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        ddof: int = 1,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the rolling covariance between the instance's data and another series
        using a specified window.

        Args:
            other (SeriesFrame): Another data series to compute covariance with.
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            ddof (int): Delta degrees of freedom.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Calculated rolling covariance values.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_cov_nb`
        """
        self_obj, other_obj = reshaping.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        if window is None:
            window = self_obj.shape[0]
        func = jit_reg.resolve_option(nb.rolling_cov_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            reshaping.to_2d_array(self_obj),
            reshaping.to_2d_array(other_obj),
            window,
            minp=minp,
            ddof=ddof,
        )
        return ArrayWrapper.from_obj(self_obj).wrap(
            out, group_by=False, **resolve_dict(wrap_kwargs)
        )

    def expanding_cov(
        self, other: tp.SeriesFrame, minp: tp.Optional[int] = 1, **kwargs
    ) -> tp.SeriesFrame:
        """Compute the expanding covariance between the instance's data and another
        series using all available data.

        Args:
            other (SeriesFrame): Another data series to compute covariance with.
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_cov`.

        Returns:
            SeriesFrame: Calculated expanding covariance values.
        """
        return self.rolling_cov(other, None, minp=minp, **kwargs)

    def rolling_corr(
        self,
        other: tp.SeriesFrame,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute rolling correlation.

        Args:
            other (SeriesFrame): Second Series or DataFrame for computing correlation.
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Rolling correlation computed over the specified window.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_corr_nb`
        """
        self_obj, other_obj = reshaping.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        if window is None:
            window = self_obj.shape[0]
        func = jit_reg.resolve_option(nb.rolling_corr_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            reshaping.to_2d_array(self_obj), reshaping.to_2d_array(other_obj), window, minp=minp
        )
        return ArrayWrapper.from_obj(self_obj).wrap(
            out, group_by=False, **resolve_dict(wrap_kwargs)
        )

    def expanding_corr(
        self, other: tp.SeriesFrame, minp: tp.Optional[int] = 1, **kwargs
    ) -> tp.SeriesFrame:
        """Compute expanding correlation by applying a growing window via `GenericAccessor.rolling_corr`.

        Args:
            other (SeriesFrame): Second Series or DataFrame for correlation computation.
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_corr`.

        Returns:
            SeriesFrame: Expanding correlation result.
        """
        return self.rolling_corr(other, None, minp=minp, **kwargs)

    def rolling_ols(
        self,
        other: tp.SeriesFrame,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Compute rolling ordinary least squares regression.

        Args:
            other (SeriesFrame): Second Series or DataFrame to perform regression against.
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Tuple[SeriesFrame, SeriesFrame]: Tuple containing the slope and intercept arrays.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_ols_nb`
        """
        self_obj, other_obj = reshaping.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        if window is None:
            window = self_obj.shape[0]
        func = jit_reg.resolve_option(nb.rolling_ols_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        slope_out, intercept_out = func(
            reshaping.to_2d_array(self_obj),
            reshaping.to_2d_array(other_obj),
            window,
            minp=minp,
        )
        return (
            ArrayWrapper.from_obj(self_obj).wrap(
                slope_out, group_by=False, **resolve_dict(wrap_kwargs)
            ),
            ArrayWrapper.from_obj(self_obj).wrap(
                intercept_out, group_by=False, **resolve_dict(wrap_kwargs)
            ),
        )

    def expanding_ols(
        self,
        other: tp.SeriesFrame,
        minp: tp.Optional[int] = 1,
        **kwargs,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Compute expanding ordinary least squares regression.

        Args:
            other (SeriesFrame): Second Series or DataFrame for regression.
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_ols`.

        Returns:
            Tuple[SeriesFrame, SeriesFrame]: Tuple containing the slope and intercept arrays.
        """
        return self.rolling_ols(other, None, minp=minp, **kwargs)

    def rolling_rank(
        self,
        window: tp.Optional[int],
        minp: tp.Optional[int] = None,
        pct: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute rolling rank.

        Args:
            window (Optional[int]): Window size.

                If None, uses the full length of the data.
            minp (Optional[int]): Minimum number of observations required.
            pct (bool): If True, compute the rank as a percentile.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Rolling rank result.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_rank_nb`
        """
        if window is None:
            window = self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.rolling_rank_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array(), window, minp=minp, pct=pct)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def expanding_rank(self, minp: tp.Optional[int] = 1, **kwargs) -> tp.SeriesFrame:
        """Compute expanding rank by applying a growing window via `GenericAccessor.rolling_rank`.

        Args:
            minp (Optional[int]): Minimum number of observations required.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_rank`.

        Returns:
            SeriesFrame: Expanding rank result.
        """
        return self.rolling_rank(None, minp=minp, **kwargs)

    def rolling_pattern_similarity(
        self,
        pattern: tp.ArrayLike,
        window: tp.Optional[int] = None,
        max_window: tp.Optional[int] = None,
        row_select_prob: float = 1.0,
        window_select_prob: float = 1.0,
        interp_mode: tp.Union[int, str] = "mixed",
        rescale_mode: tp.Union[int, str] = "minmax",
        vmin: float = np.nan,
        vmax: float = np.nan,
        pmin: float = np.nan,
        pmax: float = np.nan,
        invert: bool = False,
        error_type: tp.Union[int, str] = "absolute",
        distance_measure: tp.Union[int, str] = "mae",
        max_error: tp.ArrayLike = np.nan,
        max_error_interp_mode: tp.Union[None, int, str] = None,
        max_error_as_maxdist: bool = False,
        max_error_strict: bool = False,
        min_pct_change: float = np.nan,
        max_pct_change: float = np.nan,
        min_similarity: float = np.nan,
        minp: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute rolling pattern similarity.

        Args:
            pattern (ArrayLike): Reference pattern array for similarity comparison.
            window (Optional[int]): Window size.

                Defaults to the length of `pattern`.
            max_window (Optional[int]): Maximum length of the rolling window for matching.

                If None, defaults to `window`.
            row_select_prob (float): Probability of selecting a row.
            window_select_prob (float): Probability of selecting a window size.
            interp_mode (Union[int, str]): Interpolation mode.

                Mapped using `vectorbtpro.generic.enums.InterpMode` if provided as a string.
            rescale_mode (Union[int, str]): Rescaling mode.

                Mapped using `vectorbtpro.generic.enums.RescaleMode` if provided as a string.
            vmin (float): Minimum value used for rescaling `arr`.
            vmax (float): Maximum value used for rescaling `arr`.
            pmin (float): Minimum value used for rescaling `pattern`.
            pmax (float): Maximum value used for rescaling `pattern`.
            invert (bool): Invert the pattern by reflecting its values.
            error_type (Union[int, str]): Error calculation type.

                Mapped using `vectorbtpro.generic.enums.ErrorType` if provided as a string.
            distance_measure (Union[int, str]): Distance measure for error computation.

                Mapped using `vectorbtpro.generic.enums.DistanceMeasure` if provided as a string.
            max_error (ArrayLike): Maximum allowed error.
            max_error_interp_mode (Union[None, int, str]): Interpolation mode for `max_error`.

                Mapped using `vectorbtpro.generic.enums.InterpMode` if provided as a string.
            max_error_as_maxdist (bool): Indicates whether `max_error` represents the maximum distance at each point.
            max_error_strict (bool): If True, any instance of exceeding `max_error` results in a similarity of NaN.
            min_pct_change (float): Minimum percent change applied during rescaling.
            max_pct_change (float): Maximum percentage change allowed for a window to remain a search candidate.
            min_similarity (float): Minimum similarity threshold.
            minp (Optional[int]): Minimum number of observations required.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Rolling pattern similarity result.

        See:
            `vectorbtpro.generic.nb.rolling.rolling_pattern_similarity_nb`
        """
        if isinstance(interp_mode, str):
            interp_mode = map_enum_fields(interp_mode, InterpMode)
        if isinstance(rescale_mode, str):
            rescale_mode = map_enum_fields(rescale_mode, RescaleMode)
        if isinstance(error_type, str):
            error_type = map_enum_fields(error_type, ErrorType)
        if isinstance(distance_measure, str):
            distance_measure = map_enum_fields(distance_measure, DistanceMeasure)
        if max_error_interp_mode is not None and isinstance(max_error_interp_mode, str):
            max_error_interp_mode = map_enum_fields(max_error_interp_mode, InterpMode)
        if max_error_interp_mode is None:
            max_error_interp_mode = interp_mode

        func = jit_reg.resolve_option(nb.rolling_pattern_similarity_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            reshaping.to_1d_array(pattern),
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
            max_error=reshaping.to_1d_array(max_error),
            max_error_interp_mode=max_error_interp_mode,
            max_error_as_maxdist=max_error_as_maxdist,
            max_error_strict=max_error_strict,
            min_pct_change=min_pct_change,
            max_pct_change=max_pct_change,
            min_similarity=min_similarity,
            minp=minp,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Mapping ############# #

    @hybrid_method
    def map(
        cls_or_self,
        map_func_nb: tp.Union[str, tp.AnyMapFunc],
        *args,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Map the input arrays using the specified Numba mapping function.

        This method applies the mapping function to the underlying 2D array of data and
        supports both regular and meta mapping operations. When invoked on a class,
        it performs broadcasting and template substitution and utilizes the meta mapping version.

        Args:
            map_func_nb (Union[str, AnyMapFunc]): Callback function for mapping.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_map_nb`.
            *args: Positional arguments for `map_func_nb`.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Result of applying the mapping function to the input data, wrapped appropriately.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.map_nb` for regular mapping.
            * `vectorbtpro.generic.nb.apply_reduce.map_meta_nb` for meta mapping.

        Examples:
            Using regular function:

            ```pycon
            >>> prod_nb = njit(lambda a, x: a * x)

            >>> df.vbt.map(prod_nb, 10)
                         a   b   c
            2020-01-01  10  50  10
            2020-01-02  20  40  20
            2020-01-03  30  30  30
            2020-01-04  40  20  20
            2020-01-05  50  10  10
            ```

            Using meta function:

            ```pycon
            >>> diff_meta_nb = njit(lambda i, col, a, b: a[i, col] / b[i, col])

            >>> vbt.pd_acc.map(
            ...     diff_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper
            ... )
                               a         b         c
            2020-01-01  0.000000  0.666667  0.000000
            2020-01-02  0.333333  0.600000  0.333333
            2020-01-03  0.500000  0.500000  0.500000
            2020-01-04  0.600000  0.333333  0.333333
            2020-01-05  0.666667  0.000000  0.000000
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.map(
            ...     diff_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
                          a    b         c
            2020-01-01  1.0  0.5  0.333333
            2020-01-02  2.0  1.0  0.666667
            2020-01-03  3.0  1.5  1.000000
            2020-01-04  4.0  2.0  1.333333
            2020-01-05  5.0  2.5  1.666667
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(map_func_nb, str):
            map_func_nb = getattr(nb, map_func_nb + "_map_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            args = substitute_templates(args, template_context, eval_id="args")
            func = jit_reg.resolve_option(nb.map_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(wrapper.shape_2d, map_func_nb, *args)
        else:
            func = jit_reg.resolve_option(nb.map_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(cls_or_self.to_2d_array(), map_func_nb, *args)
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Applying ############# #

    @hybrid_method
    def apply_along_axis(
        cls_or_self,
        apply_func_nb: tp.Union[str, tp.AnyApplyFunc],
        *args,
        axis: int = 1,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Apply a function along a specified axis with optional broadcasting,
        JIT compilation, and chunked processing.

        Args:
            apply_func_nb (Union[str, AnyApplyFunc]): Callback function for applying.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_apply_nb`.
            *args: Positional arguments for `apply_func_nb`.
            axis (int): Axis along which to apply the function.

                Use 1 for column-wise and 0 for row-wise operations.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Result of applying the function to the input data.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.apply_nb` for non-meta operations and `axis=1`.
            * `vectorbtpro.generic.nb.apply_reduce.row_apply_nb` for non-meta operations and `axis=0`.
            * `vectorbtpro.generic.nb.apply_reduce.apply_meta_nb` for meta operations and `axis=1`.
            * `vectorbtpro.generic.nb.apply_reduce.row_apply_meta_nb` for meta operations and `axis=0`.

        Examples:
            Using regular function:

            ```pycon
            >>> power_nb = njit(lambda a: np.power(a, 2))

            >>> df.vbt.apply_along_axis(power_nb)
                         a   b  c
            2020-01-01   1  25  1
            2020-01-02   4  16  4
            2020-01-03   9   9  9
            2020-01-04  16   4  4
            2020-01-05  25   1  1
            ```

            Using meta function:

            ```pycon
            >>> ratio_meta_nb = njit(lambda col, a, b: a[:, col] / b[:, col])

            >>> vbt.pd_acc.apply_along_axis(
            ...     ratio_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper
            ... )
                               a         b         c
            2020-01-01  0.000000  0.666667  0.000000
            2020-01-02  0.333333  0.600000  0.333333
            2020-01-03  0.500000  0.500000  0.500000
            2020-01-04  0.600000  0.333333  0.333333
            2020-01-05  0.666667  0.000000  0.000000
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.apply_along_axis(
            ...     ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
                          a    b         c
            2020-01-01  1.0  0.5  0.333333
            2020-01-02  2.0  1.0  0.666667
            2020-01-03  3.0  1.5  1.000000
            2020-01-04  4.0  2.0  1.333333
            2020-01-05  5.0  2.5  1.666667
            ```
        """
        checks.assert_in(axis, (0, 1))
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(apply_func_nb, str):
            apply_func_nb = getattr(nb, apply_func_nb + "_apply_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper, axis=axis), template_context
            )
            args = substitute_templates(args, template_context, eval_id="args")
            if axis == 0:
                func = jit_reg.resolve_option(nb.row_apply_meta_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.apply_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(wrapper.shape_2d, apply_func_nb, *args)
        else:
            if axis == 0:
                func = jit_reg.resolve_option(nb.row_apply_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.apply_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(cls_or_self.to_2d_array(), apply_func_nb, *args)
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def row_apply(self, *args, **kwargs) -> tp.SeriesFrame:
        """Apply a function row-wise by invoking `GenericAccessor.apply_along_axis` with axis set to 0.

        Args:
            *args: Positional arguments for `GenericAccessor.apply_along_axis`.
            **kwargs: Keyword arguments for `GenericAccessor.apply_along_axis`.

        Returns:
            SeriesFrame: Result of applying the function row-wise.
        """
        return self.apply_along_axis(*args, axis=0, **kwargs)

    @hybrid_method
    def column_apply(self, *args, **kwargs) -> tp.SeriesFrame:
        """Apply a function column-wise by invoking `GenericAccessor.apply_along_axis` with axis set to 1.

        Args:
            *args: Positional arguments for `GenericAccessor.apply_along_axis`.
            **kwargs: Keyword arguments for `GenericAccessor.apply_along_axis`.

        Returns:
            SeriesFrame: Result of applying the function column-wise.
        """
        return self.apply_along_axis(*args, axis=1, **kwargs)

    # ############# Reducing ############# #

    @hybrid_method
    def rolling_apply(
        cls_or_self,
        window: tp.Optional[tp.FrequencyLike],
        reduce_func_nb: tp.Union[str, tp.AnyReduceFunc],
        *args,
        minp: tp.Optional[int] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Apply a rolling reduction function over a specified window.

        Args:
            window (Optional[FrequencyLike]): Window size as an integer or a frequency string.

                If None, an expanding window is used.
            reduce_func_nb (Union[str, AnyReduceFunc]): Callback function for reducing.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for the reduction function.
            minp (Optional[int]): Minimum number of observations required.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Result of the rolling reduction.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.rolling_reduce_nb` for integer windows.
            * `vectorbtpro.generic.nb.apply_reduce.rolling_freq_reduce_nb` for frequency windows.
            * `vectorbtpro.generic.nb.apply_reduce.rolling_reduce_meta_nb` for meta functions and integer windows.
            * `vectorbtpro.generic.nb.apply_reduce.rolling_freq_reduce_meta_nb` for meta functions and frequency windows.

        Examples:
            Using regular function:

            ```pycon
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> df.vbt.rolling_apply(3, mean_nb)
                          a    b         c
            2020-01-01  NaN  NaN       NaN
            2020-01-02  NaN  NaN       NaN
            2020-01-03  2.0  4.0  2.000000
            2020-01-04  3.0  3.0  2.333333
            2020-01-05  4.0  2.0  2.000000
            ```

            Using a frequency-based window:

            ```pycon
            >>> df.vbt.rolling_apply("3d", mean_nb)
                          a    b         c
            2020-01-01  1.0  5.0  1.000000
            2020-01-02  1.5  4.5  1.500000
            2020-01-03  2.0  4.0  2.000000
            2020-01-04  3.0  3.0  2.333333
            2020-01-05  4.0  2.0  2.000000
            ```

            Using meta function:

            ```pycon
            >>> mean_ratio_meta_nb = njit(lambda from_i, to_i, col, a, b: \\
            ...     np.mean(a[from_i:to_i, col]) / np.mean(b[from_i:to_i, col]))

            >>> vbt.pd_acc.rolling_apply(
            ...     3,
            ...     mean_ratio_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper,
            ... )
                               a         b         c
            2020-01-01       NaN       NaN       NaN
            2020-01-02       NaN       NaN       NaN
            2020-01-03  0.333333  0.600000  0.333333
            2020-01-04  0.500000  0.500000  0.400000
            2020-01-05  0.600000  0.333333  0.333333
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.rolling_apply(
            ...     2,
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
                          a     b         c
            2020-01-01  NaN   NaN       NaN
            2020-01-02  1.5  0.75  0.500000
            2020-01-03  2.5  1.25  0.833333
            2020-01-04  3.5  1.75  1.166667
            2020-01-05  4.5  2.25  1.500000
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        if window is not None:
            if not isinstance(window, int):
                window = dt.to_timedelta64(window)
        if minp is None and window is None:
            minp = 1
        if window is None:
            window = wrapper.shape[0]
        if minp is None:
            minp = window
        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            template_context = merge_dicts(
                broadcast_named_args,
                dict(wrapper=wrapper, window=window, minp=minp),
                template_context,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            if isinstance(window, int):
                func = jit_reg.resolve_option(nb.rolling_reduce_meta_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(wrapper.shape_2d, window, minp, reduce_func_nb, *args)
            else:
                func = jit_reg.resolve_option(nb.rolling_freq_reduce_meta_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(wrapper.shape_2d[1], wrapper.index.values, window, reduce_func_nb, *args)
        else:
            if isinstance(window, int):
                func = jit_reg.resolve_option(nb.rolling_reduce_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(cls_or_self.to_2d_array(), window, minp, reduce_func_nb, *args)
            else:
                func = jit_reg.resolve_option(nb.rolling_freq_reduce_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(
                    wrapper.index.values, cls_or_self.to_2d_array(), window, reduce_func_nb, *args
                )

        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def expanding_apply(cls_or_self, *args, **kwargs) -> tp.SeriesFrame:
        """Apply an expanding reduction function.

        This method is equivalent to calling `GenericAccessor.rolling_apply` with `window=None`,
        resulting in an expanding window reduction.

        Args:
            *args: Positional arguments for `GenericAccessor.rolling_apply`.
            **kwargs: Keyword arguments for `GenericAccessor.rolling_apply`.

        Returns:
            SeriesFrame: Result of the expanding reduction.
        """
        return cls_or_self.rolling_apply(None, *args, **kwargs)

    @hybrid_method
    def groupby_apply(
        cls_or_self,
        by: tp.AnyGroupByLike,
        reduce_func_nb: tp.Union[str, tp.AnyGroupByReduceFunc],
        *args,
        groupby_kwargs: tp.KwargsLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Apply a groupby reduction function on a wrapped 2D array.

        Args:
            by (AnyGroupByLike): Grouper-like specification.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            reduce_func_nb (Union[str, AnyGroupByReduceFunc]): Callback function for reducing groups.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `reduce_func_nb`.
            groupby_kwargs (KwargsLike): Keyword arguments for `pandas.Series.groupby` and
                `pandas.Series.resample` methods.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            SeriesFrame: Wrapped result after applying the groupby reduction.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.groupby_reduce_nb` for the standard version.
            * `vectorbtpro.generic.nb.apply_reduce.groupby_reduce_meta_nb` for the meta version.

        Examples:
            Using regular function:

            ```pycon
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> df.vbt.groupby_apply([1, 1, 2, 2, 3], mean_nb)
                 a    b    c
            1  1.5  4.5  1.5
            2  3.5  2.5  2.5
            3  5.0  1.0  1.0
            ```

            Using meta function:

            ```pycon
            >>> mean_ratio_meta_nb = njit(lambda idxs, group, col, a, b: \\
            ...     np.mean(a[idxs, col]) / np.mean(b[idxs, col]))

            >>> vbt.pd_acc.groupby_apply(
            ...     [1, 1, 2, 2, 3],
            ...     mean_ratio_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper
            ... )
                      a         b         c
            1  0.200000  0.636364  0.200000
            2  0.555556  0.428571  0.428571
            3  0.666667  0.000000  0.000000
            ```

            Using templates and broadcasting, let's split both input arrays into 2 groups of rows and
            run the calculation function on each group:

            ```pycon
            >>> from vectorbtpro.base.grouping.nb import group_by_evenly_nb

            >>> vbt.pd_acc.groupby_apply(
            ...     vbt.RepEval('group_by_evenly_nb(wrapper.shape[0], 2)'),
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     ),
            ...     template_context=dict(group_by_evenly_nb=group_by_evenly_nb)
            ... )
                 a     b         c
            0  2.0  1.00  0.666667
            1  4.5  2.25  1.500000
            ```

            !!! note
                The template and broadcasting approach offers flexibility by allowing arrays of any
                broadcastable shapes to be processed seamlessly.
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            by = substitute_templates(by, template_context, eval_id="by")
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        grouper = wrapper.get_index_grouper(by, **resolve_dict(groupby_kwargs))

        if isinstance(cls_or_self, type):
            group_map = grouper.get_group_map()
            template_context = merge_dicts(dict(by=by, grouper=grouper), template_context)
            args = substitute_templates(args, template_context, eval_id="args")
            func = jit_reg.resolve_option(nb.groupby_reduce_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(wrapper.shape_2d[1], group_map, reduce_func_nb, *args)
        else:
            group_map = grouper.get_group_map()
            func = jit_reg.resolve_option(nb.groupby_reduce_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(cls_or_self.to_2d_array(), group_map, reduce_func_nb, *args)

        wrap_kwargs = merge_dicts(dict(name_or_index=grouper.get_index()), wrap_kwargs)
        return wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    @hybrid_method
    def groupby_transform(
        cls_or_self,
        by: tp.AnyGroupByLike,
        transform_func_nb: tp.Union[str, tp.AnyGroupByTransformFunc],
        *args,
        groupby_kwargs: tp.KwargsLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Transform groups by applying a specified transformation function.

        This method applies a transformation function to each group defined by `by`.
        When invoked on a class, the meta version of the transformation function is used.

        Args:
            by (AnyGroupByLike): Grouper-like specification.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            transform_func_nb (Union[str, AnyGroupByTransformFunc]): Callback function for transforming groups.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_transform_nb`.
            *args: Positional arguments for `transform_func_nb`.
            groupby_kwargs (KwargsLike): Keyword arguments for `pandas.Series.groupby` and
                `pandas.Series.resample` methods.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Transformed data as a Series or DataFrame.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.groupby_transform_nb` for the standard version.
            * `vectorbtpro.generic.nb.apply_reduce.groupby_transform_meta_nb` for the meta version.

        Examples:
            Using regular function:

            ```pycon
            >>> zscore_nb = njit(lambda a: (a - np.nanmean(a)) / np.nanstd(a))

            >>> df.vbt.groupby_transform([1, 1, 2, 2, 3], zscore_nb)
                               a         b         c
            2020-01-01 -1.000000  1.666667 -1.000000
            2020-01-02 -0.333333  1.000000 -0.333333
            2020-01-03  0.242536  0.242536  0.242536
            2020-01-04  1.697749 -1.212678 -1.212678
            2020-01-05  1.414214 -0.707107 -0.707107
            ```

            Using meta function:

            ```pycon
            >>> zscore_ratio_meta_nb = njit(lambda idxs, group, a, b: \\
            ...     zscore_nb(a[idxs]) / zscore_nb(b[idxs]))

            >>> vbt.pd_acc.groupby_transform(
            ...     [1, 1, 2, 2, 3],
            ...     zscore_ratio_meta_nb,
            ...     df.vbt.to_2d_array(),
            ...     df.vbt.to_2d_array()[::-1],
            ...     wrapper=df.vbt.wrapper
            ... )
                               a         b    c
            2020-01-01 -0.600000 -1.666667  1.0
            2020-01-02 -0.333333 -3.000000  1.0
            2020-01-03  1.000000  1.000000  1.0
            2020-01-04 -1.400000 -0.714286  1.0
            2020-01-05 -2.000000 -0.500000  1.0
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        if isinstance(transform_func_nb, str):
            transform_func_nb = getattr(nb, transform_func_nb + "_transform_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            by = substitute_templates(by, template_context, eval_id="by")
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        grouper = wrapper.get_index_grouper(by, **resolve_dict(groupby_kwargs))

        if isinstance(cls_or_self, type):
            group_map = grouper.get_group_map()
            template_context = merge_dicts(dict(by=by, grouper=grouper), template_context)
            args = substitute_templates(args, template_context, eval_id="args")
            func = jit_reg.resolve_option(nb.groupby_transform_meta_nb, jitted)
            out = func(wrapper.shape_2d, group_map, transform_func_nb, *args)
        else:
            group_map = grouper.get_group_map()
            func = jit_reg.resolve_option(nb.groupby_transform_nb, jitted)
            out = func(cls_or_self.to_2d_array(), group_map, transform_func_nb, *args)

        return wrapper.wrap(out, group_by=False, **wrap_kwargs)

    @hybrid_method
    def resample_apply(
        cls_or_self,
        rule: tp.AnyRuleLike,
        reduce_func_nb: tp.Union[str, tp.AnyResampleReduceFunc],
        *args,
        use_groupby_apply: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        resample_kwargs: tp.KwargsLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Resample the underlying data using a specified rule and reduction function.

        Args:
            rule (AnyRuleLike): Resampler-like specification.

                Can be one of the following:

                * `vectorbtpro.base.resampling.base.Resampler` instance
                * Pandas `Resampler` instance
                * Instruction for any of the above
            reduce_func_nb (Union[str, AnyResampleReduceFunc]): Callback function for resampling.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `reduce_func_nb`.
            use_groupby_apply (bool): If True, apply the reduction function using
                `GenericAccessor.groupby_apply` with post-processing; otherwise, use
                `GenericAccessor.resample_to_index`.
            freq (Optional[FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            resample_kwargs (KwargsLike): Keyword arguments for `pandas.Series.resample`.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `reduce_func_nb`.

        Returns:
            SeriesFrame: Result of the resampling and reduction.

        Examples:
            Using regular function:

            ```pycon
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> df.vbt.resample_apply('2d', mean_nb)
                          a    b    c
            2020-01-01  1.5  4.5  1.5
            2020-01-03  3.5  2.5  2.5
            2020-01-05  5.0  1.0  1.0
            ```

            Using meta function:

            ```pycon
            >>> mean_ratio_meta_nb = njit(lambda idxs, group, col, a, b: \\
            ...     np.mean(a[idxs, col]) / np.mean(b[idxs, col]))

            >>> vbt.pd_acc.resample_apply(
            ...     '2d',
            ...     mean_ratio_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper
            ... )
                               a         b         c
            2020-01-01  0.200000  0.636364  0.200000
            2020-01-03  0.555556  0.428571  0.428571
            2020-01-05  0.666667  0.000000  0.000000
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.resample_apply(
            ...     '2d',
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
                          a     b         c
            2020-01-01  1.5  0.75  0.500000
            2020-01-03  3.5  1.75  1.166667
            2020-01-05  5.0  2.50  1.666667
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            rule = substitute_templates(rule, template_context, eval_id="rule")
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        if use_groupby_apply:
            if isinstance(rule, Resampler):
                raise TypeError("Resampler cannot be used with use_groupby_apply=True")
            if not isinstance(rule, PandasResampler):
                rule = pd.Series(index=wrapper.index, dtype=object).resample(
                    rule, **resolve_dict(resample_kwargs)
                )
            out_obj = cls_or_self.groupby_apply(
                rule,
                reduce_func_nb,
                *args,
                template_context=template_context,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs,
                **kwargs,
            )
            new_index = rule.count().index.rename("group")
            if pd.Index.equals(out_obj.index, new_index):
                if new_index.freq is not None:
                    try:
                        out_obj.index.freq = new_index.freq
                    except ValueError:
                        pass
                return out_obj
            resampled_arr = np.full((rule.ngroups, wrapper.shape_2d[1]), np.nan)
            resampled_obj = wrapper.wrap(
                resampled_arr,
                index=new_index,
                **resolve_dict(wrap_kwargs),
            )
            resampled_obj.loc[out_obj.index] = out_obj.values
            return resampled_obj

        if not isinstance(rule, Resampler):
            rule = wrapper.get_resampler(
                rule,
                freq=freq,
                resample_kwargs=resample_kwargs,
                return_pd_resampler=False,
            )
        return cls_or_self.resample_to_index(
            rule,
            reduce_func_nb,
            *args,
            template_context=template_context,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    @hybrid_method
    def apply_and_reduce(
        cls_or_self,
        apply_func_nb: tp.Union[str, tp.AnyApplyFunc],
        reduce_func_nb: tp.Union[str, tp.AnyReduceFunc],
        apply_args: tp.Optional[tuple] = None,
        reduce_args: tp.Optional[tuple] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Apply a function to an array and reduce its output.

        This method supports both direct and meta function application. When called on a class,
        template substitution and broadcasting are applied.

        Args:
            apply_func_nb (Union[str, AnyApplyFunc]): Callback function for applying.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_apply_nb`.
            reduce_func_nb (Union[str, AnyReduceFunc]): Callback function for reducing.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            apply_args (Optional[tuple]): Arguments passed to the apply function.
            reduce_args (Optional[tuple]): Arguments passed to the reduce function.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            Series: Result of the apply and reduce operation.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.apply_and_reduce_nb` for the standard version.
            * `vectorbtpro.generic.nb.apply_reduce.apply_and_reduce_meta_nb` for the meta version.

        Examples:
            Using regular function:

            ```pycon
            >>> greater_nb = njit(lambda a: a[a > 2])
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> df.vbt.apply_and_reduce(greater_nb, mean_nb)
            a    4.0
            b    4.0
            c    3.0
            Name: apply_and_reduce, dtype: float64
            ```

            Using meta function:

            ```pycon
            >>> and_meta_nb = njit(lambda col, a, b: a[:, col] & b[:, col])
            >>> sum_meta_nb = njit(lambda col, x: np.sum(x))

            >>> vbt.pd_acc.apply_and_reduce(
            ...     and_meta_nb,
            ...     sum_meta_nb,
            ...     apply_args=(
            ...         df.vbt.to_2d_array() > 1,
            ...         df.vbt.to_2d_array() < 4
            ...     ),
            ...     wrapper=df.vbt.wrapper
            ... )
            a    2
            b    2
            c    3
            Name: apply_and_reduce, dtype: int64
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.apply_and_reduce(
            ...     and_meta_nb,
            ...     sum_meta_nb,
            ...     apply_args=(
            ...         vbt.Rep('mask_a'),
            ...         vbt.Rep('mask_b')
            ...     ),
            ...     broadcast_named_args=dict(
            ...         mask_a=pd.Series([True, True, True, False, False], index=df.index),
            ...         mask_b=pd.DataFrame([[True, True, False]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            a    3
            b    3
            c    0
            Name: apply_and_reduce, dtype: int64
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(apply_func_nb, str):
            apply_func_nb = getattr(nb, apply_func_nb + "_apply_nb")
        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")
        if apply_args is None:
            apply_args = ()
        if reduce_args is None:
            reduce_args = ()
        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            apply_args = substitute_templates(apply_args, template_context, eval_id="apply_args")
            reduce_args = substitute_templates(reduce_args, template_context, eval_id="reduce_args")
            func = jit_reg.resolve_option(nb.apply_and_reduce_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(wrapper.shape_2d[1], apply_func_nb, apply_args, reduce_func_nb, reduce_args)
        else:
            func = jit_reg.resolve_option(nb.apply_and_reduce_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(
                cls_or_self.to_2d_array(), apply_func_nb, apply_args, reduce_func_nb, reduce_args
            )
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        wrap_kwargs = merge_dicts(dict(name_or_index="apply_and_reduce"), wrap_kwargs)
        return wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    @hybrid_method
    def reduce(
        cls_or_self,
        reduce_func_nb: tp.Union[str, tp.AnyFlexReduceFunc],
        *args,
        returns_array: bool = False,
        returns_idx: bool = False,
        flatten: bool = False,
        order: str = "C",
        to_index: bool = True,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeriesFrame:
        """Reduce data by column or for grouped data.

        The internal reduction function is chosen based on the grouping status and provided flags.

        Args:
            reduce_func_nb (Union[str, AnyFlexReduceFunc]): Callback function for flexible reducing.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `reduce_func_nb`.
            returns_array (bool): True if the reducing function returns an array.

                Determines whether the output should be processed with array-specific logic.
            returns_idx (bool): True if the reducing function returns row indices.

                When True, numeric positions may be converted to index labels if `to_index` is also True.
            flatten (bool): True to flatten grouped data before applying the reducing function.

                When operating on grouped data, a flattened array is passed to `reduce_func_nb`.
            order (str): Order in which to flatten the array ("C" for row-major or "F" for column-major).
            to_index (bool): True to convert positions to index labels when `returns_idx` is True.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeriesFrame: Reduced data as a Series or DataFrame.

        See:
            If the data is grouped and `flatten` is True:

            * `vectorbtpro.generic.nb.apply_reduce.reduce_flat_grouped_to_array_nb` is used
                when `returns_array` is True.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_flat_grouped_nb` is used
                when `returns_array` is False.

            If the data is grouped and `flatten` is False:

            * `vectorbtpro.generic.nb.apply_reduce.reduce_grouped_to_array_nb` is used
                when `returns_array` is True.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_grouped_nb` is used
                when `returns_array` is False.

            If the data is not grouped:

            * `vectorbtpro.generic.nb.apply_reduce.reduce_to_array_nb` is used
                when `returns_array` is True.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_nb` is used
                when `returns_array` is False.

            For meta reducing functions, the following internal functions are used:

            * `vectorbtpro.generic.nb.apply_reduce.reduce_grouped_to_array_meta_nb` for grouped data
                when `returns_array` is True.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_grouped_meta_nb` for grouped data
                when `returns_array` is False.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_to_array_meta_nb` for non-grouped data
                when `returns_array` is True.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_meta_nb` for non-grouped data
                when `returns_array` is False.

        Examples:
            Using regular function:

            ```pycon
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> df.vbt.reduce(mean_nb)
            a    3.0
            b    3.0
            c    1.8
            Name: reduce, dtype: float64

            >>> argmax_nb = njit(lambda a: np.argmax(a))

            >>> df.vbt.reduce(argmax_nb, returns_idx=True)
            a   2020-01-05
            b   2020-01-01
            c   2020-01-03
            Name: reduce, dtype: datetime64[ns]

            >>> df.vbt.reduce(argmax_nb, returns_idx=True, to_index=False)
            a    4
            b    0
            c    2
            Name: reduce, dtype: int64

            >>> min_max_nb = njit(lambda a: np.array([np.nanmin(a), np.nanmax(a)]))

            >>> df.vbt.reduce(min_max_nb, returns_array=True, wrap_kwargs=dict(name_or_index=['min', 'max']))
                 a  b  c
            min  1  1  1
            max  5  5  3

            >>> group_by = pd.Series(['first', 'first', 'second'], name='group')
            >>> df.vbt.reduce(mean_nb, group_by=group_by)
            group
            first     3.0
            second    1.8
            dtype: float64
            ```

            Using meta function:

            ```pycon
            >>> mean_meta_nb = njit(lambda col, a: np.nanmean(a[:, col]))

            >>> pd.Series.vbt.reduce(
            ...     mean_meta_nb,
            ...     df['a'].vbt.to_2d_array(),
            ...     wrapper=df['a'].vbt.wrapper
            ... )
            3.0

            >>> vbt.pd_acc.reduce(
            ...     mean_meta_nb,
            ...     df.vbt.to_2d_array(),
            ...     wrapper=df.vbt.wrapper
            ... )
            a    3.0
            b    3.0
            c    1.8
            Name: reduce, dtype: float64

            >>> grouped_mean_meta_nb = njit(lambda group_idxs, group, a: np.nanmean(a[:, group_idxs]))

            >>> group_by = pd.Series(['first', 'first', 'second'], name='group')
            >>> vbt.pd_acc.reduce(
            ...     grouped_mean_meta_nb,
            ...     df.vbt.to_2d_array(),
            ...     wrapper=df.vbt.wrapper,
            ...     group_by=group_by
            ... )
            group
            first     3.0
            second    1.8
            Name: reduce, dtype: float64
            ```

            Using templates and broadcasting:

            ```pycon
            >>> mean_a_b_nb = njit(lambda col, a, b: \\
            ...     np.array([np.nanmean(a[:, col]), np.nanmean(b[:, col])]))

            >>> vbt.pd_acc.reduce(
            ...     mean_a_b_nb,
            ...     vbt.Rep('arr1'),
            ...     vbt.Rep('arr2'),
            ...     returns_array=True,
            ...     broadcast_named_args=dict(
            ...         arr1=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         arr2=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     ),
            ...     wrap_kwargs=dict(name_or_index=['arr1', 'arr2'])
            ... )
                    a    b    c
            arr1  3.0  3.0  3.0
            arr2  1.0  2.0  3.0
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args,
                dict(
                    wrapper=wrapper,
                    group_by=group_by,
                    returns_array=returns_array,
                    returns_idx=returns_idx,
                    flatten=flatten,
                    order=order,
                ),
                template_context,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            if wrapper.grouper.is_grouped(group_by=group_by):
                group_map = wrapper.grouper.get_group_map(group_by=group_by)
                if returns_array:
                    func = jit_reg.resolve_option(nb.reduce_grouped_to_array_meta_nb, jitted)
                else:
                    func = jit_reg.resolve_option(nb.reduce_grouped_meta_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(group_map, reduce_func_nb, *args)
            else:
                if returns_array:
                    func = jit_reg.resolve_option(nb.reduce_to_array_meta_nb, jitted)
                else:
                    func = jit_reg.resolve_option(nb.reduce_meta_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(wrapper.shape_2d[1], reduce_func_nb, *args)
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper
            if wrapper.grouper.is_grouped(group_by=group_by):
                group_map = wrapper.grouper.get_group_map(group_by=group_by)
                if flatten:
                    checks.assert_in(order.upper(), ["C", "F"])
                    in_c_order = order.upper() == "C"
                    if returns_array:
                        func = jit_reg.resolve_option(nb.reduce_flat_grouped_to_array_nb, jitted)
                    else:
                        func = jit_reg.resolve_option(nb.reduce_flat_grouped_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(
                        cls_or_self.to_2d_array(), group_map, in_c_order, reduce_func_nb, *args
                    )
                    if returns_idx:
                        if in_c_order:
                            out //= group_map[1]  # flattened in C order
                        else:
                            out %= wrapper.shape[0]  # flattened in F order
                else:
                    if returns_array:
                        func = jit_reg.resolve_option(nb.reduce_grouped_to_array_nb, jitted)
                    else:
                        func = jit_reg.resolve_option(nb.reduce_grouped_nb, jitted)
                    func = ch_reg.resolve_option(func, chunked)
                    out = func(cls_or_self.to_2d_array(), group_map, reduce_func_nb, *args)
            else:
                if returns_array:
                    func = jit_reg.resolve_option(nb.reduce_to_array_nb, jitted)
                else:
                    func = jit_reg.resolve_option(nb.reduce_nb, jitted)
                func = ch_reg.resolve_option(func, chunked)
                out = func(cls_or_self.to_2d_array(), reduce_func_nb, *args)

        wrap_kwargs = merge_dicts(
            dict(
                name_or_index="reduce" if not returns_array else None,
                to_index=returns_idx and to_index,
                fillna=-1 if returns_idx else None,
                dtype=int_ if returns_idx else None,
            ),
            wrap_kwargs,
        )
        return wrapper.wrap_reduced(out, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def proximity_apply(
        cls_or_self,
        window: int,
        reduce_func_nb: tp.Union[str, tp.AnyProximityReduceFunc],
        *args,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Frame:
        """Apply proximity reduction on data using a sliding window.

        This method applies a reduction function on neighboring data blocks.
        The method supports broadcasting and template substitution to flexibly process input data.

        Args:
            window (int): Window size.
            reduce_func_nb (Union[str, AnyProximityReduceFunc]): Callback function for proximal reducing.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `reduce_func_nb`.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Frame: Resulting DataFrame after applying the proximity reduction.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.proximity_reduce_nb` for the standard version.
            * `vectorbtpro.generic.nb.apply_reduce.proximity_reduce_meta_nb` for the meta version.

        Examples:
            Using regular function:

            ```pycon
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> df.vbt.proximity_apply(1, mean_nb)
                          a         b         c
            2020-01-01  3.0  2.500000  3.000000
            2020-01-02  3.0  2.666667  3.000000
            2020-01-03  3.0  2.777778  2.666667
            2020-01-04  3.0  2.666667  2.000000
            2020-01-05  3.0  2.500000  1.500000
            ```

            Using meta function:

            ```pycon
            >>> @njit
            ... def mean_ratio_meta_nb(from_i, to_i, from_col, to_col, a, b):
            ...     a_mean = np.mean(a[from_i:to_i, from_col:to_col])
            ...     b_mean = np.mean(b[from_i:to_i, from_col:to_col])
            ...     return a_mean / b_mean

            >>> vbt.pd_acc.proximity_apply(
            ...     1,
            ...     mean_ratio_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper,
            ... )
                          a         b         c
            2020-01-01  0.5  0.428571  0.500000
            2020-01-02  0.5  0.454545  0.500000
            2020-01-03  0.5  0.470588  0.454545
            2020-01-04  0.5  0.454545  0.333333
            2020-01-05  0.5  0.428571  0.200000
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.proximity_apply(
            ...     1,
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
                               a     b    c
            2020-01-01  1.000000  0.75  0.6
            2020-01-02  1.333333  1.00  0.8
            2020-01-03  2.000000  1.50  1.2
            2020-01-04  2.666667  2.00  1.6
            2020-01-05  3.000000  2.25  1.8
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            template_context = merge_dicts(
                broadcast_named_args,
                dict(wrapper=wrapper, window=window),
                template_context,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            func = jit_reg.resolve_option(nb.proximity_reduce_meta_nb, jitted)
            out = func(wrapper.shape_2d, window, reduce_func_nb, *args)
        else:
            func = jit_reg.resolve_option(nb.proximity_reduce_nb, jitted)
            out = func(cls_or_self.to_2d_array(), window, reduce_func_nb, *args)

        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Squeezing ############# #

    @hybrid_method
    def squeeze_grouped(
        cls_or_self,
        squeeze_func_nb: tp.Union[str, tp.AnyGroupSqueezeFunc],
        *args,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Squeeze each group of columns into a single column.

        Args:
            squeeze_func_nb (Union[str, AnyGroupSqueezeFunc]): Callback function for squeezing groups.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `squeeze_func_nb`.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeriesFrame: Squeezed data as a Series or DataFrame.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.squeeze_grouped_nb` for the standard version.
            * `vectorbtpro.generic.nb.apply_reduce.squeeze_grouped_meta_nb` for the meta version.

        Examples:
            Using regular function:

            ```pycon
            >>> mean_nb = njit(lambda a: np.nanmean(a))

            >>> group_by = pd.Series(['first', 'first', 'second'], name='group')
            >>> df.vbt.squeeze_grouped(mean_nb, group_by=group_by)
            group       first  second
            2020-01-01    3.0     1.0
            2020-01-02    3.0     2.0
            2020-01-03    3.0     3.0
            2020-01-04    3.0     2.0
            2020-01-05    3.0     1.0
            ```

            Using meta function:

            ```pycon
            >>> mean_ratio_meta_nb = njit(lambda i, group_idxs, group, a, b: \\
            ...     np.mean(a[i][group_idxs]) / np.mean(b[i][group_idxs]))

            >>> vbt.pd_acc.squeeze_grouped(
            ...     mean_ratio_meta_nb,
            ...     df.vbt.to_2d_array() - 1,
            ...     df.vbt.to_2d_array() + 1,
            ...     wrapper=df.vbt.wrapper,
            ...     group_by=group_by
            ... )
            group       first    second
            2020-01-01    0.5  0.000000
            2020-01-02    0.5  0.333333
            2020-01-03    0.5  0.500000
            2020-01-04    0.5  0.333333
            2020-01-05    0.5  0.000000
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.squeeze_grouped(
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=pd.Series([1, 2, 3, 4, 5], index=df.index),
            ...         b=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     ),
            ...     group_by=[0, 0, 1]
            ... )
                               0         1
            2020-01-01  0.666667  0.333333
            2020-01-02  1.333333  0.666667
            2020-01-03  2.000000  1.000000
            2020-01-04  2.666667  1.333333
            2020-01-05  3.333333  1.666667
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(squeeze_func_nb, str):
            squeeze_func_nb = getattr(nb, squeeze_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args,
                dict(wrapper=wrapper, group_by=group_by),
                template_context,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            if not wrapper.grouper.is_grouped(group_by=group_by):
                raise ValueError("Grouping required")
            group_map = wrapper.grouper.get_group_map(group_by=group_by)
            func = jit_reg.resolve_option(nb.squeeze_grouped_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(wrapper.shape_2d[0], group_map, squeeze_func_nb, *args)
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper
            if not wrapper.grouper.is_grouped(group_by=group_by):
                raise ValueError("Grouping required")
            group_map = wrapper.grouper.get_group_map(group_by=group_by)
            func = jit_reg.resolve_option(nb.squeeze_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(cls_or_self.to_2d_array(), group_map, squeeze_func_nb, *args)

        return wrapper.wrap(out, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Flattening ############# #

    def flatten_grouped(
        self,
        order: str = "C",
        jitted: tp.JittedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Flatten each group of columns in the associated data.

        Args:
            order (str): Order in which to flatten the array ("C" for row-major or "F" for column-major).
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeriesFrame: Flattened data as a Series or DataFrame.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.flatten_uniform_grouped_nb` for uniform groups.
            * `vectorbtpro.generic.nb.apply_reduce.flatten_grouped_nb` for non-uniform groups.

        !!! warning
            Ensure that the distribution of group lengths is nearly uniform. Otherwise,
            groups with fewer columns may be padded with NaN values, unnecessarily consuming memory.

        Examples:
            ```pycon
            >>> group_by = pd.Series(['first', 'first', 'second'], name='group')
            >>> df.vbt.flatten_grouped(group_by=group_by, order='C')
            group       first  second
            2020-01-01    1.0     1.0
            2020-01-01    5.0     NaN
            2020-01-02    2.0     2.0
            2020-01-02    4.0     NaN
            2020-01-03    3.0     3.0
            2020-01-03    3.0     NaN
            2020-01-04    4.0     2.0
            2020-01-04    2.0     NaN
            2020-01-05    5.0     1.0
            2020-01-05    1.0     NaN

            >>> df.vbt.flatten_grouped(group_by=group_by, order='F')
            group       first  second
            2020-01-01    1.0     1.0
            2020-01-02    2.0     2.0
            2020-01-03    3.0     3.0
            2020-01-04    4.0     2.0
            2020-01-05    5.0     1.0
            2020-01-01    5.0     NaN
            2020-01-02    4.0     NaN
            2020-01-03    3.0     NaN
            2020-01-04    2.0     NaN
            2020-01-05    1.0     NaN
            ```
        """
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping required")
        checks.assert_in(order.upper(), ["C", "F"])

        group_map = self.wrapper.grouper.get_group_map(group_by=group_by)
        if np.all(group_map[1] == group_map[1].item(0)):
            func = jit_reg.resolve_option(nb.flatten_uniform_grouped_nb, jitted)
        else:
            func = jit_reg.resolve_option(nb.flatten_grouped_nb, jitted)
        if order.upper() == "C":
            out = func(self.to_2d_array(), group_map, True)
            new_index = indexes.repeat_index(self.wrapper.index, np.max(group_map[1]))
        else:
            out = func(self.to_2d_array(), group_map, False)
            new_index = indexes.tile_index(self.wrapper.index, np.max(group_map[1]))
        wrap_kwargs = merge_dicts(dict(index=new_index), wrap_kwargs)
        return self.wrapper.wrap(out, group_by=group_by, **wrap_kwargs)

    # ############# Resampling ############# #

    def realign(
        self,
        index: tp.AnyRuleLike,
        freq: tp.Union[None, bool, tp.FrequencyLike] = None,
        nan_value: tp.Optional[tp.Scalar] = None,
        ffill: bool = True,
        source_rbound: tp.Union[bool, str, tp.IndexLike] = False,
        target_rbound: tp.Union[bool, str, tp.IndexLike] = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.MaybeSeriesFrame:
        """Realign data to a new index using a resampling strategy.

        This function wraps around `vectorbtpro.generic.nb.base.realign_nb` and accepts an
        `index` that is either an instance of `vectorbtpro.base.resampling.base.Resampler` or any
        index-like object. It produces results analogous to executing
        `df.resample(closed='right', label='right').last().ffill()` on the target index.

        Args:
            index (AnyRuleLike): Target index for resampling.
            freq (Union[None, bool, FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            nan_value (Optional[Scalar]): Value used for missing data.

                Uses -1 if a mapping exists; otherwise defaults to np.nan.
            ffill (bool): Whether to forward fill missing data after resampling.
            source_rbound (Union[bool, str, IndexLike]): Determines the right boundary of the source index.

                If set to a string "pandas", the resampler uses the pandas-defined right bound.

                !!! warning
                    Make sure to use the same convention for `target_rbound` as well.
            target_rbound (Union[bool, str, IndexLike]): Determines the right boundary of the target index.

                If set to a string "pandas", the resampler uses the pandas-defined right bound.

                !!! warning
                    Make sure to use the same convention for `source_rbound` as well.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            MaybeSeriesFrame: Realigned data as a Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.base.realign_nb`

        Examples:
            Downsampling:

            ```pycon
            >>> h_index = pd.date_range('2020-01-01', '2020-01-05', freq='1h')
            >>> d_index = pd.date_range('2020-01-01', '2020-01-05', freq='1d')

            >>> h_sr = pd.Series(range(len(h_index)), index=h_index)
            >>> h_sr.vbt.realign(d_index)
            2020-01-01     0.0
            2020-01-02    24.0
            2020-01-03    48.0
            2020-01-04    72.0
            2020-01-05    96.0
            Freq: D, dtype: float64
            ```

            Upsampling:

            ```pycon
            >>> d_sr = pd.Series(range(len(d_index)), index=d_index)
            >>> d_sr.vbt.realign(h_index)
            2020-01-01 00:00:00    0.0
            2020-01-01 01:00:00    0.0
            2020-01-01 02:00:00    0.0
            2020-01-01 03:00:00    0.0
            2020-01-01 04:00:00    0.0
            ...                    ...
            2020-01-04 20:00:00    3.0
            2020-01-04 21:00:00    3.0
            2020-01-04 22:00:00    3.0
            2020-01-04 23:00:00    3.0
            2020-01-05 00:00:00    4.0
            Freq: H, Length: 97, dtype: float64
            ```
        """
        resampler = self.wrapper.get_resampler(
            index,
            freq=freq,
            return_pd_resampler=False,
            silence_warnings=silence_warnings,
        )
        one_index = False
        if len(resampler.target_index) == 1 and checks.is_dt_like(index):
            if isinstance(index, str):
                try:
                    dt.to_freq(index)
                    one_index = False
                except Exception:
                    one_index = True
            else:
                one_index = True
        if isinstance(source_rbound, bool):
            use_source_rbound = source_rbound
        else:
            use_source_rbound = False
            if isinstance(source_rbound, str):
                if source_rbound == "pandas":
                    resampler = resampler.replace(source_index=resampler.source_rbound_index)
                else:
                    raise ValueError(f"Invalid source_rbound: '{source_rbound}'")
            else:
                resampler = resampler.replace(source_index=source_rbound)
        if isinstance(target_rbound, bool):
            use_target_rbound = target_rbound
            index = resampler.target_index
        else:
            use_target_rbound = False
            index = resampler.target_index
            if isinstance(target_rbound, str):
                if target_rbound == "pandas":
                    resampler = resampler.replace(target_index=resampler.target_rbound_index)
                else:
                    raise ValueError(f"Invalid target_rbound: '{target_rbound}'")
            else:
                resampler = resampler.replace(target_index=target_rbound)

        if not use_source_rbound:
            source_freq = None
        else:
            source_freq = resampler.get_np_source_freq()
        if not use_target_rbound:
            target_freq = None
        else:
            target_freq = resampler.get_np_target_freq()
        if nan_value is None:
            if self.mapping is not None:
                nan_value = -1
            else:
                nan_value = np.nan

        func = jit_reg.resolve_option(nb.realign_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            resampler.source_index.values,
            resampler.target_index.values,
            source_freq=source_freq,
            target_freq=target_freq,
            source_rbound=use_source_rbound,
            target_rbound=use_target_rbound,
            nan_value=nan_value,
            ffill=ffill,
        )
        wrap_kwargs = merge_dicts(dict(index=index), wrap_kwargs)
        out = self.wrapper.wrap(out, group_by=False, **wrap_kwargs)
        if one_index:
            return out.iloc[0]
        return out

    def realign_opening(self, *args, **kwargs) -> tp.MaybeSeriesFrame:
        """Realign data using left boundaries for source and target indexes.

        This is a convenience wrapper around `GenericAccessor.realign` that creates a resampler
        and applies left-bound indexing to both the source and target.

        Args:
            *args: Positional arguments for `GenericAccessor.realign`.
            **kwargs: Keyword arguments for `GenericAccessor.realign`.

        Returns:
            MaybeSeriesFrame: Realigned data as a Series or DataFrame.
        """
        return self.realign(*args, source_rbound=False, target_rbound=False, **kwargs)

    def realign_closing(self, *args, **kwargs) -> tp.MaybeSeriesFrame:
        """Realign data using right boundaries for source and target indexes.

        This is a convenience wrapper around `GenericAccessor.realign` that creates a resampler
        and applies right-bound indexing to both the source and target.

        Args:
            *args: Positional arguments for `GenericAccessor.realign`.
            **kwargs: Keyword arguments for `GenericAccessor.realign`.

        Returns:
            MaybeSeriesFrame: Realigned data as a Series or DataFrame.

        !!! note
            The timestamps in the source and target indexes should denote the open time.
        """
        return self.realign(*args, source_rbound=True, target_rbound=True, **kwargs)

    @hybrid_method
    def resample_to_index(
        cls_or_self,
        index: tp.AnyRuleLike,
        reduce_func_nb: tp.Union[str, tp.AnyRangeReduceFunc],
        *args,
        freq: tp.Union[None, bool, tp.FrequencyLike] = None,
        before: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SeriesFrame:
        """Resample data to a specified target index.

        This function applies the reduction function on segments of the source data corresponding to
        index ranges derived from the target index.

        Args:
            index (AnyRuleLike): Target index for resampling.
            reduce_func_nb (Union[str, AnyRangeReduceFunc]): Callback function for reducing ranges.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `reduce_func_nb`.
            freq (Union[None, bool, FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            before (bool): If True, use index ranges preceding the target index.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            SeriesFrame: Resampled data as a Pandas Series or DataFrame.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.reduce_index_ranges_nb` for standard operations.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_index_ranges_meta_nb` for meta operations.

        Examples:
            Downsampling:

            ```pycon
            >>> h_index = pd.date_range('2020-01-01', '2020-01-05', freq='1h')
            >>> d_index = pd.date_range('2020-01-01', '2020-01-05', freq='1d')

            >>> h_sr = pd.Series(range(len(h_index)), index=h_index)
            >>> h_sr.vbt.resample_to_index(d_index, njit(lambda x: x.mean()))
            2020-01-01    11.5
            2020-01-02    35.5
            2020-01-03    59.5
            2020-01-04    83.5
            2020-01-05    96.0
            Freq: D, dtype: float64

            >>> h_sr.vbt.resample_to_index(d_index, njit(lambda x: x.mean()), before=True)
            2020-01-01     0.0
            2020-01-02    12.5
            2020-01-03    36.5
            2020-01-04    60.5
            2020-01-05    84.5
            Freq: D, dtype: float64
            ```

            Upsampling:

            ```pycon
            >>> d_sr = pd.Series(range(len(d_index)), index=d_index)
            >>> d_sr.vbt.resample_to_index(h_index, njit(lambda x: x[-1]))
            2020-01-01 00:00:00    0.0
            2020-01-01 01:00:00    NaN
            2020-01-01 02:00:00    NaN
            2020-01-01 03:00:00    NaN
            2020-01-01 04:00:00    NaN
            ...                    ...
            2020-01-04 20:00:00    NaN
            2020-01-04 21:00:00    NaN
            2020-01-04 22:00:00    NaN
            2020-01-04 23:00:00    NaN
            2020-01-05 00:00:00    4.0
            Freq: H, Length: 97, dtype: float64
            ```

            Using meta function:

            ```pycon
            >>> mean_ratio_meta_nb = njit(lambda from_i, to_i, col, a, b: \\
            ...     np.mean(a[from_i:to_i][col]) / np.mean(b[from_i:to_i][col]))

            >>> vbt.pd_acc.resample_to_index(
            ...     d_index,
            ...     mean_ratio_meta_nb,
            ...     h_sr.vbt.to_2d_array() - 1,
            ...     h_sr.vbt.to_2d_array() + 1,
            ...     wrapper=h_sr.vbt.wrapper
            ... )
            2020-01-01   -1.000000
            2020-01-02    0.920000
            2020-01-03    0.959184
            2020-01-04    0.972603
            2020-01-05    0.979381
            Freq: D, dtype: float64
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.resample_to_index(
            ...     d_index,
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=h_sr - 1,
            ...         b=h_sr + 1
            ...     )
            ... )
            2020-01-01   -1.000000
            2020-01-02    0.920000
            2020-01-03    0.959184
            2020-01-04    0.972603
            2020-01-05    0.979381
            Freq: D, dtype: float64
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            index = substitute_templates(index, template_context, eval_id="index")
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        resampler = wrapper.get_resampler(
            index,
            freq=freq,
            return_pd_resampler=False,
            silence_warnings=silence_warnings,
        )
        index_ranges = resampler.map_index_to_source_ranges(before=before, jitted=jitted)

        if isinstance(cls_or_self, type):
            template_context = merge_dicts(
                dict(
                    resampler=resampler,
                    index_ranges=index_ranges,
                ),
                template_context,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            func = jit_reg.resolve_option(nb.reduce_index_ranges_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(
                wrapper.shape_2d[1],
                index_ranges[0],
                index_ranges[1],
                reduce_func_nb,
                *args,
            )
        else:
            func = jit_reg.resolve_option(nb.reduce_index_ranges_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(
                cls_or_self.to_2d_array(),
                index_ranges[0],
                index_ranges[1],
                reduce_func_nb,
                *args,
            )

        wrap_kwargs = merge_dicts(dict(index=resampler.target_index), wrap_kwargs)
        return wrapper.wrap(out, group_by=False, **wrap_kwargs)

    @hybrid_method
    def resample_between_bounds(
        cls_or_self,
        target_lbound_index: tp.IndexLike,
        target_rbound_index: tp.IndexLike,
        reduce_func_nb: tp.Union[str, tp.AnyRangeReduceFunc],
        *args,
        closed_lbound: bool = True,
        closed_rbound: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_with_lbound: tp.Optional[bool] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Resample between target index bounds.

        Args:
            target_lbound_index (IndexLike): Target lower bound index for resampling.
            target_rbound_index (IndexLike): Target upper bound index for resampling.
            reduce_func_nb (Union[str, AnyRangeReduceFuncc]): Callback function for reducing ranges.

                For specification, see the underlying Numba function.

                If provided as a string, selects the corresponding Numba callback function
                from `vectorbtpro.generic.nb` with the suffix `_reduce_nb`.
            *args: Positional arguments for `reduce_func_nb`.
            closed_lbound (bool): Indicates if the lower bound of the interval is inclusive.
            closed_rbound (bool): Indicates if the upper bound of the interval is inclusive.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrap_with_lbound (Optional[bool]): Determines if the lower bound index is used for wrapping the output.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Resampled Series or DataFrame with aggregated values,
                wrapped based on the specified parameters.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.reduce_index_ranges_nb` for standard operations.
            * `vectorbtpro.generic.nb.apply_reduce.reduce_index_ranges_meta_nb` for meta operations.

        Examples:
            Using regular function:

            ```pycon
            >>> h_index = pd.date_range('2020-01-01', '2020-01-05', freq='1h')
            >>> d_index = pd.date_range('2020-01-01', '2020-01-05', freq='1d')

            >>> h_sr = pd.Series(range(len(h_index)), index=h_index)
            >>> h_sr.vbt.resample_between_bounds(d_index, d_index.shift(), njit(lambda x: x.mean()))
            2020-01-01    11.5
            2020-01-02    35.5
            2020-01-03    59.5
            2020-01-04    83.5
            2020-01-05    96.0
            Freq: D, dtype: float64
            ```

            Using meta function:

            ```pycon
            >>> mean_ratio_meta_nb = njit(lambda from_i, to_i, col, a, b: \\
            ...     np.mean(a[from_i:to_i][col]) / np.mean(b[from_i:to_i][col]))

            >>> vbt.pd_acc.resample_between_bounds(
            ...     d_index,
            ...     d_index.shift(),
            ...     mean_ratio_meta_nb,
            ...     h_sr.vbt.to_2d_array() - 1,
            ...     h_sr.vbt.to_2d_array() + 1,
            ...     wrapper=h_sr.vbt.wrapper
            ... )
            2020-01-01   -1.000000
            2020-01-02    0.920000
            2020-01-03    0.959184
            2020-01-04    0.972603
            2020-01-05    0.979381
            Freq: D, dtype: float64
            ```

            Using templates and broadcasting:

            ```pycon
            >>> vbt.pd_acc.resample_between_bounds(
            ...     d_index,
            ...     d_index.shift(),
            ...     mean_ratio_meta_nb,
            ...     vbt.Rep('a'),
            ...     vbt.Rep('b'),
            ...     broadcast_named_args=dict(
            ...         a=h_sr - 1,
            ...         b=h_sr + 1
            ...     )
            ... )
            2020-01-01   -1.000000
            2020-01-02    0.920000
            2020-01-03    0.959184
            2020-01-04    0.972603
            2020-01-05    0.979381
            Freq: D, dtype: float64
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(reduce_func_nb, str):
            reduce_func_nb = getattr(nb, reduce_func_nb + "_reduce_nb")

        if isinstance(cls_or_self, type):
            if len(broadcast_named_args) > 0:
                broadcast_kwargs = merge_dicts(dict(to_pd=False, min_ndim=2), broadcast_kwargs)
                if wrapper is not None:
                    broadcast_named_args = reshaping.broadcast(
                        broadcast_named_args,
                        to_shape=wrapper.shape_2d,
                        **broadcast_kwargs,
                    )
                else:
                    broadcast_named_args, wrapper = reshaping.broadcast(
                        broadcast_named_args,
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")
            template_context = merge_dicts(
                broadcast_named_args, dict(wrapper=wrapper), template_context
            )
            target_lbound_index = substitute_templates(
                target_lbound_index, template_context, eval_id="target_lbound_index"
            )
            target_rbound_index = substitute_templates(
                target_rbound_index, template_context, eval_id="target_rbound_index"
            )
        else:
            if wrapper is None:
                wrapper = cls_or_self.wrapper

        target_lbound_index = dt.prepare_dt_index(target_lbound_index)
        target_rbound_index = dt.prepare_dt_index(target_rbound_index)
        if len(target_lbound_index) == 1 and len(target_rbound_index) > 1:
            target_lbound_index = repeat_index(target_lbound_index, len(target_rbound_index))
            if wrap_with_lbound is None:
                wrap_with_lbound = False
        elif len(target_lbound_index) > 1 and len(target_rbound_index) == 1:
            target_rbound_index = repeat_index(target_rbound_index, len(target_lbound_index))
            if wrap_with_lbound is None:
                wrap_with_lbound = True
        index_ranges = Resampler.map_bounds_to_source_ranges(
            source_index=wrapper.index.values,
            target_lbound_index=target_lbound_index.values,
            target_rbound_index=target_rbound_index.values,
            closed_lbound=closed_lbound,
            closed_rbound=closed_rbound,
            skip_not_found=False,
            jitted=jitted,
        )

        if isinstance(cls_or_self, type):
            template_context = merge_dicts(
                dict(
                    target_lbound_index=target_lbound_index,
                    target_rbound_index=target_rbound_index,
                    index_ranges=index_ranges,
                ),
                template_context,
            )
            args = substitute_templates(args, template_context, eval_id="args")
            func = jit_reg.resolve_option(nb.reduce_index_ranges_meta_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(
                wrapper.shape_2d[1],
                index_ranges[0],
                index_ranges[1],
                reduce_func_nb,
                *args,
            )
        else:
            func = jit_reg.resolve_option(nb.reduce_index_ranges_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            out = func(
                cls_or_self.to_2d_array(),
                index_ranges[0],
                index_ranges[1],
                reduce_func_nb,
                *args,
            )

        if wrap_with_lbound is None:
            if closed_lbound:
                wrap_with_lbound = True
            elif closed_rbound:
                wrap_with_lbound = False
            else:
                wrap_with_lbound = True
        if wrap_with_lbound:
            wrap_kwargs = merge_dicts(dict(index=target_lbound_index), wrap_kwargs)
        else:
            wrap_kwargs = merge_dicts(dict(index=target_rbound_index), wrap_kwargs)
        return wrapper.wrap(out, group_by=False, **wrap_kwargs)

    # ############# Describing ############# #

    def min(
        self,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the minimum of non-NaN elements.

        Args:
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Minimum value computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.min_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nanmin_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="min"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.min_reduce_nb, jitted),
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nanmin_nb, jitted)
        elif arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            func = partial(np.nanmin, axis=0)
        else:
            func = partial(nanmin, axis=0)
        func = ch_reg.resolve_option(nb.nanmin_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr), group_by=False, **wrap_kwargs)

    def max(
        self,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the maximum of non-NaN elements.

        Args:
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Maximum value computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.max_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nanmax_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="max"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.max_reduce_nb, jitted),
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nanmax_nb, jitted)
        elif arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            func = partial(np.nanmax, axis=0)
        else:
            func = partial(nanmax, axis=0)
        func = ch_reg.resolve_option(nb.nanmax_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr), group_by=False, **wrap_kwargs)

    def mean(
        self,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the arithmetic mean of non-NaN elements.

        Args:
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Mean value computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.mean_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nanmean_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="mean"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.mean_reduce_nb, jitted),
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nanmean_nb, jitted)
        elif arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            func = partial(np.nanmean, axis=0)
        else:
            func = partial(nanmean, axis=0)
        func = ch_reg.resolve_option(nb.nanmean_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr), group_by=False, **wrap_kwargs)

    def median(
        self,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the median of non-NaN elements.

        Args:
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Median value computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.median_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nanmedian_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="median"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.median_reduce_nb, jitted),
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nanmedian_nb, jitted)
        elif arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            func = partial(np.nanmedian, axis=0)
        else:
            func = partial(nanmedian, axis=0)
        func = ch_reg.resolve_option(nb.nanmedian_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr), group_by=False, **wrap_kwargs)

    def std(
        self,
        ddof: int = 1,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the standard deviation of non-NaN elements.

        Args:
            ddof (int): Delta degrees of freedom.
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Standard deviation computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.std_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nanstd_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="std"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.std_reduce_nb, jitted),
                ddof,
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nanstd_nb, jitted)
        elif arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            func = partial(np.nanstd, axis=0)
        else:
            func = partial(nanstd, axis=0)
        func = ch_reg.resolve_option(nb.nanstd_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr, ddof=ddof), group_by=False, **wrap_kwargs)

    def sum(
        self,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the sum of non-NaN elements.

        Args:
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Sum computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.sum_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nansum_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="sum"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.sum_reduce_nb, jitted),
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nansum_nb, jitted)
        elif arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            func = partial(np.nansum, axis=0)
        else:
            func = partial(nansum, axis=0)
        func = ch_reg.resolve_option(nb.nansum_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr), group_by=False, **wrap_kwargs)

    def count(
        self,
        use_jitted: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return count of non-NaN elements.

        Args:
            use_jitted (Optional[bool]): Whether to use jitted execution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Count computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.count_reduce_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nancnt_nb` for non-grouped operations.

        !!! info
            For default settings, see `vectorbtpro._settings.generic`.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="count", dtype=int_), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.count_reduce_nb, jitted),
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        from vectorbtpro._settings import settings

        generic_cfg = settings["generic"]

        arr = self.to_2d_array()
        if use_jitted is None:
            use_jitted = generic_cfg["use_jitted"]
        if use_jitted:
            func = jit_reg.resolve_option(nb.nancnt_nb, jitted)
        else:
            func = lambda a: np.sum(~np.isnan(a), axis=0)
        func = ch_reg.resolve_option(nb.nancnt_nb, chunked, target_func=func)
        return self.wrapper.wrap_reduced(func(arr), group_by=False, **wrap_kwargs)

    def cov(
        self,
        other: tp.SeriesFrame,
        ddof: int = 1,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return covariance of non-NaN elements.

        Args:
            other (SeriesFrame): Series or DataFrame to compute covariance with.
            ddof (int): Delta degrees of freedom.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Covariance computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.cov_reduce_grouped_meta_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nancov_nb` for non-grouped operations.
        """
        self_obj, other_obj = reshaping.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        self_arr = reshaping.to_2d_array(self_obj)
        other_arr = reshaping.to_2d_array(other_obj)
        wrap_kwargs = merge_dicts(dict(name_or_index="cov"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return type(self).reduce(
                jit_reg.resolve_option(nb.cov_reduce_grouped_meta_nb, jitted),
                self_arr,
                other_arr,
                ddof,
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                wrapper=ArrayWrapper.from_obj(self_obj),
                group_by=self.wrapper.grouper.resolve_group_by(group_by=group_by),
                wrap_kwargs=wrap_kwargs,
            )

        func = jit_reg.resolve_option(nb.nancov_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.wrapper.wrap_reduced(
            func(self_arr, other_arr, ddof=ddof), group_by=False, **wrap_kwargs
        )

    def corr(
        self,
        other: tp.SeriesFrame,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return correlation coefficient of non-NaN elements.

        Args:
            other (SeriesFrame): Series or DataFrame to compute correlation with.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Correlation coefficient computed from non-NaN elements.

        See:
            * `vectorbtpro.generic.nb.apply_reduce.corr_reduce_grouped_meta_nb` for grouped operations.
            * `vectorbtpro.generic.nb.base.nancorr_nb` for non-grouped operations.
        """
        self_obj, other_obj = reshaping.broadcast(self.obj, other, **resolve_dict(broadcast_kwargs))
        self_arr = reshaping.to_2d_array(self_obj)
        other_arr = reshaping.to_2d_array(other_obj)
        wrap_kwargs = merge_dicts(dict(name_or_index="corr"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return type(self).reduce(
                jit_reg.resolve_option(nb.corr_reduce_grouped_meta_nb, jitted),
                self_arr,
                other_arr,
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                wrapper=ArrayWrapper.from_obj(self_obj),
                group_by=self.wrapper.grouper.resolve_group_by(group_by=group_by),
                wrap_kwargs=wrap_kwargs,
            )

        func = jit_reg.resolve_option(nb.nancorr_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        return self.wrapper.wrap_reduced(func(self_arr, other_arr), group_by=False, **wrap_kwargs)

    def rank(
        self,
        pct: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute numerical data rank.

        By default, equal values receive the average rank.

        Args:
            pct (bool): If True, compute the rank as a percentile.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Rank as a Pandas Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.base.rank_nb`
        """
        func = jit_reg.resolve_option(nb.rank_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        arr = self.to_2d_array()
        argsorted = np.argsort(arr, axis=0)
        rank = func(arr, argsorted=argsorted, pct=pct)
        return self.wrapper.wrap(rank, group_by=False, **resolve_dict(wrap_kwargs))

    def idxmin(
        self,
        order: str = "C",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return labeled index of minimum non-NaN element.

        Args:
            order (str): Order in which to flatten the array ("C" for row-major or "F" for column-major).
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            SeriesFrame: Labeled index as a Pandas Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.apply_reduce.argmin_reduce_nb`
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="idxmin"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.argmin_reduce_nb, jitted),
                returns_idx=True,
                flatten=True,
                order=order,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        def _func(arr, index):
            out = np.full(arr.shape[1], np.nan, dtype=object)
            nan_mask = np.all(np.isnan(arr), axis=0)
            out[~nan_mask] = index[nanargmin(arr[:, ~nan_mask], axis=0)]
            return out

        chunked = ch.specialize_chunked_option(chunked, arg_take_spec=dict(index=None))
        _func = ch_reg.resolve_option(nb.nanmin_nb, chunked, target_func=_func)
        out = _func(self.to_2d_array(), self.wrapper.index)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def idxmax(
        self,
        order: str = "C",
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return labeled index of maximum non-NaN element.

        Args:
            order (str): Order in which to flatten the array ("C" for row-major or "F" for column-major).
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            SeriesFrame: Labeled index as a Pandas Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.apply_reduce.argmax_reduce_nb`
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="idxmax"), wrap_kwargs)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.argmax_reduce_nb, jitted),
                returns_idx=True,
                flatten=True,
                order=order,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )

        def _func(arr, index):
            out = np.full(arr.shape[1], np.nan, dtype=object)
            nan_mask = np.all(np.isnan(arr), axis=0)
            out[~nan_mask] = index[nanargmax(arr[:, ~nan_mask], axis=0)]
            return out

        chunked = ch.specialize_chunked_option(chunked, arg_take_spec=dict(index=None))
        _func = ch_reg.resolve_option(nb.nanmax_nb, chunked, target_func=_func)
        out = _func(self.to_2d_array(), self.wrapper.index)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def describe(
        self,
        percentiles: tp.Optional[tp.ArrayLike] = None,
        ddof: int = 1,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute descriptive statistics for the data.

        This method computes statistics such as count, mean, standard deviation, minimum,
        specified percentiles, and maximum by reducing the data using
        `vectorbtpro.generic.nb.apply_reduce.describe_reduce_nb`.
        For details on the percentiles parameter, please refer to `pd.DataFrame.describe`.

        Args:
            percentiles (Optional[ArrayLike]): Percentiles to include in the summary.

                If None, defaults are used.
            ddof (int): Delta degrees of freedom.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Descriptive statistics as a Pandas Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.apply_reduce.describe_reduce_nb`

        Examples:
            ```pycon
            >>> df.vbt.describe()
                          a         b        c
            count  5.000000  5.000000  5.00000
            mean   3.000000  3.000000  1.80000
            std    1.581139  1.581139  0.83666
            min    1.000000  1.000000  1.00000
            25%    2.000000  2.000000  1.00000
            50%    3.000000  3.000000  2.00000
            75%    4.000000  4.000000  2.00000
            max    5.000000  5.000000  3.00000
            ```
        """
        if percentiles is not None:
            percentiles = reshaping.to_1d_array(percentiles)
        else:
            percentiles = np.array([0.25, 0.5, 0.75])
        percentiles = percentiles.tolist()
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.unique(percentiles)
        perc_formatted = pd.io.formats.format.format_percentiles(percentiles)
        index = pd.Index(["count", "mean", "std", "min", *perc_formatted, "max"])
        wrap_kwargs = merge_dicts(dict(name_or_index=index), wrap_kwargs)
        chunked = ch.specialize_chunked_option(
            chunked, arg_take_spec=dict(args=ch.ArgsTaker(None, None))
        )
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.reduce(
                jit_reg.resolve_option(nb.describe_reduce_nb, jitted),
                percentiles,
                ddof,
                returns_array=True,
                flatten=True,
                jitted=jitted,
                chunked=chunked,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
            )
        else:
            return self.reduce(
                jit_reg.resolve_option(nb.describe_reduce_nb, jitted),
                percentiles,
                ddof,
                returns_array=True,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
            )

    def digitize(
        self,
        bins: tp.ArrayLike = "auto",
        right: bool = False,
        return_mapping: bool = False,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.SeriesFrame, tp.Tuple[tp.SeriesFrame, dict]]:
        """Digitize the data into bins using `np.digitize`.

        This method assigns each value in the data to a bin defined by `bins`.
        If an integer is provided for `bins`, the bin edges are computed automatically based on the data range.
        Optionally, a mapping of bin indices to their corresponding bin intervals can be returned.

        Args:
            bins (ArrayLike): Either an array defining the bin edges or an integer
                specifying the number of bins.
            right (bool): Whether to treat the bins as right-inclusive.
            return_mapping (bool): Whether to return a mapping of bin indices to bin
                intervals along with the digitized data.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Union[SeriesFrame, Tuple[SeriesFrame, dict]]: Digitized data.

                If `return_mapping` is True, also returns a mapping of bin indices to bin intervals.

        Examples:
            ```pycon
            >>> df.vbt.digitize(3)
                        a  b  c
            2020-01-01  1  3  1
            2020-01-02  1  3  1
            2020-01-03  2  2  2
            2020-01-04  3  1  1
            2020-01-05  3  1  1
            ```
        """
        if wrap_kwargs is None:
            wrap_kwargs = {}
        arr = self.to_2d_array()
        if not np.iterable(bins):
            if np.isscalar(bins) and bins < 1:
                raise ValueError("Bins must be a positive integer")

            rng = (np.nanmin(self.obj.values), np.nanmax(self.obj.values))
            mn, mx = (mi + 0.0 for mi in rng)

            if np.isinf(mn) or np.isinf(mx):
                raise ValueError("Cannot specify integer bins when input data contains infinity")
            elif mn == mx:  # adjust end points before binning
                mn -= 0.001 * abs(mn) if mn != 0 else 0.001
                mx += 0.001 * abs(mx) if mx != 0 else 0.001
                bins = np.linspace(mn, mx, bins + 1, endpoint=True)
            else:  # adjust end points after binning
                bins = np.linspace(mn, mx, bins + 1, endpoint=True)
                adj = (mx - mn) * 0.001  # 0.1% of the range
                if right:
                    bins[0] -= adj
                else:
                    bins[-1] += adj
        bin_edges = reshaping.to_1d_array(bins)
        mapping = dict()
        if right:
            out = np.digitize(arr, bin_edges[1:], right=right)
            if return_mapping:
                for i in range(len(bin_edges) - 1):
                    mapping[i] = (bin_edges[i], bin_edges[i + 1])
        else:
            out = np.digitize(arr, bin_edges[:-1], right=right)
            if return_mapping:
                for i in range(1, len(bin_edges)):
                    mapping[i] = (bin_edges[i - 1], bin_edges[i])
        if return_mapping:
            return self.wrapper.wrap(out, **wrap_kwargs), mapping
        return self.wrapper.wrap(out, **wrap_kwargs)

    def value_counts(
        self,
        axis: int = 1,
        normalize: bool = False,
        sort_uniques: bool = True,
        sort: bool = False,
        ascending: bool = False,
        dropna: bool = False,
        group_by: tp.GroupByLike = None,
        mapping: tp.Union[None, bool, tp.MappingLike] = None,
        incl_all_keys: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return a Series/DataFrame containing counts of unique values.

        Args:
            axis (int): Axis along which to compute counts.

                Valid options:

                * 0: Count per row.
                * 1: Count per column.
                * -1: Count across the entire object.
            normalize (bool): If True, return relative frequencies of unique values.
            sort_uniques (bool): Sort unique values before further processing.
            sort (bool): Sort the results based on frequency counts.
            ascending (bool): True for ascending order, False for descending.
            dropna (bool): Exclude NaN values from the counts.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            mapping (Union[None, bool, MappingLike]): Mapping to relabel unique values.
            incl_all_keys (bool): Include keys from `mapping` that are missing in the data.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.mapping.apply_mapping`.

        Returns:
            SeriesFrame: Series or DataFrame with counts of unique values.

        See:
            * `vectorbtpro.generic.nb.base.value_counts_per_row_nb` for `axis=0`.
            * `vectorbtpro.generic.nb.base.value_counts_nb` for `axis=1`.
            * `vectorbtpro.generic.nb.base.value_counts_1d_nb` for `axis=-1`.

        Examples:
            ```pycon
            >>> df.vbt.value_counts()
               a  b  c
            1  1  1  2
            2  1  1  2
            3  1  1  1
            4  1  1  0
            5  1  1  0

            >>> df.vbt.value_counts(axis=-1)
            1    4
            2    4
            3    3
            4    2
            5    2
            Name: value_counts, dtype: int64

            >>> mapping = {x: 'test_' + str(x) for x in pd.unique(df.values.flatten())}
            >>> df.vbt.value_counts(mapping=mapping)
                    a  b  c
            test_1  1  1  2
            test_2  1  1  2
            test_3  1  1  1
            test_4  1  1  0
            test_5  1  1  0

            >>> sr = pd.Series([1, 2, 2, 3, 3, 3, np.nan])
            >>> sr.vbt.value_counts(mapping=mapping)
            test_1    1
            test_2    2
            test_3    3
            NaN       1
            dtype: int64

            >>> sr.vbt.value_counts(mapping=mapping, dropna=True)
            test_1    1
            test_2    2
            test_3    3
            dtype: int64

            >>> sr.vbt.value_counts(mapping=mapping, sort=True)
            test_3    3
            test_2    2
            test_1    1
            NaN       1
            dtype: int64

            >>> sr.vbt.value_counts(mapping=mapping, sort=True, ascending=True)
            test_1    1
            NaN       1
            test_2    2
            test_3    3
            dtype: int64

            >>> sr.vbt.value_counts(mapping=mapping, incl_all_keys=True)
            test_1    1
            test_2    2
            test_3    3
            test_4    0
            test_5    0
            NaN       1
            dtype: int64
            ```
        """
        checks.assert_in(axis, (-1, 0, 1))

        mapping = self.resolve_mapping(mapping=mapping)
        codes, uniques = pd.factorize(self.obj.values.flatten(), sort=False, use_na_sentinel=False)
        if axis == 0:
            func = jit_reg.resolve_option(nb.value_counts_per_row_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            value_counts = func(codes.reshape(self.wrapper.shape_2d), len(uniques))
        elif axis == 1:
            group_map = self.wrapper.grouper.get_group_map(group_by=group_by)
            func = jit_reg.resolve_option(nb.value_counts_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            value_counts = func(codes.reshape(self.wrapper.shape_2d), len(uniques), group_map)
        else:
            func = jit_reg.resolve_option(nb.value_counts_1d_nb, jitted)
            value_counts = func(codes, len(uniques))
        if incl_all_keys and mapping is not None:
            missing_keys = []
            for x in mapping:
                if pd.isnull(x) and pd.isnull(uniques).any():
                    continue
                if x not in uniques:
                    missing_keys.append(x)
            if axis == 0 or axis == 1:
                value_counts = np.vstack(
                    (value_counts, np.full((len(missing_keys), value_counts.shape[1]), 0))
                )
            else:
                value_counts = np.concatenate((value_counts, np.full(len(missing_keys), 0)))
            uniques = np.concatenate((uniques, np.array(missing_keys)))
        nan_mask = np.isnan(uniques)
        if dropna:
            value_counts = value_counts[~nan_mask]
            uniques = uniques[~nan_mask]
        if sort_uniques:
            new_indices = uniques.argsort()
            value_counts = value_counts[new_indices]
            uniques = uniques[new_indices]
        if axis == 0 or axis == 1:
            value_counts_sum = value_counts.sum(axis=1)
        else:
            value_counts_sum = value_counts
        if normalize:
            value_counts = value_counts / value_counts_sum.sum()
        if sort:
            if ascending:
                new_indices = value_counts_sum.argsort()
            else:
                new_indices = (-value_counts_sum).argsort()
            value_counts = value_counts[new_indices]
            uniques = uniques[new_indices]
        if axis == 0:
            wrapper = ArrayWrapper.from_obj(value_counts)
            value_counts_pd = wrapper.wrap(
                value_counts,
                index=uniques,
                columns=self.wrapper.index,
                **resolve_dict(wrap_kwargs),
            )
        elif axis == 1:
            value_counts_pd = self.wrapper.wrap(
                value_counts,
                index=uniques,
                group_by=group_by,
                **resolve_dict(wrap_kwargs),
            )
        else:
            wrapper = ArrayWrapper.from_obj(value_counts)
            value_counts_pd = wrapper.wrap(
                value_counts,
                index=uniques,
                **merge_dicts(dict(columns=["value_counts"]), wrap_kwargs),
            )
        if mapping is not None:
            value_counts_pd.index = apply_mapping(value_counts_pd.index, mapping, **kwargs)
        return value_counts_pd

    # ############# Transforming ############# #

    def demean(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Remove the mean value from the data array.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            group_by (GroupByLike): Grouping specification.

                If provided, the data is demeaned separately for each group.
                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Demeaned data as a Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.base.demean_nb`
        """
        func = jit_reg.resolve_option(nb.demean_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        group_map = self.wrapper.grouper.get_group_map(group_by=group_by)
        out = func(self.to_2d_array(), group_map)
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def transform(
        self, transformer: tp.TransformerT, wrap_kwargs: tp.KwargsLike = None, **kwargs
    ) -> tp.SeriesFrame:
        """Transform the data using the specified transformer.

        Applies the given transformer to the dataset. The transformer must be an instance with
        `transform` and `fit_transform` methods, ideally subclassing `sklearn.base.TransformerMixin`
        and `sklearn.base.BaseEstimator`.

        If the transformer is not already fitted, it is fitted using `fit_transform`;
        otherwise, `transform` is applied.

        Args:
            transformer (Transformer): Instance with `transform` and `fit_transform` methods.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for the transformer methods.

        Returns:
            SeriesFrame: Transformed data as a Series or DataFrame.

        Examples:
            ```pycon
            >>> from sklearn.preprocessing import MinMaxScaler

            >>> df.vbt.transform(MinMaxScaler((-1, 1)))
                          a    b    c
            2020-01-01 -1.0  1.0 -1.0
            2020-01-02 -0.5  0.5  0.0
            2020-01-03  0.0  0.0  1.0
            2020-01-04  0.5 -0.5  0.0
            2020-01-05  1.0 -1.0 -1.0

            >>> fitted_scaler = MinMaxScaler((-1, 1)).fit(np.array([[2], [4]]))
            >>> df.vbt.transform(fitted_scaler)
                          a    b    c
            2020-01-01 -2.0  2.0 -2.0
            2020-01-02 -1.0  1.0 -1.0
            2020-01-03  0.0  0.0  0.0
            2020-01-04  1.0 -1.0 -1.0
            2020-01-05  2.0 -2.0 -2.0
            ```
        """
        is_fitted = True
        try:
            check_is_fitted(transformer)
        except NotFittedError:
            is_fitted = False
        if not is_fitted:
            result = transformer.fit_transform(self.to_2d_array(), **kwargs)
        else:
            result = transformer.transform(self.to_2d_array(), **kwargs)
        return self.wrapper.wrap(result, group_by=False, **resolve_dict(wrap_kwargs))

    def zscore(self, **kwargs) -> tp.SeriesFrame:
        """Compute the z-score normalization using `sklearn.preprocessing.StandardScaler`.

        Args:
            **kwargs: Keyword arguments for `GenericAccessor.scale`.

        Returns:
            SeriesFrame: Computed z-score as a Series or DataFrame.
        """
        return self.scale(with_mean=True, with_std=True, **kwargs)

    def rebase(
        self,
        base: float,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Rebase the series to the given base.

        Scales the series relative to `base` to facilitate easier comparison and plotting.
        NaN values are filled using forward and backward fill.

        Args:
            base (float): Base value to rebase the series.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Rebased data as a Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.base.fbfill_nb`
        """
        func = jit_reg.resolve_option(nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        result = func(self.to_2d_array())
        result = result / result[0] * base
        return self.wrapper.wrap(result, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Conversion ############# #

    def drawdown(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return the drawdown series.

        Computes the drawdown series for the dataset.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Drawdown series as a Series or DataFrame.

        See:
            `vectorbtpro.generic.nb.records.drawdown_nb`
        """
        func = jit_reg.resolve_option(nb.drawdown_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(self.to_2d_array())
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def get_ranges(self, *args, wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> Ranges:
        """Generate range records from the data.

        Utilizes `vectorbtpro.generic.ranges.Ranges.from_array` to generate range records.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.ranges.Ranges.from_array`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_array`.

        Returns:
            Ranges: Ranges generated from the data.
        """
        wrapper_kwargs = merge_dicts(self.wrapper.config, wrapper_kwargs)
        return Ranges.from_array(self.obj, *args, wrapper_kwargs=wrapper_kwargs, **kwargs)

    @property
    def ranges(self) -> Ranges:
        """Range records computed with default arguments by invoking `GenericAccessor.get_ranges`.

        Returns:
            Ranges: Ranges generated from the data.
        """
        return self.get_ranges()

    def get_drawdowns(self, *args, **kwargs) -> Drawdowns:
        """Generate drawdown records from the data.

        Utilizes `vectorbtpro.generic.drawdowns.Drawdowns.from_price` to generate drawdown records.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.drawdowns.Drawdowns.from_price`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.drawdowns.Drawdowns.from_price`.

        Returns:
            Drawdowns: Drawdowns generated from the data.
        """
        return Drawdowns.from_price(self.obj, *args, wrapper=self.wrapper, **kwargs)

    @property
    def drawdowns(self) -> Drawdowns:
        """Drawdown records computed with default arguments by invoking `GenericAccessor.get_drawdowns`.

        Returns:
            Drawdowns: Drawdown records computed with default arguments.
        """
        return self.get_drawdowns()

    def to_mapped(
        self,
        dropna: bool = True,
        dtype: tp.Optional[tp.DTypeLike] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> MappedArray:
        """Convert the data into a `vectorbtpro.records.mapped_array.MappedArray` instance.

        Flattens the data and maps indices and columns. NaN values are dropped if specified.

        Args:
            dropna (bool): Whether to drop NaN values.
            dtype (Optional[DTypeLike]): Data type for the mapped array.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray`.

        Returns:
            MappedArray: Mapped array.
        """
        mapped_arr = self.to_2d_array().flatten(order="F")
        col_arr = np.repeat(np.arange(self.wrapper.shape_2d[1]), self.wrapper.shape_2d[0])
        idx_arr = np.tile(np.arange(self.wrapper.shape_2d[0]), self.wrapper.shape_2d[1])
        if dropna and np.isnan(mapped_arr).any():
            not_nan_mask = ~np.isnan(mapped_arr)
            mapped_arr = mapped_arr[not_nan_mask]
            col_arr = col_arr[not_nan_mask]
            idx_arr = idx_arr[not_nan_mask]
        return MappedArray(
            self.wrapper,
            np.asarray(mapped_arr, dtype=dtype),
            col_arr,
            idx_arr=idx_arr,
            **kwargs,
        ).regroup(group_by)

    def to_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return the returns of the data.

        Computes the returns using `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`
        with value returns enabled.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Return series as a Series or DataFrame.
        """
        from vectorbtpro.returns.accessors import ReturnsAccessor

        return ReturnsAccessor.from_value(
            self._obj,
            wrapper=self.wrapper,
            return_values=True,
            **kwargs,
        )

    def to_log_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return the logarithmic returns of the data.

        Compute log returns using `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`
        with log returns enabled.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Log return series as a Series or DataFrame.
        """
        from vectorbtpro.returns.accessors import ReturnsAccessor

        return ReturnsAccessor.from_value(
            self._obj,
            wrapper=self.wrapper,
            return_values=True,
            log_returns=True,
            **kwargs,
        )

    def to_daily_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return the daily returns of the data.

        Compute daily returns for the dataset using
        `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`
        with value returns disabled, and aggregate them by day.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Daily return series as a Series or DataFrame.
        """
        from vectorbtpro.returns.accessors import ReturnsAccessor

        return ReturnsAccessor.from_value(
            self._obj,
            wrapper=self.wrapper,
            return_values=False,
            **kwargs,
        ).daily()

    def to_daily_log_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return the daily logarithmic returns of the data.

        Compute daily log returns for the dataset using
        `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`
        with log returns enabled, and aggregate them by day.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Daily log return series as a Series or DataFrame.
        """
        from vectorbtpro.returns.accessors import ReturnsAccessor

        return ReturnsAccessor.from_value(
            self._obj,
            wrapper=self.wrapper,
            return_values=False,
            log_returns=True,
            **kwargs,
        ).daily()

    # ############# Patterns ############# #

    def find_pattern(self, *args, **kwargs) -> PatternRanges:
        """Return pattern range records from the data.

        Generate pattern range records using `vectorbtpro.generic.ranges.PatternRanges.from_pattern_search`.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.ranges.PatternRanges.from_pattern_search`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.PatternRanges.from_pattern_search`.

        Returns:
            PatternRanges: Generated pattern range records.
        """
        return PatternRanges.from_pattern_search(self.obj, *args, **kwargs)

    # ############# Crossover ############# #

    def crossed_above(
        self,
        other: tp.ArrayLike,
        wait: int = 0,
        dropna: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return a boolean series indicating where the caller crosses above the provided array.

        Args:
            other (ArrayLike): Input array to compare against.
            wait (int): Number of periods to wait before confirming a cross.
            dropna (bool): Flag to determine whether to handle missing values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Boolean Series or DataFrame indicating crossing events.

        See:
            `vectorbtpro.generic.nb.base.crossed_above_nb`

        Examples:
            ```pycon
            >>> df['b'].vbt.crossed_above(df['c'])
            2020-01-01    False
            2020-01-02    False
            2020-01-03    False
            2020-01-04    False
            2020-01-05    False
            dtype: bool

            >>> df['a'].vbt.crossed_above(df['b'])
            2020-01-01    False
            2020-01-02    False
            2020-01-03    False
            2020-01-04     True
            2020-01-05    False
            dtype: bool

            >>> df['a'].vbt.crossed_above(df['b'], wait=1)
            2020-01-01    False
            2020-01-02    False
            2020-01-03    False
            2020-01-04    False
            2020-01-05     True
            dtype: bool
            ```
        """
        broadcastable_args = dict(obj=self.obj, other=other)
        broadcast_kwargs = merge_dicts(
            dict(keep_flex=dict(obj=False, other=True)), broadcast_kwargs
        )
        broadcasted_args, wrapper = reshaping.broadcast(
            broadcastable_args,
            to_pd=False,
            min_ndim=2,
            return_wrapper=True,
            **broadcast_kwargs,
        )
        func = jit_reg.resolve_option(nb.crossed_above_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            broadcasted_args["obj"],
            broadcasted_args["other"],
            wait=wait,
            dropna=dropna,
        )
        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def crossed_below(
        self,
        other: tp.ArrayLike,
        wait: int = 0,
        dropna: bool = True,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return a boolean series indicating where the caller crosses below the provided array.

        See also `GenericAccessor.crossed_above` for similar examples.

        Args:
            other (ArrayLike): Input array to compare against.
            wait (int): Number of periods to wait before confirming a cross.
            dropna (bool): Flag to determine whether to handle missing values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Boolean Series or DataFrame indicating crossing events.

        See:
            `vectorbtpro.generic.nb.base.crossed_below_nb`
        """
        broadcastable_args = dict(obj=self.obj, other=other)
        broadcast_kwargs = merge_dicts(
            dict(keep_flex=dict(obj=False, other=True)), broadcast_kwargs
        )
        broadcasted_args, wrapper = reshaping.broadcast(
            broadcastable_args,
            to_pd=False,
            min_ndim=2,
            return_wrapper=True,
            **broadcast_kwargs,
        )
        func = jit_reg.resolve_option(nb.crossed_below_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            broadcasted_args["obj"],
            broadcasted_args["other"],
            wait=wait,
            dropna=dropna,
        )
        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Resolution ############# #

    def resolve_self(
        self: GenericAccessorT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.Optional[tp.Set[str]] = None,
        impacts_caching: bool = True,
        silence_warnings: bool = False,
    ) -> GenericAccessorT:
        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()

        reself = Wrapping.resolve_self(
            self,
            cond_kwargs=cond_kwargs,
            custom_arg_names=custom_arg_names,
            impacts_caching=impacts_caching,
            silence_warnings=silence_warnings,
        )
        if "mapping" in cond_kwargs:
            self_copy = reself.replace(mapping=cond_kwargs["mapping"])

            if not checks.is_deep_equal(self_copy.mapping, reself.mapping):
                if not silence_warnings:
                    warn(
                        "Changing the mapping will create a copy of this object. "
                        "Consider setting it upon object creation to re-use existing cache."
                    )
                for alias in reself.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs["mapping"] = self_copy.mapping
                if impacts_caching:
                    cond_kwargs["use_caching"] = False
                return self_copy
        return reself

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `GenericAccessor.stats`.

        Merges the defaults from `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.generic`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        generic_stats_cfg = settings["generic"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), generic_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            count=dict(
                title="Count",
                calc_func="count",
                inv_check_has_mapping=True,
                tags=["generic", "describe"],
            ),
            mean=dict(
                title="Mean",
                calc_func="mean",
                inv_check_has_mapping=True,
                tags=["generic", "describe"],
            ),
            std=dict(
                title="Std",
                calc_func="std",
                inv_check_has_mapping=True,
                tags=["generic", "describe"],
            ),
            min=dict(
                title="Min",
                calc_func="min",
                inv_check_has_mapping=True,
                tags=["generic", "describe"],
            ),
            median=dict(
                title="Median",
                calc_func="median",
                inv_check_has_mapping=True,
                tags=["generic", "describe"],
            ),
            max=dict(
                title="Max",
                calc_func="max",
                inv_check_has_mapping=True,
                tags=["generic", "describe"],
            ),
            idx_min=dict(
                title="Min Index",
                calc_func="idxmin",
                agg_func=None,
                inv_check_has_mapping=True,
                tags=["generic", "index"],
            ),
            idx_max=dict(
                title="Max Index",
                calc_func="idxmax",
                agg_func=None,
                inv_check_has_mapping=True,
                tags=["generic", "index"],
            ),
            value_counts=dict(
                title="Value Counts",
                calc_func=lambda value_counts: reshaping.to_dict(
                    value_counts, orient="index_series"
                ),
                resolve_value_counts=True,
                check_has_mapping=True,
                tags=["generic", "value_counts"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        trace_names: tp.TraceNames = None,
        x_labels: tp.Optional[tp.Labels] = None,
        return_fig: bool = True,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a scatter plot using `vectorbtpro.generic.plotting.Scatter` and
        return the resulting figure or trace updater.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            trace_names (TraceNames): Names for traces corresponding to data columns.
            x_labels (Optional[Labels]): Labels for the x-axis.
            return_fig (bool): If True, return the figure; otherwise, return the trace updater.
            **kwargs: Keyword arguments for `vectorbtpro.generic.plotting.Scatter`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Plot figure or trace updater.

        Examples:
            ```pycon
            >>> df.vbt.plot().show()
            ```

            ![](/assets/images/api/df_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.generic.plotting import Scatter

        if column is not None:
            _self = self.select_col(column=column)
        else:
            _self = self
        if x_labels is None:
            x_labels = _self.wrapper.index
        if trace_names is None:
            if _self.is_frame() or (_self.is_series() and _self.wrapper.name is not None):
                trace_names = _self.wrapper.columns
        scatter = Scatter(
            data=_self.to_2d_array(), trace_names=trace_names, x_labels=x_labels, **kwargs
        )
        if return_fig:
            return scatter.fig
        return scatter

    def lineplot(
        self, column: tp.Optional[tp.Column] = None, **kwargs
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot a line chart using `GenericAccessor.plot` in 'lines' mode.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `GenericAccessor.plot`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater instance.

        Examples:
            ```pycon
            >>> df.vbt.lineplot().show()
            ```

            ![](/assets/images/api/df_lineplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_lineplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        return self.plot(
            column=column, **merge_dicts(dict(trace_kwargs=dict(mode="lines")), kwargs)
        )

    def scatterplot(
        self, column: tp.Optional[tp.Column] = None, **kwargs
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot a scatter chart using `GenericAccessor.plot` in 'markers' mode.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            **kwargs: Keyword arguments for `GenericAccessor.plot`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or trace updater instance.

        Examples:
            ```pycon
            >>> df.vbt.scatterplot().show()
            ```

            ![](/assets/images/api/df_scatterplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_scatterplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        return self.plot(
            column=column, **merge_dicts(dict(trace_kwargs=dict(mode="markers")), kwargs)
        )

    def barplot(
        self,
        column: tp.Optional[tp.Column] = None,
        trace_names: tp.TraceNames = None,
        x_labels: tp.Optional[tp.Labels] = None,
        return_fig: bool = True,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a bar chart using `vectorbtpro.generic.plotting.Bar` and return the figure or chart object.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            trace_names (TraceNames): Names for traces corresponding to data columns.
            x_labels (Optional[Labels]): Labels for the x-axis.
            return_fig (bool): If True, return the figure; otherwise, return the trace updater.
            **kwargs: Keyword arguments for `vectorbtpro.generic.plotting.Bar`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or bar chart object based on `return_fig`.

        Examples:
            ```pycon
            >>> df.vbt.barplot().show()
            ```

            ![](/assets/images/api/df_barplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_barplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.generic.plotting import Bar

        if column is not None:
            _self = self.select_col(column=column)
        else:
            _self = self
        if x_labels is None:
            x_labels = _self.wrapper.index
        if trace_names is None:
            if _self.is_frame() or (_self.is_series() and _self.wrapper.name is not None):
                trace_names = _self.wrapper.columns
        bar = Bar(data=_self.to_2d_array(), trace_names=trace_names, x_labels=x_labels, **kwargs)
        if return_fig:
            return bar.fig
        return bar

    def histplot(
        self,
        column: tp.Optional[tp.Column] = None,
        by_level: tp.Optional[tp.Level] = None,
        trace_names: tp.TraceNames = None,
        group_by: tp.GroupByLike = None,
        return_fig: bool = True,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a histogram chart using `vectorbtpro.generic.plotting.Histogram` and
        return the figure or chart object.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            by_level (Optional[Level]): Level at which to unstack the data, if applicable.
            trace_names (TraceNames): Names for traces corresponding to data columns.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            return_fig (bool): If True, return the figure; otherwise, return the trace updater.
            **kwargs: Keyword arguments for `vectorbtpro.generic.plotting.Histogram`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or histogram chart object based on `return_fig`.

        Examples:
            ```pycon
            >>> df.vbt.histplot().show()
            ```

            ![](/assets/images/api/df_histplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_histplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.generic.plotting import Histogram

        if by_level is not None:
            return self.obj.unstack(by_level).vbt.histplot(
                column=column,
                trace_names=trace_names,
                group_by=group_by,
                return_fig=return_fig,
                **kwargs,
            )

        if column is not None:
            _self = self.select_col(column=column)
        else:
            _self = self

        if _self.wrapper.grouper.is_grouped(group_by=group_by):
            return _self.flatten_grouped(group_by=group_by).vbt.histplot(
                trace_names=trace_names, **kwargs
            )

        if trace_names is None:
            if _self.is_frame() or (_self.is_series() and _self.wrapper.name is not None):
                trace_names = _self.wrapper.columns
        hist = Histogram(data=_self.to_2d_array(), trace_names=trace_names, **kwargs)
        if return_fig:
            return hist.fig
        return hist

    def boxplot(
        self,
        column: tp.Optional[tp.Column] = None,
        by_level: tp.Optional[tp.Level] = None,
        trace_names: tp.TraceNames = None,
        group_by: tp.GroupByLike = None,
        return_fig: bool = True,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a box plot using `vectorbtpro.generic.plotting.Box` and
        return the figure or chart object.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            by_level (Optional[Level]): Level at which to unstack the data, if applicable.
            trace_names (TraceNames): Names for traces corresponding to data columns.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            return_fig (bool): If True, return the figure; otherwise, return the trace updater.
            **kwargs: Keyword arguments for `vectorbtpro.generic.plotting.Box`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or box plot object based on `return_fig`.

        Examples:
            ```pycon
            >>> df.vbt.boxplot().show()
            ```

            ![](/assets/images/api/df_boxplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_boxplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.generic.plotting import Box

        if by_level is not None:
            return self.obj.unstack(by_level).vbt.boxplot(
                column=column,
                trace_names=trace_names,
                group_by=group_by,
                return_fig=return_fig,
                **kwargs,
            )

        if column is not None:
            _self = self.select_col(column=column)
        else:
            _self = self

        if _self.wrapper.grouper.is_grouped(group_by=group_by):
            return _self.flatten_grouped(group_by=group_by).vbt.boxplot(
                trace_names=trace_names, **kwargs
            )

        if trace_names is None:
            if _self.is_frame() or (_self.is_series() and _self.wrapper.name is not None):
                trace_names = _self.wrapper.columns
        box = Box(data=_self.to_2d_array(), trace_names=trace_names, **kwargs)
        if return_fig:
            return box.fig
        return box

    def plot_against(
        self,
        other: tp.ArrayLike,
        column: tp.Optional[tp.Column] = None,
        trace_kwargs: tp.KwargsLike = None,
        other_trace_kwargs: tp.Union[str, tp.KwargsLike] = None,
        pos_trace_kwargs: tp.KwargsLike = None,
        neg_trace_kwargs: tp.KwargsLike = None,
        hidden_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a Series as a line against another line.

        Args:
            other (ArrayLike): Second array to compare, which will be broadcast to match the primary data.
            column (Optional[Column]): Identifier of the column to plot.
            trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the main series.
            other_trace_kwargs (Union[str, KwargsLike]): Keyword arguments for `plotly.graph_objects.Scatter` for the second series.

                Use 'hidden' to omit the secondary line.
            pos_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the positive area between the series.
            neg_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the negative area between the series.
            hidden_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for hidden lines.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Plotly figure with the main series, comparative lines, and filled areas.

        Examples:
            ```pycon
            >>> df['a'].vbt.plot_against(df['b']).show()
            ```

            ![](/assets/images/api/sr_plot_against.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/sr_plot_against.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure

        if trace_kwargs is None:
            trace_kwargs = {}
        if other_trace_kwargs is None:
            other_trace_kwargs = {}
        if pos_trace_kwargs is None:
            pos_trace_kwargs = {}
        if neg_trace_kwargs is None:
            neg_trace_kwargs = {}
        if hidden_trace_kwargs is None:
            hidden_trace_kwargs = {}

        obj = self.obj
        if isinstance(obj, pd.DataFrame):
            obj = self.select_col_from_obj(obj, column=column, group_by=False)
        if other is None:
            other = pd.Series.vbt.empty_like(obj, 1)
        else:
            other = reshaping.to_pd_array(other)
            if isinstance(other, pd.DataFrame):
                other = self.select_col_from_obj(other, column=column, group_by=False)
            obj, other = reshaping.broadcast(obj, other, columns_from="keep")
            if other.name is None:
                other = other.rename("Other")

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # TODO: Using masks feels hacky
        pos_mask = obj > other
        if pos_mask.any():
            # Fill positive area
            pos_obj = obj.copy()
            pos_obj[~pos_mask] = other[~pos_mask]
            other.vbt.lineplot(
                trace_kwargs=merge_dicts(
                    dict(
                        line=dict(color="rgba(0, 0, 0, 0)", width=0),
                        opacity=0,
                        hoverinfo="skip",
                        showlegend=False,
                        name=None,
                    ),
                    hidden_trace_kwargs,
                ),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            pos_obj.vbt.lineplot(
                trace_kwargs=merge_dicts(
                    dict(
                        fillcolor="rgba(0, 128, 0, 0.25)",
                        line=dict(color="rgba(0, 0, 0, 0)", width=0),
                        opacity=0,
                        fill="tonexty",
                        connectgaps=False,
                        hoverinfo="skip",
                        showlegend=False,
                        name=None,
                    ),
                    pos_trace_kwargs,
                ),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        neg_mask = obj < other
        if neg_mask.any():
            # Fill negative area
            neg_obj = obj.copy()
            neg_obj[~neg_mask] = other[~neg_mask]
            other.vbt.lineplot(
                trace_kwargs=merge_dicts(
                    dict(
                        line=dict(color="rgba(0, 0, 0, 0)", width=0),
                        opacity=0,
                        hoverinfo="skip",
                        showlegend=False,
                        name=None,
                    ),
                    hidden_trace_kwargs,
                ),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            neg_obj.vbt.lineplot(
                trace_kwargs=merge_dicts(
                    dict(
                        line=dict(color="rgba(0, 0, 0, 0)", width=0),
                        fillcolor="rgba(255, 0, 0, 0.25)",
                        opacity=0,
                        fill="tonexty",
                        connectgaps=False,
                        hoverinfo="skip",
                        showlegend=False,
                        name=None,
                    ),
                    neg_trace_kwargs,
                ),
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        # Plot main traces
        obj.vbt.lineplot(trace_kwargs=trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        if other_trace_kwargs == "hidden":
            other_trace_kwargs = dict(
                line=dict(color="rgba(0, 0, 0, 0)", width=0),
                opacity=0.0,
                hoverinfo="skip",
                showlegend=False,
                name=None,
            )
        other.vbt.lineplot(
            trace_kwargs=other_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig
        )
        return fig

    def overlay_with_heatmap(
        self,
        other: tp.ArrayLike,
        column: tp.Optional[tp.Column] = None,
        trace_kwargs: tp.KwargsLike = None,
        heatmap_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a Series as a line and overlay it with a heatmap.

        Args:
            other (ArrayLike): Second array to compare, which will be broadcast
                to match the primary data.
            column (Optional[Column]): Identifier of the column to plot.
            trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the line.
            heatmap_kwargs (KwargsLike): Keyword arguments for `GenericDFAccessor.heatmap` used to generate the heatmap.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Plotly figure with the line plot and heatmap overlay.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> df['a'].vbt.overlay_with_heatmap(df['b']).show()
            ```

            ![](/assets/images/api/sr_overlay_with_heatmap.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/sr_overlay_with_heatmap.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_subplots

        plotting_cfg = settings["plotting"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if heatmap_kwargs is None:
            heatmap_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        obj = self.obj
        if isinstance(obj, pd.DataFrame):
            obj = self.select_col_from_obj(obj, column=column, group_by=False)
        if other is None:
            other = pd.Series.vbt.empty_like(obj, 1)
        else:
            other = reshaping.to_pd_array(other)
            if isinstance(other, pd.DataFrame):
                other = self.select_col_from_obj(other, column=column, group_by=False)
            obj, other = reshaping.broadcast(obj, other, columns_from="keep")
            if other.name is None:
                other = other.rename("Other")

        if fig is None:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if "width" in plotting_cfg["layout"]:
                fig.update_layout(width=plotting_cfg["layout"]["width"] + 100)
        fig.update_layout(**layout_kwargs)

        other.vbt.ts_heatmap(**heatmap_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        obj.vbt.lineplot(
            trace_kwargs=merge_dicts(
                dict(line=dict(color=plotting_cfg["color_schema"]["blue"])), trace_kwargs
            ),
            add_trace_kwargs=merge_dicts(dict(secondary_y=True), add_trace_kwargs),
            fig=fig,
        )
        return fig

    def heatmap(
        self,
        column: tp.Optional[tp.Column] = None,
        x_level: tp.Optional[tp.Level] = None,
        y_level: tp.Optional[tp.Level] = None,
        symmetric: bool = False,
        sort: bool = True,
        x_labels: tp.Optional[tp.Labels] = None,
        y_labels: tp.Optional[tp.Labels] = None,
        slider_level: tp.Optional[tp.Level] = None,
        active: int = 0,
        slider_labels: tp.Optional[tp.Labels] = None,
        return_fig: bool = True,
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a heatmap figure using the object's data.

        For 2D data or non-multi-index objects, a regular heatmap is produced.
        For multi-index data with more than two levels, specify `x_level` and
        `y_level` to define the axes and optionally use a slider by setting `slider_level`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            x_level (Optional[Level]): Level for the x-axis of the heatmap.

                Accepts an integer index or string name.
            y_level (Optional[Level]): Level for the y-axis of the heatmap.

                Accepts an integer index or string name.
            symmetric (bool): Whether to apply symmetric unstacking to the data.
            sort (bool): Whether to sort the data when unstacking.
            x_labels (Optional[Labels]): Labels for the x-axis corresponding to DataFrame columns.
            y_labels (Optional[Labels]): Labels for the y-axis corresponding to DataFrame index.
            slider_level (Optional[Level]): Level to use for creating a slider in multi-index plots.
            active (int): Index of the trace to display initially in slider-based plots.
            slider_labels (Optional[Labels]): Labels for the slider steps.
            return_fig (bool): If True, return the figure; otherwise, return the trace updater.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **kwargs: Keyword arguments for `vectorbtpro.generic.plotting.Heatmap`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or heatmap object based on `return_fig`.

        Examples:
            Plotting a figure based on a regular index:

            ```pycon
            >>> df = pd.DataFrame([
            ...     [0, np.nan, np.nan],
            ...     [np.nan, 1, np.nan],
            ...     [np.nan, np.nan, 2]
            ... ])
            >>> df.vbt.heatmap().show()
            ```

            ![](/assets/images/api/df_heatmap.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_heatmap.dark.svg#only-dark){: .iimg loading=lazy }

            Plotting a figure based on a multi-index:

            ```pycon
            >>> multi_index = pd.MultiIndex.from_tuples([
            ...     (1, 1),
            ...     (2, 2),
            ...     (3, 3)
            ... ])
            >>> sr = pd.Series(np.arange(len(multi_index)), index=multi_index)
            >>> sr
            1  1    0
            2  2    1
            3  3    2
            dtype: int64

            >>> sr.vbt.heatmap().show()
            ```

            ![](/assets/images/api/sr_heatmap.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/sr_heatmap.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.generic.plotting import Heatmap

        if column is not None:
            _self = self.select_col(column=column)
        else:
            _self = self
        if _self.ndim == 2 or not isinstance(self.wrapper.index, pd.MultiIndex):
            if x_labels is None:
                x_labels = _self.wrapper.columns
            if y_labels is None:
                y_labels = _self.wrapper.index
            heatmap = Heatmap(
                data=_self.to_2d_array(),
                x_labels=x_labels,
                y_labels=y_labels,
                fig=fig,
                **kwargs,
            )
            if return_fig:
                return heatmap.fig
            return heatmap

        (x_level, y_level), (slider_level,) = indexes.pick_levels(
            _self.wrapper.index,
            required_levels=(x_level, y_level),
            optional_levels=(slider_level,),
        )

        x_level_vals = _self.wrapper.index.get_level_values(x_level)
        y_level_vals = _self.wrapper.index.get_level_values(y_level)
        x_name = x_level_vals.name if x_level_vals.name is not None else "x"
        y_name = y_level_vals.name if y_level_vals.name is not None else "y"
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    hovertemplate=f"{x_name}: %{{x}}<br>"
                    + f"{y_name}: %{{y}}<br>"
                    + "value: %{z}<extra></extra>"
                ),
                xaxis_title=x_level_vals.name,
                yaxis_title=y_level_vals.name,
            ),
            kwargs,
        )

        if slider_level is None:
            # No grouping
            df = _self.unstack_to_df(
                index_levels=y_level,
                column_levels=x_level,
                symmetric=symmetric,
                sort=sort,
            )
            return df.vbt.heatmap(
                x_labels=x_labels,
                y_labels=y_labels,
                fig=fig,
                return_fig=return_fig,
                **kwargs,
            )

        # Requires grouping
        # See https://plotly.com/python/sliders/
        if not return_fig:
            raise ValueError("Cannot use return_fig=False and slider_level simultaneously")
        _slider_labels = []
        for i, (name, group) in enumerate(_self.obj.groupby(level=slider_level)):
            if slider_labels is not None:
                name = slider_labels[i]
            _slider_labels.append(name)
            df = group.vbt.unstack_to_df(
                index_levels=y_level,
                column_levels=x_level,
                symmetric=symmetric,
                sort=sort,
            )
            if x_labels is None:
                x_labels = df.columns
            if y_labels is None:
                y_labels = df.index
            _kwargs = merge_dicts(
                dict(
                    trace_kwargs=dict(name=str(name) if name is not None else None, visible=False),
                ),
                kwargs,
            )
            default_size = fig is None and "height" not in _kwargs
            fig = Heatmap(
                data=reshaping.to_2d_array(df),
                x_labels=x_labels,
                y_labels=y_labels,
                fig=fig,
                **_kwargs,
            ).fig
            if default_size:
                fig.layout["height"] += 100  # slider takes up space
        fig.data[active].visible = True
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}, {}],
                label=str(_slider_labels[i]) if _slider_labels[i] is not None else None,
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        prefix = (
            f"{_self.wrapper.index.names[slider_level]}: "
            if _self.wrapper.index.names[slider_level] is not None
            else None
        )
        sliders = [
            dict(
                active=active,
                currentvalue={"prefix": prefix},
                pad={"t": 50},
                steps=steps,
            )
        ]
        fig.update_layout(sliders=sliders)
        return fig

    def ts_heatmap(
        self,
        column: tp.Optional[tp.Column] = None,
        is_y_category: bool = True,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a heatmap for time-series data.

        The function selects a column from the object's data if specified,
        converts a Series to a DataFrame if necessary, and transposes the data
        to produce a heatmap with reversed vertical order.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            is_y_category (bool): Flag indicating whether to treat the y-axis as categorical.
            **kwargs: Keyword arguments for `GenericAccessor.heatmap`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Figure or heatmap object based on `return_fig`.
        """
        if column is not None:
            obj = self.select_col_from_obj(self.obj, column=column)
        else:
            obj = self.obj
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        return obj.transpose().iloc[::-1].vbt.heatmap(is_y_category=is_y_category, **kwargs)

    def volume(
        self,
        column: tp.Optional[tp.Column] = None,
        x_level: tp.Optional[tp.Level] = None,
        y_level: tp.Optional[tp.Level] = None,
        z_level: tp.Optional[tp.Level] = None,
        x_labels: tp.Optional[tp.Labels] = None,
        y_labels: tp.Optional[tp.Labels] = None,
        z_labels: tp.Optional[tp.Labels] = None,
        slider_level: tp.Optional[tp.Level] = None,
        slider_labels: tp.Optional[tp.Labels] = None,
        active: int = 0,
        scene_name: str = "scene",
        fillna: tp.Optional[tp.Number] = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        return_fig: bool = True,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Create a 3D volume figure using the object's multi-index and values.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            x_level (Optional[Level]): Level to use for the x-axis.

                Provide an integer index or a level name.
            y_level (Optional[Level]): Level to use for the y-axis.

                Provide an integer index or a level name.
            z_level (Optional[Level]): Level to use for the z-axis.

                Provide an integer index or a level name.
            x_labels (Optional[Labels]): Labels for the x-axis.

                If not provided, unique x-level values are used.
            y_labels (Optional[Labels]): Labels for the y-axis.

                If not provided, unique y-level values are used.
            z_labels (Optional[Labels]): Labels for the z-axis.

                If not provided, unique z-level values are used.
            slider_level (Optional[Level]): Level to use for grouping data with a slider.
            slider_labels (Optional[Labels]): Labels to override default slider group names.
            active (int): Index of the active slider step.
            scene_name (str): Name of the 3D scene.
            fillna (Optional[Number]): Value to replace NaNs in the data.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            return_fig (bool): If True, return the figure; otherwise, return the trace updater.
            **kwargs: Keyword arguments for `vectorbtpro.generic.plotting.Volume`.

        Returns:
            Union[BaseFigure, TraceUpdater]: Created 3D volume figure or trace updater object.

        Examples:
            ```pycon
            >>> multi_index = pd.MultiIndex.from_tuples([
            ...     (1, 1, 1),
            ...     (2, 2, 2),
            ...     (3, 3, 3)
            ... ])
            >>> sr = pd.Series(np.arange(len(multi_index)), index=multi_index)
            >>> sr
            1  1  1    0
            2  2  2    1
            3  3  3    2
            dtype: int64

            >>> sr.vbt.volume().show()
            ```

            ![](/assets/images/api/sr_volume.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/sr_volume.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.generic.plotting import Volume

        self_col = self.select_col(column=column, group_by=False)

        (x_level, y_level, z_level), (slider_level,) = indexes.pick_levels(
            self_col.wrapper.index,
            required_levels=(x_level, y_level, z_level),
            optional_levels=(slider_level,),
        )

        x_level_vals = self_col.wrapper.index.get_level_values(x_level)
        y_level_vals = self_col.wrapper.index.get_level_values(y_level)
        z_level_vals = self_col.wrapper.index.get_level_values(z_level)
        # Labels are just unique level values
        if x_labels is None:
            x_labels = np.unique(x_level_vals)
        if y_labels is None:
            y_labels = np.unique(y_level_vals)
        if z_labels is None:
            z_labels = np.unique(z_level_vals)

        x_name = x_level_vals.name if x_level_vals.name is not None else "x"
        y_name = y_level_vals.name if y_level_vals.name is not None else "y"
        z_name = z_level_vals.name if z_level_vals.name is not None else "z"
        def_kwargs = dict()
        def_kwargs["trace_kwargs"] = dict(
            hovertemplate=f"{x_name}: %{{x}}<br>"
            + f"{y_name}: %{{y}}<br>"
            + f"{z_name}: %{{z}}<br>"
            + "value: %{value}<extra></extra>"
        )
        def_kwargs[scene_name] = dict(
            xaxis_title=x_level_vals.name,
            yaxis_title=y_level_vals.name,
            zaxis_title=z_level_vals.name,
        )
        def_kwargs["scene_name"] = scene_name
        kwargs = merge_dicts(def_kwargs, kwargs)

        contains_nan = False
        if slider_level is None:
            # No grouping
            v = self_col.unstack_to_array(levels=(x_level, y_level, z_level))
            if fillna is not None:
                v = np.nan_to_num(v, nan=fillna)
            if np.isnan(v).any():
                contains_nan = True
            volume = Volume(
                data=v, x_labels=x_labels, y_labels=y_labels, z_labels=z_labels, fig=fig, **kwargs
            )
            if return_fig:
                fig = volume.fig
            else:
                fig = volume
        else:
            # Requires grouping
            # See https://plotly.com/python/sliders/
            if not return_fig:
                raise ValueError("Cannot use return_fig=False and slider_level simultaneously")
            _slider_labels = []
            for i, (name, group) in enumerate(self_col.obj.groupby(level=slider_level)):
                if slider_labels is not None:
                    name = slider_labels[i]
                _slider_labels.append(name)
                v = group.vbt.unstack_to_array(levels=(x_level, y_level, z_level))
                if fillna is not None:
                    v = np.nan_to_num(v, nan=fillna)
                if np.isnan(v).any():
                    contains_nan = True
                _kwargs = merge_dicts(
                    dict(
                        trace_kwargs=dict(
                            name=str(name) if name is not None else None, visible=False
                        )
                    ),
                    kwargs,
                )
                default_size = fig is None and "height" not in _kwargs
                fig = Volume(
                    data=v,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    z_labels=z_labels,
                    fig=fig,
                    **_kwargs,
                ).fig
                if default_size:
                    fig.layout["height"] += 100  # slider takes up space
            fig.data[active].visible = True
            steps = []
            for i in range(len(fig.data)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)}, {}],
                    label=str(_slider_labels[i]) if _slider_labels[i] is not None else None,
                )
                step["args"][0]["visible"][i] = True
                steps.append(step)
            prefix = (
                f"{self_col.wrapper.index.names[slider_level]}: "
                if self_col.wrapper.index.names[slider_level] is not None
                else None
            )
            sliders = [
                dict(active=active, currentvalue={"prefix": prefix}, pad={"t": 50}, steps=steps)
            ]
            fig.update_layout(sliders=sliders)

        if contains_nan:
            warn(
                "Data contains NaNs. Use `fillna` argument or `show` method in case of visualization issues."
            )
        return fig

    def qqplot(
        self,
        column: tp.Optional[tp.Column] = None,
        sparams: tp.Union[tp.Iterable, tuple, None] = (),
        dist: str = "norm",
        plot_line: bool = True,
        line_shape_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot a probability plot using `scipy.stats.probplot` and display the results as a scatter plot.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            sparams (Union[Iterable, tuple, None]): Parameters to pass to `scipy.stats.probplot`.
            dist (str): Distribution name for generating the probability plot.
            plot_line (bool): Whether to add a fitted reference line based on the probability plot.
            line_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for the reference line.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **kwargs: Keyword arguments for `GenericAccessor.scatterplot`.

        Returns:
            BaseFigure: Generated probability plot figure with an optional reference line.

        Examples:
            ```pycon
            >>> pd.Series(np.random.standard_normal(100)).vbt.qqplot().show()
            ```

            ![](/assets/images/api/sr_qqplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/sr_qqplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        import scipy.stats as st

        obj = self.select_col_from_obj(self.obj, column=column, group_by=False)
        qq = st.probplot(obj, sparams=sparams, dist=dist)
        fig = pd.Series(qq[0][1], index=qq[0][0]).vbt.scatterplot(fig=fig, **kwargs)

        if plot_line:
            if line_shape_kwargs is None:
                line_shape_kwargs = {}
            x = np.array([qq[0][0][0], qq[0][0][-1]])
            y = qq[1][1] + qq[1][0] * x
            fig.add_shape(
                **merge_dicts(
                    dict(
                        type="line",
                        xref=xref,
                        yref=yref,
                        x0=x[0],
                        y0=y[0],
                        x1=x[1],
                        y1=y[1],
                        line=dict(color="red"),
                    ),
                    line_shape_kwargs,
                )
            )

        return fig

    def areaplot(
        self,
        line_shape: str = "spline",
        line_visible: bool = False,
        colorway: tp.Union[None, str, tp.Sequence[str]] = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot a stacked area chart.

        Args:
            line_shape (str): Shape of the plot line (e.g. "spline").
            line_visible (bool): Determines whether the line outlining the area is displayed.
            colorway (Union[None, str, Sequence[str]]): Name of a built-in qualitative
                color palette or a sequence of colors.
            trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the area.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Generated stacked area chart.

        Examples:
            ```pycon
            >>> df.vbt.areaplot().show()
            ```

            ![](/assets/images/api/df_areaplot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_areaplot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.express as px

        from vectorbtpro.utils.figure import make_figure

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if colorway is None:
            if fig.layout.colorway is not None:
                colorway = fig.layout.colorway
            else:
                colorway = fig.layout.template.layout.colorway
            if len(self.wrapper.columns) > len(colorway):
                colorway = px.colors.qualitative.Alphabet
        elif isinstance(colorway, str):
            colorway = getattr(px.colors.qualitative, colorway)

        pos_mask = self.obj.values > 0
        pos_mask_any = pos_mask.any()
        neg_mask = self.obj.values < 0
        neg_mask_any = neg_mask.any()
        pos_showlegend = False
        neg_showlegend = False
        if pos_mask_any:
            pos_showlegend = True
        elif neg_mask_any:
            neg_showlegend = True
        line_width = None if line_visible else 0
        line_opacity = 0.3 if line_visible else 0.8
        if pos_mask_any:
            pos_df = self.obj.copy()
            pos_df[neg_mask] = 0.0
            fig = pos_df.vbt.lineplot(
                trace_kwargs=[
                    merge_dicts(
                        dict(
                            legendgroup="area_" + str(c),
                            stackgroup="one",
                            line=dict(
                                width=line_width,
                                color=colorway[c % len(colorway)],
                                shape=line_shape,
                            ),
                            fillcolor=adjust_opacity(colorway[c % len(colorway)], line_opacity),
                            showlegend=pos_showlegend,
                        ),
                        resolve_dict(trace_kwargs, i=c),
                    )
                    for c in range(len(self.wrapper.columns))
                ],
                add_trace_kwargs=add_trace_kwargs,
                use_gl=False,
                fig=fig,
                **layout_kwargs,
            )
        if neg_mask_any:
            neg_df = self.obj.copy()
            neg_df[pos_mask] = 0.0
            fig = neg_df.vbt.lineplot(
                trace_kwargs=[
                    merge_dicts(
                        dict(
                            legendgroup="area_" + str(c),
                            stackgroup="two",
                            line=dict(
                                width=line_width,
                                color=colorway[c % len(colorway)],
                                shape=line_shape,
                            ),
                            fillcolor=adjust_opacity(colorway[c % len(colorway)], line_opacity),
                            showlegend=neg_showlegend,
                        ),
                        resolve_dict(trace_kwargs, i=c),
                    )
                    for c in range(len(self.wrapper.columns))
                ],
                add_trace_kwargs=add_trace_kwargs,
                use_gl=False,
                fig=fig,
                **layout_kwargs,
            )
        return fig

    def plot_pattern(
        self,
        pattern: tp.ArrayLike,
        interp_mode: tp.Union[int, str] = "mixed",
        rescale_mode: tp.Union[int, str] = "minmax",
        vmin: float = np.nan,
        vmax: float = np.nan,
        pmin: float = np.nan,
        pmax: float = np.nan,
        invert: bool = False,
        error_type: tp.Union[int, str] = "absolute",
        max_error: tp.ArrayLike = np.nan,
        max_error_interp_mode: tp.Union[None, int, str] = None,
        column: tp.Optional[tp.Column] = None,
        plot_obj: bool = True,
        fill_distance: bool = False,
        obj_trace_kwargs: tp.KwargsLike = None,
        pattern_trace_kwargs: tp.KwargsLike = None,
        lower_max_error_trace_kwargs: tp.KwargsLike = None,
        upper_max_error_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the pattern with computed error bounds.

        Perform a pattern similarity calculation analogous to
        `vectorbtpro.generic.nb.patterns.pattern_similarity_nb` and plot the original data,
        fitted pattern, and error bands.

        Args:
            pattern (ArrayLike): Reference pattern array for similarity comparison.
            interp_mode (Union[int, str]): Interpolation mode.

                Mapped using `vectorbtpro.generic.enums.InterpMode` if provided as a string.
            rescale_mode (Union[int, str]): Rescaling mode.

                Mapped using `vectorbtpro.generic.enums.RescaleMode` if provided as a string.
            vmin (float): Minimum value used for rescaling `arr`.
            vmax (float): Maximum value used for rescaling `arr`.
            pmin (float): Minimum value used for rescaling `pattern`.
            pmax (float): Maximum value used for rescaling `pattern`.
            invert (bool): Invert the pattern by reflecting its values.
            error_type (Union[int, str]): Error calculation type.

                Mapped using `vectorbtpro.generic.enums.ErrorType` if provided as a string.
            max_error (ArrayLike): Maximum allowed error.
            max_error_interp_mode (Union[None, int, str]): Interpolation mode for `max_error`.

                Mapped using `vectorbtpro.generic.enums.InterpMode` if provided as a string.
            column (Optional[Column]): Identifier of the column to plot.
            plot_obj (bool): If True, includes the original object data in the plot.
            fill_distance (bool): Controls whether to fill the space between close and pattern.

                Visible for every interpolation mode except discrete.
            obj_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the original data.
            pattern_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the pattern.
            lower_max_error_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the lower error bound.
            upper_max_error_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the upper error bound.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object with the plotted pattern and error bands.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> sr = pd.Series([10, 11, 12, 13, 12, 13, 14, 15, 13, 14, 11])
            >>> sr.vbt.plot_pattern([1, 2, 3, 2, 1]).show()
            ```

            ![](/assets/images/api/sr_plot_pattern.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/sr_plot_pattern.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if isinstance(interp_mode, str):
            interp_mode = map_enum_fields(interp_mode, InterpMode)
        if isinstance(rescale_mode, str):
            rescale_mode = map_enum_fields(rescale_mode, RescaleMode)
        if isinstance(error_type, str):
            error_type = map_enum_fields(error_type, ErrorType)
        if max_error_interp_mode is not None and isinstance(max_error_interp_mode, str):
            max_error_interp_mode = map_enum_fields(max_error_interp_mode, InterpMode)
        if max_error_interp_mode is None:
            max_error_interp_mode = interp_mode

        obj_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"])),
            obj_trace_kwargs,
        )
        if pattern_trace_kwargs is None:
            pattern_trace_kwargs = {}
        if lower_max_error_trace_kwargs is None:
            lower_max_error_trace_kwargs = {}
        if upper_max_error_trace_kwargs is None:
            upper_max_error_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        self_col = self.select_col(column=column, group_by=False)
        if plot_obj:
            # Plot object
            fig = self_col.lineplot(
                trace_kwargs=obj_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        # Reconstruct pattern and max error bands
        pattern_sr, max_error_sr = self_col.fit_pattern(
            pattern,
            interp_mode=interp_mode,
            rescale_mode=rescale_mode,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            invert=invert,
            error_type=error_type,
            max_error=max_error,
            max_error_interp_mode=max_error_interp_mode,
        )

        # Plot pattern and max error bands
        def_pattern_trace_kwargs = dict(
            name="Pattern",
            connectgaps=True,
        )
        if interp_mode == InterpMode.Discrete:
            _pattern_trace_kwargs = merge_dicts(
                def_pattern_trace_kwargs,
                dict(
                    mode="lines+markers",
                    marker=dict(color=adjust_opacity(plotting_cfg["color_schema"]["cyan"], 0.75)),
                    line=dict(
                        color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.75), dash="dot"
                    ),
                ),
                pattern_trace_kwargs,
            )
        else:
            if fill_distance:
                _pattern_trace_kwargs = merge_dicts(
                    def_pattern_trace_kwargs,
                    dict(
                        mode="lines",
                        line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["cyan"], 0.75)),
                        fill="tonexty",
                        fillcolor=adjust_opacity(plotting_cfg["color_schema"]["cyan"], 0.25),
                    ),
                    pattern_trace_kwargs,
                )
            else:
                _pattern_trace_kwargs = merge_dicts(
                    def_pattern_trace_kwargs,
                    dict(
                        mode="lines",
                        line=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["cyan"], 0.75),
                            dash="dot",
                        ),
                    ),
                    pattern_trace_kwargs,
                )
        fig = pattern_sr.rename(None).vbt.plot(
            trace_kwargs=_pattern_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        # Plot max error bounds
        if not np.isnan(max_error).all():
            def_max_error_trace_kwargs = dict(
                name="Max error",
                connectgaps=True,
            )
            if max_error_interp_mode == InterpMode.Discrete:
                _lower_max_error_trace_kwargs = merge_dicts(
                    def_max_error_trace_kwargs,
                    dict(
                        mode="markers+lines",
                        marker=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.5)
                        ),
                        line=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5),
                            dash="dot",
                        ),
                    ),
                    lower_max_error_trace_kwargs,
                )
                _upper_max_error_trace_kwargs = merge_dicts(
                    def_max_error_trace_kwargs,
                    dict(
                        mode="markers+lines",
                        marker=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.5)
                        ),
                        line=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5),
                            dash="dot",
                        ),
                        showlegend=False,
                    ),
                    upper_max_error_trace_kwargs,
                )
            else:
                _lower_max_error_trace_kwargs = merge_dicts(
                    def_max_error_trace_kwargs,
                    dict(
                        mode="lines",
                        line=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.5),
                            dash="dot",
                        ),
                    ),
                    lower_max_error_trace_kwargs,
                )
                _upper_max_error_trace_kwargs = merge_dicts(
                    def_max_error_trace_kwargs,
                    dict(
                        mode="lines",
                        line=dict(
                            color=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.5),
                            dash="dot",
                        ),
                        fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.1),
                        fill="tonexty",
                        showlegend=False,
                    ),
                    upper_max_error_trace_kwargs,
                )
            fig = (
                (pattern_sr - max_error_sr)
                .rename(None)
                .vbt.plot(
                    trace_kwargs=_lower_max_error_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )
            )
            fig = (
                (pattern_sr + max_error_sr)
                .rename(None)
                .vbt.plot(
                    trace_kwargs=_upper_max_error_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )
            )

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `GenericAccessor.plots`.

        Merges the defaults from `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.generic`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        generic_plots_cfg = settings["generic"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), generic_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                check_is_not_grouped=True,
                plot_func="plot",
                pass_trace_names=False,
                tags="generic",
            )
        ),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


if settings["importing"]["sklearn"]:
    from sklearn.exceptions import NotFittedError
    from sklearn.preprocessing import (
        Binarizer,
        MaxAbsScaler,
        MinMaxScaler,
        Normalizer,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )
    from sklearn.utils.validation import check_is_fitted

    transform_config = ReadonlyConfig(
        {
            "binarize": dict(
                transformer=Binarizer, docstring="See `sklearn.preprocessing.Binarizer`."
            ),
            "minmax_scale": dict(
                transformer=MinMaxScaler, docstring="See `sklearn.preprocessing.MinMaxScaler`."
            ),
            "maxabs_scale": dict(
                transformer=MaxAbsScaler, docstring="See `sklearn.preprocessing.MaxAbsScaler`."
            ),
            "normalize": dict(
                transformer=Normalizer, docstring="See `sklearn.preprocessing.Normalizer`."
            ),
            "robust_scale": dict(
                transformer=RobustScaler, docstring="See `sklearn.preprocessing.RobustScaler`."
            ),
            "scale": dict(
                transformer=StandardScaler, docstring="See `sklearn.preprocessing.StandardScaler`."
            ),
            "quantile_transform": dict(
                transformer=QuantileTransformer,
                docstring="See `sklearn.preprocessing.QuantileTransformer`.",
            ),
            "power_transform": dict(
                transformer=PowerTransformer,
                docstring="See `sklearn.preprocessing.PowerTransformer`.",
            ),
        }
    )
    """_"""

    __pdoc__[
        "transform_config"
    ] = f"""Configuration for transformation methods to be attached to `GenericAccessor`.

```python
{transform_config.prettify_doc()}
```
"""

    GenericAccessor = attach_transform_methods(transform_config)(GenericAccessor)

GenericAccessor.override_metrics_doc(__pdoc__)
GenericAccessor.override_subplots_doc(__pdoc__)


class GenericSRAccessor(GenericAccessor, BaseSRAccessor):
    """Class representing an accessor for Series of any data type.

    Accessible via `pd.Series.vbt`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying data array.
        mapping (Optional[MappingLike]): Mapping configuration.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseSRAccessor` and `GenericAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        mapping: tp.Optional[tp.MappingLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        BaseSRAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            GenericAccessor.__init__(self, wrapper, obj=obj, mapping=mapping, **kwargs)

    def fit_pattern(
        self,
        pattern: tp.ArrayLike,
        interp_mode: tp.Union[int, str] = "mixed",
        rescale_mode: tp.Union[int, str] = "minmax",
        vmin: float = np.nan,
        vmax: float = np.nan,
        pmin: float = np.nan,
        pmax: float = np.nan,
        invert: bool = False,
        error_type: tp.Union[int, str] = "absolute",
        max_error: tp.ArrayLike = np.nan,
        max_error_interp_mode: tp.Union[None, int, str] = None,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Series, tp.Series]:
        """Fit a pattern on the series data using `vectorbtpro.generic.nb.patterns.fit_pattern_nb`.

        Args:
            pattern (ArrayLike): Reference pattern array for similarity comparison.
            interp_mode (Union[int, str]): Interpolation mode.

                Mapped using `vectorbtpro.generic.enums.InterpMode` if provided as a string.
            rescale_mode (Union[int, str]): Rescaling mode.

                Mapped using `vectorbtpro.generic.enums.RescaleMode` if provided as a string.
            vmin (float): Minimum value used for rescaling `arr`.
            vmax (float): Maximum value used for rescaling `arr`.
            pmin (float): Minimum value used for rescaling `pattern`.
            pmax (float): Maximum value used for rescaling `pattern`.
            invert (bool): Invert the pattern by reflecting its values.
            error_type (Union[int, str]): Error calculation type.

                Mapped using `vectorbtpro.generic.enums.ErrorType` if provided as a string.
            max_error (ArrayLike): Maximum allowed error.
            max_error_interp_mode (Union[None, int, str]): Interpolation mode for `max_error`.

                Mapped using `vectorbtpro.generic.enums.InterpMode` if provided as a string.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Tuple[Series, Series]: Tuple containing the fitted pattern series and the maximum error series.
        """
        if isinstance(interp_mode, str):
            interp_mode = map_enum_fields(interp_mode, InterpMode)
        if isinstance(rescale_mode, str):
            rescale_mode = map_enum_fields(rescale_mode, RescaleMode)
        if isinstance(error_type, str):
            error_type = map_enum_fields(error_type, ErrorType)
        if max_error_interp_mode is not None and isinstance(max_error_interp_mode, str):
            max_error_interp_mode = map_enum_fields(max_error_interp_mode, InterpMode)
        if max_error_interp_mode is None:
            max_error_interp_mode = interp_mode
        pattern = reshaping.to_1d_array(pattern)
        max_error = reshaping.broadcast_array_to(max_error, len(pattern))

        func = jit_reg.resolve_option(nb.fit_pattern_nb, jitted)
        fit_pattern, fit_max_error = func(
            self.to_1d_array(),
            pattern,
            interp_mode=interp_mode,
            rescale_mode=rescale_mode,
            vmin=vmin,
            vmax=vmax,
            pmin=pmin,
            pmax=pmax,
            invert=invert,
            error_type=error_type,
            max_error=max_error,
            max_error_interp_mode=max_error_interp_mode,
        )
        pattern_sr = self.wrapper.wrap(fit_pattern)
        max_error_sr = self.wrapper.wrap(fit_max_error)
        return pattern_sr, max_error_sr

    def to_renko(
        self,
        brick_size: tp.ArrayLike,
        relative: tp.ArrayLike = False,
        start_value: tp.Optional[float] = None,
        max_out_len: tp.Optional[int] = None,
        reset_index: bool = False,
        return_uptrend: bool = False,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.Series, tp.Tuple[tp.Series, tp.Series]]:
        """Generate a Renko chart from the series data using
        `vectorbtpro.generic.nb.base.to_renko_1d_nb`.

        Args:
            brick_size (ArrayLike): Brick size for constructing the Renko chart.
            relative (ArrayLike): Indicator specifying if the brick size is relative.
            start_value (Optional[float]): Starting value for the Renko chart.
            max_out_len (Optional[int]): Maximum length of the output.
            reset_index (bool): Whether to reset the index in the resulting Series.
            return_uptrend (bool): Whether to return the uptrend series along with the Renko series.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Union[Series, Tuple[Series, Series]]: Renko series or a tuple containing
                the Renko series and the uptrend series.
        """
        func = jit_reg.resolve_option(nb.to_renko_1d_nb, jitted)
        arr_out, idx_out, uptrend_out = func(
            self.to_1d_array(),
            reshaping.broadcast_array_to(brick_size, self.wrapper.shape[0]),
            relative=reshaping.broadcast_array_to(relative, self.wrapper.shape[0]),
            start_value=start_value,
            max_out_len=max_out_len,
        )
        if reset_index:
            new_index = pd.RangeIndex(stop=len(idx_out))
        else:
            new_index = self.wrapper.index[idx_out]
        wrap_kwargs = merge_dicts(
            dict(index=new_index),
            wrap_kwargs,
        )
        sr_out = self.wrapper.wrap(arr_out, group_by=False, **wrap_kwargs)
        if return_uptrend:
            uptrend_out = self.wrapper.wrap(uptrend_out, group_by=False, **wrap_kwargs)
            return sr_out, uptrend_out
        return sr_out

    def to_renko_ohlc(
        self,
        brick_size: tp.ArrayLike,
        relative: tp.ArrayLike = False,
        start_value: tp.Optional[float] = None,
        max_out_len: tp.Optional[int] = None,
        reset_index: bool = False,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Frame:
        """Generate OHLC Renko bars from the series data using
        `vectorbtpro.generic.nb.base.to_renko_ohlc_1d_nb`.

        Args:
            brick_size (ArrayLike): Brick size for constructing the OHLC Renko chart.
            relative (ArrayLike): Indicator specifying if the brick size is relative.
            start_value (Optional[float]): Starting value for the Renko chart.
            max_out_len (Optional[int]): Maximum length of the output.
            reset_index (bool): Whether to reset the index in the resulting DataFrame.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Frame: DataFrame representing the OHLC Renko bars with columns
                ["Open", "High", "Low", "Close"].
        """
        func = jit_reg.resolve_option(nb.to_renko_ohlc_1d_nb, jitted)
        arr_out, idx_out = func(
            self.to_1d_array(),
            reshaping.broadcast_array_to(brick_size, self.wrapper.shape[0]),
            relative=reshaping.broadcast_array_to(relative, self.wrapper.shape[0]),
            start_value=start_value,
            max_out_len=max_out_len,
        )
        if reset_index:
            new_index = pd.RangeIndex(stop=len(idx_out))
        else:
            new_index = self.wrapper.index[idx_out]
        wrap_kwargs = merge_dicts(
            dict(index=new_index, columns=["Open", "High", "Low", "Close"]),
            wrap_kwargs,
        )
        return self.wrapper.wrap(arr_out, group_by=False, **wrap_kwargs)


class GenericDFAccessor(GenericAccessor, BaseDFAccessor):
    """Class representing an accessor for DataFrames of any data type.

    Accessible via `pd.DataFrame.vbt`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying data object.
        mapping (Optional[MappingLike]): Mapping configuration.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseDFAccessor` and `GenericAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        mapping: tp.Optional[tp.MappingLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        BaseDFAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            GenericAccessor.__init__(self, wrapper, obj=obj, mapping=mapping, **kwargs)

    def band(self, band_name: str, return_meta: bool = False) -> tp.Union[tp.Series, dict]:
        """Calculate the band value based on the given band name.

        Args:
            band_name (str): Band name determining the calculation.

                Possible values include:

                * "50%": 50th quantile.
                * "Q=50%": 50th quantile.
                * "Q=0.5": 50th quantile.
                * "Z=1.96": Z-score of 1.96.
                * "P=95%": One-tailed significance level of 0.95 (translated into z-score).
                * "P=0.95": One-tailed significance level of 0.95 (translated into z-score).
                * "median": Median (50th quantile).
                * "mean": Mean across all columns.
                * "min": Minimum across all columns.
                * "max": Maximum across all columns.
                * "lowest": Column with the lowest final value.
                * "highest": Column with the highest final value.
            return_meta (bool): Whether to return metadata including band details.

        Returns:
            Union[Series, dict]: Computed band as a Series if `return_meta` is False, or a dict containing
                band metadata (`band_name`, `band_title`, and `band_func`) if `return_meta` is True.
        """
        band_name = band_name.lower().replace(" ", "")
        if band_name == "median":
            band_name = "50%"
        if "%" in band_name and not band_name.startswith("q=") and not band_name.startswith("p="):
            band_name = f"q={band_name}"
        if band_name.startswith("q="):
            if "%" in band_name:
                q = float(band_name.replace("q=", "").replace("%", "")) / 100
            else:
                q = float(band_name.replace("q=", ""))
            q_readable = np.around(q * 100, decimals=2)
            if q_readable.is_integer():
                q_readable = int(q_readable)
            band_title = f"Q={q_readable}% (proj)"

            def _band_func(df, _q=q):
                return df.quantile(_q, axis=1)

        elif band_name.startswith("z="):
            z = float(band_name.replace("z=", ""))
            z_readable = np.around(z, decimals=2)
            if z_readable.is_integer():
                z_readable = int(z_readable)
            band_title = f"Z={z_readable} (proj)"

            def _band_func(df, _z=z):
                return df.mean(axis=1) + _z * df.std(axis=1)

        elif band_name.startswith("p="):
            import scipy.stats as st

            if "%" in band_name:
                p = float(band_name.replace("p=", "").replace("%", "")) / 100
            else:
                p = float(band_name.replace("p=", ""))
            p_readable = np.around(p * 100, decimals=2)
            if p_readable.is_integer():
                p_readable = int(p_readable)
            band_title = f"P={p_readable}% (proj)"
            z = st.norm.ppf(p)

            def _band_func(df, _z=z):
                return df.mean(axis=1) + _z * df.std(axis=1)

        elif band_name == "mean":
            band_title = "Mean (proj)"

            def _band_func(df):
                return df.mean(axis=1)

        elif band_name == "min":
            band_title = "Min (proj)"

            def _band_func(df):
                return df.min(axis=1)

        elif band_name == "max":
            band_title = "Max (proj)"

            def _band_func(df):
                return df.max(axis=1)

        elif band_name == "lowest":
            band_title = "Lowest (proj)"

            def _band_func(df):
                return df[df.ffill().iloc[-1].idxmin()]

        elif band_name == "highest":
            band_title = "Highest (proj)"

            def _band_func(df):
                return df[df.ffill().iloc[-1].idxmax()]

        else:
            raise ValueError(f"Invalid band_name: '{band_name}'")
        if return_meta:
            return dict(band_name=band_name, band_title=band_title, band_func=_band_func)
        return _band_func(self.obj)

    def plot_projections(
        self,
        plot_projections: bool = True,
        plot_bands: bool = True,
        plot_lower: tp.Union[bool, str, tp.Callable] = True,
        plot_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_upper: tp.Union[bool, str, tp.Callable] = True,
        plot_aux_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_fill: bool = True,
        colorize: tp.Union[bool, str, tp.Callable] = True,
        rename_levels: tp.Union[None, dict, tp.Sequence] = None,
        projection_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        aux_middle_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot DataFrame columns as projections with optional bands and colorization.

        Plots each column of the DataFrame as a projection line. Optionally, bands
        (lower, middle, upper, and auxiliary middle) are computed and plotted to provide
        additional context. Projections are drawn with semi-transparent lines, and
        colorization is applied based on band metrics if specified.

        Args:
            plot_projections (bool): Plot each projection as a semi-transparent line if True.
            plot_bands (bool): Plot computed bands if True.

                Disabled for single-column DataFrames.
            plot_lower (Union[bool, str, Callable]): Specification for the lower band.

                Accepts:

                * True: Compute using the default "20%" quantile.
                * False: Do not plot the band.
                * str: Use a custom band identifier.
                * Callable: Function to compute the band by reducing the DataFrame.
            plot_middle (Union[bool, str, Callable]): Specification for the middle band.

                Accepts:

                * True: Compute using the default "50%" quantile.
                * False: Do not plot the band.
                * str: Use a custom band identifier.
                * Callable: Function to compute the band by reducing the DataFrame.
            plot_upper (Union[bool, str, Callable]): Specification for the upper band.

                Accepts:

                * True: Compute using the default "80%" quantile.
                * False: Do not plot the band.
                * str: Use a custom band identifier.
                * Callable: Function to compute the band by reducing the DataFrame.
            plot_aux_middle (Union[bool, str, Callable]): Specification for an auxiliary middle band.

                Accepts:

                * True: Compute using "mean".
                * False: Do not plot the band.
                * str: Use a custom band identifier.
                * Callable: Function to compute the band by reducing the DataFrame.
            plot_fill (bool): Fill the area between band traces if True.
            colorize (Union[bool, str, Callable]): Strategy for colorizing projections or bands.

                Accepts:

                * False: Do not colorize.
                * True or "median": Colorize by median.
                * "mean": Colorize by mean.
                * "last": Colorize by last value.
                * Callable: Function that reduces a Series or DataFrame to compute a metric.
            rename_levels (Union[None, dict, Sequence]): Mapping or sequence to rename multi-index legend levels.
            projection_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the projections.
            lower_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the lower band.
            middle_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the middle band.
            upper_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the upper band.
            aux_middle_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the auxiliary middle band.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Updated figure with plotted projections and bands.

        !!! note
            When using z-scores, the upper band should be positive, the middle computed as "mean", and
            the lower negative. For significance levels, the middle remains "mean" whereas the lower should be
            positive and lower than the upper (e.g., 25% and 75%).

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> df = pd.DataFrame({
            ...     0: [10, 11, 12, 11, 10],
            ...     1: [10, 12, 14, np.nan, np.nan],
            ...     2: [10, 12, 11, 12, np.nan],
            ...     3: [10, 9, 8, 9, 8],
            ...     4: [10, 11, np.nan, np.nan, np.nan],
            ... })
            >>> df.vbt.plot_projections().show()
            ```

            ![](/assets/images/api/df_plot_projections.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/df_plot_projections.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        if projection_trace_kwargs is None:
            projection_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # Resolve band functions and names
        if len(self.obj.columns) == 1:
            plot_bands = False
        if not plot_bands:
            plot_lower = False
            plot_middle = False
            plot_upper = False
            plot_aux_middle = False
        if isinstance(plot_lower, bool):
            if plot_lower:
                plot_lower = "20%"
            else:
                plot_lower = None
        if isinstance(plot_middle, bool):
            if plot_middle:
                plot_middle = "50%"
            else:
                plot_middle = None
        if isinstance(plot_upper, bool):
            if plot_upper:
                plot_upper = "80%"
            else:
                plot_upper = None
        if isinstance(plot_aux_middle, bool):
            if plot_aux_middle:
                plot_aux_middle = "mean"
            else:
                plot_aux_middle = None

        def _resolve_band_and_name(band_func, arg_name):
            band_title = None
            if isinstance(band_func, str):
                band_func_meta = self.band(band_func, return_meta=True)
                band_title = band_func_meta["band_title"]
                band_func = band_func_meta["band_func"]
            if band_func is not None and not callable(band_func):
                raise TypeError(f"Argument {arg_name} has wrong type '{type(band_func)}'")
            return band_func, band_title

        plot_lower, lower_name = _resolve_band_and_name(plot_lower, "plot_lower")
        if lower_name is None:
            lower_name = "Lower (proj)"
        plot_middle, middle_name = _resolve_band_and_name(plot_middle, "plot_middle")
        if middle_name is None:
            middle_name = "Middle (proj)"
        plot_upper, upper_name = _resolve_band_and_name(plot_upper, "plot_upper")
        if upper_name is None:
            upper_name = "Upper (proj)"
        plot_aux_middle, aux_middle_name = _resolve_band_and_name(
            plot_aux_middle, "plot_aux_middle"
        )
        if aux_middle_name is None:
            aux_middle_name = "Aux middle (proj)"

        if isinstance(colorize, bool):
            if colorize:
                colorize = "median"
            else:
                colorize = None
        if colorize is not None:
            if isinstance(colorize, str):
                colorize = colorize.lower().replace(" ", "")
                if colorize == "median":
                    colorize = lambda x: x.median()
                elif colorize == "mean":
                    colorize = lambda x: x.mean()
                elif colorize == "last":
                    colorize = lambda x: x.ffill().iloc[-1]
                else:
                    raise ValueError(f"Argument colorize has wrong value '{colorize}'")
            if colorize is not None and not callable(colorize):
                raise TypeError(f"Argument colorize has wrong type '{type(colorize)}'")
        if colorize is not None:
            proj_min = colorize(self.obj - self.obj.iloc[0]).min()
            proj_max = colorize(self.obj - self.obj.iloc[0]).max()
        else:
            proj_min = None
            proj_max = None

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if len(self.obj.columns) > 0:
            if plot_projections:
                # Plot projections
                for col in range(self.wrapper.shape[1]):
                    proj_sr = self.obj.iloc[:, col].dropna()
                    hovertemplate = "(%{x}, %{y})"
                    if not checks.is_default_index(self.wrapper.columns):
                        level_names = []
                        level_values = []
                        if isinstance(self.wrapper.columns, pd.MultiIndex):
                            for l in range(self.wrapper.columns.nlevels):
                                level_names.append(self.wrapper.columns.names[l])
                                level_values.append(self.wrapper.columns.get_level_values(l)[col])
                        else:
                            level_names.append(self.wrapper.columns.name)
                            level_values.append(self.wrapper.columns[col])
                        for l in range(len(level_names)):
                            level_name = level_names[l]
                            level_value = level_values[l]
                            if rename_levels is not None:
                                if isinstance(rename_levels, dict):
                                    if level_name in rename_levels:
                                        level_name = rename_levels[level_name]
                                    elif l in rename_levels:
                                        level_name = rename_levels[l]
                                else:
                                    level_name = rename_levels[l]
                            if level_name is None:
                                level_name = f"Level {l}"
                            hovertemplate += f"<br>{level_name}: {level_value}"

                    if colorize is not None:
                        proj_color = map_value_to_cmap(
                            colorize(proj_sr - proj_sr.iloc[0]),
                            [
                                plotting_cfg["color_schema"]["red"],
                                plotting_cfg["color_schema"]["yellow"],
                                plotting_cfg["color_schema"]["green"],
                            ],
                            vmin=proj_min,
                            vcenter=0,
                            vmax=proj_max,
                        )
                    else:
                        proj_color = plotting_cfg["color_schema"]["gray"]
                    if not plot_bands:
                        proj_opacity = 0.5
                    else:
                        proj_opacity = 0.1
                    _projection_trace_kwargs = merge_dicts(
                        dict(
                            name=f"proj ({self.obj.shape[1]})",
                            line=dict(color=proj_color),
                            opacity=proj_opacity,
                            legendgroup="proj",
                            showlegend=col == 0,
                            hovertemplate=hovertemplate,
                        ),
                        projection_trace_kwargs,
                    )
                    proj_sr.rename(None).vbt.lineplot(
                        trace_kwargs=_projection_trace_kwargs,
                        add_trace_kwargs=add_trace_kwargs,
                        fig=fig,
                    )

        if plot_bands and len(self.obj.columns) > 1:
            # Calculate bands
            if plot_lower is not None:
                lower_band = plot_lower(self.obj).dropna()
            else:
                lower_band = None
            if plot_middle is not None:
                middle_band = plot_middle(self.obj).dropna()
            else:
                middle_band = None
            if plot_upper is not None:
                upper_band = plot_upper(self.obj).dropna()
            else:
                upper_band = None
            if plot_aux_middle is not None:
                aux_middle_band = plot_aux_middle(self.obj).dropna()
            else:
                aux_middle_band = None

            if lower_band is not None:
                # Plot lower band
                def_lower_trace_kwargs = dict(name=lower_name)
                if colorize is not None:
                    lower_color = map_value_to_cmap(
                        colorize(lower_band - lower_band.iloc[0]),
                        [
                            plotting_cfg["color_schema"]["red"],
                            plotting_cfg["color_schema"]["yellow"],
                            plotting_cfg["color_schema"]["green"],
                        ],
                        vmin=proj_min,
                        vcenter=0,
                        vmax=proj_max,
                    )
                    def_lower_trace_kwargs["line"] = dict(color=adjust_opacity(lower_color, 0.75))
                else:
                    lower_color = plotting_cfg["color_schema"]["gray"]
                    def_lower_trace_kwargs["line"] = dict(color=adjust_opacity(lower_color, 0.5))
                lower_band.rename(None).vbt.lineplot(
                    trace_kwargs=merge_dicts(def_lower_trace_kwargs, lower_trace_kwargs),
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

            if middle_band is not None:
                # Plot middle band
                def_middle_trace_kwargs = dict(name=middle_name)
                if colorize is not None:
                    middle_color = map_value_to_cmap(
                        colorize(middle_band - middle_band.iloc[0]),
                        [
                            plotting_cfg["color_schema"]["red"],
                            plotting_cfg["color_schema"]["yellow"],
                            plotting_cfg["color_schema"]["green"],
                        ],
                        vmin=proj_min,
                        vcenter=0,
                        vmax=proj_max,
                    )
                else:
                    middle_color = plotting_cfg["color_schema"]["gray"]
                def_middle_trace_kwargs["line"] = dict(color=middle_color)
                if plot_fill and lower_band is not None:
                    def_middle_trace_kwargs["fill"] = "tonexty"
                    def_middle_trace_kwargs["fillcolor"] = adjust_opacity(
                        plotting_cfg["color_schema"]["gray"], 0.25
                    )
                middle_band.rename(None).vbt.lineplot(
                    trace_kwargs=merge_dicts(def_middle_trace_kwargs, middle_trace_kwargs),
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

            if upper_band is not None:
                # Plot upper band
                def_upper_trace_kwargs = dict(name=upper_name)
                if colorize is not None:
                    upper_color = map_value_to_cmap(
                        colorize(upper_band - upper_band.iloc[0]),
                        [
                            plotting_cfg["color_schema"]["red"],
                            plotting_cfg["color_schema"]["yellow"],
                            plotting_cfg["color_schema"]["green"],
                        ],
                        vmin=proj_min,
                        vcenter=0,
                        vmax=proj_max,
                    )
                    def_upper_trace_kwargs["line"] = dict(color=adjust_opacity(upper_color, 0.75))
                else:
                    upper_color = plotting_cfg["color_schema"]["gray"]
                    def_upper_trace_kwargs["line"] = dict(color=adjust_opacity(upper_color, 0.5))
                if plot_fill and (lower_band is not None or middle_band is not None):
                    def_upper_trace_kwargs["fill"] = "tonexty"
                    def_upper_trace_kwargs["fillcolor"] = adjust_opacity(
                        plotting_cfg["color_schema"]["gray"], 0.25
                    )
                upper_band.rename(None).vbt.lineplot(
                    trace_kwargs=merge_dicts(def_upper_trace_kwargs, upper_trace_kwargs),
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

            if aux_middle_band is not None:
                # Plot auxiliary band
                def_aux_middle_trace_kwargs = dict(name=aux_middle_name)
                if colorize is not None:
                    aux_middle_color = map_value_to_cmap(
                        colorize(aux_middle_band - aux_middle_band.iloc[0]),
                        [
                            plotting_cfg["color_schema"]["red"],
                            plotting_cfg["color_schema"]["yellow"],
                            plotting_cfg["color_schema"]["green"],
                        ],
                        vmin=proj_min,
                        vcenter=0,
                        vmax=proj_max,
                    )
                else:
                    aux_middle_color = plotting_cfg["color_schema"]["gray"]
                def_aux_middle_trace_kwargs["line"] = dict(dash="dot", color=aux_middle_color)
                aux_middle_band.rename(None).vbt.lineplot(
                    trace_kwargs=merge_dicts(def_aux_middle_trace_kwargs, aux_middle_trace_kwargs),
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        return fig
