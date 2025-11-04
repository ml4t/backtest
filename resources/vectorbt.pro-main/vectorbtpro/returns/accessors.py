# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom Pandas accessors for returns.

Access methods are available as follows:

* `ReturnsSRAccessor` can be accessed via `pd.Series.vbt.returns.*`
* `ReturnsDFAccessor` can be accessed via `pd.DataFrame.vbt.returns.*`

!!! note
    The underlying Series/DataFrame must represent returns. To convert price data to returns, use `ReturnsAccessor.from_value`.

    Grouping is supported only by methods accepting the `group_by` argument.
    Accessor methods do not use caching.

There are three ways to compute returns and obtain the accessor:

```pycon
>>> from vectorbtpro import *

>>> price = pd.Series([1.1, 1.2, 1.3, 1.2, 1.1])

>>> # 1. Using pd.Series.pct_change
>>> rets = price.pct_change()
>>> ret_acc = rets.vbt.returns(freq='d')

>>> # 2. Using vectorbtpro.generic.accessors.GenericAccessor.to_returns
>>> rets = price.vbt.to_returns()
>>> ret_acc = rets.vbt.returns(freq='d')

>>> # 3. Using vectorbtpro.returns.accessors.ReturnsAccessor.from_value
>>> ret_acc = pd.Series.vbt.returns.from_value(price, freq='d')

>>> # Access total returns using vectorbtpro.returns.accessors.ReturnsAccessor.total
>>> ret_acc.total()
0.0
```

The accessors extend `vectorbtpro.generic.accessors`.

```pycon
>>> # Inherited from GenericAccessor
>>> ret_acc.max()
0.09090909090909083
```

## Defaults

`ReturnsAccessor` accepts a `defaults` dictionary that provides default parameters used throughout the accessor, such as:

* `start_value`: Starting value.
* `window`: Window length.
* `minp`: Minimum number of observations required in a window.
* `ddof`: Delta Degrees of Freedom.
* `risk_free`: Constant risk-free return during the period.
* `levy_alpha`: Scaling relation (Levy stability exponent).
* `required_return`: Minimum acceptable return.
* `cutoff`: Decimal representing the cutoff percentage for the lowest returns.
* `periods`: Number of observations for annualization (can be an integer or "dt_periods").

Global settings, including `bm_returns` and `year_freq`, can be configured through:

```pycon
>>> benchmark = pd.Series([1.05, 1.1, 1.15, 1.1, 1.05])
>>> bm_returns = benchmark.vbt.to_returns()

>>> vbt.settings.returns['bm_returns'] = bm_returns
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `ReturnsAccessor.metrics`.

```pycon
>>> ret_acc.stats()
Start                                      0
End                                        4
Duration                     5 days 00:00:00
Total Return [%]                           0
Benchmark Return [%]                       0
Annualized Return [%]                      0
Annualized Volatility [%]            184.643
Sharpe Ratio                        0.691185
Calmar Ratio                               0
Max Drawdown [%]                     15.3846
Omega Ratio                          1.08727
Sortino Ratio                        1.17805
Skew                              0.00151002
Kurtosis                            -5.94737
Tail Ratio                           1.08985
Common Sense Ratio                   1.08985
Value at Risk                     -0.0823718
Alpha                                0.78789
Beta                                 1.83864
dtype: object
```

!!! note
    `ReturnsAccessor.stats` does not support grouping.

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `ReturnsAccessor.subplots`.

The `ReturnsAccessor` class provides a single subplot based on `ReturnsAccessor.plot_cumulative`:

```pycon
>>> ret_acc.plots().show()
```

![](/assets/images/api/returns_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/returns_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from vectorbtpro import _typing as tp
from vectorbtpro.accessors import (
    register_df_vbt_accessor,
    register_sr_vbt_accessor,
    register_vbt_accessor,
)
from vectorbtpro.base.reshaping import broadcast_array_to, broadcast_to, to_1d_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic.accessors import GenericAccessor, GenericDFAccessor, GenericSRAccessor
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.sim_range import SimRangeMixin
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.returns import nb
from vectorbtpro.utils import checks
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import hybrid_method, hybrid_property
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from vectorbtpro.returns.qs_adapter import QSAdapter as QSAdapterT
else:
    QSAdapterT = "vectorbtpro.returns.qs_adapter.QSAdapter"

__all__ = [
    "ReturnsAccessor",
    "ReturnsSRAccessor",
    "ReturnsDFAccessor",
]

__pdoc__ = {}

ReturnsAccessorT = tp.TypeVar("ReturnsAccessorT", bound="ReturnsAccessor")


@register_vbt_accessor("returns")
class ReturnsAccessor(GenericAccessor, SimRangeMixin):
    """Class representing an accessor on top of return series for both Pandas `Series` and `DataFrame`.

    Accessible via `pd.Series.vbt.returns` and `pd.DataFrame.vbt.returns`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Pandas object containing return data.

            Represents the return series.
        bm_returns (Optional[ArrayLike]): Benchmark returns.
        log_returns (bool): Flag indicating whether returns are logarithmic.
        year_freq (Optional[FrequencyLike]): Year frequency for annualization (e.g., "252 days", "auto").
        defaults (KwargsLike): Dictionary of default parameters.
        sim_start (Optional[ArrayLike]): Start index of the simulation range.
        sim_end (Optional[ArrayLike]): End index of the simulation range.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor`.

    !!! info
        For default settings, see `vectorbtpro._settings.returns`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        log_returns: bool = False,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        defaults: tp.KwargsLike = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        GenericAccessor.__init__(
            self,
            wrapper,
            obj=obj,
            bm_returns=bm_returns,
            log_returns=log_returns,
            year_freq=year_freq,
            defaults=defaults,
            sim_start=sim_start,
            sim_end=sim_end,
            **kwargs,
        )
        SimRangeMixin.__init__(self, sim_start=sim_start, sim_end=sim_end)

        self._bm_returns = bm_returns
        self._log_returns = log_returns
        self._year_freq = year_freq
        self._defaults = defaults

    @classmethod
    def from_value(
        cls: tp.Type[ReturnsAccessorT],
        value: tp.ArrayLike,
        init_value: tp.ArrayLike = np.nan,
        log_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        return_values: bool = False,
        **kwargs,
    ) -> tp.Union[ReturnsAccessorT, tp.SeriesFrame]:
        """Return a new `ReturnsAccessor` instance with return calculations derived from `value`.

        Args:
            value (ArrayLike): Input data from which returns are computed.
            init_value (ArrayLike): Initial value to broadcast for each column.
            log_returns (bool): Flag indicating whether returns are logarithmic.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            return_values (bool): If True, return wrapped return values instead of a `ReturnsAccessor` instance.
            **kwargs: Keyword arguments for `ReturnsAccessor`.

        Returns:
            Union[ReturnsAccessor, SeriesFrame]: New accessor instance or wrapped return values.

        See:
            `vectorbtpro.returns.nb.returns_nb`
        """
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not checks.is_any_array(value):
            value = np.asarray(value)
        if wrapper is None:
            wrapper = ArrayWrapper.from_obj(value, **wrapper_kwargs)
        elif len(wrapper_kwargs) > 0:
            wrapper = wrapper.replace(**wrapper_kwargs)
        value = to_2d_array(value)
        init_value = broadcast_array_to(init_value, value.shape[1])
        sim_start = cls.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        returns = func(
            value,
            init_value=init_value,
            log_returns=log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        if return_values:
            return wrapper.wrap(returns, group_by=False)
        return cls(wrapper, returns, sim_start=sim_start, sim_end=sim_end, **kwargs)

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[ReturnsAccessorT],
        *objs: tp.MaybeSequence[ReturnsAccessorT],
        **kwargs,
    ) -> tp.Kwargs:
        kwargs = GenericAccessor.resolve_row_stack_kwargs(*objs, **kwargs)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, ReturnsAccessor):
                raise TypeError("Each object to be merged must be an instance of ReturnsAccessor")
        if "bm_returns" not in kwargs:
            bm_returns = []
            stack_bm_returns = True
            for obj in objs:
                if obj.config["bm_returns"] is not None:
                    bm_returns.append(obj.config["bm_returns"])
                else:
                    stack_bm_returns = False
                    break
            if stack_bm_returns:
                kwargs["bm_returns"] = kwargs["wrapper"].row_stack_arrs(
                    *bm_returns,
                    group_by=False,
                    wrap=False,
                )
        if "sim_start" not in kwargs:
            kwargs["sim_start"] = cls.row_stack_sim_start(kwargs["wrapper"], *objs)
        if "sim_end" not in kwargs:
            kwargs["sim_end"] = cls.row_stack_sim_end(kwargs["wrapper"], *objs)
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[ReturnsAccessorT],
        *objs: tp.MaybeSequence[ReturnsAccessorT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing a `ReturnsAccessor` after stacking along columns.

        Args:
            *objs (MaybeSequence[ReturnsAccessor]): Instances of `ReturnsAccessor` to be merged.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            **kwargs: Additional keyword arguments.

        Returns:
            Kwargs: Resolved keyword arguments for creating a new `ReturnsAccessor` instance.
        """
        kwargs = GenericAccessor.resolve_column_stack_kwargs(
            *objs, reindex_kwargs=reindex_kwargs, **kwargs
        )
        kwargs.pop("reindex_kwargs", None)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, ReturnsAccessor):
                raise TypeError("Each object to be merged must be an instance of ReturnsAccessor")
        if "bm_returns" not in kwargs:
            bm_returns = []
            stack_bm_returns = True
            for obj in objs:
                if obj.bm_returns is not None:
                    bm_returns.append(obj.bm_returns)
                else:
                    stack_bm_returns = False
                    break
            if stack_bm_returns:
                kwargs["bm_returns"] = kwargs["wrapper"].column_stack_arrs(
                    *bm_returns,
                    reindex_kwargs=reindex_kwargs,
                    group_by=False,
                    wrap=False,
                )
        if "sim_start" not in kwargs:
            kwargs["sim_start"] = cls.column_stack_sim_start(kwargs["wrapper"], *objs)
        if "sim_end" not in kwargs:
            kwargs["sim_end"] = cls.column_stack_sim_end(kwargs["wrapper"], *objs)
        return kwargs

    @hybrid_property
    def sr_accessor_cls(cls_or_self) -> tp.Type["ReturnsSRAccessor"]:
        return ReturnsSRAccessor

    @hybrid_property
    def df_accessor_cls(cls_or_self) -> tp.Type["ReturnsDFAccessor"]:
        return ReturnsDFAccessor

    def indexing_func(
        self: ReturnsAccessorT,
        *args,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> ReturnsAccessorT:
        """Perform indexing on a `ReturnsAccessor` instance.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.

        Returns:
            ReturnsAccessor: New accessor instance with the sliced data,
                benchmark returns, and simulation indices.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_obj = wrapper_meta["new_wrapper"].wrap(
            self.to_2d_array()[wrapper_meta["row_idxs"], :][:, wrapper_meta["col_idxs"]],
            group_by=False,
        )
        if self._bm_returns is not None:
            new_bm_returns = ArrayWrapper.select_from_flex_array(
                self._bm_returns,
                row_idxs=wrapper_meta["row_idxs"],
                col_idxs=wrapper_meta["col_idxs"],
                rows_changed=wrapper_meta["rows_changed"],
                columns_changed=wrapper_meta["columns_changed"],
            )
        else:
            new_bm_returns = None
        new_sim_start = self.sim_start_indexing_func(wrapper_meta)
        new_sim_end = self.sim_end_indexing_func(wrapper_meta)

        if checks.is_series(new_obj):
            return self.replace(
                cls_=self.sr_accessor_cls,
                wrapper=wrapper_meta["new_wrapper"],
                obj=new_obj,
                bm_returns=new_bm_returns,
                sim_start=new_sim_start,
                sim_end=new_sim_end,
            )
        return self.replace(
            cls_=self.df_accessor_cls,
            wrapper=wrapper_meta["new_wrapper"],
            obj=new_obj,
            bm_returns=new_bm_returns,
            sim_start=new_sim_start,
            sim_end=new_sim_end,
        )

    # ############# Properties ############# #

    @property
    def bm_returns(self) -> tp.Optional[tp.SeriesFrame]:
        """Benchmark returns wrapped using the assigned array wrapper.

        Returns:
            Optional[SeriesFrame]: Benchmark returns if available; otherwise, None.

        !!! info
            For default settings, see `vectorbtpro._settings.returns`.
        """
        from vectorbtpro._settings import settings

        returns_cfg = settings["returns"]

        bm_returns = self._bm_returns
        if bm_returns is None:
            bm_returns = returns_cfg["bm_returns"]
        if bm_returns is not None:
            bm_returns = self.wrapper.wrap(bm_returns, group_by=False)
        return bm_returns

    def get_bm_returns_acc(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
    ) -> tp.Optional[ReturnsAccessorT]:
        """Return a returns accessor for benchmark returns based on provided data and simulation range.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.

        Returns:
            Optional[ReturnsAccessorT]: Returns accessor instance for benchmark returns,
                or None if benchmark returns are unavailable.
        """
        if bm_returns is None:
            bm_returns = self.bm_returns
        if bm_returns is None:
            return None
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)
        return self.replace(
            obj=bm_returns,
            bm_returns=None,
            sim_start=sim_start,
            sim_end=sim_end,
        )

    @property
    def bm_returns_acc(self) -> tp.Optional[ReturnsAccessorT]:
        """Benchmark returns accessor using default arguments.

        Returns:
            Optional[ReturnsAccessor]: Returns accessor instance for benchmark returns,
                or None if benchmark returns are unavailable.
        """
        return self.get_bm_returns_acc()

    @property
    def log_returns(self) -> bool:
        """Flag indicating if returns and benchmark returns are provided as log returns.

        Returns:
            bool: True if returns are log returns; otherwise, False.
        """
        return self._log_returns

    @classmethod
    def auto_detect_ann_factor(cls, index: tp.DatetimeIndex) -> tp.Optional[float]:
        """Auto-detect the annualization factor using a datetime index.

        Args:
            index (DatetimeIndex): Datetime index used to determine the annualization factor.

        Returns:
            Optional[float]: Computed annualization factor, or None if indeterminable.
        """
        checks.assert_instance_of(index, pd.DatetimeIndex, arg_name="index")
        if len(index) == 1:
            return None
        offset = index[0] + pd.offsets.YearBegin() - index[0]
        first_date = index[0] + offset
        last_date = index[-1] + offset
        next_year_date = last_date + pd.offsets.YearBegin()
        ratio = (last_date.value - first_date.value) / (next_year_date.value - first_date.value)
        ann_factor = len(index) / ratio
        ann_factor /= next_year_date.year - first_date.year
        return ann_factor

    @classmethod
    def parse_ann_factor(
        cls, index: tp.DatetimeIndex, method_name: str = "max"
    ) -> tp.Optional[float]:
        """Parse the annualization factor from a datetime index using a specified method.

        Args:
            index (DatetimeIndex): Datetime index used to determine the annualization factor.
            method_name (str): Method name to apply on yearly counts (e.g., "max").

        Returns:
            Optional[float]: Parsed annualization factor, or None if not determinable.
        """
        checks.assert_instance_of(index, pd.DatetimeIndex, arg_name="index")
        if len(index) == 1:
            return None
        offset = index[0] + pd.offsets.YearBegin() - index[0]
        shifted_index = index + offset
        years = shifted_index.year
        full_years = years[years < years.max()]
        if len(full_years) == 0:
            return None
        return getattr(full_years.value_counts(), method_name.lower())()

    @classmethod
    def ann_factor_to_year_freq(
        cls,
        ann_factor: float,
        freq: tp.PandasFrequency,
        method_name: tp.Optional[str] = None,
    ) -> tp.PandasFrequency:
        """Convert an annualization factor into a year frequency.

        Args:
            ann_factor (float): Annualization factor.
            freq (PandasFrequency): Pandas-friendly frequency or offset.
            method_name (Optional[str]): Name of the NumPy method to apply to the annualization factor.

        Returns:
            PandasFrequency: Resulting year frequency.
        """
        if method_name not in (None, False):
            if method_name is True:
                ann_factor = round(ann_factor)
            else:
                ann_factor = getattr(np, method_name.lower())(ann_factor)
        if checks.is_float(ann_factor) and float.is_integer(ann_factor):
            ann_factor = int(ann_factor)
        if checks.is_float(ann_factor) and isinstance(freq, BaseOffset):
            freq = dt.offset_to_timedelta(freq)
        return ann_factor * freq

    @classmethod
    def year_freq_depends_on_index(cls, year_freq: tp.FrequencyLike) -> bool:
        """Determine whether the specified year frequency depends on the index.

        Args:
            year_freq (FrequencyLike): Year frequency for annualization (e.g., "252 days", "auto").

        Returns:
            bool: True if the frequency depends on the index; otherwise, False.
        """
        if isinstance(year_freq, str):
            year_freq = " ".join(year_freq.strip().split())
            if year_freq == "auto" or year_freq.startswith("auto_"):
                return True
            if year_freq.startswith("index_"):
                return True
        return False

    @hybrid_method
    def get_year_freq(
        cls_or_self,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Optional[tp.PandasFrequency]:
        """Resolve and return the year frequency based on provided input and index information.

        If `year_freq` is a string starting with "auto", the annualization factor is auto-detected using
        `ReturnsAccessor.auto_detect_ann_factor`. If it has the form "auto_[method_name]", the corresponding
        NumPy method is applied to the annualization factor. Similarly, if `year_freq` starts with "index_", the
        annualization factor is parsed using `ReturnsAccessor.parse_ann_factor`.

        Args:
            year_freq (Optional[FrequencyLike]): Year frequency for annualization (e.g., "252 days", "auto").
            index (Optional[Index]): Datetime index used for annualization factor detection.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min").

                See `vectorbtpro.utils.datetime_.to_freq`.

        Returns:
            Optional[PandasFrequency]: Resolved year frequency, or None if it cannot be determined.

        !!! info
            For default settings, see `vectorbtpro._settings.returns`.
        """
        if not isinstance(cls_or_self, type):
            if year_freq is None:
                year_freq = cls_or_self._year_freq
        if year_freq is None:
            from vectorbtpro._settings import settings

            returns_cfg = settings["returns"]

            year_freq = returns_cfg["year_freq"]
        if year_freq is None:
            return None

        if isinstance(year_freq, str):
            year_freq = " ".join(year_freq.strip().split())
            if cls_or_self.year_freq_depends_on_index(year_freq):
                if not isinstance(cls_or_self, type):
                    if index is None:
                        index = cls_or_self.wrapper.index
                    if freq is None:
                        freq = cls_or_self.wrapper.freq
                if index is None or not isinstance(index, pd.DatetimeIndex) or freq is None:
                    return None

                if year_freq == "auto" or year_freq.startswith("auto_"):
                    ann_factor = cls_or_self.auto_detect_ann_factor(index)
                    if year_freq == "auto":
                        method_name = None
                    else:
                        method_name = year_freq.replace("auto_", "")
                    year_freq = cls_or_self.ann_factor_to_year_freq(
                        ann_factor,
                        dt.to_freq(freq),
                        method_name=method_name,
                    )
                else:
                    method_name = year_freq.replace("index_", "")
                    ann_factor = cls_or_self.parse_ann_factor(index, method_name=method_name)
                    year_freq = cls_or_self.ann_factor_to_year_freq(
                        ann_factor,
                        dt.to_freq(freq),
                        method_name=None,
                    )

        return dt.to_freq(year_freq)

    @property
    def year_freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Year frequency.

        Returns:
            Optional[PandasFrequency]: Year frequency if available; otherwise, None.
        """
        return self.get_year_freq()

    @hybrid_method
    def get_ann_factor(
        cls_or_self,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        raise_error: bool = False,
    ) -> tp.Optional[float]:
        """Return the annualization factor based on the given year frequency and data frequency.

        Args:
            year_freq (Optional[FrequencyLike]): Year frequency for annualization (e.g., "252 days", "auto").
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min").

                See `vectorbtpro.utils.datetime_.to_timedelta`.
            raise_error (bool): Flag indicating whether to raise an error if the frequencies are None.

        Returns:
            Optional[float]: Computed annualization factor, or None if not determinable.

        !!! info
            For default settings, see `vectorbtpro._settings.returns` and `vectorbtpro._settings.wrapping`.
        """
        if isinstance(cls_or_self, type):
            from vectorbtpro._settings import settings

            returns_cfg = settings["returns"]
            wrapping_cfg = settings["wrapping"]

            if year_freq is None:
                year_freq = returns_cfg["year_freq"]
            if freq is None:
                freq = wrapping_cfg["freq"]
            if freq is not None and dt.freq_depends_on_index(freq):
                freq = None
        else:
            if year_freq is None:
                year_freq = cls_or_self.year_freq
            if freq is None:
                freq = cls_or_self.wrapper.freq
        if year_freq is None:
            if not raise_error:
                return None
            raise ValueError(
                "Year frequency is None. "
                "Pass it as `year_freq` or define it globally under `settings.returns`. "
                "To determine year frequency automatically, use 'auto'."
            )
        if freq is None:
            if not raise_error:
                return None
            raise ValueError(
                "Index frequency is None. "
                "Pass it as `freq` or define it globally under `settings.wrapping`. "
                "To determine frequency automatically, use 'auto'."
            )
        return dt.to_timedelta(year_freq, approximate=True) / dt.to_timedelta(
            freq, approximate=True
        )

    @property
    def ann_factor(self) -> float:
        """Annualization factor.

        Returns:
            float: Annualization factor.
        """
        return self.get_ann_factor(raise_error=True)

    @hybrid_method
    def get_periods(
        cls_or_self,
        periods: tp.Union[None, str, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Optional[tp.ArrayLike]:
        """Prepare periods using simulation parameters and a wrapper if necessary.

        Args:
            periods (Union[None, str, ArrayLike]): Period specification.

                If None, defaults are used.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Optional[ArrayLike]: Prepared periods array or None.
        """
        if not isinstance(cls_or_self, type) and periods is None:
            periods = cls_or_self.defaults["periods"]
        if isinstance(periods, str) and periods.lower() == "dt_periods":
            if not isinstance(cls_or_self, type):
                if wrapper is None:
                    wrapper = cls_or_self.wrapper
            else:
                checks.assert_not_none(wrapper, arg_name="wrapper")

            sim_start = cls_or_self.resolve_sim_start(
                sim_start=sim_start,
                allow_none=True,
                wrapper=wrapper,
                group_by=group_by,
            )
            sim_end = cls_or_self.resolve_sim_end(
                sim_end=sim_end,
                allow_none=True,
                wrapper=wrapper,
                group_by=group_by,
            )
            if sim_start is not None or sim_end is not None:
                if sim_start is None:
                    sim_start = cls_or_self.resolve_sim_start(
                        sim_start=sim_start,
                        allow_none=False,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
                if sim_end is None:
                    sim_end = cls_or_self.resolve_sim_end(
                        sim_end=sim_end,
                        allow_none=False,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
                periods = []
                for i in range(len(sim_start)):
                    sim_index = wrapper.index[sim_start[i] : sim_end[i]]
                    if len(sim_index) == 0:
                        periods.append(0)
                    else:
                        periods.append(wrapper.index_acc.get_dt_periods(index=sim_index))
                periods = np.asarray(periods)
            else:
                periods = wrapper.dt_periods
        return periods

    @property
    def periods(self) -> tp.Optional[tp.ArrayLike]:
        """Periods computed for the returns accessor.

        Returns:
            Optional[ArrayLike]: Periods array, or None if not applicable.
        """
        return self.get_periods()

    def deannualize(self, value: float) -> float:
        """Deannualize a value based on the annualization factor.

        Args:
            value (float): Annual return value.

        Returns:
            float: Deannualized return value.
        """
        return np.power(1 + value, 1.0 / self.ann_factor) - 1.0

    @property
    def defaults(self) -> tp.Kwargs:
        """Defaults for ReturnsAccessor.

        Merges defaults from `vectorbtpro._settings.returns` with the accessor's own defaults.

        Returns:
            Kwargs: Merged defaults dictionary.

        !!! info
            For default settings, see `defaults` in `vectorbtpro._settings.returns`.
        """
        from vectorbtpro._settings import settings

        returns_defaults_cfg = settings["returns"]["defaults"]

        return merge_dicts(returns_defaults_cfg, self._defaults)

    # ############# Transforming ############# #

    def mirror(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Mirror returns based on simulation indices.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Mirrored returns.

        See:
            `vectorbtpro.returns.nb.mirror_returns_nb`
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.mirror_returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mirrored_returns = func(
            self.to_2d_array(),
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(mirrored_returns, group_by=False, **resolve_dict(wrap_kwargs))

    def cumulative(
        self,
        start_value: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Calculate cumulative returns.

        Args:
            start_value (Optional[float]): Initial value for cumulative returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Cumulative returns.

        See:
            `vectorbtpro.returns.nb.cumulative_returns_nb`
        """
        if start_value is None:
            start_value = self.defaults["start_value"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.cumulative_returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        cumulative = func(
            self.to_2d_array(),
            start_value=start_value,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(cumulative, group_by=False, **resolve_dict(wrap_kwargs))

    def resample(
        self: ReturnsAccessorT,
        *args,
        fill_with_zero: bool = True,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> ReturnsAccessorT:
        """Resample returns in the accessor.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.
            fill_with_zero (bool): Flag indicating whether to fill missing values with zero.
            wrapper_meta (DictLike): Metadata from the resampling operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.

        Returns:
            ReturnsAccessor: New resampled returns accessor.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        new_wrapper = wrapper_meta["new_wrapper"]

        new_obj = self.resample_apply(
            wrapper_meta["resampler"],
            nb.total_return_1d_nb,
            self.log_returns,
        )
        if fill_with_zero:
            new_obj = new_obj.vbt.fillna(0.0)
        if self._bm_returns is not None:
            new_bm_returns = self.bm_returns.vbt.resample_apply(
                wrapper_meta["resampler"],
                nb.total_return_1d_nb,
                self.log_returns,
            )
            if fill_with_zero:
                new_bm_returns = new_bm_returns.vbt.fillna(0.0)
        else:
            new_bm_returns = None
        new_sim_start = self.resample_sim_start(new_wrapper)
        new_sim_end = self.resample_sim_end(new_wrapper)

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            obj=new_obj,
            bm_returns=new_bm_returns,
            sim_start=new_sim_start,
            sim_end=new_sim_end,
        )

    def resample_returns(
        self,
        rule: tp.AnyRuleLike,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Resample returns to a custom frequency, date offset, or index.

        Args:
            rule (AnyRuleLike): Resampler-like specification.

                See `ReturnsAccessor.resample_apply`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `ReturnsAccessor.resample_apply`.

        Returns:
            SeriesFrame: Resampled returns.

        See:
            `vectorbtpro.returns.nb.total_return_1d_nb`
        """
        checks.assert_instance_of(self.obj.index, (pd.DatetimeIndex, pd.PeriodIndex))

        func = jit_reg.resolve_option(nb.total_return_1d_nb, jitted)
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(
                args=ch.ArgsTaker(
                    None,
                )
            ),
        )
        return self.resample_apply(
            rule,
            func,
            self.log_returns,
            jitted=jitted,
            chunked=chunked,
            **kwargs,
        )

    def daily(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Calculate daily returns.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `ReturnsAccessor.resample_returns`.

        Returns:
            SeriesFrame: Daily returns.
        """
        return self.resample_returns("1D", jitted=jitted, chunked=chunked, **kwargs)

    def annual(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Calculate annual returns using the defined year frequency.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `ReturnsAccessor.resample_returns`.

        Returns:
            SeriesFrame: Annual returns.
        """
        return self.resample_returns(self.year_freq, jitted=jitted, chunked=chunked, **kwargs)

    # ############# Metrics ############# #

    def final_value(
        self,
        start_value: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate the final value of returns over a simulation period.

        Args:
            start_value (Optional[float]): Initial value for cumulative returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Final cumulative return value.

        See:
            `vectorbtpro.returns.nb.final_value_nb`
        """
        if start_value is None:
            start_value = self.defaults["start_value"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.final_value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            start_value=start_value,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="final_value"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_final_value(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        start_value: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute rolling final value.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            start_value (Optional[float]): Starting value for the calculation.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped array of rolling final values.

        See:
            `vectorbtpro.returns.nb.rolling_final_value_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if start_value is None:
            start_value = self.defaults["start_value"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_final_value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            start_value=start_value,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def total(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute total return.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed total return.

        See:
            `vectorbtpro.returns.nb.total_return_nb`
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.total_return_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="total_return"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_total(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute rolling total return.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Wrapped array of rolling total returns.

        See:
            `vectorbtpro.returns.nb.rolling_total_return_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_total_return_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def annualized(
        self,
        periods: tp.Union[None, str, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute annualized return.

        Args:
            periods (Union[None, str, ArrayLike]): Period specification.

                See `ReturnsAccessor.get_periods`.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed annualized return.

        See:
            `vectorbtpro.returns.nb.annualized_return_nb`
        """
        periods = self.get_periods(periods=periods, sim_start=sim_start, sim_end=sim_end)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.annualized_return_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            self.ann_factor,
            periods=periods,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="annualized_return"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_annualized(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling annualized return.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Computed rolling annualized return.

        See:
            `vectorbtpro.returns.nb.rolling_annualized_return_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_annualized_return_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            self.ann_factor,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def annualized_volatility(
        self,
        levy_alpha: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute annualized volatility.

        Args:
            levy_alpha (Optional[float]): Alpha parameter for Lévy correction.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed annualized volatility.

        See:
            `vectorbtpro.returns.nb.annualized_volatility_nb`
        """
        if levy_alpha is None:
            levy_alpha = self.defaults["levy_alpha"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.annualized_volatility_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            self.ann_factor,
            levy_alpha=levy_alpha,
            ddof=ddof,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="annualized_volatility"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_annualized_volatility(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        levy_alpha: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate rolling annualized volatility.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            levy_alpha (Optional[float]): Alpha parameter for Lévy correction.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Series of rolling annualized volatility values.

        See:
            `vectorbtpro.returns.nb.rolling_annualized_volatility_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if levy_alpha is None:
            levy_alpha = self.defaults["levy_alpha"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_annualized_volatility_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            self.ann_factor,
            levy_alpha=levy_alpha,
            ddof=ddof,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def calmar_ratio(
        self,
        periods: tp.Union[None, str, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate Calmar ratio.

        Args:
            periods (Union[None, str, ArrayLike]): Period specification.

                See `ReturnsAccessor.get_periods`.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Series of Calmar ratio values.

        See:
            `vectorbtpro.returns.nb.calmar_ratio_nb`
        """
        periods = self.get_periods(periods=periods, sim_start=sim_start, sim_end=sim_end)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.calmar_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            self.ann_factor,
            periods=periods,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="calmar_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_calmar_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate rolling Calmar ratio.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Series of rolling Calmar ratio values.

        See:
            `vectorbtpro.returns.nb.rolling_calmar_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_calmar_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            self.ann_factor,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def omega_ratio(
        self,
        risk_free: tp.Optional[float] = None,
        required_return: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate Omega ratio.

        Args:
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            required_return (Optional[float]): Required return threshold used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Series of Omega ratio values.

        See:
            `vectorbtpro.returns.nb.omega_ratio_nb`
        """
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if required_return is None:
            required_return = self.defaults["required_return"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.deannualized_return_nb, jitted)
        required_return = func(required_return, self.ann_factor)
        func = jit_reg.resolve_option(nb.omega_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - risk_free - required_return,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="omega_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_omega_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        risk_free: tp.Optional[float] = None,
        required_return: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate rolling Omega ratio.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            required_return (Optional[float]): Required return threshold used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Series of rolling Omega ratio values.

        See:
            `vectorbtpro.returns.nb.rolling_omega_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if required_return is None:
            required_return = self.defaults["required_return"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.deannualized_return_nb, jitted)
        required_return = func(required_return, self.ann_factor)
        func = jit_reg.resolve_option(nb.rolling_omega_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - risk_free - required_return,
            window,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def sharpe_ratio(
        self,
        annualized: bool = True,
        risk_free: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate Sharpe ratio.

        Args:
            annualized (bool): Whether to annualize the Sharpe ratio.
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Series of Sharpe ratio values.

        See:
            `vectorbtpro.returns.nb.sharpe_ratio_nb`
        """
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.sharpe_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        if annualized:
            ann_factor = self.ann_factor
        else:
            ann_factor = 1
        out = func(
            self.to_2d_array() - risk_free,
            ann_factor,
            ddof=ddof,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="sharpe_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_sharpe_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        risk_free: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        annualized: bool = True,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        stream_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate rolling Sharpe ratio.

        Computes the rolling Sharpe ratio over a specified window using excess returns relative
        to the risk-free rate.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            annualized (bool): Whether to annualize the Sharpe ratio.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            stream_mode (bool): Flag indicating whether to use stream mode.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Series representing the rolling Sharpe ratio.

        See:
            `vectorbtpro.returns.nb.rolling_sharpe_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_sharpe_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        if annualized:
            ann_factor = self.ann_factor
        else:
            ann_factor = 1
        out = func(
            self.to_2d_array() - risk_free,
            window,
            ann_factor,
            ddof=ddof,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
            stream_mode=stream_mode,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def sharpe_ratio_std(
        self,
        risk_free: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        bias: bool = True,
        wrap_kwargs: tp.KwargsLike = None,
    ):
        """Calculate the standard deviation of the Sharpe ratio estimation.

        Computes the standard deviation (or standard error) of the Sharpe ratio estimate using
        the sample's skewness and kurtosis, adjusted by the degrees of freedom.

        Args:
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bias (bool): Flag indicating whether to apply bias correction for skewness and kurtosis.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Standard deviation of the Sharpe ratio estimation.
        """
        from scipy import stats as scipy_stats

        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        returns = to_2d_array(self.obj)
        nanmask = np.isnan(returns)
        if nanmask.any():
            returns = returns.copy()
            returns[nanmask] = 0.0
        n = len(returns)
        skew = scipy_stats.skew(returns, axis=0, bias=bias)
        kurtosis = scipy_stats.kurtosis(returns, axis=0, bias=bias)
        sr = to_1d_array(self.sharpe_ratio(annualized=False, risk_free=risk_free, ddof=ddof))
        out = np.sqrt((1 + (0.5 * sr**2) - (skew * sr) + (((kurtosis - 3) / 4) * sr**2)) / (n - 1))
        wrap_kwargs = merge_dicts(dict(name_or_index="sharpe_ratio_std"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def prob_sharpe_ratio(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        risk_free: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        bias: bool = True,
        wrap_kwargs: tp.KwargsLike = None,
    ):
        """Calculate the probabilistic Sharpe Ratio (PSR).

        Evaluates the probability that the strategy's Sharpe ratio exceeds that of a benchmark.
        If benchmark returns are provided, the benchmark Sharpe ratio is computed for comparison.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bias (bool): Flag indicating whether to apply bias correction for skewness and kurtosis.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed probabilistic Sharpe ratio.
        """
        from scipy import stats as scipy_stats

        if bm_returns is None:
            bm_returns = self.bm_returns
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        if bm_returns is not None:
            bm_sr = to_1d_array(
                self.replace(obj=bm_returns, bm_returns=None).sharpe_ratio(
                    annualized=False,
                    risk_free=risk_free,
                    ddof=ddof,
                )
            )
        else:
            bm_sr = 0
        sr = to_1d_array(self.sharpe_ratio(annualized=False, risk_free=risk_free, ddof=ddof))
        sr_std = to_1d_array(self.sharpe_ratio_std(risk_free=risk_free, ddof=ddof, bias=bias))
        out = scipy_stats.norm.cdf((sr - bm_sr) / sr_std)
        wrap_kwargs = merge_dicts(dict(name_or_index="prob_sharpe_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def deflated_sharpe_ratio(
        self,
        risk_free: tp.Optional[float] = None,
        ddof: tp.Optional[int] = None,
        bias: bool = True,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate the deflated Sharpe Ratio (DSR).

        Adjusts the Sharpe ratio for bias and sample variability, expressing the probability that
        the strategy's true Sharpe ratio is positive.

        Args:
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bias (bool): Flag indicating whether to apply bias correction for skewness and kurtosis.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Deflated Sharpe ratio.
        """
        from scipy import stats as scipy_stats

        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        sharpe_ratio = to_1d_array(
            self.sharpe_ratio(annualized=False, risk_free=risk_free, ddof=ddof)
        )
        var_sharpe = np.nanvar(sharpe_ratio, ddof=ddof)
        returns = to_2d_array(self.obj)
        nanmask = np.isnan(returns)
        if nanmask.any():
            returns = returns.copy()
            returns[nanmask] = 0.0
        skew = scipy_stats.skew(returns, axis=0, bias=bias)
        kurtosis = scipy_stats.kurtosis(returns, axis=0, bias=bias)
        SR0 = sharpe_ratio + np.sqrt(var_sharpe) * (
            (1 - np.euler_gamma) * scipy_stats.norm.ppf(1 - 1 / self.wrapper.shape_2d[1])
            + np.euler_gamma * scipy_stats.norm.ppf(1 - 1 / (self.wrapper.shape_2d[1] * np.e))
        )
        out = scipy_stats.norm.cdf(
            ((sharpe_ratio - SR0) * np.sqrt(self.wrapper.shape_2d[0] - 1))
            / np.sqrt(1 - skew * sharpe_ratio + ((kurtosis - 1) / 4) * sharpe_ratio**2)
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="deflated_sharpe_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def downside_risk(
        self,
        required_return: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate downside risk.

        Computes the downside risk of the returns relative to a specified required return.

        Args:
            required_return (Optional[float]): Required return threshold used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Calculated downside risk.

        See:
            `vectorbtpro.returns.nb.downside_risk_nb`
        """
        if required_return is None:
            required_return = self.defaults["required_return"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.downside_risk_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - required_return,
            self.ann_factor,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="downside_risk"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_downside_risk(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        required_return: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling downside risk.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            required_return (Optional[float]): Required return threshold used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling downside risk series wrapped with the configured wrapper.

        See:
            `vectorbtpro.returns.nb.rolling_downside_risk_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if required_return is None:
            required_return = self.defaults["required_return"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_downside_risk_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - required_return,
            window,
            self.ann_factor,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def sortino_ratio(
        self,
        required_return: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute sortino ratio.

        Args:
            required_return (Optional[float]): Required return threshold used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Sortino ratio values wrapped with the configured wrapper.

        See:
            `vectorbtpro.returns.nb.sortino_ratio_nb`
        """
        if required_return is None:
            required_return = self.defaults["required_return"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.sortino_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - required_return,
            self.ann_factor,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="sortino_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_sortino_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        required_return: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling sortino ratio.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            required_return (Optional[float]): Required return threshold used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling sortino ratio series wrapped with the configured wrapper.

        See:
            `vectorbtpro.returns.nb.rolling_sortino_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if required_return is None:
            required_return = self.defaults["required_return"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_sortino_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - required_return,
            window,
            self.ann_factor,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def information_ratio(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute information ratio.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Information ratio series wrapped with the configured wrapper.

        See:
            `vectorbtpro.returns.nb.information_ratio_nb`
        """
        if ddof is None:
            ddof = self.defaults["ddof"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.information_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - to_2d_array(bm_returns),
            ddof=ddof,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="information_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_information_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling information ratio.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling information ratio series wrapped with the configured wrapper.

        See:
            `vectorbtpro.returns.nb.rolling_information_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_information_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - to_2d_array(bm_returns),
            window,
            ddof=ddof,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def beta(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute beta.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Beta values wrapped with the configured wrapper.

        See:
            `vectorbtpro.returns.nb.beta_nb`
        """
        if ddof is None:
            ddof = self.defaults["ddof"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.beta_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            ddof=ddof,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="beta"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_beta(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        ddof: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling beta values.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            ddof (Optional[int]): Delta degrees of freedom.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Computed rolling beta values.

        See:
            `vectorbtpro.returns.nb.rolling_beta_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if ddof is None:
            ddof = self.defaults["ddof"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_beta_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            window,
            ddof=ddof,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def alpha(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        risk_free: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute alpha values.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed alpha values.

        See:
            `vectorbtpro.returns.nb.alpha_nb`
        """
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.alpha_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - risk_free,
            to_2d_array(bm_returns) - risk_free,
            self.ann_factor,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="alpha"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_alpha(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        risk_free: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling alpha values.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            risk_free (Optional[float]): Risk-free return used to compute excess returns.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Computed rolling alpha values.

        See:
            `vectorbtpro.returns.nb.rolling_alpha_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if risk_free is None:
            risk_free = self.defaults["risk_free"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_alpha_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array() - risk_free,
            to_2d_array(bm_returns) - risk_free,
            window,
            self.ann_factor,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def tail_ratio(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        noarr_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute tail ratio values.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed tail ratio values.

        See:
            `vectorbtpro.returns.nb.tail_ratio_nb`
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.tail_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            sim_start=sim_start,
            sim_end=sim_end,
            noarr_mode=noarr_mode,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="tail_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_tail_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        noarr_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling tail ratio values.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Computed rolling tail ratio values.

        See:
            `vectorbtpro.returns.nb.rolling_tail_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_tail_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
            noarr_mode=noarr_mode,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def profit_factor(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the profit factor.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed profit factor values.

        See:
            `vectorbtpro.returns.nb.profit_factor_nb`
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.profit_factor_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="profit_factor"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_profit_factor(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling profit factor.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling profit factor.

        See:
            `vectorbtpro.returns.nb.rolling_profit_factor_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_profit_factor_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def common_sense_ratio(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute Common Sense Ratio (CSR).

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed Common Sense Ratio.

        See:
            `vectorbtpro.returns.nb.common_sense_ratio_nb`
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.common_sense_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="common_sense_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_common_sense_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling Common Sense Ratio (CSR) over a specified window.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling Common Sense Ratio.

        See:
            `vectorbtpro.returns.nb.rolling_common_sense_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_common_sense_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def value_at_risk(
        self,
        cutoff: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        noarr_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute Value at Risk (VaR).

        Args:
            cutoff (Optional[float]): Fractional cutoff level.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed Value at Risk.

        See:
            `vectorbtpro.returns.nb.value_at_risk_nb`
        """
        if cutoff is None:
            cutoff = self.defaults["cutoff"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.value_at_risk_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            cutoff=cutoff,
            sim_start=sim_start,
            sim_end=sim_end,
            noarr_mode=noarr_mode,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="value_at_risk"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_value_at_risk(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        cutoff: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        noarr_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute rolling Value at Risk (VaR) over a specified window.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            cutoff (Optional[float]): Fractional cutoff level.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling Value at Risk.

        See:
            `vectorbtpro.returns.nb.rolling_value_at_risk_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if cutoff is None:
            cutoff = self.defaults["cutoff"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_value_at_risk_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            cutoff=cutoff,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
            noarr_mode=noarr_mode,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def cond_value_at_risk(
        self,
        cutoff: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        noarr_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute Conditional Value at Risk (CVaR).

        Args:
            cutoff (Optional[float]): Fractional cutoff level.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed Conditional Value at Risk.

        See:
            `vectorbtpro.returns.nb.cond_value_at_risk_nb`
        """
        if cutoff is None:
            cutoff = self.defaults["cutoff"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.cond_value_at_risk_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            cutoff=cutoff,
            sim_start=sim_start,
            sim_end=sim_end,
            noarr_mode=noarr_mode,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="cond_value_at_risk"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_cond_value_at_risk(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        cutoff: tp.Optional[float] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        noarr_mode: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Rolling Conditional Value at Risk (CVaR).

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            cutoff (Optional[float]): Fractional cutoff level.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            noarr_mode (bool): Flag indicating whether to avoid allocating new arrays.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling CVaR values.

        See:
            `vectorbtpro.returns.nb.rolling_cond_value_at_risk_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if cutoff is None:
            cutoff = self.defaults["cutoff"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_cond_value_at_risk_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            cutoff=cutoff,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
            noarr_mode=noarr_mode,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def capture_ratio(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        periods: tp.Union[None, str, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the capture ratio.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            periods (Union[None, str, ArrayLike]): Period specification.

                See `ReturnsAccessor.get_periods`.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed capture ratio.

        See:
            `vectorbtpro.returns.nb.capture_ratio_nb`
        """
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        periods = self.get_periods(periods=periods, sim_start=sim_start, sim_end=sim_end)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.capture_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            self.ann_factor,
            periods=periods,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="capture_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_capture_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the rolling capture ratio.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling capture ratio values.

        See:
            `vectorbtpro.returns.nb.rolling_capture_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_capture_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            window,
            self.ann_factor,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def up_capture_ratio(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        periods: tp.Union[None, str, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the up-market capture ratio.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            periods (Union[None, str, ArrayLike]): Period specification.

                See `ReturnsAccessor.get_periods`.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Up-market capture ratio values.

        See:
            `vectorbtpro.returns.nb.up_capture_ratio_nb`
        """
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        periods = self.get_periods(periods=periods, sim_start=sim_start, sim_end=sim_end)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.up_capture_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            self.ann_factor,
            periods=periods,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="up_capture_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_up_capture_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the rolling up-market capture ratio.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling up-market capture ratio values.

        See:
            `vectorbtpro.returns.nb.rolling_up_capture_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_up_capture_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            window,
            self.ann_factor,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def down_capture_ratio(
        self,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        periods: tp.Union[None, str, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the down-market capture ratio against benchmark returns.

        Args:
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            periods (Union[None, str, ArrayLike]): Period specification.

                See `ReturnsAccessor.get_periods`.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Wrapped Series or DataFrame containing the down-market capture ratio.

        See:
            `vectorbtpro.returns.nb.down_capture_ratio_nb`
        """
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        periods = self.get_periods(periods=periods, sim_start=sim_start, sim_end=sim_end)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.down_capture_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            self.ann_factor,
            periods=periods,
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="down_capture_ratio"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_down_capture_ratio(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Compute the rolling down-market capture ratio over a specified window.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Defaults to `ReturnsAccessor.bm_returns` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling down-market capture ratio values.

        See:
            `vectorbtpro.returns.nb.rolling_down_capture_ratio_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        if bm_returns is None:
            bm_returns = self.bm_returns
        checks.assert_not_none(bm_returns, arg_name="bm_returns")
        bm_returns = broadcast_to(bm_returns, self.obj)
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_down_capture_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            to_2d_array(bm_returns),
            window,
            self.ann_factor,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def drawdown(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Compute the relative decline from a peak based on cumulative returns.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Series or DataFrame representing the relative drawdown.
        """
        return self.cumulative(
            start_value=1,
            sim_start=sim_start,
            sim_end=sim_end,
            jitted=jitted,
            chunked=chunked,
        ).vbt.drawdown(
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
        )

    def max_drawdown(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate the maximum drawdown (MDD) from returns, yielding the same output as
        `vectorbtpro.generic.drawdowns.Drawdowns.max_drawdown`.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Maximum drawdown values.

        See:
            `vectorbtpro.returns.nb.max_drawdown_nb`
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.max_drawdown_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            log_returns=self.log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="max_drawdown"), wrap_kwargs)
        return self.wrapper.wrap_reduced(out, group_by=False, **wrap_kwargs)

    def rolling_max_drawdown(
        self,
        window: tp.Optional[int] = None,
        *,
        minp: tp.Optional[int] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Calculate rolling maximum drawdown (MDD) over a specified window.

        Args:
            window (Optional[int]): Window size.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            minp (Optional[int]): Minimum number of observations required.

                Defaults to the value in `ReturnsAccessor.defaults` if not provided.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            MaybeSeries: Rolling maximum drawdown values.

        See:
            `vectorbtpro.returns.nb.rolling_max_drawdown_nb`
        """
        if window is None:
            window = self.defaults["window"]
        if minp is None:
            minp = self.defaults["minp"]
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        func = jit_reg.resolve_option(nb.rolling_max_drawdown_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        out = func(
            self.to_2d_array(),
            window,
            log_returns=self.log_returns,
            minp=minp,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return self.wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    def get_drawdowns(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> Drawdowns:
        """Generate drawdown records from cumulative returns.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.drawdowns.Drawdowns.from_price`.

        Returns:
            Drawdowns: Instance containing drawdown records computed from cumulative returns.
        """
        sim_start = self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = self.resolve_sim_end(sim_end=sim_end, group_by=False)

        return Drawdowns.from_price(
            self.cumulative(
                start_value=1.0,
                sim_start=sim_start,
                sim_end=sim_end,
                jitted=jitted,
            ),
            sim_start=sim_start,
            sim_end=sim_end,
            wrapper=self.wrapper,
            **kwargs,
        )

    @property
    def qs(self) -> QSAdapterT:
        """Quantstats adapter for performance analysis.

        Returns:
            QSAdapter: Instance of `vectorbtpro.returns.qs_adapter.QSAdapter`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("quantstats")
        from vectorbtpro.returns.qs_adapter import QSAdapter

        return QSAdapter(self)

    def resolve_self(
        self: ReturnsAccessorT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.Optional[tp.Set[str]] = None,
        impacts_caching: bool = True,
        silence_warnings: bool = False,
    ) -> ReturnsAccessorT:
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
        if "year_freq" in cond_kwargs:
            self_copy = reself.replace(year_freq=cond_kwargs["year_freq"])

            if self_copy.year_freq != reself.year_freq:
                if not silence_warnings:
                    warn(
                        "Changing the year frequency will create a copy of this object. "
                        "Consider setting it upon object creation to re-use existing cache."
                    )
                for alias in reself.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs["year_freq"] = self_copy.year_freq
                if impacts_caching:
                    cond_kwargs["use_caching"] = False
                return self_copy
        return reself

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `ReturnsAccessor.stats`.

        Merges the defaults from:

        * `vectorbtpro.generic.accessors.GenericAccessor.stats_defaults`
        * `ReturnsAccessor.defaults` (acting as settings)
        * `stats` configuration from `vectorbtpro._settings.returns`

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        returns_stats_cfg = settings["returns"]["stats"]

        return merge_dicts(
            GenericAccessor.stats_defaults.__get__(self),
            dict(settings=self.defaults),
            dict(settings=dict(year_freq=self.year_freq)),
            returns_stats_cfg,
        )

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func="sim_start_index",
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func="sim_end_index",
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func="sim_duration",
                apply_to_timedelta=True,
                tags="wrapper",
            ),
            total_return=dict(
                title="Total Return [%]",
                calc_func="total",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="returns",
            ),
            bm_return=dict(
                title="Benchmark Return [%]",
                calc_func="bm_returns_acc.total",
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_bm_returns=True,
                tags="returns",
            ),
            ann_return=dict(
                title="Annualized Return [%]",
                calc_func="annualized",
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            ann_volatility=dict(
                title="Annualized Volatility [%]",
                calc_func="annualized_volatility",
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            max_dd=dict(
                title="Max Drawdown [%]",
                calc_func="drawdowns.get_max_drawdown",
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=["returns", "drawdowns"],
            ),
            max_dd_duration=dict(
                title="Max Drawdown Duration",
                calc_func="drawdowns.get_max_duration",
                fill_wrap_kwargs=True,
                tags=["returns", "drawdowns", "duration"],
            ),
            sharpe_ratio=dict(
                title="Sharpe Ratio",
                calc_func="sharpe_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            calmar_ratio=dict(
                title="Calmar Ratio",
                calc_func="calmar_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            omega_ratio=dict(
                title="Omega Ratio",
                calc_func="omega_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            sortino_ratio=dict(
                title="Sortino Ratio",
                calc_func="sortino_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            skew=dict(
                title="Skew",
                calc_func="obj.skew",
                tags="returns",
            ),
            kurtosis=dict(
                title="Kurtosis",
                calc_func="obj.kurtosis",
                tags="returns",
            ),
            tail_ratio=dict(
                title="Tail Ratio",
                calc_func="tail_ratio",
                tags="returns",
            ),
            common_sense_ratio=dict(
                title="Common Sense Ratio",
                calc_func="common_sense_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags="returns",
            ),
            value_at_risk=dict(
                title="Value at Risk",
                calc_func="value_at_risk",
                tags="returns",
            ),
            alpha=dict(
                title="Alpha",
                calc_func="alpha",
                check_has_freq=True,
                check_has_year_freq=True,
                check_has_bm_returns=True,
                tags="returns",
            ),
            beta=dict(
                title="Beta",
                calc_func="beta",
                check_has_bm_returns=True,
                tags="returns",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_cumulative(
        self,
        column: tp.Optional[tp.Column] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        start_value: float = 1,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        fit_sim_range: bool = True,
        fill_to_benchmark: bool = False,
        main_kwargs: tp.KwargsLike = None,
        bm_kwargs: tp.KwargsLike = None,
        pct_scale: bool = False,
        hline_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot cumulative returns.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            bm_returns (Optional[ArrayLike]): Benchmark returns.

                Will be broadcast per element.
            start_value (float): Starting value for cumulative returns.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.

                May be an int, datetime-like, or array-like.
            sim_end (Optional[ArrayLike]): End index of the simulation range.

                May be an int, datetime-like, or array-like.
            fit_sim_range (bool): Flag indicating whether to fit the figure to the simulation range.
            fill_to_benchmark (bool): Whether to fill between the main plot and the benchmark plot or
                between the main plot and the start value.
            main_kwargs (KwargsLike): Keyword arguments for
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot` for the main plot.
            bm_kwargs (KwargsLike): Keyword arguments for
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot` for the benchmark plot.
            pct_scale (bool): Flag to display the y-axis on a percentage scale.
            hline_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for the horizontal line.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure displaying the cumulative returns plot.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> np.random.seed(0)
            >>> rets = pd.Series(np.random.uniform(-0.05, 0.05, size=100))
            >>> bm_returns = pd.Series(np.random.uniform(-0.05, 0.05, size=100))
            >>> rets.vbt.returns.plot_cumulative(bm_returns=bm_returns).show()
            ```

            ![](/assets/images/api/plot_cumulative.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/plot_cumulative.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import get_domain, make_figure

        plotting_cfg = settings["plotting"]

        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        def_layout_kwargs = {xaxis: {}, yaxis: {}}
        if pct_scale:
            start_value = 0
            def_layout_kwargs[yaxis]["tickformat"] = ".2%"
        if fig is None:
            fig = make_figure()
        fig.update_layout(**def_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        y_domain = get_domain(yref, fig)

        if bm_returns is None:
            bm_returns = self.bm_returns
        fill_to_benchmark = fill_to_benchmark and bm_returns is not None

        if bm_returns is not None:
            # Plot benchmark
            bm_returns = broadcast_to(bm_returns, self.obj)
            bm_returns = self.select_col_from_obj(bm_returns, column=column, group_by=False)
            if bm_kwargs is None:
                bm_kwargs = {}
            bm_kwargs = merge_dicts(
                dict(
                    trace_kwargs=dict(
                        line=dict(
                            color=plotting_cfg["color_schema"]["gray"],
                        ),
                        name="Benchmark",
                    )
                ),
                bm_kwargs,
            )
            bm_cumulative_returns = bm_returns.vbt.returns.cumulative(
                start_value=start_value,
                sim_start=sim_start,
                sim_end=sim_end,
            )
            bm_cumulative_returns.vbt.lineplot(
                **bm_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig
            )
        else:
            bm_cumulative_returns = None

        if main_kwargs is None:
            main_kwargs = {}
        cumulative_returns = self.cumulative(
            start_value=start_value,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        cumulative_returns = self.select_col_from_obj(
            cumulative_returns, column=column, group_by=False
        )
        main_kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["purple"],
                    ),
                ),
                other_trace_kwargs="hidden",
            ),
            main_kwargs,
        )
        if fill_to_benchmark:
            cumulative_returns.vbt.plot_against(
                bm_cumulative_returns, add_trace_kwargs=add_trace_kwargs, fig=fig, **main_kwargs
            )
        else:
            cumulative_returns.vbt.plot_against(
                start_value, add_trace_kwargs=add_trace_kwargs, fig=fig, **main_kwargs
            )

        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=start_value,
                    x1=x_domain[1],
                    y1=start_value,
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                group_by=False,
                xref=xref,
            )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `ReturnsAccessor.plots`.

        Merges the defaults from:

        * `vectorbtpro.generic.accessors.GenericAccessor.plots_defaults`
        * `ReturnsAccessor.defaults` (acting as settings)
        * `plots` configuration from `vectorbtpro._settings.returns`

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        returns_plots_cfg = settings["returns"]["plots"]

        return merge_dicts(
            GenericAccessor.plots_defaults.__get__(self),
            dict(settings=self.defaults),
            dict(settings=dict(year_freq=self.year_freq)),
            returns_plots_cfg,
        )

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot_cumulative=dict(
                title="Cumulative Returns",
                yaxis_kwargs=dict(title="Cumulative returns"),
                plot_func="plot_cumulative",
                pass_hline_shape_kwargs=True,
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags="returns",
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


ReturnsAccessor.override_metrics_doc(__pdoc__)
ReturnsAccessor.override_subplots_doc(__pdoc__)


@register_sr_vbt_accessor("returns")
class ReturnsSRAccessor(ReturnsAccessor, GenericSRAccessor):
    """Class representing an accessor on top of return series for Series only.

    Accessible via `pd.Series.vbt.returns`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying data for the Series.
        bm_returns (Optional[ArrayLike]): Benchmark returns.
        year_freq (Optional[FrequencyLike]): Year frequency for annualization (e.g., "252 days", "auto").
        defaults (KwargsLike): Dictionary of default parameters.
        sim_start (Optional[ArrayLike]): Start index of the simulation range.
        sim_end (Optional[ArrayLike]): End index of the simulation range.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericSRAccessor` and `ReturnsAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        defaults: tp.KwargsLike = None,
        sim_start: tp.Optional[tp.Array1d] = None,
        sim_end: tp.Optional[tp.Array1d] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        GenericSRAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            ReturnsAccessor.__init__(
                self,
                wrapper,
                obj=obj,
                bm_returns=bm_returns,
                year_freq=year_freq,
                defaults=defaults,
                sim_start=sim_start,
                sim_end=sim_end,
                **kwargs,
            )


@register_df_vbt_accessor("returns")
class ReturnsDFAccessor(ReturnsAccessor, GenericDFAccessor):
    """Class representing an accessor on top of return series for DataFrames only.

    Accessible via `pd.DataFrame.vbt.returns`.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Underlying data for the DataFrame.
        bm_returns (Optional[ArrayLike]): Benchmark returns.
        year_freq (Optional[FrequencyLike]): Year frequency for annualization (e.g., "252 days", "auto").
        defaults (KwargsLike): Dictionary of default parameters.
        sim_start (Optional[ArrayLike]): Start index of the simulation range.
        sim_end (Optional[ArrayLike]): End index of the simulation range.
        **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericDFAccessor` and `ReturnsAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        bm_returns: tp.Optional[tp.ArrayLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        defaults: tp.KwargsLike = None,
        sim_start: tp.Optional[tp.Array1d] = None,
        sim_end: tp.Optional[tp.Array1d] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        GenericDFAccessor.__init__(self, wrapper, obj=obj, _full_init=False, **kwargs)

        if _full_init:
            ReturnsAccessor.__init__(
                self,
                wrapper,
                obj=obj,
                bm_returns=bm_returns,
                year_freq=year_freq,
                defaults=defaults,
                sim_start=sim_start,
                sim_end=sim_end,
                **kwargs,
            )
