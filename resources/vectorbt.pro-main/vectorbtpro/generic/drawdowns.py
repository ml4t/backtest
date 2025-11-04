# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the base class for managing drawdown records.

Drawdown records capture critical information about drawdown intervals and are
implemented as a subclass of `vectorbtpro.generic.ranges.Ranges`.

!!! warning
    `Drawdowns` include both recovered and active drawdowns, which may skew performance
    results.

    To focus on recovered drawdowns only, explicitly use the `status_recovered` attribute.

Call `Drawdowns.from_price` to generate drawdown records from any time series for
immediate analysis.

```pycon
>>> from vectorbtpro import *

>>> price = vbt.YFData.pull(
...     "BTC-USD",
...     start="2019-10 UTC",
...     end="2020-01 UTC"
... ).get('Close')
```

[=100% "100%"]{: .candystripe .candystripe-animate }

```pycon
>>> price = price.rename(None)

>>> drawdowns = vbt.Drawdowns.from_price(price, wrapper_kwargs=dict(freq='d'))

>>> drawdowns.readable
   Drawdown Id  Column               Start Index              Valley Index  \\
0            0       0 2019-10-02 00:00:00+00:00 2019-10-06 00:00:00+00:00
1            1       0 2019-10-09 00:00:00+00:00 2019-10-24 00:00:00+00:00
2            2       0 2019-10-27 00:00:00+00:00 2019-12-17 00:00:00+00:00

                  End Index   Peak Value  Valley Value    End Value     Status
0 2019-10-09 00:00:00+00:00  8393.041992   7988.155762  8595.740234  Recovered
1 2019-10-25 00:00:00+00:00  8595.740234   7493.488770  8660.700195  Recovered
2 2019-12-31 00:00:00+00:00  9551.714844   6640.515137  7193.599121     Active

>>> drawdowns.duration.max(wrap_kwargs=dict(to_timedelta=True))
Timedelta('66 days 00:00:00')
```

## From accessors

Generic accessors provide a `drawdowns` property and a `get_drawdowns` method:

```pycon
>>> # vectorbtpro.generic.accessors.GenericAccessor.drawdowns.coverage
>>> price.vbt.drawdowns.coverage
0.967391304347826
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Drawdowns.metrics`.

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 2, 1, 3, 2],
...     'b': [2, 3, 1, 2, 1]
... })

>>> drawdowns = df.vbt(freq='d').drawdowns

>>> drawdowns['a'].stats()
Start                                        0
End                                          4
Period                         5 days 00:00:00
Coverage [%]                              80.0
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                2 days 00:00:00
Active Recovery [%]                        0.0
Active Recovery Return [%]                 0.0
Active Recovery Duration       0 days 00:00:00
Max Drawdown [%]                          50.0
Avg Drawdown [%]                          50.0
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          2 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                1.0
Name: a, dtype: object
```

By default, metrics `max_dd`, `avg_dd`, `max_dd_duration`, and `avg_dd_duration`
exclude active drawdowns. To include active drawdowns, pass `incl_active=True`:

```pycon
>>> drawdowns['a'].stats(settings=dict(incl_active=True))
Start                                        0
End                                          4
Period                         5 days 00:00:00
Coverage [%]                              80.0
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                2 days 00:00:00
Active Recovery [%]                        0.0
Active Recovery Return [%]                 0.0
Active Recovery Duration       0 days 00:00:00
Max Drawdown [%]                          50.0
Avg Drawdown [%]                     41.666667
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          2 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                1.0
Name: a, dtype: object
```

`Drawdowns.stats` also supports (re-)grouping:

```pycon
>>> drawdowns['a'].stats(group_by=True)
UserWarning: Metric 'active_dd' does not support grouped data
UserWarning: Metric 'active_duration' does not support grouped data
UserWarning: Metric 'active_recovery' does not support grouped data
UserWarning: Metric 'active_recovery_return' does not support grouped data
UserWarning: Metric 'active_recovery_duration' does not support grouped data

Start                                        0
End                                          4
Period                         5 days 00:00:00
Coverage [%]                              80.0
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Max Drawdown [%]                          50.0
Avg Drawdown [%]                          50.0
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          2 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                1.0
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Drawdowns.subplots`.

The `Drawdowns` class provides a single subplot through `Drawdowns.plot`:

```pycon
>>> drawdowns['a'].plots().show()
```

![](/assets/images/api/drawdowns_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/drawdowns_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb
from vectorbtpro.generic.enums import DrawdownStatus, drawdown_dt, range_dt
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.records.decorators import (
    attach_fields,
    attach_shortcut_properties,
    override_field_config,
)
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import Config, HybridConfig, ReadonlyConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.template import RepEval, RepFunc

__all__ = [
    "Drawdowns",
]

__pdoc__ = {}

dd_field_config = ReadonlyConfig(
    dict(
        dtype=drawdown_dt,
        settings=dict(
            id=dict(title="Drawdown Id"),
            valley_idx=dict(title="Valley Index", mapping="index"),
            start_val=dict(
                title="Start Value",
            ),
            valley_val=dict(
                title="Valley Value",
            ),
            end_val=dict(
                title="End Value",
            ),
            status=dict(mapping=DrawdownStatus),
        ),
    )
)
"""_"""

__pdoc__["dd_field_config"] = f"""Field configuration for `Drawdowns`.

```python
{dd_field_config.prettify_doc()}
```
"""

dd_attach_field_config = ReadonlyConfig(dict(status=dict(attach_filters=True)))
"""_"""

__pdoc__["dd_attach_field_config"] = f"""Field attachment configuration for `Drawdowns`.

```python
{dd_attach_field_config.prettify_doc()}
```
"""

dd_shortcut_config = ReadonlyConfig(
    dict(
        ranges=dict(),
        decline_ranges=dict(),
        recovery_ranges=dict(),
        drawdown=dict(obj_type="mapped_array"),
        avg_drawdown=dict(obj_type="red_array"),
        max_drawdown=dict(obj_type="red_array"),
        recovery_return=dict(obj_type="mapped_array"),
        avg_recovery_return=dict(obj_type="red_array"),
        max_recovery_return=dict(obj_type="red_array"),
        decline_duration=dict(obj_type="mapped_array"),
        recovery_duration=dict(obj_type="mapped_array"),
        recovery_duration_ratio=dict(obj_type="mapped_array"),
        active_drawdown=dict(obj_type="red_array"),
        active_duration=dict(obj_type="red_array"),
        active_recovery=dict(obj_type="red_array"),
        active_recovery_return=dict(obj_type="red_array"),
        active_recovery_duration=dict(obj_type="red_array"),
    )
)
"""_"""

__pdoc__["dd_shortcut_config"] = f"""Shortcut properties configuration for `Drawdowns`.

```python
{dd_shortcut_config.prettify_doc()}
```
"""

DrawdownsT = tp.TypeVar("DrawdownsT", bound="Drawdowns")


@attach_shortcut_properties(dd_shortcut_config)
@attach_fields(dd_attach_field_config)
@override_field_config(dd_field_config)
class Drawdowns(Ranges):
    """Class extending `vectorbtpro.generic.ranges.Ranges` for working with drawdown records.

    Requires `records_arr` to have all fields defined in `vectorbtpro.generic.enums.drawdown_dt`.

    !!! info
        For default settings, see `vectorbtpro._settings.drawdowns`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_price(
        cls: tp.Type[DrawdownsT],
        close: tp.ArrayLike,
        *,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        attach_data: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DrawdownsT:
        """Create a new `Drawdowns` instance from price data.

        Args:
            close (ArrayLike): Array of close prices.
            open (Optional[ArrayLike]): Array of open prices.
            high (Optional[ArrayLike]): Array of high prices.
            low (Optional[ArrayLike]): Array of low prices.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            attach_data (bool): Flag indicating whether to attach the OHLC data.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Drawdowns`.

        Returns:
            Drawdowns: New instance of `Drawdowns` created from the provided price data.

        See:
            `vectorbtpro.generic.nb.records.get_drawdowns_nb`
        """
        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        close_arr = to_2d_array(close)
        open_arr = to_2d_array(open) if open is not None else None
        high_arr = to_2d_array(high) if high is not None else None
        low_arr = to_2d_array(low) if low is not None else None

        func = jit_reg.resolve_option(nb.get_drawdowns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        records_arr = func(
            open=open_arr,
            high=high_arr,
            low=low_arr,
            close=close_arr,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        if wrapper is None:
            wrapper = ArrayWrapper.from_obj(close, **resolve_dict(wrapper_kwargs))
        elif len(wrapper_kwargs) > 0:
            wrapper = wrapper.replace(**wrapper_kwargs)
        return cls(
            wrapper,
            records_arr,
            open=open if attach_data else None,
            high=high if attach_data else None,
            low=low if attach_data else None,
            close=close if attach_data else None,
            **kwargs,
        )

    def get_ranges(self, **kwargs) -> Ranges:
        """Return peak-to-end range records.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: Instance of `Ranges` representing peak-to-end records.

                Has the `vectorbtpro.generic.enums.range_dt` dtype.
        """
        new_records_arr = np.empty(self.values.shape, dtype=range_dt)
        new_records_arr["id"][:] = self.get_field_arr("id").copy()
        new_records_arr["col"][:] = self.get_field_arr("col").copy()
        new_records_arr["start_idx"][:] = self.get_field_arr("start_idx").copy()
        new_records_arr["end_idx"][:] = self.get_field_arr("end_idx").copy()
        new_records_arr["status"][:] = self.get_field_arr("status").copy()
        return Ranges.from_records(
            self.wrapper,
            new_records_arr,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            **kwargs,
        )

    def get_decline_ranges(self, **kwargs) -> Ranges:
        """Return peak-to-valley range records.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: Instance of `Ranges` representing peak-to-valley records.

                Has the `vectorbtpro.generic.enums.range_dt` dtype.
        """
        new_records_arr = np.empty(self.values.shape, dtype=range_dt)
        new_records_arr["id"][:] = self.get_field_arr("id").copy()
        new_records_arr["col"][:] = self.get_field_arr("col").copy()
        new_records_arr["start_idx"][:] = self.get_field_arr("start_idx").copy()
        new_records_arr["end_idx"][:] = self.get_field_arr("valley_idx").copy()
        new_records_arr["status"][:] = self.get_field_arr("status").copy()
        return Ranges.from_records(
            self.wrapper,
            new_records_arr,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            **kwargs,
        )

    def get_recovery_ranges(self, **kwargs) -> Ranges:
        """Return valley-to-end range records.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.generic.ranges.Ranges.from_records`.

        Returns:
            Ranges: Instance of `Ranges` representing valley-to-end records.

                Has the `vectorbtpro.generic.enums.range_dt` dtype.
        """
        new_records_arr = np.empty(self.values.shape, dtype=range_dt)
        new_records_arr["id"][:] = self.get_field_arr("id").copy()
        new_records_arr["col"][:] = self.get_field_arr("col").copy()
        new_records_arr["start_idx"][:] = self.get_field_arr("valley_idx").copy()
        new_records_arr["end_idx"][:] = self.get_field_arr("end_idx").copy()
        new_records_arr["status"][:] = self.get_field_arr("status").copy()
        return Ranges.from_records(
            self.wrapper,
            new_records_arr,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            **kwargs,
        )

    # ############# Drawdown ############# #

    def get_drawdown(
        self, jitted: tp.JittedOption = None, chunked: tp.ChunkedOption = None, **kwargs
    ) -> MappedArray:
        """Return the drawdown values.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Drawdowns.map_array`.

        Returns:
            MappedArray: Array of computed drawdown values.

        See:
            `vectorbtpro.generic.nb.records.dd_drawdown_nb`

        !!! note
            Both recovered and active drawdowns are considered.
        """
        func = jit_reg.resolve_option(nb.dd_drawdown_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        drawdown = func(self.get_field_arr("start_val"), self.get_field_arr("valley_val"))
        return self.map_array(drawdown, **kwargs)

    def get_avg_drawdown(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the average drawdown computed from `Drawdowns.drawdown`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.mean`.

        Returns:
            MaybeSeries: Average drawdown value.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="avg_drawdown"), wrap_kwargs)
        return self.drawdown.mean(
            group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs
        )

    def get_max_drawdown(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the maximum drawdown computed from `Drawdowns.drawdown`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.min`.

        Returns:
            MaybeSeries: Maximum drawdown value.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="max_drawdown"), wrap_kwargs)
        return self.drawdown.min(
            group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs
        )

    # ############# Recovery ############# #

    def get_recovery_return(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Return the recovery return values.

        !!! note
            Both recovered and active drawdowns are considered.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Drawdowns.map_array`.

        Returns:
            MappedArray: Array containing the computed recovery return values.

        See:
            `vectorbtpro.generic.nb.records.dd_recovery_return_nb`
        """
        func = jit_reg.resolve_option(nb.dd_recovery_return_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        recovery_return = func(self.get_field_arr("valley_val"), self.get_field_arr("end_val"))
        return self.map_array(recovery_return, **kwargs)

    def get_avg_recovery_return(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the average recovery return computed from `Drawdowns.recovery_return`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.mean`.

        Returns:
            MaybeSeries: Average recovery return.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="avg_recovery_return"), wrap_kwargs)
        return self.recovery_return.mean(
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_max_recovery_return(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the maximum recovery return computed from `Drawdowns.recovery_return`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.max`.

        Returns:
            MaybeSeries: Maximum recovery return.
        """
        wrap_kwargs = merge_dicts(dict(name_or_index="max_recovery_return"), wrap_kwargs)
        return self.recovery_return.max(
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    # ############# Duration ############# #

    def get_decline_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Compute the decline duration.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Drawdowns.map_array`.

        Returns:
            MappedArray: Computed decline duration.

        See:
            `vectorbtpro.generic.nb.records.dd_decline_duration_nb`

        !!! note
            Calculation accounts for both recovered and active drawdowns.
        """
        func = jit_reg.resolve_option(nb.dd_decline_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        decline_duration = func(self.get_field_arr("start_idx"), self.get_field_arr("valley_idx"))
        return self.map_array(decline_duration, **kwargs)

    def get_recovery_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Compute the recovery duration.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Drawdowns.map_array`.

        Returns:
            MappedArray: Computed recovery duration.

        See:
            `vectorbtpro.generic.nb.records.dd_recovery_duration_nb`

        !!! note
            A value higher than 1 indicates that recovery was slower than the decline.
            Calculation accounts for both recovered and active drawdowns.
        """
        func = jit_reg.resolve_option(nb.dd_recovery_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        recovery_duration = func(self.get_field_arr("valley_idx"), self.get_field_arr("end_idx"))
        return self.map_array(recovery_duration, **kwargs)

    def get_recovery_duration_ratio(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Compute the recovery duration ratio.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Drawdowns.map_array`.

        Returns:
            MappedArray: Computed recovery duration ratio.

        See:
            `vectorbtpro.generic.nb.records.dd_recovery_duration_ratio_nb`

        !!! note
            Calculation accounts for both recovered and active drawdowns.
        """
        func = jit_reg.resolve_option(nb.dd_recovery_duration_ratio_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        recovery_duration_ratio = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("valley_idx"),
            self.get_field_arr("end_idx"),
        )
        return self.map_array(recovery_duration_ratio, **kwargs)

    # ############# Status: Active ############# #

    def get_active_drawdown(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the drawdown of the last active drawdown.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Active drawdown.

        !!! note
            Grouping is not supported by this method.
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(dict(name_or_index="active_drawdown"), wrap_kwargs)
        active = self.status_active
        curr_end_val = active.end_val.nth(-1, group_by=group_by, jitted=jitted, chunked=chunked)
        curr_start_val = active.start_val.nth(-1, group_by=group_by, jitted=jitted, chunked=chunked)
        curr_drawdown = (curr_end_val - curr_start_val) / curr_start_val
        return self.wrapper.wrap_reduced(curr_drawdown, group_by=group_by, **wrap_kwargs)

    def get_active_duration(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the duration of the last active drawdown.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.nth`.

        Returns:
            MaybeSeries: Duration of the active drawdown.

        !!! note
            Grouping is not supported by this method.
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(
            dict(to_timedelta=True, name_or_index="active_duration"), wrap_kwargs
        )
        return self.status_active.duration.nth(
            -1,
            jitted=jitted,
            chunked=chunked,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_active_recovery(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the recovery of the last active drawdown.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Active recovery.

        !!! note
            Grouping is not supported by this method.
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(dict(name_or_index="active_recovery"), wrap_kwargs)
        active = self.status_active
        curr_start_val = active.start_val.nth(-1, group_by=group_by, jitted=jitted, chunked=chunked)
        curr_end_val = active.end_val.nth(-1, group_by=group_by, jitted=jitted, chunked=chunked)
        curr_valley_val = active.valley_val.nth(
            -1, group_by=group_by, jitted=jitted, chunked=chunked
        )
        curr_recovery = (curr_end_val - curr_valley_val) / (curr_start_val - curr_valley_val)
        return self.wrapper.wrap_reduced(curr_recovery, group_by=group_by, **wrap_kwargs)

    def get_active_recovery_return(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the recovery return of the last active drawdown.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.nth`.

        Returns:
            MaybeSeries: Recovery return of the active drawdown.

        !!! note
            Grouping is not supported by this method.
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(dict(name_or_index="active_recovery_return"), wrap_kwargs)
        return self.status_active.recovery_return.nth(
            -1,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    def get_active_recovery_duration(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the recovery duration of the last active drawdown.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.records.mapped_array.MappedArray.nth`.

        Returns:
            MaybeSeries: Recovery duration of the active drawdown.

        !!! note
            Grouping is not supported by this method.
        """
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(
            dict(to_timedelta=True, name_or_index="active_recovery_duration"), wrap_kwargs
        )
        return self.status_active.recovery_duration.nth(
            -1,
            group_by=group_by,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        )

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `Drawdowns.stats`.

        Merges the defaults from `vectorbtpro.generic.ranges.Ranges.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.drawdowns`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        drawdowns_stats_cfg = settings["drawdowns"]["stats"]

        return merge_dicts(Ranges.stats_defaults.__get__(self), drawdowns_stats_cfg)

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
            coverage=dict(
                title="Coverage [%]",
                calc_func="coverage",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=["ranges", "duration"],
            ),
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            total_recovered=dict(
                title="Total Recovered Drawdowns",
                calc_func="status_recovered.count",
                tags="drawdowns",
            ),
            total_active=dict(
                title="Total Active Drawdowns", calc_func="status_active.count", tags="drawdowns"
            ),
            active_dd=dict(
                title="Active Drawdown [%]",
                calc_func="active_drawdown",
                post_calc_func=lambda self, out, settings: -out * 100,
                check_is_not_grouped=True,
                tags=["drawdowns", "active"],
            ),
            active_duration=dict(
                title="Active Duration",
                calc_func="active_duration",
                fill_wrap_kwargs=True,
                check_is_not_grouped=True,
                tags=["drawdowns", "active", "duration"],
            ),
            active_recovery=dict(
                title="Active Recovery [%]",
                calc_func="active_recovery",
                post_calc_func=lambda self, out, settings: out * 100,
                check_is_not_grouped=True,
                tags=["drawdowns", "active"],
            ),
            active_recovery_return=dict(
                title="Active Recovery Return [%]",
                calc_func="active_recovery_return",
                post_calc_func=lambda self, out, settings: out * 100,
                check_is_not_grouped=True,
                tags=["drawdowns", "active"],
            ),
            active_recovery_duration=dict(
                title="Active Recovery Duration",
                calc_func="active_recovery_duration",
                fill_wrap_kwargs=True,
                check_is_not_grouped=True,
                tags=["drawdowns", "active", "duration"],
            ),
            max_dd=dict(
                title="Max Drawdown [%]",
                calc_func=RepEval(
                    "'max_drawdown' if incl_active else 'status_recovered.get_max_drawdown'"
                ),
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=RepEval("['drawdowns'] if incl_active else ['drawdowns', 'recovered']"),
            ),
            avg_dd=dict(
                title="Avg Drawdown [%]",
                calc_func=RepEval(
                    "'avg_drawdown' if incl_active else 'status_recovered.get_avg_drawdown'"
                ),
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=RepEval("['drawdowns'] if incl_active else ['drawdowns', 'recovered']"),
            ),
            max_dd_duration=dict(
                title="Max Drawdown Duration",
                calc_func=RepEval(
                    "'max_duration' if incl_active else 'status_recovered.get_max_duration'"
                ),
                fill_wrap_kwargs=True,
                tags=RepEval(
                    "['drawdowns', 'duration'] if incl_active else ['drawdowns', 'recovered', 'duration']"
                ),
            ),
            avg_dd_duration=dict(
                title="Avg Drawdown Duration",
                calc_func=RepEval(
                    "'avg_duration' if incl_active else 'status_recovered.get_avg_duration'"
                ),
                fill_wrap_kwargs=True,
                tags=RepEval(
                    "['drawdowns', 'duration'] if incl_active else ['drawdowns', 'recovered', 'duration']"
                ),
            ),
            max_return=dict(
                title="Max Recovery Return [%]",
                calc_func="status_recovered.recovery_return.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=["drawdowns", "recovered"],
            ),
            avg_return=dict(
                title="Avg Recovery Return [%]",
                calc_func="status_recovered.recovery_return.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=["drawdowns", "recovered"],
            ),
            max_recovery_duration=dict(
                title="Max Recovery Duration",
                calc_func="status_recovered.recovery_duration.max",
                apply_to_timedelta=True,
                tags=["drawdowns", "recovered", "duration"],
            ),
            avg_recovery_duration=dict(
                title="Avg Recovery Duration",
                calc_func="status_recovered.recovery_duration.mean",
                apply_to_timedelta=True,
                tags=["drawdowns", "recovered", "duration"],
            ),
            recovery_duration_ratio=dict(
                title="Avg Recovery Duration Ratio",
                calc_func="status_recovered.recovery_duration_ratio.mean",
                tags=["drawdowns", "recovered"],
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
        top_n: tp.Optional[int] = 5,
        plot_ohlc: bool = True,
        plot_close: bool = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        peak_trace_kwargs: tp.KwargsLike = None,
        valley_trace_kwargs: tp.KwargsLike = None,
        recovery_trace_kwargs: tp.KwargsLike = None,
        active_trace_kwargs: tp.KwargsLike = None,
        decline_shape_kwargs: tp.KwargsLike = None,
        recovery_shape_kwargs: tp.KwargsLike = None,
        active_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot drawdowns.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            top_n (Optional[int]): Display only the top N records sorted by maximum duration.
            plot_ohlc (bool): Whether to plot the OHLC data.
            plot_close (bool): Whether to plot the close price if OHLC data is not plotted.
            plot_markers (bool): Plot markers for significant drawdown events.
            plot_zones (bool): Plot zones delineating drawdown phases.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
            peak_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for peak markers.
            valley_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for valley markers.
            recovery_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for recovery markers.
            active_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for active recovery markers.
            decline_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for shapes representing decline zones.
            recovery_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for shapes representing recovery zones.
            active_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for shapes representing active recovery zones.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object containing the plotted drawdowns and price data.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> index = pd.date_range("2020", periods=8)
            >>> price = pd.Series([1, 2, 1, 2, 3, 2, 1, 2], index=index)
            >>> vbt.Drawdowns.from_price(price, wrapper_kwargs=dict(freq='1 day')).plot().show()
            ```

            ![](/assets/images/api/drawdowns_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/drawdowns_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import get_domain, make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        if top_n is not None:
            # Drawdowns is negative, thus top_n becomes bottom_n
            self_col = self_col.apply_mask(self_col.drawdown.bottom_n_mask(top_n))

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if peak_trace_kwargs is None:
            peak_trace_kwargs = {}
        if valley_trace_kwargs is None:
            valley_trace_kwargs = {}
        if recovery_trace_kwargs is None:
            recovery_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if decline_shape_kwargs is None:
            decline_shape_kwargs = {}
        if recovery_shape_kwargs is None:
            recovery_shape_kwargs = {}
        if active_shape_kwargs is None:
            active_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        y_domain = get_domain(yref, fig)

        plotting_ohlc = False
        if (
            plot_ohlc
            and self_col._open is not None
            and self_col._high is not None
            and self_col._low is not None
            and self_col._close is not None
        ):
            plotting_ohlc = True
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and self_col._close is not None:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            id_ = self_col.get_field_arr("id")
            start_idx = self_col.get_map_field_to_index("start_idx")
            if not plotting_ohlc and self_col._close is not None:
                start_val = self_col.close.loc[start_idx]
            else:
                start_val = self_col.get_field_arr("start_val")
            valley_idx = self_col.get_map_field_to_index("valley_idx")
            if not plotting_ohlc and self_col._close is not None:
                valley_val = self_col.close.loc[valley_idx]
            else:
                valley_val = self_col.get_field_arr("valley_val")
            end_idx = self_col.get_map_field_to_index("end_idx")
            if not plotting_ohlc and self_col._close is not None:
                end_val = self_col.close.loc[end_idx]
            else:
                end_val = self_col.get_field_arr("end_val")
            drawdown = self_col.drawdown.values
            recovery_return = self_col.recovery_return.values
            status = self_col.get_field_arr("status")

            decline_duration = to_1d_array(
                self_col.wrapper.arr_to_timedelta(
                    self_col.decline_duration.values, to_pd=True, silence_warnings=True
                ).astype(str)
            )
            recovery_duration = to_1d_array(
                self_col.wrapper.arr_to_timedelta(
                    self_col.recovery_duration.values, to_pd=True, silence_warnings=True
                ).astype(str)
            )
            duration = to_1d_array(
                self_col.wrapper.arr_to_timedelta(
                    self_col.duration.values, to_pd=True, silence_warnings=True
                ).astype(str)
            )

            # Peak and recovery at same time -> recovery wins
            peak_mask = (start_val != np.roll(end_val, 1)) | (start_idx != np.roll(end_idx, 1))
            if peak_mask.any():
                if plot_markers:
                    # Plot peak markers
                    peak_customdata, peak_hovertemplate = self_col.prepare_customdata(
                        incl_fields=["id", "start_idx", "start_val"], mask=peak_mask
                    )
                    _peak_trace_kwargs = merge_dicts(
                        dict(
                            x=start_idx[peak_mask],
                            y=start_val[peak_mask],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                color=plotting_cfg["contrast_color_schema"]["blue"],
                                size=7,
                                line=dict(
                                    width=1,
                                    color=adjust_lightness(
                                        plotting_cfg["contrast_color_schema"]["blue"]
                                    ),
                                ),
                            ),
                            name="Peak",
                            customdata=peak_customdata,
                            hovertemplate=peak_hovertemplate,
                        ),
                        peak_trace_kwargs,
                    )
                    peak_scatter = go.Scatter(**_peak_trace_kwargs)
                    fig.add_trace(peak_scatter, **add_trace_kwargs)

            recovered_mask = status == DrawdownStatus.Recovered
            if recovered_mask.any():
                if plot_markers:
                    # Plot valley markers
                    valley_customdata, valley_hovertemplate = self_col.prepare_customdata(
                        incl_fields=["id", "valley_idx", "valley_val"],
                        append_info=[
                            (drawdown, "Drawdown", "$title: %{customdata[$index]:,%}"),
                            (decline_duration, "Decline duration"),
                        ],
                        mask=recovered_mask,
                    )
                    _valley_trace_kwargs = merge_dicts(
                        dict(
                            x=valley_idx[recovered_mask],
                            y=valley_val[recovered_mask],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                color=plotting_cfg["contrast_color_schema"]["red"],
                                size=7,
                                line=dict(
                                    width=1,
                                    color=adjust_lightness(
                                        plotting_cfg["contrast_color_schema"]["red"]
                                    ),
                                ),
                            ),
                            name="Valley",
                            customdata=valley_customdata,
                            hovertemplate=valley_hovertemplate,
                        ),
                        valley_trace_kwargs,
                    )
                    valley_scatter = go.Scatter(**_valley_trace_kwargs)
                    fig.add_trace(valley_scatter, **add_trace_kwargs)

                if plot_markers:
                    # Plot recovery markers
                    recovery_customdata, recovery_hovertemplate = self_col.prepare_customdata(
                        incl_fields=["id", "end_idx", "end_val"],
                        append_info=[
                            (drawdown, "Drawdown", "$title: %{customdata[$index]:,%}"),
                            (duration, "Drawdown duration"),
                            (
                                recovery_return,
                                "Recovery return",
                                "$title: %{customdata[$index]:,%}",
                            ),
                            (recovery_duration, "Recovery duration"),
                        ],
                        mask=recovered_mask,
                    )
                    _recovery_trace_kwargs = merge_dicts(
                        dict(
                            x=end_idx[recovered_mask],
                            y=end_val[recovered_mask],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                color=plotting_cfg["contrast_color_schema"]["green"],
                                size=7,
                                line=dict(
                                    width=1,
                                    color=adjust_lightness(
                                        plotting_cfg["contrast_color_schema"]["green"]
                                    ),
                                ),
                            ),
                            name="Recovery/Peak",
                            customdata=recovery_customdata,
                            hovertemplate=recovery_hovertemplate,
                        ),
                        recovery_trace_kwargs,
                    )
                    recovery_scatter = go.Scatter(**_recovery_trace_kwargs)
                    fig.add_trace(recovery_scatter, **add_trace_kwargs)

            active_mask = status == DrawdownStatus.Active
            if active_mask.any():
                if plot_markers:
                    # Plot active markers
                    active_customdata, active_hovertemplate = self_col.prepare_customdata(
                        incl_fields=["id"],
                        append_info=[
                            (drawdown, "Drawdown", "$title: %{customdata[$index]:,%}"),
                            (duration, "Drawdown duration"),
                        ],
                        mask=active_mask,
                    )
                    _active_trace_kwargs = merge_dicts(
                        dict(
                            x=end_idx[active_mask],
                            y=end_val[active_mask],
                            mode="markers",
                            marker=dict(
                                symbol="diamond",
                                color=plotting_cfg["contrast_color_schema"]["orange"],
                                size=7,
                                line=dict(
                                    width=1,
                                    color=adjust_lightness(
                                        plotting_cfg["contrast_color_schema"]["orange"]
                                    ),
                                ),
                            ),
                            name="Active",
                            customdata=active_customdata,
                            hovertemplate=active_hovertemplate,
                        ),
                        active_trace_kwargs,
                    )
                    active_scatter = go.Scatter(**_active_trace_kwargs)
                    fig.add_trace(active_scatter, **add_trace_kwargs)

            if plot_zones:
                # Plot drawdown zones
                self_col.status_recovered.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(
                            x0=RepFunc(lambda i: start_idx[recovered_mask][i]),
                            x1=RepFunc(lambda i: valley_idx[recovered_mask][i]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["red"],
                        ),
                        decline_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

                # Plot recovery zones
                self_col.status_recovered.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(
                            x0=RepFunc(lambda i: valley_idx[recovered_mask][i]),
                            x1=RepFunc(lambda i: end_idx[recovered_mask][i]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["green"],
                        ),
                        recovery_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

                # Plot active drawdown zones
                self_col.status_active.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(
                            x0=RepFunc(lambda i: start_idx[active_mask][i]),
                            x1=RepFunc(lambda i: end_idx[active_mask][i]),
                            fillcolor=plotting_cfg["contrast_color_schema"]["orange"],
                        ),
                        active_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `Drawdowns.plots`.

        Merges the defaults from `vectorbtpro.generic.ranges.Ranges.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.drawdowns`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        drawdowns_plots_cfg = settings["drawdowns"]["plots"]

        return merge_dicts(Ranges.plots_defaults.__get__(self), drawdowns_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Drawdowns",
                check_is_not_grouped=True,
                plot_func="plot",
                tags="drawdowns",
            )
        ),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Drawdowns.override_field_config_doc(__pdoc__)
Drawdowns.override_metrics_doc(__pdoc__)
Drawdowns.override_subplots_doc(__pdoc__)
