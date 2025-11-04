# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module for working with range records.

Range records capture information on time intervals. They are useful for analyzing durations
of processes, such as drawdowns, trades, and positions, as well as evaluating time between events
(e.g., entry and exit signals).

Each range has a starting and an ending point. For example, `range(20)` yields a start of 0 and
an end of 20 (not 19).

!!! note
    If a range remains open in a column, its `end_idx` points to the latest index.
    Ensure you account for this when calculating custom duration metrics.

```pycon
>>> from vectorbtpro import *

>>> start = '2019-01-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> price = vbt.YFData.pull('BTC-USD', start=start, end=end).get('Close')
```

[=100% "100%"]{: .candystripe .candystripe-animate }

```pycon
>>> fast_ma = vbt.MA.run(price, 10)
>>> slow_ma = vbt.MA.run(price, 50)
>>> fast_below_slow = fast_ma.ma_above(slow_ma)

>>> ranges = vbt.Ranges.from_array(fast_below_slow, wrapper_kwargs=dict(freq='d'))

>>> ranges.readable
   Range Id  Column           Start Timestamp             End Timestamp  \\
0         0       0 2019-02-19 00:00:00+00:00 2019-07-25 00:00:00+00:00
1         1       0 2019-08-08 00:00:00+00:00 2019-08-19 00:00:00+00:00
2         2       0 2019-11-01 00:00:00+00:00 2019-11-20 00:00:00+00:00

   Status
0  Closed
1  Closed
2  Closed

>>> ranges.duration.max(wrap_kwargs=dict(to_timedelta=True))
Timedelta('156 days 00:00:00')
```

## From accessors

All generic accessors have a `ranges` property and a `get_ranges` method:

```pycon
>>> # vectorbtpro.generic.accessors.GenericAccessor.ranges.coverage
>>> fast_below_slow.vbt.ranges.coverage
0.5081967213114754
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Ranges.metrics`.

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 2, np.nan, np.nan, 5, 6],
...     'b': [np.nan, 2, np.nan, 4, np.nan, 6]
... })
>>> ranges = df.vbt(freq='d').ranges

>>> ranges['a'].stats()
Start                             0
End                               5
Period              6 days 00:00:00
Total Records                     2
Coverage                   0.666667
Overlap Coverage                0.0
Duration: Min       2 days 00:00:00
Duration: Median    2 days 00:00:00
Duration: Max       2 days 00:00:00
Name: a, dtype: object
```

`Ranges.stats` also supports re-grouping:

```pycon
>>> ranges.stats(group_by=True)
Start                                       0
End                                         5
Period                        6 days 00:00:00
Total Records                               5
Coverage                             0.416667
Overlap Coverage                          0.4
Duration: Min                 1 days 00:00:00
Duration: Median              1 days 00:00:00
Duration: Max                 2 days 00:00:00
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Ranges.subplots`.

The `Ranges` class provides a single subplot via `Ranges.plot`:

```pycon
>>> ranges['a'].plots().show()
```

![](/assets/images/api/ranges_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/ranges_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexes import combine_indexes, stack_indexes, tile_index
from vectorbtpro.base.reshaping import tile, to_1d_array, to_2d_array, to_pd_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import enums, nb
from vectorbtpro.generic.price_records import PriceRecords
from vectorbtpro.records.base import Records
from vectorbtpro.records.decorators import (
    attach_fields,
    attach_shortcut_properties,
    override_field_config,
)
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.attr_ import MISSING, DefineMixin, define
from vectorbtpro.utils.colors import adjust_lightness
from vectorbtpro.utils.config import Config, HybridConfig, ReadonlyConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.params import Param, combine_params
from vectorbtpro.utils.parsing import get_func_kwargs
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import substitute_templates
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "Ranges",
    "PatternRanges",
    "PSC",
]

__pdoc__ = {}

# ############# Ranges ############# #

ranges_field_config = ReadonlyConfig(
    dict(
        dtype=enums.range_dt,
        settings=dict(
            id=dict(title="Range Id"),
            idx=dict(name="end_idx"),  # remap field of Records
            start_idx=dict(title="Start Index", mapping="index"),
            end_idx=dict(title="End Index", mapping="index"),
            status=dict(title="Status", mapping=enums.RangeStatus),
        ),
    )
)
"""_"""

__pdoc__["ranges_field_config"] = f"""Field configuration for `Ranges`.

```python
{ranges_field_config.prettify_doc()}
```
"""

ranges_attach_field_config = ReadonlyConfig(dict(status=dict(attach_filters=True)))
"""_"""

__pdoc__["ranges_attach_field_config"] = f"""Configuration for fields to be attached to `Ranges`.

```python
{ranges_attach_field_config.prettify_doc()}
```
"""

ranges_shortcut_config = ReadonlyConfig(
    dict(
        valid=dict(),
        invalid=dict(),
        first_pd_mask=dict(obj_type="array"),
        last_pd_mask=dict(obj_type="array"),
        ranges_pd_mask=dict(obj_type="array"),
        first_idx=dict(obj_type="mapped_array"),
        last_idx=dict(obj_type="mapped_array"),
        duration=dict(obj_type="mapped_array"),
        real_duration=dict(obj_type="mapped_array"),
        avg_duration=dict(obj_type="red_array"),
        max_duration=dict(obj_type="red_array"),
        coverage=dict(obj_type="red_array"),
        overlap_coverage=dict(
            method_name="get_coverage", obj_type="red_array", method_kwargs=dict(overlapping=True)
        ),
        projections=dict(obj_type="array"),
    )
)
"""_"""

__pdoc__["ranges_shortcut_config"] = f"""Configuration for shortcut properties attached to `Ranges`.

```python
{ranges_shortcut_config.prettify_doc()}
```
"""

RangesT = tp.TypeVar("RangesT", bound="Ranges")


@attach_shortcut_properties(ranges_shortcut_config)
@attach_fields(ranges_attach_field_config)
@override_field_config(ranges_field_config)
class Ranges(PriceRecords):
    """Class for handling range records that extends `vectorbtpro.generic.price_records.PriceRecords`.

    Requires `records_arr` to have all fields defined in `vectorbtpro.generic.enums.range_dt`.

    !!! info
        For default settings, see `vectorbtpro._settings.ranges`.
    """

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_array(
        cls: tp.Type[RangesT],
        arr: tp.ArrayLike,
        gap_value: tp.Optional[tp.Scalar] = None,
        attach_as_close: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RangesT:
        """Build `Ranges` from an array.

        Constructs a `Ranges` instance by identifying consecutive sequences of valid data points.

        It searches for sequences of:

        * True values in boolean arrays (with False as a gap),
        * Positive integers in integer arrays (with -1 as a gap), and
        * Non-NaN values in other data types (with NaN acting as a gap).

        Args:
            arr (ArrayLike): Input array to analyze.
            gap_value (Optional[Scalar]): Value indicating a gap in the data.
            attach_as_close (bool): Whether to attach the input array as the `close` field.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Ranges`.

        Returns:
            Ranges: New instance constructed from the array.

        See:
            `vectorbtpro.generic.nb.records.get_ranges_nb`
        """
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        wrapper = ArrayWrapper.from_obj(arr, **wrapper_kwargs)

        arr = to_2d_array(arr)
        if gap_value is None:
            if np.issubdtype(arr.dtype, np.bool_):
                gap_value = False
            elif np.issubdtype(arr.dtype, np.integer):
                gap_value = -1
            else:
                gap_value = np.nan
        func = jit_reg.resolve_option(nb.get_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        records_arr = func(arr, gap_value)
        if attach_as_close and "close" not in kwargs:
            kwargs["close"] = arr
        return cls(wrapper, records_arr, **kwargs)

    @classmethod
    def from_delta(
        cls,
        records_or_mapped: tp.Union[Records, MappedArray],
        delta: tp.Union[str, int, tp.FrequencyLike],
        shift: tp.Optional[int] = None,
        idx_field_or_arr: tp.Union[None, str, tp.Array1d] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> "Ranges":
        """Build `Ranges` from a record or mapped array with an applied time delta on its index.

        This method applies a time delta to the index field of the input records. When `delta`
        is an integer, it represents a number of rows; otherwise, it is converted to a timedelta,
        requiring the index to be datetime-like or have a defined frequency. Conversion is performed
        using `vectorbtpro.utils.datetime_.to_timedelta64`.

        Args:
            records_or_mapped (Union[Records, MappedArray]): Record or mapped array to convert into ranges.
            delta (Union[str, int, FrequencyLike]): Time difference to apply, as a row count or timedelta.
            shift (Optional[int]): Number of rows to shift the delta application.
            idx_field_or_arr (Union[None, str, Array1d]): Field name or array for extracting index values.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Ranges`.

        Returns:
            Ranges: New instance constructed based on the applied delta.

        See:
            `vectorbtpro.generic.nb.records.get_ranges_from_delta_nb`
        """
        if idx_field_or_arr is None:
            if isinstance(records_or_mapped, Records):
                idx_field_or_arr = records_or_mapped.get_field_arr("idx")
            else:
                idx_field_or_arr = records_or_mapped.idx_arr
        if isinstance(idx_field_or_arr, str):
            if isinstance(records_or_mapped, Records):
                idx_field_or_arr = records_or_mapped.get_field_arr(idx_field_or_arr)
            else:
                raise ValueError("Providing an index field is allowed for records only")
        if isinstance(records_or_mapped, Records):
            id_arr = records_or_mapped.get_field_arr("id")
        else:
            id_arr = records_or_mapped.id_arr
        if isinstance(delta, int):
            delta_use_index = False
            index = None
        else:
            delta = dt.to_ns(dt.to_timedelta64(delta))
            if isinstance(records_or_mapped.wrapper.index, pd.DatetimeIndex):
                index = dt.to_ns(records_or_mapped.wrapper.index)
            else:
                freq = dt.to_ns(dt.to_timedelta64(records_or_mapped.wrapper.freq))
                index = np.arange(records_or_mapped.wrapper.shape[0]) * freq
            delta_use_index = True
        if shift is None:
            shift = 0
        col_map = records_or_mapped.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.get_ranges_from_delta_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        new_records_arr = func(
            records_or_mapped.wrapper.shape[0],
            idx_field_or_arr,
            id_arr,
            col_map,
            index=index,
            delta=delta,
            delta_use_index=delta_use_index,
            shift=shift,
        )
        if isinstance(records_or_mapped, PriceRecords):
            kwargs = merge_dicts(
                dict(
                    open=records_or_mapped._open,
                    high=records_or_mapped._high,
                    low=records_or_mapped._low,
                    close=records_or_mapped._close,
                ),
                kwargs,
            )
        return Ranges.from_records(records_or_mapped.wrapper, new_records_arr, **kwargs)

    def with_delta(self, *args, **kwargs) -> "Ranges":
        """Call `Ranges.from_delta` using the current instance.

        Args:
            *args: Positional arguments for `Ranges.from_delta`.
            **kwargs: Keyword arguments for `Ranges.from_delta`.

        Returns:
            Ranges: New instance resulting from the `from_delta` conversion.
        """
        return Ranges.from_delta(self, *args, **kwargs)

    def crop(self: RangesT) -> RangesT:
        """Crop the range records to the valid index span.

        Trims the data to include only records between the minimum start index and
        one more than the maximum end index.

        Returns:
            Ranges: New instance containing the cropped data.
        """
        min_start_idx = np.min(self.get_field_arr("start_idx"))
        max_start_idx = np.max(self.get_field_arr("end_idx")) + 1
        return self.iloc[min_start_idx:max_start_idx]

    # ############# Filtering ############# #

    def filter_min_duration(
        self: RangesT,
        min_duration: tp.Union[str, int, tp.FrequencyLike],
        real: bool = False,
        **kwargs,
    ) -> RangesT:
        """Filter out ranges shorter than a specified minimum duration.

        Args:
            min_duration (Union[str, int, FrequencyLike]): Minimum allowed duration.
            real (bool): If True, use real durations; otherwise, use effective durations.
            **kwargs: Keyword arguments for `Ranges.apply_mask`.

        Returns:
            Ranges: New instance with ranges meeting the minimum duration criteria.
        """
        if isinstance(min_duration, int):
            return self.apply_mask(self.duration.values >= min_duration, **kwargs)
        min_duration = dt.to_timedelta64(min_duration)
        if real:
            return self.apply_mask(self.real_duration.values >= min_duration, **kwargs)
        return self.apply_mask(self.duration.values * self.wrapper.freq >= min_duration, **kwargs)

    def filter_max_duration(
        self: RangesT,
        max_duration: tp.Union[str, int, tp.FrequencyLike],
        real: bool = False,
        **kwargs,
    ) -> RangesT:
        """Filter out ranges longer than a specified maximum duration.

        Args:
            max_duration (Union[str, int, FrequencyLike]): Maximum allowed duration.
            real (bool): If True, use real durations; otherwise, use effective durations.
            **kwargs: Keyword arguments for `Ranges.apply_mask`.

        Returns:
            Ranges: New instance with ranges meeting the maximum duration criteria.
        """
        if isinstance(max_duration, int):
            return self.apply_mask(self.duration.values <= max_duration, **kwargs)
        max_duration = dt.to_timedelta64(max_duration)
        if real:
            return self.apply_mask(self.real_duration.values <= max_duration, **kwargs)
        return self.apply_mask(self.duration.values * self.wrapper.freq <= max_duration, **kwargs)

    # ############# Masking ############# #

    def get_first_pd_mask(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Generate a mask based on the first index of each range.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `Ranges.get_pd_mask`.

        Returns:
            SeriesFrame: Mask indicating the first indices of each range.
        """
        return self.get_pd_mask(idx_arr=self.first_idx.values, group_by=group_by, **kwargs)

    def get_last_pd_mask(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Generate a mask based on the last index of each range.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `Ranges.get_pd_mask`.

        Returns:
            SeriesFrame: Mask indicating the last indices of each range.
        """
        out = self.get_pd_mask(idx_arr=self.last_idx.values, group_by=group_by, **kwargs)
        return out

    def get_ranges_pd_mask(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Generate a boolean mask for all ranges.

        This method converts range start and end indices, along with their statuses,
        into a boolean mask.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Boolean mask representing the ranges.

        See:
            `vectorbtpro.generic.nb.records.ranges_to_mask_nb`
        """
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.ranges_to_mask_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mask = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            col_map,
            len(self.wrapper.index),
        )
        return self.wrapper.wrap(mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Stats ############# #

    def get_valid(self: RangesT, **kwargs) -> RangesT:
        """Extract valid ranges by filtering out records with invalid indices.

        A valid range is defined as having both the start and end indices not equal to -1.

        Args:
            **kwargs: Keyword arguments for `Ranges.apply_mask`.

        Returns:
            Ranges: New instance containing only valid ranges.
        """
        filter_mask = (self.get_field_arr("start_idx") != -1) & (
            self.get_field_arr("end_idx") != -1
        )
        return self.apply_mask(filter_mask, **kwargs)

    def get_invalid(self: RangesT, **kwargs) -> RangesT:
        """Return invalid ranges.

        A range is considered invalid if its start or end index equals -1.

        Args:
            **kwargs: Keyword arguments for `Ranges.apply_mask`.

        Returns:
            Ranges: Ranges with indices marked as invalid.
        """
        filter_mask = (self.get_field_arr("start_idx") == -1) | (
            self.get_field_arr("end_idx") == -1
        )
        return self.apply_mask(filter_mask, **kwargs)

    def get_first_idx(self, **kwargs) -> MappedArray:
        """Return the first index for each range.

        Args:
            **kwargs: Keyword arguments for `Ranges.map_field`.

        Returns:
            MappedArray: First index in each range as a mapped array.
        """
        return self.map_field("start_idx", **kwargs)

    def get_last_idx(self, **kwargs) -> MappedArray:
        """Return the last index for each range.

        Adjust the end index for ranges with a closed status by subtracting one.

        Args:
            **kwargs: Keyword arguments for `Ranges.map_array`.

        Returns:
            MappedArray: Adjusted last index for each range as a mapped array.
        """
        last_idx = self.get_field_arr("end_idx", copy=True)
        status = self.get_field_arr("status")
        last_idx[status == enums.RangeStatus.Closed] -= 1
        return self.map_array(last_idx, **kwargs)

    def get_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Return the effective duration of each range in integer format.

        The duration is computed based on the start and end indices and is adjusted by the range status.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Ranges.map_array`.

        Returns:
            MappedArray: Array of effective durations for each range.

        See:
            `vectorbtpro.generic.nb.records.range_duration_nb`
        """
        func = jit_reg.resolve_option(nb.range_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        duration = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            freq=1,
        )
        return self.map_array(duration, **kwargs)

    def get_real_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Return the real duration of each range in timedelta format.

        The duration is calculated by converting start and end indices to nanoseconds and then
        computing the difference based on the given frequency.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            **kwargs: Keyword arguments for `Ranges.map_array`.

        Returns:
            MappedArray: Array of durations for each range expressed as timedelta.

        See:
            `vectorbtpro.generic.nb.records.range_duration_nb`
        """
        func = jit_reg.resolve_option(nb.range_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        duration = func(
            dt.to_ns(self.get_map_field_to_index("start_idx")),
            dt.to_ns(self.get_map_field_to_index("end_idx")),
            self.get_field_arr("status"),
            freq=dt.to_ns(dt.to_timedelta64(self.wrapper.freq)),
        ).astype("timedelta64[ns]")
        return self.map_array(duration, **kwargs)

    def get_avg_duration(
        self,
        real: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the average duration of ranges as a timedelta.

        The duration is computed as the mean of either the real or effective range durations.

        Args:
            real (bool): If True, use real durations; otherwise, use effective durations.
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
            MaybeSeries: Average duration of the ranges in timedelta.
        """
        if real:
            duration = self.real_duration
            duration = duration.replace(mapped_arr=dt.to_ns(duration.mapped_arr))
            wrap_kwargs = merge_dicts(
                dict(name_or_index="avg_real_duration", dtype="timedelta64[ns]"), wrap_kwargs
            )
        else:
            duration = self.duration
            wrap_kwargs = merge_dicts(
                dict(to_timedelta=True, name_or_index="avg_duration"), wrap_kwargs
            )
        return duration.mean(
            group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs
        )

    def get_max_duration(
        self,
        real: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the maximum duration among ranges as a timedelta.

        The duration is computed as the maximum of either the real or effective range durations.

        Args:
            real (bool): If True, use real durations; otherwise, use effective durations.
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
            MaybeSeries: Maximum duration among the ranges in timedelta.
        """
        if real:
            duration = self.real_duration
            duration = duration.replace(mapped_arr=dt.to_ns(duration.mapped_arr))
            wrap_kwargs = merge_dicts(
                dict(name_or_index="max_real_duration", dtype="timedelta64[ns]"), wrap_kwargs
            )
        else:
            duration = self.duration
            wrap_kwargs = merge_dicts(
                dict(to_timedelta=True, name_or_index="max_duration"), wrap_kwargs
            )
        return duration.max(
            group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs
        )

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the coverage of ranges, defined as the total number of steps covered by all ranges.

        Args:
            overlapping (bool): Whether to consider overlapping ranges.
            normalize (bool): Whether to normalize the coverage.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            MaybeSeries: Computed coverage of the ranges.

        See:
            `vectorbtpro.generic.nb.records.range_coverage_nb`
        """
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        index_lens = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.range_coverage_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        coverage = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            col_map,
            index_lens,
            overlapping=overlapping,
            normalize=normalize,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="coverage"), wrap_kwargs)
        return self.wrapper.wrap_reduced(coverage, group_by=group_by, **wrap_kwargs)

    def get_projections(
        self,
        close: tp.Optional[tp.ArrayLike] = None,
        proj_start: tp.Union[None, str, int, tp.FrequencyLike] = None,
        proj_period: tp.Union[None, str, int, tp.FrequencyLike] = None,
        incl_end_idx: bool = True,
        extend: bool = False,
        rebase: bool = True,
        start_value: tp.ArrayLike = 1.0,
        ffill: bool = False,
        remove_empty: bool = True,
        return_raw: bool = False,
        start_index: tp.Optional[tp.Timestamp] = None,
        id_level: tp.Union[None, str, tp.IndexLike] = None,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        clean_index_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.Tuple[tp.Array1d, tp.Array2d], tp.Frame]:
        """Generate projections for each range record.

        This method generates projections based on range records and provided close price data.

        Args:
            close (Optional[ArrayLike]): Array of close price values.

                If None, uses `Ranges.close`.
            proj_start (Union[None, str, int, FrequencyLike]): Defines when to start the projection.

                Use an integer for a specified number of rows after the start row or a timedelta-like value.
                The second option requires the index to be datetime-like, or at least the frequency to be set.
                The conversion uses `vectorbtpro.utils.datetime_.to_timedelta64`.
            proj_period (Union[None, str, int, FrequencyLike]): Defines the projection length.

                Use an integer for row count or a timedelta-like value.
            incl_end_idx (bool): Whether the end index of a range is inclusive.
            extend (bool): Whether to extend the projection to a fixed length (beyond the end).

                The extension period is taken from the longest range duration if `proj_period`
                is None, and from the longest `proj_period` if not None.
            rebase (bool): Rebase projection values so that each projection starts with `start_value`.

                If False, the projection retains the original close values.
            start_value (ArrayLike): Initial value for rebasing.

                Can be a scalar or an array per column. If set to -1, uses the latest close value.
            ffill (bool): Forward fill NaN values in the projection, even if they are NaN in `close`.
            remove_empty (bool): Remove projections that are NaN or contain only a single element.
            return_raw (bool): Return the raw output (range indices and projections)
                instead of a wrapped DataFrame.
            start_index (Optional[Timestamp]): Starting timestamp to define
                the DataFrame index of the projection.
            id_level (Union[None, str, IndexLike]): Identifier or key for naming range IDs.

                If a string, it may refer to a field mapping.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.

        Returns:
            Union[Tuple[Array1d, Array2d], Frame]: If `return_raw` is True, returns a tuple with
                a 1D array of record indices and a 2D array of projections, with each row corresponding
                to a range; otherwise, returns a DataFrame with projections.

        See:
            `vectorbtpro.generic.nb.records.map_ranges_to_projections_nb`

        !!! note
            As opposed to the Numba-compiled function, the returned DataFrame has projections
            stacked along columns rather than rows. Set `return_raw` to True to obtain the original format.
        """
        if close is None:
            close = self.close
            checks.assert_not_none(close, arg_name="close")
        else:
            close = self.wrapper.wrap(close, group_by=False)
        if proj_start is None:
            proj_start = 0
        if isinstance(proj_start, int):
            proj_start_use_index = False
            index = None
        else:
            proj_start = dt.to_ns(dt.to_timedelta64(proj_start))
            if isinstance(self.wrapper.index, pd.DatetimeIndex):
                index = dt.to_ns(self.wrapper.index)
            else:
                freq = dt.to_ns(dt.to_timedelta64(self.wrapper.freq))
                index = np.arange(self.wrapper.shape[0]) * freq
            proj_start_use_index = True
        if proj_period is not None:
            if isinstance(proj_period, int):
                proj_period_use_index = False
            else:
                proj_period = dt.to_ns(dt.to_timedelta64(proj_period))
                if index is None:
                    if isinstance(self.wrapper.index, pd.DatetimeIndex):
                        index = dt.to_ns(self.wrapper.index)
                    else:
                        freq = dt.to_ns(dt.to_timedelta64(self.wrapper.freq))
                        index = np.arange(self.wrapper.shape[0]) * freq
                proj_period_use_index = True
        else:
            proj_period_use_index = False

        func = jit_reg.resolve_option(nb.map_ranges_to_projections_nb, jitted)
        ridxs, projections = func(
            to_2d_array(close),
            self.get_field_arr("col"),
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            index=index,
            proj_start=proj_start,
            proj_start_use_index=proj_start_use_index,
            proj_period=proj_period,
            proj_period_use_index=proj_period_use_index,
            incl_end_idx=incl_end_idx,
            extend=extend,
            rebase=rebase,
            start_value=to_1d_array(start_value),
            ffill=ffill,
            remove_empty=remove_empty,
        )
        if return_raw:
            return ridxs, projections
        projections = projections.T
        wrapper = ArrayWrapper.from_obj(projections, freq=self.wrapper.freq)
        if id_level is None:
            id_level = pd.Index(self.id_arr, name="range_id")
        elif isinstance(id_level, str):
            mapping = self.get_field_mapping(id_level)
            if isinstance(mapping, str) and mapping == "index":
                id_level = self.get_map_field_to_index(id_level).rename(id_level)
            else:
                id_level = pd.Index(self.get_apply_mapping_arr(id_level), name=id_level)
        else:
            if not isinstance(id_level, pd.Index):
                id_level = pd.Index(id_level, name="range_id")
        if start_index is None:
            start_index = close.index[-1]
        wrap_kwargs = merge_dicts(
            dict(
                index=pd.date_range(
                    start=start_index,
                    periods=projections.shape[0],
                    freq=self.wrapper.freq,
                ),
                columns=stack_indexes(
                    self.wrapper.columns[self.col_arr[ridxs]],
                    id_level[ridxs],
                    **resolve_dict(clean_index_kwargs),
                ),
            ),
            wrap_kwargs,
        )
        return wrapper.wrap(projections, **wrap_kwargs)

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `Data.stats`.

        Merges the defaults from `vectorbtpro.records.base.Records.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.ranges`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        ranges_stats_cfg = settings["ranges"]["stats"]

        return merge_dicts(Records.stats_defaults.__get__(self), ranges_stats_cfg)

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
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            coverage=dict(
                title="Coverage",
                calc_func="coverage",
                overlapping=False,
                tags=["ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="coverage",
                overlapping=True,
                tags=["ranges", "coverage"],
            ),
            duration=dict(
                title="Duration",
                calc_func="duration.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                },
                apply_to_timedelta=True,
                tags=["ranges", "duration"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_projections(
        self,
        column: tp.Optional[tp.Column] = None,
        min_duration: tp.Union[str, int, tp.FrequencyLike] = None,
        max_duration: tp.Union[str, int, tp.FrequencyLike] = None,
        last_n: tp.Optional[int] = None,
        top_n: tp.Optional[int] = None,
        random_n: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        proj_start: tp.Union[None, str, int, tp.FrequencyLike] = "current_or_0",
        proj_period: tp.Union[None, str, int, tp.FrequencyLike] = "max",
        incl_end_idx: bool = True,
        extend: bool = False,
        ffill: bool = False,
        plot_past_period: tp.Union[None, str, int, tp.FrequencyLike] = "current_or_proj_period",
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        plot_projections: bool = True,
        plot_bands: bool = True,
        plot_lower: tp.Union[bool, str, tp.Callable] = True,
        plot_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_upper: tp.Union[bool, str, tp.Callable] = True,
        plot_aux_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_fill: bool = True,
        colorize: tp.Union[bool, str, tp.Callable] = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        projection_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        aux_middle_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot projections.

        Generate and plot projection lines and bands based on filtered range records.
        This method filters range records by duration constraints and selection criteria,
        computes projections using `Ranges.get_projections`, and visualizes the results with
        `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
        It also overlays OHLC or close price data if available.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            min_duration (Union[str, int, FrequencyLike]): Filter range records by minimum duration.
            max_duration (Union[str, int, FrequencyLike]): Filter range records by maximum duration.
            last_n (Optional[int]): Number of most recent range records to select.
            top_n (Optional[int]): Display only the top N records sorted by maximum duration.
            random_n (Optional[int]): Number of range records to select randomly.
            seed (Optional[int]): Random seed for deterministic output.
            proj_start (Union[None, str, int, FrequencyLike]): Defines when to start the projection.

                See `Ranges.get_projections`.

                Allows an option "current_or_{value}", which sets `proj_start` to the duration
                of the current open range or to the specified value if no open range exists.
            proj_period (Union[None, str, int, FrequencyLike]): Defines the projection length.

                See `Ranges.get_projections`.

                Allows options such as "current_or_{option}", "mean", "min", "max", "median",
                or a percentage like "50%" representing a quantile derived from closed ranges.
            incl_end_idx (bool): Whether the end index of a range is inclusive.

                See `Ranges.get_projections`.
            extend (bool): Whether to extend the projection to a fixed length (beyond the end).

                See `Ranges.get_projections`.
            ffill (bool): Forward fill NaN values in the projection, even if they are NaN in `close`.

                See `Ranges.get_projections`.
            plot_past_period (Union[None, str, int, FrequencyLike]): Past period for plotting.

                Accepts the same options as `proj_period` plus "proj_period" and "current_or_proj_period".
            plot_ohlc (Union[bool, DataFrame]): Flag or data specifying whether to plot OHLC.
            plot_close (Union[bool, Series]): Flag or data specifying whether to plot close prices.
            plot_projections (bool): Plot each projection as a semi-transparent line if True.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`
            plot_bands (bool): Plot computed bands if True.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_lower (Union[bool, str, Callable]): Specification for the lower band.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`
            plot_middle (Union[bool, str, Callable]): Specification for the middle band.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`
            plot_upper (Union[bool, str, Callable]): Specification for the upper band.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`
            plot_aux_middle (Union[bool, str, Callable]): Specification for an auxiliary middle band.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`
            plot_fill (bool): Fill the area between band traces if True.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`
            colorize (Union[bool, str, Callable]): Strategy for colorizing projections or bands.

                See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
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
            BaseFigure: Figure object containing the plotted projections and price data.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> price = pd.Series(
            ...     [11, 12, 13, 14, 11, 12, 13, 12, 11, 12],
            ...     index=pd.date_range("2020", periods=10),
            ... )
            >>> vbt.Ranges.from_array(
            ...     price >= 12,
            ...     attach_as_close=False,
            ...     close=price,
            ... ).plot_projections(
            ...     proj_start=0,
            ...     proj_period=4,
            ...     extend=True,
            ...     plot_past_period=None
            ... ).show()
            ```

            ![](/assets/images/api/ranges_plot_projections.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ranges_plot_projections.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        self_col_open = self_col.status_open
        self_col = self_col.status_closed
        if proj_start is not None:
            if isinstance(proj_start, str) and proj_start.startswith("current_or_"):
                if self_col_open.count() > 0:
                    if self_col_open.count() > 1:
                        raise ValueError("Only one open range is allowed")
                    proj_start = int(self_col_open.duration.values[0])
                else:
                    proj_start = proj_start.replace("current_or_", "")
                    if proj_start.isnumeric():
                        proj_start = int(proj_start)
            if proj_start != 0:
                self_col = self_col.filter_min_duration(proj_start, real=True)
        if min_duration is not None:
            self_col = self_col.filter_min_duration(min_duration, real=True)
        if max_duration is not None:
            self_col = self_col.filter_max_duration(max_duration, real=True)
        if last_n is not None:
            self_col = self_col.last_n(last_n)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))
        if random_n is not None:
            self_col = self_col.random_n(random_n, seed=seed)
        if self_col.count() == 0:
            warn("No ranges to plot. Relax the requirements.")

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True
        if close is None:
            raise ValueError("Close cannot be None")

        # Resolve windows
        def _resolve_period(period):
            if self_col.count() == 0:
                period = None
            if period is not None:
                if isinstance(period, str):
                    period = period.lower().replace(" ", "")
                    if period == "median":
                        period = "50%"
                    if "%" in period:
                        period = int(
                            np.quantile(
                                self_col.duration.values,
                                float(period.replace("%", "")) / 100,
                            )
                        )
                    elif period.startswith("current_or_"):
                        if self_col_open.count() > 0:
                            if self_col_open.count() > 1:
                                raise ValueError("Only one open range is allowed")
                            period = int(self_col_open.duration.values[0])
                        else:
                            period = period.replace("current_or_", "")
                            return _resolve_period(period)
                    elif period == "mean":
                        period = int(np.mean(self_col.duration.values))
                    elif period == "min":
                        period = int(np.min(self_col.duration.values))
                    elif period == "max":
                        period = int(np.max(self_col.duration.values))
            return period

        proj_period = _resolve_period(proj_period)
        if isinstance(proj_period, int) and proj_period == 0:
            warn("Projection period is zero. Setting to maximum.")
            proj_period = int(np.max(self_col.duration.values))
        if plot_past_period is not None and isinstance(plot_past_period, str):
            plot_past_period = plot_past_period.lower().replace(" ", "")
            if plot_past_period == "proj_period":
                plot_past_period = proj_period
            elif plot_past_period == "current_or_proj_period":
                if self_col_open.count() > 0:
                    if self_col_open.count() > 1:
                        raise ValueError("Only one open range is allowed")
                    plot_past_period = int(self_col_open.duration.values[0])
                else:
                    plot_past_period = proj_period
        plot_past_period = _resolve_period(plot_past_period)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot OHLC/close
        if plot_ohlc and ohlc is not None:
            if plot_past_period is not None:
                if isinstance(plot_past_period, int):
                    _ohlc = ohlc.iloc[-plot_past_period:]
                else:
                    plot_past_period = dt.to_timedelta(plot_past_period)
                    _ohlc = ohlc[ohlc.index > ohlc.index[-1] - plot_past_period]
            else:
                _ohlc = ohlc
            if _ohlc.size > 0:
                if "opacity" not in ohlc_trace_kwargs:
                    ohlc_trace_kwargs["opacity"] = 0.5
                fig = _ohlc.vbt.ohlcv.plot(
                    ohlc_type=ohlc_type,
                    plot_volume=False,
                    ohlc_trace_kwargs=ohlc_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )
        elif plot_close:
            if plot_past_period is not None:
                if isinstance(plot_past_period, int):
                    _close = close.iloc[-plot_past_period:]
                else:
                    plot_past_period = dt.to_timedelta(plot_past_period)
                    _close = close[close.index > close.index[-1] - plot_past_period]
            else:
                _close = close
            if _close.size > 0:
                fig = _close.vbt.lineplot(
                    trace_kwargs=close_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        if self_col.count() > 0:
            # Get projections
            projections = self_col.get_projections(
                close=close,
                proj_start=proj_start,
                proj_period=proj_period,
                incl_end_idx=incl_end_idx,
                extend=extend,
                rebase=True,
                start_value=close.iloc[-1],
                ffill=ffill,
                remove_empty=True,
                return_raw=False,
            )

            if len(projections.columns) > 0:
                # Plot projections
                rename_levels = dict(range_id=self_col.get_field_title("id"))
                fig = projections.vbt.plot_projections(
                    plot_projections=plot_projections,
                    plot_bands=plot_bands,
                    plot_lower=plot_lower,
                    plot_middle=plot_middle,
                    plot_upper=plot_upper,
                    plot_aux_middle=plot_aux_middle,
                    plot_fill=plot_fill,
                    colorize=colorize,
                    rename_levels=rename_levels,
                    projection_trace_kwargs=projection_trace_kwargs,
                    lower_trace_kwargs=lower_trace_kwargs,
                    middle_trace_kwargs=middle_trace_kwargs,
                    upper_trace_kwargs=upper_trace_kwargs,
                    aux_middle_trace_kwargs=aux_middle_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        return fig

    def plot_shapes(
        self,
        column: tp.Optional[tp.Column] = None,
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot range shapes on a figure.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            plot_ohlc (Union[bool, DataFrame]): Flag or data specifying whether to plot OHLC.
            plot_close (Union[bool, Series]): Flag or data specifying whether to plot close prices.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
            add_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for each shape.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object containing the plotted shapes.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            Plot zones colored by duration:

            ```pycon
            >>> price = pd.Series(
            ...     [1, 2, 1, 2, 3, 2, 1, 2, 3],
            ...     index=pd.date_range("2020", periods=9),
            ... )

            >>> def get_opacity(self_col, i):
            ...     real_duration = self_col.get_real_duration().values
            ...     return real_duration[i] / real_duration.max() * 0.5

            >>> vbt.Ranges.from_array(price >= 2).plot_shapes(
            ...     add_shape_kwargs=dict(fillcolor="teal", opacity=vbt.RepFunc(get_opacity))
            ... ).show()
            ```

            ![](/assets/images/api/ranges_plot_shapes.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ranges_plot_shapes.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import get_domain, make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if add_shape_kwargs is None:
            add_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(yref, fig)
        y_domain = get_domain(yref, fig)

        # Plot OHLC/close
        if plot_ohlc and ohlc is not None:
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        elif plot_close and close is not None:
            fig = close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            start_idx = self_col.get_map_field_to_index("start_idx", minus_one_to_zero=True)
            end_idx = self_col.get_map_field_to_index("end_idx")
            for i in range(len(self_col.values)):
                start_index = start_idx[i]
                end_index = end_idx[i]
                _shape_kwargs = substitute_templates(
                    add_shape_kwargs,
                    context=dict(
                        self_col=self_col,
                        i=i,
                        record=self_col.values[i],
                        start_index=start_index,
                        end_index=end_index,
                        xref=xref,
                        yref=yref,
                        x_domain=x_domain,
                        y_domain=y_domain,
                        close=close,
                        ohlc=ohlc,
                    ),
                    eval_id="add_shape_kwargs",
                )
                _shape_kwargs = merge_dicts(
                    dict(
                        type="rect",
                        xref=xref,
                        yref="paper",
                        x0=start_index,
                        y0=y_domain[0],
                        x1=end_index,
                        y1=y_domain[1],
                        fillcolor="gray",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    ),
                    _shape_kwargs,
                )
                fig.add_shape(**_shape_kwargs)

        return fig

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        top_n: tp.Optional[int] = None,
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        start_trace_kwargs: tp.KwargsLike = None,
        end_trace_kwargs: tp.KwargsLike = None,
        open_shape_kwargs: tp.KwargsLike = None,
        closed_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        return_close: bool = False,
        **layout_kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.Tuple[tp.BaseFigure, tp.Series]]:
        """Plot range data including OHLC, close prices, start/end markers,
        and shaded zones for open and closed ranges.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            top_n (Optional[int]): Display only the top N records sorted by maximum duration.
            plot_ohlc (Union[bool, Frame]): If True, plot OHLC data or pass a DataFrame to be used as OHLC.
            plot_close (Union[bool, Series]): Flag or data specifying whether to plot close prices.
            plot_markers (bool): Whether to plot markers at the start and end of ranges.
            plot_zones (bool): Whether to plot shaded zones for open and closed ranges.
            ohlc_type (Union[None, str, BaseTraceType]): Specifies the OHLC plot type.

                Use 'OHLC', 'Candlestick', or a Plotly trace type. Pass None to use the default.
            ohlc_trace_kwargs (KwargsLike): Keyword arguments for `ohlc_type` for the OHLC data.
            close_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the close price.
            start_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for start markers.
            end_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for end markers.
            open_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for open zones.
            closed_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for closed zones.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            return_close (bool): Whether to return the close Series along with the figure.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure object containing the plotted ranges.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> price = pd.Series(
            ...     [1, 2, 1, 2, 3, 2, 1, 2, 3],
            ...     index=pd.date_range("2020", periods=9),
            ... )
            >>> vbt.Ranges.from_array(price >= 2).plot().show()
            ```

            ![](/assets/images/api/ranges_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ranges_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go

        from vectorbtpro._settings import settings
        from vectorbtpro.utils.figure import get_domain, make_figure

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"]), name="Close"),
            close_trace_kwargs,
        )
        if start_trace_kwargs is None:
            start_trace_kwargs = {}
        if end_trace_kwargs is None:
            end_trace_kwargs = {}
        if open_shape_kwargs is None:
            open_shape_kwargs = {}
        if closed_shape_kwargs is None:
            closed_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        y_domain = get_domain(yref, fig)

        # Plot OHLC/close
        plotting_ohlc = False
        if plot_ohlc and ohlc is not None:
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            plotting_ohlc = True
        elif plot_close and close is not None:
            fig = close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            start_idx = self_col.get_map_field_to_index("start_idx", minus_one_to_zero=True)
            if plotting_ohlc and self_col.open is not None:
                start_val = self_col.open.loc[start_idx]
            elif close is not None:
                start_val = close.loc[start_idx]
            else:
                start_val = np.full(len(start_idx), 0)
            end_idx = self_col.get_map_field_to_index("end_idx")
            if close is not None:
                end_val = close.loc[end_idx]
            else:
                end_val = np.full(len(end_idx), 0)
            status = self_col.get_field_arr("status")

            if plot_markers:
                # Plot start markers
                start_customdata, start_hovertemplate = self_col.prepare_customdata(
                    incl_fields=["id", "start_idx"]
                )
                _start_trace_kwargs = merge_dicts(
                    dict(
                        x=start_idx,
                        y=start_val,
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
                        name="Start",
                        customdata=start_customdata,
                        hovertemplate=start_hovertemplate,
                    ),
                    start_trace_kwargs,
                )
                start_scatter = go.Scatter(**_start_trace_kwargs)
                fig.add_trace(start_scatter, **add_trace_kwargs)

            closed_mask = status == enums.RangeStatus.Closed
            if closed_mask.any():
                if plot_markers:
                    # Plot end markers
                    closed_end_customdata, closed_end_hovertemplate = self_col.prepare_customdata(
                        mask=closed_mask
                    )
                    _end_trace_kwargs = merge_dicts(
                        dict(
                            x=end_idx[closed_mask],
                            y=end_val[closed_mask],
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
                            name="Closed",
                            customdata=closed_end_customdata,
                            hovertemplate=closed_end_hovertemplate,
                        ),
                        end_trace_kwargs,
                    )
                    closed_end_scatter = go.Scatter(**_end_trace_kwargs)
                    fig.add_trace(closed_end_scatter, **add_trace_kwargs)

            open_mask = status == enums.RangeStatus.Open
            if open_mask.any():
                if plot_markers:
                    # Plot end markers
                    open_end_customdata, open_end_hovertemplate = self_col.prepare_customdata(
                        excl_fields=["end_idx"], mask=open_mask
                    )
                    _end_trace_kwargs = merge_dicts(
                        dict(
                            x=end_idx[open_mask],
                            y=end_val[open_mask],
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
                            name="Open",
                            customdata=open_end_customdata,
                            hovertemplate=open_end_hovertemplate,
                        ),
                        end_trace_kwargs,
                    )
                    open_end_scatter = go.Scatter(**_end_trace_kwargs)
                    fig.add_trace(open_end_scatter, **add_trace_kwargs)

            if plot_zones:
                # Plot closed range zones
                self_col.status_closed.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(fillcolor=plotting_cfg["contrast_color_schema"]["green"]),
                        closed_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

                # Plot open range zones
                self_col.status_open.plot_shapes(
                    plot_ohlc=False,
                    plot_close=False,
                    add_shape_kwargs=merge_dicts(
                        dict(fillcolor=plotting_cfg["contrast_color_schema"]["orange"]),
                        open_shape_kwargs,
                    ),
                    add_trace_kwargs=add_trace_kwargs,
                    xref=xref,
                    yref=yref,
                    fig=fig,
                )

        if return_close:
            return fig, close
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `Data.plots`.

        Merges the defaults from `vectorbtpro.records.base.Records.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.ranges`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        ranges_plots_cfg = settings["ranges"]["plots"]

        return merge_dicts(Records.plots_defaults.__get__(self), ranges_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Ranges",
                check_is_not_grouped=True,
                plot_func="plot",
                tags="ranges",
            )
        ),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Ranges.override_field_config_doc(__pdoc__)
Ranges.override_metrics_doc(__pdoc__)
Ranges.override_subplots_doc(__pdoc__)

# ############# Pattern ranges ############# #

PatternRangesT = tp.TypeVar("PatternRangesT", bound="PatternRanges")


@define
class PSC(DefineMixin):
    """Class representing a pattern search configuration.

    Every field is resolved into a format suitable for Numba.
    """

    pattern: tp.Union[tp.ArrayLike] = define.required_field()
    """Flexible array representing the pattern to locate.

    Can be smaller or larger than the source array. In such cases,
    the smaller array is stretched using the interpolation mode specified by `PSC.interp_mode`.
    """

    window: tp.Optional[int] = define.optional_field()
    """Base length of the rolling window for matching.

    If None, defaults to the length of `PSC.pattern`.
    """

    max_window: tp.Optional[int] = define.optional_field()
    """Maximum length of the rolling window for matching.

    If None, defaults to `PSC.window`.
    """

    row_select_prob: tp.Union[float] = define.optional_field()
    """Probability of selecting a row."""

    window_select_prob: tp.Union[float] = define.optional_field()
    """Probability of selecting a window size."""

    roll_forward: tp.Union[bool] = define.optional_field()
    """Process windows in forward direction if True; otherwise, in backward direction."""

    interp_mode: tp.Union[int, str] = define.optional_field()
    """Interpolation mode for mapping array values.

    See `vectorbtpro.generic.enums.InterpMode`.
    """

    rescale_mode: tp.Union[int, str] = define.optional_field()
    """Rescaling mode for adjusting the ranges of the input array and `PSC.pattern`.

    See `vectorbtpro.generic.enums.RescaleMode`.
    """

    vmin: tp.Union[float] = define.optional_field()
    """Minimum value used for rescaling the input array.

    Use only when the array has fixed bounds. Used in rescaling with `RescaleMode.MinMax`
    and for verifying `PSC.min_pct_change` and `PSC.max_pct_change`.

    If set to NaN, it is calculated dynamically.
    """

    vmax: tp.Union[float] = define.optional_field()
    """Maximum value used for rescaling the input array.

    Use only when the array has fixed bounds. Used in rescaling with `RescaleMode.MinMax`
    and for verifying `PSC.min_pct_change` and `PSC.max_pct_change`.

    If set to NaN, it is calculated dynamically.
    """

    pmin: tp.Union[float] = define.optional_field()
    """Minimum value used for rescaling `PSC.pattern`.

    Used in rescaling with `RescaleMode.MinMax` and for computing the maximum distance
    at each point when `PSC.max_error_as_maxdist` is disabled.

    If set to NaN, it is calculated dynamically.
    """

    pmax: tp.Union[float] = define.optional_field()
    """Maximum value used for rescaling `PSC.pattern`.

    Used in rescaling with `RescaleMode.MinMax` and for computing the maximum distance
    at each point when `PSC.max_error_as_maxdist` is disabled.

    If set to NaN, it is calculated dynamically.
    """

    invert: tp.Union[bool] = define.optional_field()
    """Invert the pattern by reflecting its values."""

    error_type: tp.Union[int, str] = define.optional_field()
    """Error computation mode.

    See `vectorbtpro.generic.enums.ErrorType`.
    """

    distance_measure: tp.Union[int, str] = define.optional_field()
    """Method for measuring distance (e.g., MAE, MSE, RMSE).

    See `vectorbtpro.generic.enums.DistanceMeasure`.
    """

    max_error: tp.Union[tp.ArrayLike] = define.optional_field()
    """Maximum error threshold for normalization.

    If provided as an array, it must match the size of the pattern and be on the same scale.
    """

    max_error_interp_mode: tp.Union[None, int, str] = define.optional_field()
    """Interpolation mode for `PSC.max_error`.

    If None, defaults to `PSC.interp_mode`.

    See `vectorbtpro.generic.enums.InterpMode`.
    """

    max_error_as_maxdist: tp.Union[bool] = define.optional_field()
    """Indicates whether `PSC.max_error` represents the maximum distance at each point.

    If False, exceeding `PSC.max_error` sets the distance to the maximum derived from
    `PSC.pmin`, `PSC.pmax`, and the pattern value at that point. If True and any point
    in a window is NaN, that point is skipped.
    """

    max_error_strict: tp.Union[bool] = define.optional_field()
    """If True, any instance of exceeding `PSC.max_error` results in a similarity of NaN."""

    min_pct_change: tp.Union[float] = define.optional_field()
    """Minimum percentage change required for a window to remain a search candidate.

    Window similarity is set to NaN if this threshold is not met.
    """

    max_pct_change: tp.Union[float] = define.optional_field()
    """Maximum percentage change allowed for a window to remain a search candidate.

    Window similarity is set to NaN if this threshold is exceeded.
    """

    min_similarity: tp.Union[float] = define.optional_field()
    """Minimum similarity threshold.

    If the computed similarity falls below this, returns NaN.
    """

    minp: tp.Optional[int] = define.optional_field()
    """Minimum number of observations in the price window required to yield a value."""

    overlap_mode: tp.Union[int, str] = define.optional_field()
    """Mode for handling overlapping matches.

    See `vectorbtpro.generic.enums.OverlapMode`.
    """

    max_records: tp.Optional[int] = define.optional_field()
    """Maximum number of records to be filled.

    If None, defaults to the number of rows in the source array.
    """

    name: tp.Optional[str] = define.field(default=None)
    """Optional name assigned to the configuration."""

    def __eq__(self, other):
        return checks.is_deep_equal(self, other)

    def __hash__(self):
        dct = self.asdict()
        if isinstance(dct["pattern"], np.ndarray):
            dct["pattern"] = tuple(dct["pattern"])
        else:
            dct["pattern"] = (dct["pattern"],)
        if isinstance(dct["max_error"], np.ndarray):
            dct["max_error"] = tuple(dct["max_error"])
        else:
            dct["max_error"] = (dct["max_error"],)
        return hash(tuple(dct.items()))


pattern_ranges_field_config = ReadonlyConfig(
    dict(
        dtype=enums.pattern_range_dt,
        settings=dict(
            id=dict(title="Pattern Range Id"),
            similarity=dict(title="Similarity"),
        ),
    )
)
"""_"""

__pdoc__["pattern_ranges_field_config"] = f"""Field configuration for `PatternRanges`.

```python
{pattern_ranges_field_config.prettify_doc()}
```
"""


@attach_fields
@override_field_config(pattern_ranges_field_config)
class PatternRanges(Ranges):
    """Class for working with range records generated from pattern search, extending `Ranges`.

    Requires `records_arr` to have all fields defined in `vectorbtpro.generic.enums.pattern_range_dt`.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        records_arr (RecordArray): Array of records.

            Must adhere to the `vectorbtpro.generic.enums.pattern_range_dt` dtype.
        search_configs (List[PSC]): List of `PSC` instances.
        **kwargs: Keyword arguments for `Ranges`.
    """

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        search_configs: tp.List[PSC],
        **kwargs,
    ) -> None:
        Ranges.__init__(
            self,
            wrapper,
            records_arr,
            search_configs=search_configs,
            **kwargs,
        )

        self._search_configs = search_configs

    @property
    def search_configs(self) -> tp.List[PSC]:
        """List of `PSC` instances, one for each column.

        Returns:
            List[PSC]: List of `PSC` instances.
        """
        return self._search_configs

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def resolve_search_config(
        cls, search_config: tp.Union[None, dict, PSC] = None, **kwargs
    ) -> PSC:
        """Resolve search configuration for `PatternRanges.from_pattern_search`.

        Convert array-like inputs to one-dimensional arrays and map enumerated fields to integer values.

        Args:
            search_config (Union[None, dict, PSC]): Search configuration for the pattern search.
            **kwargs: Keyword arguments for the search configuration.

        Returns:
            PSC: Resolved search configuration.
        """
        if search_config is None:
            search_config = dict()
        if isinstance(search_config, dict):
            search_config = PSC(**search_config)
        search_config = search_config.asdict()
        defaults = {}
        for k, v in get_func_kwargs(cls.from_pattern_search).items():
            if k in search_config:
                defaults[k] = v
        defaults = merge_dicts(defaults, kwargs)
        for k, v in search_config.items():
            if v is MISSING:
                v = defaults[k]
            if k == "pattern":
                if v is None:
                    raise ValueError("Must provide pattern")
                v = to_1d_array(v)
            elif k == "max_error":
                v = to_1d_array(v)
            elif k == "interp_mode":
                v = map_enum_fields(v, enums.InterpMode)
            elif k == "rescale_mode":
                v = map_enum_fields(v, enums.RescaleMode)
            elif k == "error_type":
                v = map_enum_fields(v, enums.ErrorType)
            elif k == "distance_measure":
                v = map_enum_fields(v, enums.DistanceMeasure)
            elif k == "max_error_interp_mode":
                if v is None:
                    v = search_config["interp_mode"]
                else:
                    v = map_enum_fields(v, enums.InterpMode)
            elif k == "overlap_mode":
                v = map_enum_fields(v, enums.OverlapMode)
            search_config[k] = v
        return PSC(**search_config)

    @classmethod
    def from_pattern_search(
        cls: tp.Type[PatternRangesT],
        arr: tp.ArrayLike,
        pattern: tp.Union[Param, tp.ArrayLike] = None,
        window: tp.Union[Param, None, int] = None,
        max_window: tp.Union[Param, None, int] = None,
        row_select_prob: tp.Union[Param, float] = 1.0,
        window_select_prob: tp.Union[Param, float] = 1.0,
        roll_forward: tp.Union[Param, bool] = False,
        interp_mode: tp.Union[Param, int, str] = "mixed",
        rescale_mode: tp.Union[Param, int, str] = "minmax",
        vmin: tp.Union[Param, float] = np.nan,
        vmax: tp.Union[Param, float] = np.nan,
        pmin: tp.Union[Param, float] = np.nan,
        pmax: tp.Union[Param, float] = np.nan,
        invert: bool = False,
        error_type: tp.Union[Param, int, str] = "absolute",
        distance_measure: tp.Union[Param, int, str] = "mae",
        max_error: tp.Union[Param, tp.ArrayLike] = np.nan,
        max_error_interp_mode: tp.Union[Param, None, int, str] = None,
        max_error_as_maxdist: tp.Union[Param, bool] = False,
        max_error_strict: tp.Union[Param, bool] = False,
        min_pct_change: tp.Union[Param, float] = np.nan,
        max_pct_change: tp.Union[Param, float] = np.nan,
        min_similarity: tp.Union[Param, float] = 0.85,
        minp: tp.Union[Param, None, int] = None,
        overlap_mode: tp.Union[Param, int, str] = "disallow",
        max_records: tp.Union[Param, None, int] = None,
        random_subset: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        search_configs: tp.Optional[tp.Sequence[tp.MaybeSequence[PSC]]] = None,
        jitted: tp.JittedOption = None,
        execute_kwargs: tp.KwargsLike = None,
        attach_as_close: bool = True,
        clean_index_kwargs: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PatternRangesT:
        """Build `PatternRanges` from all occurrences of a pattern in an array.

        This class method searches for parameters of type `vectorbtpro.utils.params.Param` among the
        provided arguments. If such parameters are found, they are broadcasted and combined via
        `vectorbtpro.utils.params.combine_params` into one or more search configurations. If no `Param`
        arguments are detected, a single search configuration is built from the given parameters.
        When `search_configs` is supplied, it is used directly without modification. For example,
        passing `min_similarity` of 95% will use it in all search configurations except where
        it was explicitly overridden.

        Each search configuration is resolved using `PatternRanges.resolve_search_config`.
        The configurations are executed with `vectorbtpro.utils.execution.execute`, the resulting
        records arrays are concatenated, and the outcome is wrapped in a `PatternRanges` instance.

        Args:
            arr (ArrayLike): Input array for pattern search.
            pattern (Union[Param, ArrayLike]): See `PSC.pattern`.
            window (Union[Param, None, int]): See `PSC.window`.
            max_window (Union[Param, None, int]): See `PSC.max_window`.
            row_select_prob (Union[Param, float]): See `PSC.row_select_prob`.
            window_select_prob (Union[Param, float]): See `PSC.window_select_prob`.
            roll_forward (Union[Param, bool]): See `PSC.roll_forward`.
            interp_mode (Union[Param, int, str]): See `PSC.interp_mode`.
            rescale_mode (Union[Param, int, str]): See `PSC.rescale_mode`.
            vmin (Union[Param, float]): See `PSC.vmin`.
            vmax (Union[Param, float]): See `PSC.vmax`.
            pmin (Union[Param, float]): See `PSC.pmin`.
            pmax (Union[Param, float]): See `PSC.pmax`.
            invert (bool): See `PSC.invert`.
            error_type (Union[Param, int, str]): See `PSC.error_type`.
            distance_measure (Union[Param, int, str]): See `PSC.distance_measure`.
            max_error (Union[Param, ArrayLike]): See `PSC.max_error`.
            max_error_interp_mode (Union[Param, None, int, str]): See `PSC.max_error_interp_mode`.
            max_error_as_maxdist (Union[Param, bool]): See `PSC.max_error_as_maxdist`.
            max_error_strict (Union[Param, bool]): See `PSC.max_error_strict`.
            min_pct_change (Union[Param, float]): See `PSC.min_pct_change`.
            max_pct_change (Union[Param, float]): See `PSC.max_pct_change`.
            min_similarity (Union[Param, float]): See `PSC.min_similarity`.
            minp (Union[Param, None, int]): See `PSC.minp`.
            overlap_mode (Union[Param, int, str]): See `PSC.overlap_mode`.
            max_records (Union[Param, None, int]): See `PSC.max_records`.
            random_subset (Optional[int]): Select a random subset of parameter combinations.

                Set the seed for reproducibility.
            seed (Optional[int]): Random seed for deterministic output.
            search_configs (Optional[Sequence[MaybeSequence[PSC]]]): Sequence of search configuration instances.

                If a configuration is a list of `PSC` instances, it is applied per column in `arr`;
                otherwise, per array.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            attach_as_close (bool): Whether to attach the input array as the `close` field.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `PatternRanges`.

        Returns:
            PatternRanges: New `PatternRanges` instance with found pattern ranges.

        See:
            `vectorbtpro.generic.nb.records.find_pattern_1d_nb`
        """
        if seed is not None:
            set_seed(seed)
        if clean_index_kwargs is None:
            clean_index_kwargs = {}
        arr = to_pd_array(arr)
        arr_2d = to_2d_array(arr)
        arr_wrapper = ArrayWrapper.from_obj(arr)
        psc_keys = [a.name for a in PSC.fields if a.name != "name"]
        method_locals = locals()
        method_locals = {k: v for k, v in method_locals.items() if k in psc_keys}

        # Flatten search configs
        flat_search_configs = []
        psc_names = []
        psc_names_none = True
        n_configs = 0
        if search_configs is not None:
            for maybe_search_config in search_configs:
                if isinstance(maybe_search_config, dict):
                    maybe_search_config = PSC(**maybe_search_config)
                if isinstance(maybe_search_config, PSC):
                    for col in range(arr_2d.shape[1]):
                        flat_search_configs.append(maybe_search_config)
                        if maybe_search_config.name is not None:
                            psc_names.append(maybe_search_config.name)
                            psc_names_none = False
                        else:
                            psc_names.append(n_configs)
                    n_configs += 1
                else:
                    if len(maybe_search_config) != arr_2d.shape[1]:
                        raise ValueError(
                            "Sub-list with PSC instances must match the number of columns"
                        )
                    for col, search_config in enumerate(maybe_search_config):
                        if isinstance(search_config, dict):
                            search_config = PSC(**search_config)
                        flat_search_configs.append(search_config)
                        if search_config.name is not None:
                            psc_names.append(search_config.name)
                            psc_names_none = False
                        else:
                            psc_names.append(n_configs)
                        n_configs += 1

        # Combine parameters
        param_dct = {}
        for k, v in method_locals.items():
            if k in psc_keys and isinstance(v, Param):
                param_dct[k] = v
        param_columns = None
        if len(param_dct) > 0:
            param_product, param_columns = combine_params(
                param_dct,
                random_subset=random_subset,
                clean_index_kwargs=clean_index_kwargs,
            )
            if len(flat_search_configs) == 0:
                flat_search_configs = []
                for i in range(len(param_columns)):
                    search_config = dict()
                    for k, v in param_product.items():
                        search_config[k] = v[i]
                    for col in range(arr_2d.shape[1]):
                        flat_search_configs.append(PSC(**search_config))
            else:
                new_flat_search_configs = []
                for i in range(len(param_columns)):
                    for search_config in flat_search_configs:
                        new_search_config = dict()
                        for k, v in search_config.asdict().items():
                            if v is not MISSING:
                                if k in param_product:
                                    raise ValueError(
                                        f"Parameter '{k}' is re-defined in a search configuration"
                                    )
                                new_search_config[k] = v
                            if k in param_product:
                                new_search_config[k] = param_product[k][i]
                        new_flat_search_configs.append(PSC(**new_search_config))
                flat_search_configs = new_flat_search_configs

        # Create config from arguments if empty
        if len(flat_search_configs) == 0:
            single_group = True
            for col in range(arr_2d.shape[1]):
                flat_search_configs.append(PSC())
        else:
            single_group = False

        # Prepare function and arguments
        tasks = []
        func = jit_reg.resolve_option(nb.find_pattern_1d_nb, jitted)
        def_func_kwargs = get_func_kwargs(func)
        new_search_configs = []
        for c in range(len(flat_search_configs)):
            func_kwargs = {
                "col": c,
                "arr": arr_2d[:, c % arr_2d.shape[1]],
            }
            new_search_config = cls.resolve_search_config(flat_search_configs[c], **method_locals)
            for k, v in new_search_config.asdict().items():
                if k == "name":
                    continue
                if isinstance(v, Param):
                    raise TypeError("Cannot use Param inside search configs")
                if k in def_func_kwargs:
                    if v is not def_func_kwargs[k]:
                        func_kwargs[k] = v
                else:
                    func_kwargs[k] = v
            tasks.append(Task(func, **func_kwargs))
            new_search_configs.append(new_search_config)

        # Build column hierarchy
        n_config_params = len(psc_names) // arr_2d.shape[1]
        if param_columns is not None:
            if n_config_params == 0 or (n_config_params == 1 and psc_names_none):
                new_columns = combine_indexes(
                    (param_columns, arr_wrapper.columns), **clean_index_kwargs
                )
            else:
                search_config_index = pd.Index(psc_names, name="search_config")
                base_columns = stack_indexes(
                    (search_config_index, tile_index(arr_wrapper.columns, n_config_params)),
                    **clean_index_kwargs,
                )
                new_columns = combine_indexes((param_columns, base_columns), **clean_index_kwargs)
        else:
            if n_config_params == 0 or (n_config_params == 1 and psc_names_none):
                new_columns = arr_wrapper.columns
            else:
                search_config_index = pd.Index(psc_names, name="search_config")
                new_columns = stack_indexes(
                    (search_config_index, tile_index(arr_wrapper.columns, n_config_params)),
                    **clean_index_kwargs,
                )

        # Execute each configuration
        execute_kwargs = merge_dicts(
            dict(show_progress=False if single_group else None), execute_kwargs
        )
        result_list = execute(tasks, keys=new_columns, **execute_kwargs)
        records_arr = np.concatenate(result_list)

        # Wrap with class
        wrapper = ArrayWrapper(
            **merge_dicts(
                dict(
                    index=arr_wrapper.index,
                    columns=new_columns,
                ),
                wrapper_kwargs,
            )
        )
        if attach_as_close and "close" not in kwargs:
            kwargs["close"] = arr
        if "open" in kwargs and kwargs["open"] is not None:
            kwargs["open"] = to_2d_array(kwargs["open"])
            kwargs["open"] = tile(kwargs["open"], len(wrapper.columns) // kwargs["open"].shape[1])
        if "high" in kwargs and kwargs["high"] is not None:
            kwargs["high"] = to_2d_array(kwargs["high"])
            kwargs["high"] = tile(kwargs["high"], len(wrapper.columns) // kwargs["high"].shape[1])
        if "low" in kwargs and kwargs["low"] is not None:
            kwargs["low"] = to_2d_array(kwargs["low"])
            kwargs["low"] = tile(kwargs["low"], len(wrapper.columns) // kwargs["low"].shape[1])
        if "close" in kwargs and kwargs["close"] is not None:
            kwargs["close"] = to_2d_array(kwargs["close"])
            kwargs["close"] = tile(
                kwargs["close"], len(wrapper.columns) // kwargs["close"].shape[1]
            )
        return cls(wrapper, records_arr, new_search_configs, **kwargs)

    def with_delta(self, *args, **kwargs) -> Ranges:
        """Return a new range by calling `Ranges.from_delta` with the instance's index set to its last index.

        Args:
            *args: Positional arguments for `Ranges.from_delta`.
            **kwargs: Keyword arguments for `Ranges.from_delta`.

                If 'idx_field_or_arr' is not provided, it defaults to the instance's last index values.

        Returns:
            Ranges: Resulting range object from `Ranges.from_delta`.
        """
        if "idx_field_or_arr" not in kwargs:
            kwargs["idx_field_or_arr"] = self.last_idx.values
        return Ranges.from_delta(self, *args, **kwargs)

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[PatternRangesT],
        *objs: tp.MaybeSequence[PatternRangesT],
        **kwargs,
    ) -> tp.Kwargs:
        kwargs = Ranges.resolve_row_stack_kwargs(*objs, **kwargs)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PatternRanges):
                raise TypeError("Each object to be merged must be an instance of PatternRanges")
        new_search_configs = []
        for obj in objs:
            if len(obj.search_configs) == 1:
                new_search_configs.append(obj.search_configs * len(kwargs["wrapper"].columns))
            else:
                new_search_configs.append(obj.search_configs)
            if len(new_search_configs) >= 2:
                if new_search_configs[-1] != new_search_configs[0]:
                    raise ValueError(
                        "Objects to be merged must have compatible PSC instances. Pass to override."
                    )
        kwargs["search_configs"] = new_search_configs[0]
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[PatternRangesT],
        *objs: tp.MaybeSequence[PatternRangesT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing a `PatternRanges` instance after column stacking.

        Args:
            *objs (MaybeSequence[PatternRanges]): `PatternRanges` instances to be stacked.
            **kwargs: Keyword arguments for `Ranges.resolve_column_stack_kwargs`.

        Returns:
            Kwargs: Resolved keyword arguments with an aggregated 'search_configs'
                field from the merged objects.
        """
        kwargs = Ranges.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs.pop("reindex_kwargs", None)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PatternRanges):
                raise TypeError("Each object to be merged must be an instance of PatternRanges")
        kwargs["search_configs"] = [
            search_config for obj in objs for search_config in obj.search_configs
        ]
        return kwargs

    def indexing_func(
        self: PatternRangesT, *args, ranges_meta: tp.DictLike = None, **kwargs
    ) -> PatternRangesT:
        """Index a `PatternRanges` instance and return a new instance with updated indexing
        and search configurations.

        Args:
            *args: Positional arguments for `Ranges.indexing_func_meta`.
            ranges_meta (DictLike): Metadata for indexing.

                If None, it is generated using `Ranges.indexing_func_meta`.
            **kwargs: Keyword arguments for `Ranges.indexing_func_meta`.

        Returns:
            PatternRanges: New `PatternRanges` instance with updated indexing information.
        """
        if ranges_meta is None:
            ranges_meta = Ranges.indexing_func_meta(self, *args, **kwargs)
        col_idxs = ranges_meta["wrapper_meta"]["col_idxs"]
        if not isinstance(col_idxs, slice):
            col_idxs = to_1d_array(col_idxs)
        col_idxs = np.arange(self.wrapper.shape_2d[1])[col_idxs]
        new_search_configs = []
        for i in col_idxs:
            new_search_configs.append(self.search_configs[i])
        return self.replace(
            wrapper=ranges_meta["wrapper_meta"]["new_wrapper"],
            records_arr=ranges_meta["new_records_arr"],
            search_configs=new_search_configs,
            open=ranges_meta["open"],
            high=ranges_meta["high"],
            low=ranges_meta["low"],
            close=ranges_meta["close"],
        )

    # ############# Stats ############# #

    _metrics: tp.ClassVar[Config] = HybridConfig(
        {
            **Ranges.metrics,
            "similarity": dict(
                title="Similarity",
                calc_func="similarity.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                },
                tags=["pattern_ranges", "similarity"],
            ),
        }
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plots ############# #

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        top_n: tp.Optional[int] = None,
        fit_ranges: tp.Union[bool, tp.MaybeSequence[int]] = False,
        plot_patterns: bool = True,
        plot_max_error: bool = False,
        fill_distance: bool = True,
        pattern_trace_kwargs: tp.KwargsLike = None,
        lower_max_error_trace_kwargs: tp.KwargsLike = None,
        upper_max_error_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot pattern ranges.

        Based on `Ranges.plot` and `vectorbtpro.generic.accessors.GenericSRAccessor.plot_pattern`.

        Args:
            column (Optional[Column]): Identifier of the column to plot.
            top_n (Optional[int]): Display only the top N records sorted by maximum duration.
            fit_ranges (Union[bool, MaybeSequence[int]]): Select the range records to fit.

                Use True to fit all records, or provide an integer or sequence of
                integers to select specific records.
            plot_patterns (bool): Indicates whether to plot the `PSC.pattern`.
            plot_max_error (bool): Indicates whether to plot the `PSC.max_error`.
            fill_distance (bool): Controls whether to fill the space between close and pattern.

                Visible for every interpolation mode except discrete.
            pattern_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the pattern.
            lower_max_error_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the lower error bound.
            upper_max_error_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the upper error bound.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            xref (str): Reference for the x-axis (e.g., "x", "x2").
            yref (str): Reference for the y-axis (e.g., "y", "y2").
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **kwargs: Keyword arguments for `Ranges.plot`.

        Returns:
            BaseFigure: Figure with plotted pattern ranges.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))
        search_config = self_col.search_configs[0]
        if isinstance(fit_ranges, bool) and not fit_ranges:
            fit_ranges = None
        if fit_ranges is not None:
            if fit_ranges is True:
                self_col = self_col.iloc[
                    self_col.values["start_idx"][0] : self_col.values["end_idx"][-1] + 1
                ]
            elif checks.is_int(fit_ranges):
                self_col = self_col.apply_mask(self_col.id_arr == fit_ranges)
                self_col = self_col.iloc[
                    self_col.values["start_idx"][0] : self_col.values["end_idx"][0] + 1
                ]
            else:
                self_col = self_col.apply_mask(np.isin(self_col.id_arr, fit_ranges))
                self_col = self_col.iloc[
                    self_col.values["start_idx"][0] : self_col.values["end_idx"][0] + 1
                ]
        if pattern_trace_kwargs is None:
            pattern_trace_kwargs = {}
        if lower_max_error_trace_kwargs is None:
            lower_max_error_trace_kwargs = {}
        if upper_max_error_trace_kwargs is None:
            upper_max_error_trace_kwargs = {}

        open_shape_kwargs = merge_dicts(
            dict(fillcolor=plotting_cfg["contrast_color_schema"]["blue"]),
            kwargs.pop("open_shape_kwargs", None),
        )
        closed_shape_kwargs = merge_dicts(
            dict(fillcolor=plotting_cfg["contrast_color_schema"]["blue"]),
            kwargs.pop("closed_shape_kwargs", None),
        )
        fig, close = Ranges.plot(
            self_col,
            return_close=True,
            open_shape_kwargs=open_shape_kwargs,
            closed_shape_kwargs=closed_shape_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            xref=xref,
            yref=yref,
            fig=fig,
            **kwargs,
        )

        if self_col.count() > 0:
            # Extract information
            start_idx = self_col.get_map_field_to_index("start_idx", minus_one_to_zero=True)
            end_idx = self_col.get_map_field_to_index("end_idx")
            status = self_col.get_field_arr("status")

            if plot_patterns:
                # Plot pattern
                for r in range(len(start_idx)):
                    _start_idx = start_idx[r]
                    _end_idx = end_idx[r]
                    if close is None:
                        raise ValueError("Must provide close to overlay patterns")
                    arr_sr = close.loc[_start_idx:_end_idx]
                    if status[r] == enums.RangeStatus.Closed:
                        arr_sr = arr_sr.iloc[:-1]
                    if fill_distance:
                        obj_trace_kwargs = dict(
                            line=dict(color="rgba(0, 0, 0, 0)", width=0),
                            opacity=0,
                            hoverinfo="skip",
                            showlegend=False,
                            name=None,
                        )
                    else:
                        obj_trace_kwargs = None
                    _pattern_trace_kwargs = merge_dicts(
                        dict(
                            legendgroup="pattern",
                            showlegend=r == 0,
                        ),
                        pattern_trace_kwargs,
                    )
                    _lower_max_error_trace_kwargs = merge_dicts(
                        dict(
                            legendgroup="max_error",
                            showlegend=r == 0,
                        ),
                        lower_max_error_trace_kwargs,
                    )
                    _upper_max_error_trace_kwargs = merge_dicts(
                        dict(
                            legendgroup="max_error",
                            showlegend=False,
                        ),
                        upper_max_error_trace_kwargs,
                    )

                    fig = arr_sr.vbt.plot_pattern(
                        pattern=search_config.pattern,
                        interp_mode=search_config.interp_mode,
                        rescale_mode=search_config.rescale_mode,
                        vmin=search_config.vmin,
                        vmax=search_config.vmax,
                        pmin=search_config.pmin,
                        pmax=search_config.pmax,
                        invert=search_config.invert,
                        error_type=search_config.error_type,
                        max_error=search_config.max_error if plot_max_error else np.nan,
                        max_error_interp_mode=search_config.max_error_interp_mode,
                        plot_obj=fill_distance,
                        fill_distance=fill_distance,
                        obj_trace_kwargs=obj_trace_kwargs,
                        pattern_trace_kwargs=_pattern_trace_kwargs,
                        lower_max_error_trace_kwargs=_lower_max_error_trace_kwargs,
                        upper_max_error_trace_kwargs=_upper_max_error_trace_kwargs,
                        add_trace_kwargs=add_trace_kwargs,
                        fig=fig,
                    )

        return fig


PatternRanges.override_field_config_doc(__pdoc__)
PatternRanges.override_metrics_doc(__pdoc__)
