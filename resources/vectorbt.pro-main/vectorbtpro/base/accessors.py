# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing custom Pandas accessors for base operations with Pandas objects."""

import ast
import inspect

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler

from vectorbtpro import _typing as tp
from vectorbtpro.base import combining, indexes, reshaping
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.indexes import IndexApplier
from vectorbtpro.base.indexing import (
    get_index_points,
    get_index_ranges,
    point_idxr_defaults,
    range_idxr_defaults,
)
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.chunking import (
    ArraySelector,
    ArraySlicer,
    ChunkMeta,
    get_chunk_meta_key,
    iter_chunk_meta,
)
from vectorbtpro.utils.config import Configured, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import hybrid_method, hybrid_property
from vectorbtpro.utils.eval_ import evaluate
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.magic_decorators import (
    attach_binary_magic_methods,
    attach_unary_magic_methods,
)
from vectorbtpro.utils.parsing import get_context_vars, get_func_arg_names
from vectorbtpro.utils.template import substitute_templates
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from vectorbtpro.data.base import Data as DataT
else:
    DataT = "vectorbtpro.data.base.Data"
if tp.TYPE_CHECKING:
    from vectorbtpro.generic.splitting.base import Splitter as SplitterT
else:
    SplitterT = "vectorbtpro.generic.splitting.base.Splitter"

__all__ = ["BaseIDXAccessor", "BaseAccessor", "BaseSRAccessor", "BaseDFAccessor"]

BaseIDXAccessorT = tp.TypeVar("BaseIDXAccessorT", bound="BaseIDXAccessor")


class BaseIDXAccessor(Configured, IndexApplier):
    """Class representing an accessor on top of Index.

    Accessible via `pd.Index.vbt` and all child accessors.

    Args:
        obj (Index): Pandas Index object to be wrapped by the accessor.
        freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

            See `vectorbtpro.utils.datetime_.infer_index_freq`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(self, obj: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None, **kwargs) -> None:
        checks.assert_instance_of(obj, pd.Index)

        Configured.__init__(self, obj=obj, freq=freq, **kwargs)

        self._obj = obj
        self._freq = freq

    @property
    def obj(self) -> tp.Index:
        """Pandas Index object.

        Returns:
            Index: Underlying Pandas Index object.
        """
        return self._obj

    def get(self) -> tp.Index:
        """Return Pandas Index object.

        Returns:
            Index: Underlying Pandas Index object.
        """
        return self.obj

    # ############# Index ############# #

    def to_ns(self) -> tp.Array1d:
        """Convert the index to a 64-bit integer array.

        Timestamps are converted to nanoseconds.

        Returns:
            Array1d: Resulting 64-bit integer array.
        """
        return dt.to_ns(self.obj)

    def to_period(self, freq: tp.FrequencyLike, shift: bool = False) -> tp.PeriodIndex:
        """Convert the index to a PeriodIndex.

        Args:
            freq (FrequencyLike): Frequency of the period index.
            shift (bool): If True, shift the resulting period.

        Returns:
            PeriodIndex: Converted PeriodIndex.
        """
        index = self.obj
        if isinstance(index, pd.DatetimeIndex):
            index = index.tz_localize(None).to_period(freq)
            if shift:
                index = index.shift()
        if not isinstance(index, pd.PeriodIndex):
            raise TypeError(f"Cannot convert index of type {type(index)} to period")
        return index

    def to_period_ts(self, *args, **kwargs) -> tp.DatetimeIndex:
        """Convert the index to a DatetimeIndex.

        The index is first converted to a PeriodIndex and then to timestamps.

        Args:
            *args: Positional arguments for `BaseIDXAccessor.to_period`.
            **kwargs: Keyword arguments for `BaseIDXAccessor.to_period`.

        Returns:
            DatetimeIndex: Resulting timestamp index.
        """
        new_index = self.to_period(*args, **kwargs).to_timestamp()
        if self.obj.tz is not None:
            new_index = new_index.tz_localize(self.obj.tz)
        return new_index

    def to_period_ns(self, *args, **kwargs) -> tp.Array1d:
        """Convert the index to a 64-bit integer array via timestamps.

        The index is first converted to a PeriodIndex, then to a DatetimeIndex,
        and finally to nanoseconds.

        Args:
            *args: Positional arguments for `BaseIDXAccessor.to_period_ts`.
            **kwargs: Keyword arguments for `BaseIDXAccessor.to_period_ts`.

        Returns:
            Array1d: Resulting 64-bit integer array.
        """
        return dt.to_ns(self.to_period_ts(*args, **kwargs))

    @classmethod
    def from_values(cls, *args, **kwargs) -> tp.Index:
        """Return an index created from values using `vectorbtpro.base.indexes.index_from_values`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexes.index_from_values`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.index_from_values`.

        Returns:
            Index: Generated index.
        """
        return indexes.index_from_values(*args, **kwargs)

    def repeat(self, *args, **kwargs) -> tp.Index:
        """Return an index with repeated values using `vectorbtpro.base.indexes.repeat_index`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexes.repeat_index`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.repeat_index`.

        Returns:
            Index: Index with repeated values.
        """
        return indexes.repeat_index(self.obj, *args, **kwargs)

    def tile(self, *args, **kwargs) -> tp.Index:
        """Return an index tiled using `vectorbtpro.base.indexes.tile_index`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexes.tile_index`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.tile_index`.

        Returns:
            Index: Tiled index.
        """
        return indexes.tile_index(self.obj, *args, **kwargs)

    @hybrid_method
    def stack(
        cls_or_self,
        *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"],
        on_top: bool = False,
        **kwargs,
    ) -> tp.Index:
        """Stack multiple indexes using `vectorbtpro.base.indexes.stack_indexes`.

        Args:
            *others (Union[IndexLike, BaseIDXAccessor]): (Additional) indexes or accessors to stack.
            on_top (bool): If True, add the new levels before the existing index; if False, add them after.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.stack_indexes`.

        Returns:
            Index: Stacked index.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            if on_top:
                objs = (*others, cls_or_self.obj)
            else:
                objs = (cls_or_self.obj, *others)
        return indexes.stack_indexes(*objs, **kwargs)

    @hybrid_method
    def combine(
        cls_or_self,
        *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"],
        on_top: bool = False,
        **kwargs,
    ) -> tp.Index:
        """Combine multiple indexes using `vectorbtpro.base.indexes.combine_indexes`.

        Args:
            *others (Union[IndexLike, BaseIDXAccessor]): (Additional) indexes or accessors to combine.
            on_top (bool): If True, add the new levels before the existing index; if False, add them after.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.combine_indexes`.

        Returns:
            Index: Combined index.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            if on_top:
                objs = (*others, cls_or_self.obj)
            else:
                objs = (cls_or_self.obj, *others)
        return indexes.combine_indexes(*objs, **kwargs)

    @hybrid_method
    def concat(
        cls_or_self, *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"], **kwargs
    ) -> tp.Index:
        """Concatenate multiple indexes using `vectorbtpro.base.indexes.concat_indexes`.

        Args:
            *others (Union[IndexLike, BaseIDXAccessor]): (Additional) indexes or accessors to concatenate.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.concat_indexes`.

        Returns:
            Index: Concatenated index.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        return indexes.concat_indexes(*objs, **kwargs)

    def apply_to_index(
        self: BaseIDXAccessorT,
        apply_func: tp.Callable,
        *args,
        **kwargs,
    ) -> tp.Index:
        return self.replace(obj=apply_func(self.obj, *args, **kwargs)).obj

    def align_to(self, *args, **kwargs) -> tp.IndexSlice:
        """Align the index to a target index using `vectorbtpro.base.indexes.align_index_to`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexes.align_index_to`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.align_index_to`.

        Returns:
            IndexSlice: Aligned index slice.
        """
        return indexes.align_index_to(self.obj, *args, **kwargs)

    @hybrid_method
    def align(
        cls_or_self,
        *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"],
        **kwargs,
    ) -> tp.Tuple[tp.IndexSlice, ...]:
        """Align multiple indexes using `vectorbtpro.base.indexes.align_indexes`.

        Args:
            *others (Union[IndexLike, BaseIDXAccessor]): (Additional) indexes or accessors to align.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.align_indexes`.

        Returns:
            Tuple[IndexSlice, ...]: Tuple of aligned index slices.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        return indexes.align_indexes(*objs, **kwargs)

    def cross_with(self, *args, **kwargs) -> tp.Tuple[tp.IndexSlice, tp.IndexSlice]:
        """Cross the current index with another index using `vectorbtpro.base.indexes.cross_index_with`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexes.cross_index_with`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.cross_index_with`.

        Returns:
            Tuple[IndexSlice, IndexSlice]: Resulting pair of index slices.
        """
        return indexes.cross_index_with(self.obj, *args, **kwargs)

    @hybrid_method
    def cross(
        cls_or_self,
        *others: tp.Union[tp.IndexLike, "BaseIDXAccessor"],
        **kwargs,
    ) -> tp.Tuple[tp.IndexSlice, ...]:
        """Cross multiple indexes using `vectorbtpro.base.indexes.cross_indexes`.

        Args:
            *others (Union[IndexLike, BaseIDXAccessor]): (Additional) indexes or accessors to cross.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.cross_indexes`.

        Returns:
            Tuple[IndexSlice, ...]: Resulting crossed indexes.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseIDXAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        return indexes.cross_indexes(*objs, **kwargs)

    x = cross

    def find_first_occurrence(self, *args, **kwargs) -> int:
        """Find the first occurrence of a value in the index using `vectorbtpro.base.indexes.find_first_occurrence`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexes.find_first_occurrence`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.find_first_occurrence`.

        Returns:
            int: Index of the first occurrence.
        """
        return indexes.find_first_occurrence(self.obj, *args, **kwargs)

    # ############# Frequency ############# #

    @hybrid_method
    def get_freq(
        cls_or_self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> tp.Union[None, float, tp.PandasFrequency]:
        """Determine the frequency of the index as a Pandas Timedelta or frequency.

        Args:
            index (Optional[Index]): Index from which to infer the frequency.

                If None, the accessor's index is used.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            Union[None, float, PandasFrequency]: Inferred frequency, or None if conversion fails.

        !!! info
            For default settings, see `vectorbtpro._settings.wrapping`.
        """
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if not isinstance(cls_or_self, type):
            if index is None:
                index = cls_or_self.obj
            if freq is None:
                freq = cls_or_self._freq
        else:
            checks.assert_not_none(index, arg_name="index")

        if freq is None:
            freq = wrapping_cfg["freq"]
        try:
            return dt.infer_index_freq(index, freq=freq, **kwargs)
        except Exception:
            return None

    @property
    def freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Index frequency excluding date offsets and numeric frequencies.

        Uses `BaseIDXAccessor.get_freq` with specific restrictions.

        Returns:
            Optional[PandasFrequency]: Frequency of the index, or None if not set.
        """
        return self.get_freq(allow_offset=True, allow_numeric=False)

    @property
    def ns_freq(self) -> tp.Optional[int]:
        """Frequency of the index as a 64-bit integer in nanoseconds.

        Timestamps are converted to nanoseconds via Timedelta.

        Returns:
            Optional[int]: Frequency in nanoseconds, or None if not set.
        """
        freq = self.get_freq(allow_offset=False, allow_numeric=True)
        if freq is not None:
            freq = dt.to_ns(dt.to_timedelta64(freq))
        return freq

    @property
    def any_freq(self) -> tp.Union[None, float, tp.PandasFrequency]:
        """Frequency of the index of any type using `BaseIDXAccessor.get_freq`.

        Returns:
            Union[None, float, PandasFrequency]: Frequency of the index, or None if not set.
        """
        return self.get_freq()

    @hybrid_method
    def get_periods(cls_or_self, index: tp.Optional[tp.Index] = None) -> int:
        """Return the number of periods in the index without considering datetime-like properties.

        Args:
            index (Optional[Index]): Index for which to count periods.

                If None, the accessor's index is used.

        Returns:
            int: Number of periods in the index.
        """
        if not isinstance(cls_or_self, type):
            if index is None:
                index = cls_or_self.obj
        else:
            checks.assert_not_none(index, arg_name="index")
        return len(index)

    @property
    def periods(self) -> int:
        """Number of periods in the index.

        Returns:
            int: Computed number of periods.
        """
        return len(self.obj)

    @hybrid_method
    def get_dt_periods(
        cls_or_self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> float:
        """Return the number of periods in the index, accounting for its datetime-like properties.

        Args:
            index (Optional[Index]): Index to process.

                If omitted and invoked on an instance, the object's index is used.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            float: Calculated number of periods.

        !!! info
            For default settings, see `vectorbtpro._settings.wrapping`.
        """
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if not isinstance(cls_or_self, type):
            if index is None:
                index = cls_or_self.obj
        else:
            checks.assert_not_none(index, arg_name="index")

        if isinstance(index, pd.DatetimeIndex):
            freq = cls_or_self.get_freq(
                index=index, freq=freq, allow_offset=True, allow_numeric=False
            )
            if freq is not None:
                if not isinstance(freq, pd.Timedelta):
                    freq = dt.to_timedelta(freq, approximate=True)
                return (index[-1] - index[0]) / freq + 1
            if not wrapping_cfg["silence_warnings"]:
                warn(
                    "Couldn't parse the frequency of index. Pass it as `freq` or "
                    "define it globally under `settings.wrapping`."
                )
        if checks.is_number(index[0]) and checks.is_number(index[-1]):
            freq = cls_or_self.get_freq(
                index=index, freq=freq, allow_offset=False, allow_numeric=True
            )
            if checks.is_number(freq):
                return (index[-1] - index[0]) / freq + 1
            return index[-1] - index[0] + 1
        if not wrapping_cfg["silence_warnings"]:
            warn("Index is neither datetime-like nor integer")
        return cls_or_self.get_periods(index=index)

    @property
    def dt_periods(self) -> float:
        """Datetime period count in the index using default parameters.

        Returns:
            float: Computed number of datetime periods.
        """
        return self.get_dt_periods()

    def arr_to_timedelta(
        self,
        a: tp.MaybeArray,
        to_pd: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[pd.Index, tp.MaybeArray]:
        """Convert an array of values to a duration based on the index frequency from `BaseIDXAccessor.freq`.

        Args:
            a (MaybeArray): Input array containing numerical values.
            to_pd (bool): Determines whether to return a Pandas timedelta representation.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            Union[pd.Index, MaybeArray]: Array converted to time durations.

        !!! info
            For default settings, see `vectorbtpro._settings.wrapping`.
        """
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        freq = self.freq
        if freq is None:
            if not silence_warnings:
                warn(
                    "Couldn't parse the frequency of index. Pass it as `freq` or "
                    "define it globally under `settings.wrapping`."
                )
            return a
        if not isinstance(freq, pd.Timedelta):
            freq = dt.to_timedelta(freq, approximate=True)
        if to_pd:
            out = pd.to_timedelta(a * freq)
        else:
            out = a * freq
        return out

    # ############# Grouping ############# #

    def get_grouper(
        self, by: tp.AnyGroupByLike, groupby_kwargs: tp.KwargsLike = None, **kwargs
    ) -> Grouper:
        """Return an index grouper of type `vectorbtpro.base.grouping.base.Grouper`.

        Args:
            by (AnyGroupByLike): Grouper-like specification.

                Can be one of the following:

                * `vectorbtpro.base.grouping.base.Grouper` instance
                * Pandas `GroupBy` instance
                * Pandas `Resampler` instance
                * Instruction for any of the above
            groupby_kwargs (KwargsLike): Keyword arguments for `pandas.Series.groupby` and
                `pandas.Series.resample` methods.
            **kwargs: Keyword arguments for initializing `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Grouper: Constructed index grouper.
        """
        if groupby_kwargs is None:
            groupby_kwargs = {}
        if isinstance(by, Grouper):
            if len(kwargs) > 0:
                return by.replace(**kwargs)
            return by
        if isinstance(by, (PandasGroupBy, PandasResampler)):
            return Grouper.from_pd_group_by(by, **kwargs)
        try:
            return Grouper(index=self.obj, group_by=by, **kwargs)
        except Exception:
            pass
        if isinstance(self.obj, pd.DatetimeIndex):
            try:
                return Grouper(index=self.obj, group_by=self.to_period(dt.to_freq(by)), **kwargs)
            except Exception:
                pass
            try:
                pd_group_by = pd.Series(index=self.obj, dtype=object).resample(
                    dt.to_freq(by), **groupby_kwargs
                )
                return Grouper.from_pd_group_by(pd_group_by, **kwargs)
            except Exception:
                pass
        pd_group_by = pd.Series(index=self.obj, dtype=object).groupby(by, axis=0, **groupby_kwargs)
        return Grouper.from_pd_group_by(pd_group_by, **kwargs)

    def get_resampler(
        self,
        rule: tp.AnyRuleLike,
        freq: tp.Optional[tp.FrequencyLike] = None,
        resample_kwargs: tp.KwargsLike = None,
        return_pd_resampler: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[Resampler, tp.PandasResampler]:
        """Return an index resampler of type `vectorbtpro.base.resampling.base.Resampler`.

        Args:
            rule (AnyRuleLike): Resampler-like specification.

                Can be one of the following:

                * `vectorbtpro.base.resampling.base.Resampler` instance
                * Pandas `Resampler` instance
                * Instruction for any of the above
                * Datetime-like object or iterable used to create a new index
            freq (Optional[FrequencyLike]): Frequency of the target index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            resample_kwargs (KwargsLike): Keyword arguments for `pandas.Series.resample`.
            return_pd_resampler (bool): Flag indicating whether to return a Pandas resampler.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            Union[Resampler, PandasResampler]: Constructed index resampler.
        """
        if checks.is_frequency_like(rule):
            try:
                rule = dt.to_freq(rule)
                is_td = True
            except Exception:
                is_td = False
            if is_td:
                resample_kwargs = merge_dicts(
                    dict(closed="left", label="left"),
                    resample_kwargs,
                )
                rule = pd.Series(index=self.obj, dtype=object).resample(
                    rule, **resolve_dict(resample_kwargs)
                )
        if isinstance(rule, PandasResampler):
            if return_pd_resampler:
                return rule
            if silence_warnings is None:
                silence_warnings = True
            rule = Resampler.from_pd_resampler(
                rule, source_freq=self.freq, silence_warnings=silence_warnings
            )
        if return_pd_resampler:
            raise TypeError("Cannot convert Resampler to Pandas Resampler")
        if checks.is_dt_like(rule) or checks.is_iterable(rule):
            rule = dt.prepare_dt_index(rule)
            rule = Resampler(
                source_index=self.obj,
                target_index=rule,
                source_freq=self.freq,
                target_freq=freq,
                silence_warnings=silence_warnings,
            )
        if isinstance(rule, Resampler):
            if freq is not None:
                rule = rule.replace(target_freq=freq)
            return rule
        raise ValueError(f"Cannot build Resampler from {rule}")

    # ############# Points and ranges ############# #

    def get_points(self, *args, **kwargs) -> tp.Array1d:
        """Return index points using `vectorbtpro.base.indexing.get_index_points`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexing.get_index_points`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexing.get_index_points`.

        Returns:
            Array1d: Array of index points.
        """
        return get_index_points(self.obj, *args, **kwargs)

    def get_ranges(self, *args, **kwargs) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Return index ranges using `vectorbtpro.base.indexing.get_index_ranges`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.indexing.get_index_ranges`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexing.get_index_ranges`.

        Returns:
            Tuple[Array1d, Array1d]: Tuple of arrays representing the index ranges.
        """
        return get_index_ranges(self.obj, self.any_freq, *args, **kwargs)

    # ############# Splitting ############# #

    def split(
        self, *args, splitter_cls: tp.Optional[tp.Type[SplitterT]] = None, **kwargs
    ) -> tp.Any:
        """Return the result of splitting the Pandas object using
        `vectorbtpro.generic.splitting.base.Splitter.split_and_take`.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_take`.
            splitter_cls (Optional[Type[Splitter]]): Splitter class to use.

                Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_take`.

        Returns:
            Any: Result of the split operation.

        !!! note
            Splits the Pandas object itself, not the accessor.
        """
        from vectorbtpro.generic.splitting.base import Splitter

        if splitter_cls is None:
            splitter_cls = Splitter
        return splitter_cls.split_and_take(self.obj, self.obj, *args, **kwargs)

    def split_apply(
        self,
        apply_func: tp.Callable,
        *args,
        splitter_cls: tp.Optional[tp.Type[SplitterT]] = None,
        **kwargs,
    ) -> tp.Any:
        """Return the result of splitting and applying a function using
        `vectorbtpro.generic.splitting.base.Splitter.split_and_apply`.

        Args:
            apply_func (Callable): Function to apply to each split.
            *args: Positional arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_apply`.
            splitter_cls (Optional[Type[Splitter]]): Splitter class to use.

                Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
            **kwargs: Keyword arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_apply`.

        Returns:
            Any: Result after applying the split and apply operation.

        !!! note
            Splits the Pandas object itself, not the accessor.
        """
        from vectorbtpro.generic.splitting.base import Splitter, Takeable

        if splitter_cls is None:
            splitter_cls = Splitter
        return splitter_cls.split_and_apply(
            self.obj, apply_func, Takeable(self.obj), *args, **kwargs
        )

    # ############# Chunking ############# #

    def chunk(
        self: BaseIDXAccessorT,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Union[None, int, str] = None,
        chunk_len: tp.Union[None, int, str] = None,
        chunk_meta: tp.Optional[tp.Iterable[ChunkMeta]] = None,
        select: bool = False,
        return_chunk_meta: bool = False,
    ) -> tp.Iterator[tp.Union[tp.Index, tp.Tuple[ChunkMeta, tp.Index]]]:
        """Chunk the instance and yield segments from its underlying Pandas object.

        The chunking is performed along the appropriate axis: if the object is one-dimensional,
        the axis is set to 0; otherwise, 1 is used.

        Args:
            min_size (Optional[int]): Minimum number of elements to split.
            n_chunks (Union[None, int, str]): Specification for the number of chunks.
            chunk_len (Union[None, int, str]): Specification for the length of each chunk.
            chunk_meta (Optional[Iterable[ChunkMeta]]): Iterable containing metadata for each chunk.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.
            select (bool): Determines whether to use `ArraySelector` (if True) or
                `ArraySlicer` (if False) for extracting the chunk.
            return_chunk_meta (bool): Flag indicating whether to yield chunk metadata alongside each chunk.

        Yields:
            Union[Index, Tuple[ChunkMeta, Index]]: Index chunk or a tuple with
                the chunk metadata and index chunk.

        !!! note
            Splits the underlying Pandas object, not the accessor.
        """
        if chunk_meta is None:
            chunk_meta = iter_chunk_meta(
                size=len(self.obj), min_size=min_size, n_chunks=n_chunks, chunk_len=chunk_len
            )
        for _chunk_meta in chunk_meta:
            if select:
                array_taker = ArraySelector()
            else:
                array_taker = ArraySlicer()
            if return_chunk_meta:
                yield _chunk_meta, array_taker.take(self.obj, _chunk_meta)
            else:
                yield array_taker.take(self.obj, _chunk_meta)

    def chunk_apply(
        self: BaseIDXAccessorT,
        apply_func: tp.Union[str, tp.Callable],
        *args,
        chunk_kwargs: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MergeableResults:
        """Chunk the instance and apply a function to each generated chunk.

        The function can be specified as a callable or as a string representing a method name.
        Each chunk is derived from the instance's underlying Pandas object.

        Args:
            apply_func (Union[str, Callable]): Function or method name to apply to each chunk.
            *args: Positional arguments for `apply_func`.
            chunk_kwargs (KwargsLike): Keyword arguments for the chunking handler.

                See `BaseIDXAccessor.chunk`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            MergeableResults: Merged results from applying the function on all chunks.

        !!! note
            Splits the underlying Pandas object, not the accessor.
        """
        if isinstance(apply_func, str):
            apply_func = getattr(type(self), apply_func)
        if chunk_kwargs is None:
            chunk_arg_names = set(get_func_arg_names(self.chunk))
            chunk_kwargs = {}
            for k in list(kwargs.keys()):
                if k in chunk_arg_names:
                    chunk_kwargs[k] = kwargs.pop(k)
        if execute_kwargs is None:
            execute_kwargs = {}
        chunks = self.chunk(return_chunk_meta=True, **chunk_kwargs)
        tasks = []
        keys = []
        for _chunk_meta, chunk in chunks:
            tasks.append(Task(apply_func, chunk, *args, **kwargs))
            keys.append(get_chunk_meta_key(_chunk_meta))
        keys = pd.Index(keys, name="chunk_indices")
        return execute(tasks, size=len(tasks), keys=keys, **execute_kwargs)


BaseAccessorT = tp.TypeVar("BaseAccessorT", bound="BaseAccessor")


@attach_binary_magic_methods(lambda self, other, np_func: self.combine(other, combine_func=np_func))
@attach_unary_magic_methods(lambda self, np_func: self.apply(apply_func=np_func))
class BaseAccessor(Wrapping):
    """Class representing a base accessor for Series and DataFrame objects.

    Accessible via `pd.Series.vbt` and `pd.DataFrame.vbt` and their descendants.

    Because a Series is simply a DataFrame with a single column, it is converted to a DataFrame
    for matrix operations, and the resulting 2-dimensional output is converted back to a Series using
    `BaseAccessor.wrapper`.

    Args:
        wrapper (Union[ArrayWrapper, tp.ArrayLike]): Wrapper instance or array if `obj` is not provided.
        obj (Optional[tp.ArrayLike]): Array if `wrapper` is an array wrapper.
        **kwargs: Keyword arguments distributed between
            `vectorbtpro.base.wrapping.ArrayWrapper` and `vectorbtpro.base.wrapping.Wrapping`.

    !!! note
        When using magic methods, ensure that `.vbt` is called on the operand on the left
        if the other operand is an array.

        Accessors do not use caching.

        Grouping is supported only by methods that accept the `group_by` argument.

    Examples:
        Build a symmetric matrix:

        ```pycon
        >>> from vectorbtpro import *

        >>> # vectorbtpro.base.accessors.BaseAccessor.make_symmetric
        >>> pd.Series([1, 2, 3]).vbt.make_symmetric()
             0    1    2
        0  1.0  2.0  3.0
        1  2.0  NaN  NaN
        2  3.0  NaN  NaN
        ```

        Broadcast Pandas objects:

        ```pycon
        >>> sr = pd.Series([1])
        >>> df = pd.DataFrame([1, 2, 3])

        >>> vbt.base.reshaping.broadcast_to(sr, df)
           0
        0  1
        1  1
        2  1

        >>> sr.vbt.broadcast_to(df)
           0
        0  1
        1  1
        2  1
        ```

        Use class and instance methods like `BaseAccessor.broadcast`:

        ```pycon
        >>> from vectorbtpro.base.accessors import BaseAccessor

        >>> # Same as sr.vbt.broadcast(df)
        >>> new_sr, new_df = BaseAccessor.broadcast(sr, df)
        >>> new_sr
           0
        0  1
        1  1
        2  1
        >>> new_df
           0
        0  1
        1  2
        2  3
        ```

        Instead of explicitly importing an accessor, use `pd_acc`:

        ```pycon
        >>> vbt.pd_acc.broadcast(sr, df)
        >>> new_sr
           0
        0  1
        1  1
        2  1
        >>> new_df
           0
        0  1
        1  2
        2  3
        ```

        Leverage arithmetic (e.g. `+`), comparison (e.g. `>`), and logical (e.g. `&`) operators that
        forward operations to `BaseAccessor.combine`:

        ```pycon
        >>> sr.vbt + df
           0
        0  2
        1  3
        2  4
        ```

        Many interesting use cases can be implemented this way.

        Compare an array with different thresholds:

        ```pycon
        >>> df.vbt > vbt.Param(np.arange(3), name='threshold')
        threshold     0                  1                  2
                     a2    b2    c2     a2    b2    c2     a2     b2    c2
        x2         True  True  True  False  True  True  False  False  True
        y2         True  True  True   True  True  True   True   True  True
        z2         True  True  True   True  True  True   True   True  True
        ```

        Use the broadcasting mechanism:

        ```pycon
        >>> df.vbt > vbt.Param(np.arange(3), name='threshold')
        threshold     0                  1                  2
                     a2    b2    c2     a2    b2    c2     a2     b2    c2
        x2         True  True  True  False  True  True  False  False  True
        y2         True  True  True   True  True  True   True   True  True
        z2         True  True  True   True  True  True   True   True  True
        ```
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        if len(kwargs) > 0:
            wrapper_kwargs, kwargs = ArrayWrapper.extract_init_kwargs(**kwargs)
        else:
            wrapper_kwargs, kwargs = {}, {}
        if not isinstance(wrapper, ArrayWrapper):
            if obj is not None:
                raise ValueError("Must either provide wrapper and object, or only object")
            wrapper, obj = ArrayWrapper.from_obj(wrapper, **wrapper_kwargs), wrapper
        else:
            if obj is None:
                raise ValueError("Must either provide wrapper and object, or only object")
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)

        Wrapping.__init__(self, wrapper, obj=obj, **kwargs)

        self._obj = obj

    def __call__(self: BaseAccessorT, **kwargs) -> BaseAccessorT:
        """Pass additional keyword arguments to the initializer and return a new accessor instance.

        Args:
            **kwargs: Keyword arguments for `BaseAccessor.replace`.

        Returns:
            BaseAccessor: New accessor instance with updated parameters.
        """
        return self.replace(**kwargs)

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[BaseAccessorT],
        *objs: BaseAccessorT,
        **kwargs,
    ) -> tp.Kwargs:
        if "obj" not in kwargs:
            kwargs["obj"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.obj for obj in objs],
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[BaseAccessorT],
        *objs: BaseAccessorT,
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `BaseAccessor` after stacking along columns.

        Args:
            *objs (MaybeTuple[BaseAccessor]): Accessor instances to be stacked.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Resolved keyword arguments for initializing `BaseAccessor`.
        """
        if "obj" not in kwargs:
            kwargs["obj"] = kwargs["wrapper"].column_stack_arrs(
                *[obj.obj for obj in objs],
                reindex_kwargs=reindex_kwargs,
                group_by=False,
                wrap=False,
            )
        return kwargs

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[BaseAccessorT],
        *objs: tp.MaybeSequence[BaseAccessorT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> BaseAccessorT:
        """Stack multiple `BaseAccessor` instances along rows.

        Stacks accessor instances along the row axis using
        `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` and configures a new accessor.

        Args:
            *objs (MaybeSequence[BaseAccessor]): (Additional) accessor instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `BaseAccessor` through
                `BaseAccessor.resolve_row_stack_kwargs` and `BaseAccessor.resolve_stack_kwargs`.

        Returns:
            BaseAccessor: New accessor instance with row-stacked data.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, BaseAccessor):
                raise TypeError("Each object to be merged must be an instance of BaseAccessor")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "wrapper" in kwargs and kwargs["wrapper"] is not None:
            wrapper = kwargs["wrapper"]
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)
        else:
            wrapper = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)
        kwargs["wrapper"] = wrapper

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        if kwargs["wrapper"].ndim == 1:
            return cls.sr_accessor_cls(**kwargs)
        return cls.df_accessor_cls(**kwargs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[BaseAccessorT],
        *objs: tp.MaybeSequence[BaseAccessorT],
        wrapper_kwargs: tp.KwargsLike = None,
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> BaseAccessorT:
        """Stack multiple `BaseAccessor` instances along columns.

        Stacks accessor instances along the column axis using
        `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` and configures a new accessor.

        Args:
            *objs (MaybeSequence[BaseAccessor]): (Additional) accessor instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            **kwargs: Keyword arguments for `BaseAccessor` through
                `BaseAccessor.resolve_column_stack_kwargs` and `BaseAccessor.resolve_stack_kwargs`.

        Returns:
            BaseAccessor: New accessor instance with column-stacked data.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, BaseAccessor):
                raise TypeError("Each object to be merged must be an instance of BaseAccessor")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "wrapper" in kwargs and kwargs["wrapper"] is not None:
            wrapper = kwargs["wrapper"]
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)
        else:
            wrapper = ArrayWrapper.column_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)
        kwargs["wrapper"] = wrapper

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls.df_accessor_cls(**kwargs)

    @hybrid_property
    def sr_accessor_cls(cls_or_self) -> tp.Type["BaseSRAccessor"]:
        """Pandas Series accessor class.

        Returns:
            Type[BaseSRAccessor]: Class of the Series accessor.
        """
        return BaseSRAccessor

    @hybrid_property
    def df_accessor_cls(cls_or_self) -> tp.Type["BaseDFAccessor"]:
        """Pandas DataFrame accessor class.

        Returns:
            Type[BaseDFAccessor]: Class of the DataFrame accessor.
        """
        return BaseDFAccessor

    def indexing_func(
        self: BaseAccessorT, *args, wrapper_meta: tp.DictLike = None, **kwargs
    ) -> BaseAccessorT:
        """Perform indexing on the accessor instance and return a new accessor with the selected data.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func`.
        Returns:
            BaseAccessor: New accessor instance with the selected subset of data.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_obj = ArrayWrapper.select_from_flex_array(
            self._obj,
            row_idxs=wrapper_meta["row_idxs"],
            col_idxs=wrapper_meta["col_idxs"],
            rows_changed=wrapper_meta["rows_changed"],
            columns_changed=wrapper_meta["columns_changed"],
        )
        if checks.is_series(new_obj):
            return self.replace(
                cls_=self.sr_accessor_cls, wrapper=wrapper_meta["new_wrapper"], obj=new_obj
            )
        return self.replace(
            cls_=self.df_accessor_cls, wrapper=wrapper_meta["new_wrapper"], obj=new_obj
        )

    def indexing_setter_func(self, pd_indexing_setter_func: tp.Callable, **kwargs) -> None:
        pd_indexing_setter_func(self._obj)

    @property
    def obj(self) -> tp.SeriesFrame:
        """Underlying Pandas object.

        Returns:
            SeriesFrame: Pandas Series or DataFrame that matches the wrapper's configuration,
                or a wrapped version if it does not.
        """
        if isinstance(self._obj, (pd.Series, pd.DataFrame)):
            if self._obj.shape == self.wrapper.shape:
                if self._obj.index is self.wrapper.index:
                    if isinstance(self._obj, pd.Series) and self._obj.name == self.wrapper.name:
                        return self._obj
                    if (
                        isinstance(self._obj, pd.DataFrame)
                        and self._obj.columns is self.wrapper.columns
                    ):
                        return self._obj
        return self.wrapper.wrap(self._obj, group_by=False)

    def get(
        self, key: tp.Optional[tp.Hashable] = None, default: tp.Optional[tp.Any] = None
    ) -> tp.SeriesFrame:
        """Retrieve the underlying Pandas object or a specific element by key.

        Args:
            key (Optional[Hashable]): Key to access from the underlying object.
            default (Optional[Any]): Default value if the key is not found.

        Returns:
            SeriesFrame: Underlying Pandas object if no key is provided,
                or the value associated with the key.
        """
        if key is None:
            return self.obj
        return self.obj.get(key, default=default)

    @property
    def unwrapped(self) -> tp.SeriesFrame:
        return self.obj

    @hybrid_method
    def should_wrap(cls_or_self) -> bool:
        return False

    @hybrid_property
    def ndim(cls_or_self) -> tp.Optional[int]:
        """Number of dimensions in the underlying object.

        For example:

        * 1 corresponds to a Series.
        * 2 corresponds to a DataFrame.

        Returns:
            Optional[int]: Number of dimensions in the underlying object.
        """
        if isinstance(cls_or_self, type):
            return None
        return cls_or_self.obj.ndim

    @hybrid_method
    def is_series(cls_or_self) -> bool:
        """Determine whether the underlying object is a Pandas Series.

        Returns:
            bool: True if the underlying object is a Series, False otherwise.

        !!! abstract
            This method should be overridden in a subclass.
        """
        if isinstance(cls_or_self, type):
            raise NotImplementedError
        return isinstance(cls_or_self.obj, pd.Series)

    @hybrid_method
    def is_frame(cls_or_self) -> bool:
        """Determine whether the underlying object is a Pandas DataFrame.

        Returns:
            bool: True if the underlying object is a DataFrame, False otherwise.

        !!! abstract
            This method should be overridden in a subclass.
        """
        if isinstance(cls_or_self, type):
            raise NotImplementedError
        return isinstance(cls_or_self.obj, pd.DataFrame)

    @classmethod
    def resolve_shape(cls, shape: tp.ShapeLike) -> tp.Shape:
        """Resolve the given shape to a two-dimensional format.

        Args:
            shape (ShapeLike): Input shape.

        Returns:
            Shape: Resolved two-dimensional shape.
        """
        shape_2d = reshaping.to_2d_shape(shape)
        try:
            if cls.is_series() and shape_2d[1] > 1:
                raise ValueError("Use DataFrame accessor")
        except NotImplementedError:
            pass
        return shape_2d

    # ############# Creation ############# #

    @classmethod
    def empty(cls, shape: tp.Shape, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """Generate an empty Pandas Series or DataFrame.

        Args:
            shape (Shape): Tuple specifying the dimensions of the output array.
            fill_value (Scalar): Value to fill the object.
            **kwargs: Keyword arguments for the Pandas constructor.

        Returns:
            SeriesFrame: Empty Series or DataFrame with the specified shape.
        """
        if not isinstance(shape, tuple) or (isinstance(shape, tuple) and len(shape) == 1):
            return pd.Series(np.full(shape, fill_value), **kwargs)
        return pd.DataFrame(np.full(shape, fill_value), **kwargs)

    @classmethod
    def empty_like(
        cls, other: tp.SeriesFrame, fill_value: tp.Scalar = np.nan, **kwargs
    ) -> tp.SeriesFrame:
        """Generate an empty Pandas Series or DataFrame based on another object.

        Args:
            other (SeriesFrame): Reference Pandas object to mimic.
            fill_value (Scalar): Value to fill the new object.
            **kwargs: Keyword arguments for the Pandas constructor.

        Returns:
            SeriesFrame: Empty Series or DataFrame matching the structure of `other`.
        """
        if checks.is_series(other):
            return cls.empty(
                other.shape, fill_value=fill_value, index=other.index, name=other.name, **kwargs
            )
        return cls.empty(
            other.shape, fill_value=fill_value, index=other.index, columns=other.columns, **kwargs
        )

    # ############# Indexes ############# #

    def apply_to_index(
        self: BaseAccessorT,
        *args,
        wrap: bool = False,
        **kwargs,
    ) -> tp.Union[BaseAccessorT, tp.SeriesFrame]:
        """Apply an indexing operation to the accessor.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.Wrapping.apply_to_index`.
            wrap (bool): If False, return the underlying Pandas object; if True, return an accessor.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.Wrapping.apply_to_index`.

        Returns:
            Union[BaseAccessor, SeriesFrame]: Accessor instance with the applied indexing,
                or the underlying Pandas object if `wrap` is False.

        !!! note
            If `wrap` is False, returns Pandas object, not accessor!
        """
        result = Wrapping.apply_to_index(self, *args, **kwargs)
        if wrap:
            return result
        return result.obj

    # ############# Setting ############# #

    def set(
        self,
        value_or_func: tp.Union[tp.MaybeArray, tp.Callable],
        *args,
        inplace: bool = False,
        columns: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Set values at specified index points of the data.

        Determine index points using `vectorbtpro.base.indexing.get_index_points` and assign values accordingly.
        If `value_or_func` is callable, it is invoked at each index point with a context that includes:

        * `i`: iteration index.
        * `index_point`: absolute position in the index.
        * `wrapper`: the associated data wrapper.
        * `obj`: the underlying data object.

        Keyword arguments not used by `get_index_points` are filtered, template-substituted,
        and passed to the callable.

        Args:
            value_or_func (Union[MaybeArray, Callable]): Value to assign or a function that
                computes the value based on context.
            *args: Positional arguments for `value_or_func` if it is callable.
            inplace (bool): If True, modify the data object in place.
            columns (Optional[MaybeSequence[Hashable]]): Column(s) where the value should be set.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexing.get_index_points`.

                These are also passed to `value_or_func` if it is callable.

        Returns:
            Optional[SeriesFrame]: Pandas Series or DataFrame if `inplace` is False.
        """
        if inplace:
            obj = self.obj
        else:
            obj = self.obj.copy()
        index_points = get_index_points(self.wrapper.index, **kwargs)

        if callable(value_or_func):
            func_kwargs = {k: v for k, v in kwargs.items() if k not in point_idxr_defaults}
            template_context = merge_dicts(kwargs, template_context)
        else:
            func_kwargs = None
        if callable(value_or_func):
            for i in range(len(index_points)):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_point=index_points[i],
                        index_points=index_points,
                        wrapper=self.wrapper,
                        obj=self.obj,
                        columns=columns,
                        args=args,
                        kwargs=kwargs,
                    ),
                    template_context,
                )
                _func_args = substitute_templates(args, _template_context, eval_id="func_args")
                _func_kwargs = substitute_templates(
                    func_kwargs, _template_context, eval_id="func_kwargs"
                )
                v = value_or_func(*_func_args, **_func_kwargs)
                if self.is_series() or columns is None:
                    obj.iloc[index_points[i]] = v
                elif is_scalar(columns):
                    obj.iloc[index_points[i], obj.columns.get_indexer([columns])[0]] = v
                else:
                    obj.iloc[index_points[i], obj.columns.get_indexer(columns)] = v
        elif checks.is_sequence(value_or_func) and not is_scalar(value_or_func):
            if self.is_series():
                obj.iloc[index_points] = reshaping.to_1d_array(value_or_func)
            elif columns is None:
                obj.iloc[index_points] = reshaping.to_2d_array(value_or_func)
            elif is_scalar(columns):
                obj.iloc[index_points, obj.columns.get_indexer([columns])[0]] = (
                    reshaping.to_1d_array(value_or_func)
                )
            else:
                obj.iloc[index_points, obj.columns.get_indexer(columns)] = reshaping.to_2d_array(
                    value_or_func
                )
        else:
            if self.is_series() or columns is None:
                obj.iloc[index_points] = value_or_func
            elif is_scalar(columns):
                obj.iloc[index_points, obj.columns.get_indexer([columns])[0]] = value_or_func
            else:
                obj.iloc[index_points, obj.columns.get_indexer(columns)] = value_or_func
        if inplace:
            return None
        return obj

    def set_between(
        self,
        value_or_func: tp.Union[tp.MaybeArray, tp.Callable],
        *args,
        inplace: bool = False,
        columns: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Set values over specified index ranges of the data.

        Determine index ranges using `vectorbtpro.base.indexing.get_index_ranges` and assign values accordingly.
        If `value_or_func` is callable, it is invoked for each range with a context that includes:

        * `i`: iteration index.
        * `index_slice`: a slice representing the absolute range in the index.
        * `range_starts`: list of range starting indices.
        * `range_ends`: list of range ending indices.
        * `wrapper`: the associated data wrapper.
        * `obj`: the underlying data object.

        Keyword arguments not used by `get_index_ranges` are filtered, template-substituted,
        and passed to the callable.

        Args:
            value_or_func (Union[MaybeArray, Callable]): Value to assign or a function that
                computes the value based on context.
            *args: Positional arguments for `value_or_func` if it is callable.
            inplace (bool): If True, modify the data object in place.
            columns (Optional[MaybeSequence[Hashable]]): Column(s) where the value should be set.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexing.get_index_ranges`.

                These are also passed to `value_or_func` if it is callable.

        Returns:
            Optional[SeriesFrame]: Pandas Series or DataFrame if `inplace` is False.
        """
        if inplace:
            obj = self.obj
        else:
            obj = self.obj.copy()
        index_ranges = get_index_ranges(self.wrapper.index, **kwargs)

        if callable(value_or_func):
            func_kwargs = {k: v for k, v in kwargs.items() if k not in range_idxr_defaults}
            template_context = merge_dicts(kwargs, template_context)
        else:
            func_kwargs = None
        for i in range(len(index_ranges[0])):
            if callable(value_or_func):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_slice=slice(index_ranges[0][i], index_ranges[1][i]),
                        range_starts=index_ranges[0],
                        range_ends=index_ranges[1],
                        wrapper=self.wrapper,
                        obj=self.obj,
                        columns=columns,
                        args=args,
                        kwargs=kwargs,
                    ),
                    template_context,
                )
                _func_args = substitute_templates(args, _template_context, eval_id="func_args")
                _func_kwargs = substitute_templates(
                    func_kwargs, _template_context, eval_id="func_kwargs"
                )
                v = value_or_func(*_func_args, **_func_kwargs)
            elif checks.is_sequence(value_or_func) and not isinstance(value_or_func, str):
                v = value_or_func[i]
            else:
                v = value_or_func
            if self.is_series() or columns is None:
                obj.iloc[index_ranges[0][i] : index_ranges[1][i]] = v
            elif is_scalar(columns):
                obj.iloc[
                    index_ranges[0][i] : index_ranges[1][i], obj.columns.get_indexer([columns])[0]
                ] = v
            else:
                obj.iloc[
                    index_ranges[0][i] : index_ranges[1][i], obj.columns.get_indexer(columns)
                ] = v
        if inplace:
            return None
        return obj

    # ############# Splitting ############# #

    def split(
        self,
        *args,
        splitter_cls: tp.Optional[tp.Type[SplitterT]] = None,
        wrap: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        from vectorbtpro.generic.splitting.base import Splitter

        if splitter_cls is None:
            splitter_cls = Splitter
        if wrap is None:
            wrap = self.should_wrap()
        wrapped_self = self if wrap else self.unwrapped
        return splitter_cls.split_and_take(
            self.wrapper.index,
            wrapped_self,
            *args,
            _take_kwargs=dict(into="reset_stacked"),
            **kwargs,
        )

    # ############# Reshaping ############# #

    def to_1d_array(self) -> tp.Array1d:
        """Convert the data to a one-dimensional array.

        Calls `vectorbtpro.base.reshaping.to_1d` with `raw=True` on the underlying data.

        Returns:
            Array1d: One-dimensional representation of the data.
        """
        return reshaping.to_1d_array(self.obj)

    def to_2d_array(self) -> tp.Array2d:
        """Convert the data to a two-dimensional array.

        Calls `vectorbtpro.base.reshaping.to_2d` with `raw=True` on the underlying data.

        Returns:
            Array2d: Two-dimensional representation of the data.
        """
        return reshaping.to_2d_array(self.obj)

    def tile(
        self,
        n: int,
        keys: tp.Optional[tp.IndexLike] = None,
        axis: int = 1,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Tile the data by repeating it along a specified axis.

        Calls `vectorbtpro.base.reshaping.tile` to generate tiled data.
        Set `axis` to 1 for columns and 0 for index. If `keys` is provided,
        it is used as the outermost level when combining indexes.

        Args:
            n (int): Number of times to tile the data.
            keys (Optional[IndexLike]): Outer-level keys used to combine indexes.
            axis (int): Axis along which to tile the data (1 for columns, 0 for index).
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Tiled data, with updated index or columns if `keys` is provided.
        """
        tiled = reshaping.tile(self.obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = indexes.combine_indexes([keys, self.wrapper.columns])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values,
                    **merge_dicts(dict(columns=new_columns), wrap_kwargs),
                )
            else:
                new_index = indexes.combine_indexes([keys, self.wrapper.index])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values,
                    **merge_dicts(dict(index=new_index), wrap_kwargs),
                )
        return tiled

    def repeat(
        self,
        n: int,
        keys: tp.Optional[tp.IndexLike] = None,
        axis: int = 1,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Repeat the data along a specified axis.

        Calls `vectorbtpro.base.reshaping.repeat` to generate repeated data.
        Set `axis` to 1 for columns and 0 for index. If `keys` is provided,
        it is used as the outermost level when combining indexes.

        Args:
            n (int): Number of times to repeat the data.
            keys (Optional[IndexLike]): Outer-level keys used to combine indexes.
            axis (int): Axis along which to repeat the data (1 for columns, 0 for index).
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Repeated data, with updated index or columns if `keys` is provided.
        """
        repeated = reshaping.repeat(self.obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = indexes.combine_indexes([self.wrapper.columns, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values,
                    **merge_dicts(dict(columns=new_columns), wrap_kwargs),
                )
            else:
                new_index = indexes.combine_indexes([self.wrapper.index, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values,
                    **merge_dicts(dict(index=new_index), wrap_kwargs),
                )
        return repeated

    def align_to(
        self, other: tp.SeriesFrame, wrap_kwargs: tp.KwargsLike = None, **kwargs
    ) -> tp.SeriesFrame:
        """Align input object to match the axes of a given object.

        Args:
            other (SeriesFrame): Object to align to.

                Must be a `pd.Series` or `pd.DataFrame`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.align_index_to`.

        Returns:
            SeriesFrame: Aligned object.

        Example:
            ```pycon
            >>> df1 = pd.DataFrame(
            ...     [[1, 2], [3, 4]],
            ...     index=['x', 'y'],
            ...     columns=['a', 'b']
            ... )
            >>> df1
               a  b
            x  1  2
            y  3  4

            >>> df2 = pd.DataFrame(
            ...     [[5, 6, 7, 8], [9, 10, 11, 12]],
            ...     index=['x', 'y'],
            ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']])
            ... )
            >>> df2
                   1       2
               a   b   a   b
            x  5   6   7   8
            y  9  10  11  12

            >>> df1.vbt.align_to(df2)
                  1     2
               a  b  a  b
            x  1  2  1  2
            y  3  4  3  4
            ```
        """
        checks.assert_instance_of(other, (pd.Series, pd.DataFrame))
        obj = reshaping.to_2d(self.obj)
        other = reshaping.to_2d(other)

        aligned_index = indexes.align_index_to(obj.index, other.index, **kwargs)
        aligned_columns = indexes.align_index_to(obj.columns, other.columns, **kwargs)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrapper.wrap(
            obj.values,
            group_by=False,
            **merge_dicts(dict(index=other.index, columns=other.columns), wrap_kwargs),
        )

    @hybrid_method
    def align(
        cls_or_self,
        *others: tp.Union[tp.SeriesFrame, "BaseAccessor"],
        **kwargs,
    ) -> tp.Tuple[tp.SeriesFrame, ...]:
        """Align input objects to a common index and columns.

        Args:
            *others (Union[SeriesFrame, BaseAccessor]): (Additional) objects to align.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexes.align_indexes`.

        Returns:
            Tuple[SeriesFrame, ...]: Aligned objects.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        objs_2d = list(map(reshaping.to_2d, objs))
        index_slices, new_index = indexes.align_indexes(
            *map(lambda x: x.index, objs_2d),
            return_new_index=True,
            **kwargs,
        )
        column_slices, new_columns = indexes.align_indexes(
            *map(lambda x: x.columns, objs_2d),
            return_new_index=True,
            **kwargs,
        )
        new_objs = []
        for i in range(len(objs_2d)):
            new_obj = objs_2d[i].iloc[index_slices[i], column_slices[i]].copy(deep=False)
            if objs[i].ndim == 1 and new_obj.shape[1] == 1:
                new_obj = new_obj.iloc[:, 0].rename(objs[i].name)
            new_obj.index = new_index
            new_obj.columns = new_columns
            new_objs.append(new_obj)
        return tuple(new_objs)

    def cross_with(
        self, other: tp.SeriesFrame, wrap_kwargs: tp.KwargsLike = None
    ) -> tp.SeriesFrame:
        """Cross align the input object with another object on both axes.

        Args:
            other (SeriesFrame): Object to cross align with.

                Must be a `pd.Series` or `pd.DataFrame`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Cross aligned object.

        Example:
            ```pycon
            >>> df1 = pd.DataFrame(
            ...     [[1, 2, 3, 4], [5, 6, 7, 8]],
            ...     index=['x', 'y'],
            ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']])
            ... )
            >>> df1
               1     2
               a  b  a  b
            x  1  2  3  4
            y  5  6  7  8

            >>> df2 = pd.DataFrame(
            ...     [[9, 10, 11, 12], [13, 14, 15, 16]],
            ...     index=['x', 'y'],
            ...     columns=pd.MultiIndex.from_arrays([[3, 3, 4, 4], ['a', 'b', 'a', 'b']])
            ... )
            >>> df2
                3       4
                a   b   a   b
            x   9  10  11  12
            y  13  14  15  16

            >>> df1.vbt.cross_with(df2)
               1           2
               3     4     3     4
               a  b  a  b  a  b  a  b
            x  1  2  1  2  3  4  3  4
            y  5  6  5  6  7  8  7  8
            ```
        """
        checks.assert_instance_of(other, (pd.Series, pd.DataFrame))
        obj = reshaping.to_2d(self.obj)
        other = reshaping.to_2d(other)

        index_slices, new_index = indexes.cross_index_with(
            obj.index,
            other.index,
            return_new_index=True,
        )
        column_slices, new_columns = indexes.cross_index_with(
            obj.columns,
            other.columns,
            return_new_index=True,
        )
        obj = obj.iloc[index_slices[0], column_slices[0]]
        return self.wrapper.wrap(
            obj.values,
            group_by=False,
            **merge_dicts(dict(index=new_index, columns=new_columns), wrap_kwargs),
        )

    @hybrid_method
    def cross(
        cls_or_self, *others: tp.Union[tp.SeriesFrame, "BaseAccessor"]
    ) -> tp.Tuple[tp.SeriesFrame, ...]:
        """Cross align input objects on both axes.

        Args:
            *others (Union[SeriesFrame, BaseAccessor]): (Additional) objects to cross align.

        Returns:
            Tuple[SeriesFrame, ...]: Cross aligned objects.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        objs_2d = list(map(reshaping.to_2d, objs))
        index_slices, new_index = indexes.cross_indexes(
            *map(lambda x: x.index, objs_2d),
            return_new_index=True,
        )
        column_slices, new_columns = indexes.cross_indexes(
            *map(lambda x: x.columns, objs_2d),
            return_new_index=True,
        )
        new_objs = []
        for i in range(len(objs_2d)):
            new_obj = objs_2d[i].iloc[index_slices[i], column_slices[i]].copy(deep=False)
            if objs[i].ndim == 1 and new_obj.shape[1] == 1:
                new_obj = new_obj.iloc[:, 0].rename(objs[i].name)
            new_obj.index = new_index
            new_obj.columns = new_columns
            new_objs.append(new_obj)
        return tuple(new_objs)

    x = cross

    @hybrid_method
    def broadcast(cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """Broadcast input arrays to a common shape.

        Args:
            *others (Union[ArrayLike, BaseAccessor]): (Additional) arrays for broadcasting.

            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.broadcast`.

        Returns:
            Any: Broadcasted result.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        return reshaping.broadcast(*objs, **kwargs)

    def broadcast_to(self, other: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """Broadcast the input object to the shape of another array.

        Args:
            other (Union[ArrayLike, BaseAccessor]): Array to broadcast to.
            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.broadcast_to`.

        Returns:
            Any: Broadcasted result.
        """
        if isinstance(other, BaseAccessor):
            other = other.obj
        return reshaping.broadcast_to(self.obj, other, **kwargs)

    @hybrid_method
    def broadcast_combs(
        cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs
    ) -> tp.Any:
        """Compute broadcast combinations for input arrays.

        Args:
            *others (Union[ArrayLike, BaseAccessor]): (Additional) arrays for broadcasting combinations.

            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.broadcast_combs`.

        Returns:
            Any: Result of broadcast combinations.
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj, *others)
        return reshaping.broadcast_combs(*objs, **kwargs)

    def make_symmetric(self, *args, **kwargs) -> tp.Frame:
        """Return a symmetric version of the DataFrame using `vectorbtpro.base.reshaping.make_symmetric`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.reshaping.make_symmetric`.
            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.make_symmetric`.

        Returns:
            Frame: Symmetric reshaped DataFrame.
        """
        return reshaping.make_symmetric(self.obj, *args, **kwargs)

    def unstack_to_array(self, *args, **kwargs) -> tp.Array:
        """Return an array by unstacking data using `vectorbtpro.base.reshaping.unstack_to_array`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.reshaping.unstack_to_array`.
            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.unstack_to_array`.

        Returns:
            Array: Resulting unstacked array.
        """
        return reshaping.unstack_to_array(self.obj, *args, **kwargs)

    def unstack_to_df(self, *args, **kwargs) -> tp.Frame:
        """Return a DataFrame by unstacking data using `vectorbtpro.base.reshaping.unstack_to_df`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.reshaping.unstack_to_df`.
            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.unstack_to_df`.

        Returns:
            Frame: Resulting unstacked DataFrame.
        """
        return reshaping.unstack_to_df(self.obj, *args, **kwargs)

    def to_dict(self, *args, **kwargs) -> tp.Mapping:
        """Convert the object to a dictionary using `vectorbtpro.base.reshaping.to_dict`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.reshaping.to_dict`.
            **kwargs: Keyword arguments for `vectorbtpro.base.reshaping.to_dict`.

        Returns:
            Mapping: Dictionary representation of the data.
        """
        return reshaping.to_dict(self.obj, *args, **kwargs)

    # ############# Conversion ############# #

    def to_data(
        self,
        data_cls: tp.Optional[tp.Type[DataT]] = None,
        columns_are_symbols: bool = True,
        **kwargs,
    ) -> DataT:
        """Convert the object to a `vectorbtpro.data.base.Data` instance.

        Args:
            data_cls (Optional[Type[Data]]): Data class to use for conversion.
            columns_are_symbols (bool): Flag indicating whether the columns represent symbols.
            **kwargs: Keyword arguments for `Data.from_data`.

        Returns:
            Data: `vectorbtpro.data.base.Data` instance representing the converted data.
        """
        if data_cls is None:
            from vectorbtpro.data.base import Data

            data_cls = Data

        return data_cls.from_data(self.obj, columns_are_symbols=columns_are_symbols, **kwargs)

    # ############# Combining ############# #

    def apply(
        self,
        apply_func: tp.Callable,
        *args,
        keep_pd: bool = False,
        to_2d: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Apply a function to the object with broadcasting and template substitution support.

        Args:
            apply_func (Callable): Function to apply.
            *args: Positional arguments for `apply_func`.
            keep_pd (bool): If True, retain inputs as Pandas objects; otherwise, convert them to NumPy arrays.
            to_2d (bool): If True, reshapes inputs to two-dimensional arrays.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            SeriesFrame: Result after applying the function.

        !!! note
            The resulting array must have the same shape as the original array.

        Examples:
            Using instance method:

            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> sr.vbt.apply(lambda x: x ** 2)
            x    1
            y    4
            dtype: int64
            ```

            Using class method, templates, and broadcasting:

            ```pycon
            >>> sr.vbt.apply(
            ...     lambda x, y: x + y,
            ...     vbt.Rep('y'),
            ...     broadcast_named_args=dict(
            ...         y=pd.DataFrame([[3, 4]], columns=['a', 'b'])
            ...     )
            ... )
               a  b
            x  4  5
            y  5  6
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        if to_2d:
            broadcast_named_args = {
                k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()
            }
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, template_context)
        args = substitute_templates(args, template_context, eval_id="args")
        kwargs = substitute_templates(kwargs, template_context, eval_id="kwargs")
        out = apply_func(broadcast_named_args["obj"], *args, **kwargs)
        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def concat(
        cls_or_self,
        *others: tp.ArrayLike,
        broadcast_kwargs: tp.KwargsLike = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Frame:
        """Concatenate the object with additional arrays along columns.

        Args:
            *others (ArrayLike): (Additional) arrays or objects to concatenate.

                If an element is an instance of `BaseAccessor`, its `.obj` attribute is used.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            keys (Optional[IndexLike]): Keys to label the columns in the resulting DataFrame.

        Returns:
            Frame: Concatenated DataFrame.

        Example:
            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> sr.vbt.concat(df, keys=['c', 'd'])
                  c     d
               a  b  a  b
            x  1  1  3  4
            y  2  2  5  6
            ```
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj,) + others
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        broadcasted = reshaping.broadcast(*objs, **broadcast_kwargs)
        broadcasted = tuple(map(reshaping.to_2d, broadcasted))
        out = pd.concat(broadcasted, axis=1, keys=keys)
        if not isinstance(out.columns, pd.MultiIndex) and np.all(out.columns == 0):
            out.columns = pd.RangeIndex(start=0, stop=len(out.columns), step=1)
        return out

    def apply_and_concat(
        self,
        ntimes: int,
        apply_func: tp.Callable,
        *args,
        keep_pd: bool = False,
        to_2d: bool = False,
        keys: tp.Optional[tp.IndexLike] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.Frame]:
        """Apply the given function `apply_func` multiple times and concatenate the results along columns.

        This method applies `apply_func` `ntimes` times on the parent object,
        passing additional arguments and keyword arguments as provided.
        If `keys` is specified, the output columns are labeled accordingly;
        otherwise, a default numeric index is created. The method also handles broadcasting,
        template substitutions, and optional reshaping based on the flags `keep_pd` and `to_2d`.

        See also `vectorbtpro.base.combining.apply_and_concat`.

        Args:
            ntimes (int): Number of times to execute `apply_func`.
            apply_func (Callable): Function to execute, with the iteration index as its first parameter.
            *args: Positional arguments for `apply_func`.
            keep_pd (bool): If True, retain inputs as Pandas objects; otherwise, convert them to NumPy arrays.
            to_2d (bool): If True, reshapes inputs to two-dimensional arrays.
            keys (Optional[IndexLike]): Labels for the concatenated results along columns.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            MaybeTuple[Frame]: Wrapped result as a single DataFrame or a tuple of frames,
                or None if no result is produced.

        !!! note
            The arrays to be concatenated must have the same shape as the broadcast input arrays.

        Examples:
            Using instance method:

            ```pycon
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> df.vbt.apply_and_concat(
            ...     3,
            ...     lambda i, a, b: a * b[i],
            ...     [1, 2, 3],
            ...     keys=['c', 'd', 'e']
            ... )
                  c       d       e
               a  b   a   b   a   b
            x  3  4   6   8   9  12
            y  5  6  10  12  15  18
            ```

            Using class method, templates, and broadcasting:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> sr.vbt.apply_and_concat(
            ...     3,
            ...     lambda i, a, b: a * b + i,
            ...     vbt.Rep('df'),
            ...     broadcast_named_args=dict(
            ...         df=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            apply_idx        0         1         2
                       a  b  c  a  b   c  a  b   c
            x          1  2  3  2  3   4  3  4   5
            y          2  4  6  3  5   7  4  6   8
            z          3  6  9  4  7  10  5  8  11
            ```

            To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def apply_func(i, a):
            ...     time.sleep(1)
            ...     return a

            >>> sr = pd.Series([1, 2, 3])

            >>> %timeit sr.vbt.apply_and_concat(3, apply_func)
            3.02 s ± 3.76 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit sr.vbt.apply_and_concat(3, apply_func, execute_kwargs=dict(engine='dask'))
            1.02 s ± 927 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        if to_2d:
            broadcast_named_args = {
                k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()
            }
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, dict(ntimes=ntimes), template_context)
        args = substitute_templates(args, template_context, eval_id="args")
        kwargs = substitute_templates(kwargs, template_context, eval_id="kwargs")
        out = combining.apply_and_concat(
            ntimes, apply_func, broadcast_named_args["obj"], *args, **kwargs
        )
        if keys is not None:
            new_columns = indexes.combine_indexes([keys, wrapper.columns])
        else:
            top_columns = pd.Index(np.arange(ntimes), name="apply_idx")
            new_columns = indexes.combine_indexes([top_columns, wrapper.columns])
        if out is None:
            return None
        wrap_kwargs = merge_dicts(dict(columns=new_columns), wrap_kwargs)
        if isinstance(out, list):
            return tuple(map(lambda x: wrapper.wrap(x, group_by=False, **wrap_kwargs), out))
        return wrapper.wrap(out, group_by=False, **wrap_kwargs)

    @hybrid_method
    def combine(
        cls_or_self,
        obj: tp.MaybeTupleList[tp.Union[tp.ArrayLike, "BaseAccessor"]],
        combine_func: tp.Callable,
        *args,
        allow_multiple: bool = True,
        keep_pd: bool = False,
        to_2d: bool = False,
        concat: tp.Optional[bool] = None,
        keys: tp.Optional[tp.IndexLike] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Combine this object with additional objects using a specified combine function.

        Args:
            obj (array_like): Object or sequence of objects to combine with.
            combine_func (callable): Function used to combine two arrays.

                The function may be Numba-compiled.
            *args: Positional arguments for `combine_func`.
            allow_multiple (bool): Determines if a tuple, list, or Index is interpreted as multiple objects.

                Applicable only when using the instance method.
            keep_pd (bool): If True, retain inputs as Pandas objects; otherwise, convert them to NumPy arrays.
            to_2d (bool): If True, reshapes inputs to two-dimensional arrays.
            concat (bool): Determines whether to concatenate the results along the column axis or
                combine objects pairwise.

                * If True, see `vectorbtpro.base.combining.combine_and_concat`.
                * If False, see `vectorbtpro.base.combining.combine_multiple`.
                * If None, defaults to True when combining multiple objects.

                Can only be used with the instance method.
            keys (index_like): Label(s) for the outermost column level.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments for `combine_func`.

        Returns:
            SeriesFrame: Result after combining the objects.

        !!! note
            If `combine_func` is Numba-compiled, inputs are broadcast using `WRITEABLE` and
            `C_CONTIGUOUS` flags, which may incur significant overhead for large objects with
            differing shapes or memory orders. Ensure that all objects have the same data type
            and that each argument in `*args` is Numba-compatible.

        Examples:
            Using instance method:

            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> sr.vbt.combine(df, np.add)
               a  b
            x  4  5
            y  7  8

            >>> sr.vbt.combine([df, df * 2], np.add, concat=False)
                a   b
            x  10  13
            y  17  20

            >>> sr.vbt.combine([df, df * 2], np.add)
            combine_idx     0       1
                         a  b   a   b
            x            4  5   7   9
            y            7  8  12  14

            >>> sr.vbt.combine([df, df * 2], np.add, keys=['c', 'd'])
                  c       d
               a  b   a   b
            x  4  5   7   9
            y  7  8  12  14

            >>> sr.vbt.combine(vbt.Param([1, 2], name='param'), np.add)
            param  1  2
            x      2  3
            y      3  4

            >>> # using class method
            >>> sr.vbt.combine([df, df * 2], np.add, concat=False)
                a   b
            x  10  13
            y  17  20
            ```

            Using class method, templates, and broadcasting:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> sr.vbt.combine(
            ...     [1, 2, 3],
            ...     lambda x, y, z: x + y + z,
            ...     vbt.Rep('df'),
            ...     broadcast_named_args=dict(
            ...         df=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            combine_idx        0        1        2
                         a  b  c  a  b  c  a  b  c
            x            3  4  5  4  5  6  5  6  7
            y            4  5  6  5  6  7  6  7  8
            z            5  6  7  6  7  8  7  8  9
            ```

            To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def combine_func(a, b):
            ...     time.sleep(1)
            ...     return a + b

            >>> sr = pd.Series([1, 2, 3])

            >>> %timeit sr.vbt.combine([1, 1, 1], combine_func)
            3.01 s ± 2.98 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit sr.vbt.combine([1, 1, 1], combine_func, execute_kwargs=dict(engine='dask'))
            1.02 s ± 2.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        from vectorbtpro.indicators.factory import IndicatorBase

        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(cls_or_self, type):
            objs = obj
        else:
            if allow_multiple and isinstance(obj, (tuple, list)):
                objs = obj
                if concat is None:
                    concat = True
            else:
                objs = (obj,)
        new_objs = []
        for obj in objs:
            if isinstance(obj, BaseAccessor):
                obj = obj.obj
            elif isinstance(obj, IndicatorBase):
                obj = obj.main_output
            new_objs.append(obj)
        objs = tuple(new_objs)
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self.obj,) + objs
        if checks.is_numba_func(combine_func):
            # Numba requires writeable arrays and in the same order
            broadcast_kwargs = merge_dicts(
                dict(require_kwargs=dict(requirements=["W", "C"])), broadcast_kwargs
            )

        # Broadcast and substitute templates
        broadcast_named_args = {
            **{"obj_" + str(i): obj for i, obj in enumerate(objs)},
            **broadcast_named_args,
        }
        broadcast_named_args, wrapper = reshaping.broadcast(
            broadcast_named_args,
            return_wrapper=True,
            **broadcast_kwargs,
        )
        if to_2d:
            broadcast_named_args = {
                k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()
            }
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, template_context)
        args = substitute_templates(args, template_context, eval_id="args")
        kwargs = substitute_templates(kwargs, template_context, eval_id="kwargs")
        inputs = [broadcast_named_args["obj_" + str(i)] for i in range(len(objs))]

        if concat is None:
            concat = len(inputs) > 2
        if concat:
            # Concat the results horizontally
            if isinstance(cls_or_self, type):
                raise TypeError("Use instance method to concatenate")
            out = combining.combine_and_concat(inputs[0], inputs[1:], combine_func, *args, **kwargs)
            if keys is not None:
                new_columns = indexes.combine_indexes([keys, wrapper.columns])
            else:
                top_columns = pd.Index(np.arange(len(objs) - 1), name="combine_idx")
                new_columns = indexes.combine_indexes([top_columns, wrapper.columns])
            return wrapper.wrap(
                out, **merge_dicts(dict(columns=new_columns, force_2d=True), wrap_kwargs)
            )
        else:
            # Combine arguments pairwise into one object
            out = combining.combine_multiple(inputs, combine_func, *args, **kwargs)
            return wrapper.wrap(out, **resolve_dict(wrap_kwargs))

    @classmethod
    def eval(
        cls,
        expr: str,
        frames_back: int = 1,
        use_numexpr: bool = False,
        numexpr_kwargs: tp.KwargsLike = None,
        local_dict: tp.Optional[tp.Mapping] = None,
        global_dict: tp.Optional[tp.Mapping] = None,
        broadcast_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Evaluate a simple array expression element-wise using NumExpr or NumPy.

        Perform element-wise evaluation of the provided array expression.
        If `use_numexpr` is enabled, only one-line expressions are supported.
        Otherwise, the evaluation is performed via `vectorbtpro.utils.eval_.evaluate`.

        Args:
            expr (str): Expression string.

                Must contain valid Python code and can be single-line or multi-line.
            frames_back (int): Number of frames to go back from the current frame.
            use_numexpr (bool): Flag indicating whether to use NumExpr for evaluation.
            numexpr_kwargs (KwargsLike): Keyword arguments for `numexpr.evaluate`.
            local_dict (Optional[Mapping]): Dictionary of local variables.

                If not provided, uses the calling frame's local variables.
            global_dict (Optional[Mapping]): Dictionary of global variables.

                If not provided, uses the calling frame's global variables.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

        Returns:
            Any: Evaluated expression wrapped using the broadcast wrapper.

        !!! note
            All required variables will broadcast against each other prior to the evaluation.

        Example:
            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> df = pd.DataFrame([[4, 5, 6]], index=['x', 'y', 'z'], columns=['a', 'b', 'c'])
            >>> vbt.pd_acc.eval('sr + df')
               a  b  c
            x  5  6  7
            y  6  7  8
            z  7  8  9
            ```
        """
        if numexpr_kwargs is None:
            numexpr_kwargs = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        expr = inspect.cleandoc(expr)
        parsed = ast.parse(expr)
        body_nodes = list(parsed.body)

        load_vars = set()
        store_vars = set()
        for body_node in body_nodes:
            for child_node in ast.walk(body_node):
                if type(child_node) is ast.Name:
                    if isinstance(child_node.ctx, ast.Load):
                        if child_node.id not in store_vars:
                            load_vars.add(child_node.id)
                    if isinstance(child_node.ctx, ast.Store):
                        store_vars.add(child_node.id)
        load_vars = list(load_vars)
        objs = get_context_vars(
            load_vars, frames_back=frames_back, local_dict=local_dict, global_dict=global_dict
        )
        objs = dict(zip(load_vars, objs))
        objs, wrapper = reshaping.broadcast(objs, return_wrapper=True, **broadcast_kwargs)
        objs = {k: np.asarray(v) for k, v in objs.items()}

        if use_numexpr:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("numexpr")
            import numexpr

            out = numexpr.evaluate(expr, local_dict=objs, **numexpr_kwargs)
        else:
            out = evaluate(expr, context=objs)
        return wrapper.wrap(out, **wrap_kwargs)


class BaseSRAccessor(BaseAccessor):
    """Class representing an accessor for one-dimensional Series data.

    Accessible via `pd.Series.vbt` and its child accessors.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Optional array-like object.
        **kwargs: Keyword arguments for `BaseAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        if _full_init:
            if isinstance(wrapper, ArrayWrapper):
                if wrapper.ndim == 2:
                    if wrapper.shape[1] == 1:
                        wrapper = wrapper.replace(ndim=1)
                    else:
                        raise TypeError("Series accessors work only one one-dimensional data")

            BaseAccessor.__init__(self, wrapper, obj=obj, **kwargs)

    @hybrid_property
    def ndim(cls_or_self) -> int:
        return 1

    @hybrid_method
    def is_series(cls_or_self) -> bool:
        return True

    @hybrid_method
    def is_frame(cls_or_self) -> bool:
        return False


class BaseDFAccessor(BaseAccessor):
    """Class representing an accessor for two-dimensional DataFrame data.

    Accessible via `pd.DataFrame.vbt` and its child accessors.

    Args:
        wrapper (Union[ArrayWrapper, ArrayLike]): Array wrapper instance or array-like object.
        obj (Optional[ArrayLike]): Optional array-like object.
        **kwargs: Keyword arguments for `BaseAccessor`.
    """

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        _full_init: bool = True,
        **kwargs,
    ) -> None:
        if _full_init:
            if isinstance(wrapper, ArrayWrapper):
                if wrapper.ndim == 1:
                    wrapper = wrapper.replace(ndim=2)

            BaseAccessor.__init__(self, wrapper, obj=obj, **kwargs)

    @hybrid_property
    def ndim(cls_or_self) -> int:
        return 2

    @hybrid_method
    def is_series(cls_or_self) -> bool:
        return False

    @hybrid_method
    def is_frame(cls_or_self) -> bool:
        return True
