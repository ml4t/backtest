# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base classes and functions for resampling."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexes import repeat_index
from vectorbtpro.base.resampling import nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.decorators import cached_property, hybrid_method
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "Resampler",
]

ResamplerT = tp.TypeVar("ResamplerT", bound="Resampler")


class Resampler(Configured):
    """Class for resampling an index.

    Args:
        source_index (IndexLike): Source index to be resampled.
        target_index (IndexLike): Target index produced by resampling.
        source_freq (Union[None, bool, FrequencyLike]): Frequency of the source index
            (e.g., "daily", "15 min", "index_mean").

            See `vectorbtpro.utils.datetime_.infer_index_freq`.

            Set to False to disable automatic frequency inference.
        target_freq (Union[None, bool, FrequencyLike]): Frequency of the target index
            (e.g., "daily", "15 min", "index_mean").

            See `vectorbtpro.utils.datetime_.infer_index_freq`.

            Set to False to disable automatic frequency inference.
        silence_warnings (bool): Flag to suppress warning messages.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(
        self,
        source_index: tp.IndexLike,
        target_index: tp.IndexLike,
        source_freq: tp.Union[None, bool, tp.FrequencyLike] = None,
        target_freq: tp.Union[None, bool, tp.FrequencyLike] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        source_index = dt.prepare_dt_index(source_index)
        target_index = dt.prepare_dt_index(target_index)
        infer_source_freq = True
        if isinstance(source_freq, bool):
            if not source_freq:
                infer_source_freq = False
            source_freq = None
        infer_target_freq = True
        if isinstance(target_freq, bool):
            if not target_freq:
                infer_target_freq = False
            target_freq = None
        if infer_source_freq:
            source_freq = dt.infer_index_freq(source_index, freq=source_freq)
        if infer_target_freq:
            target_freq = dt.infer_index_freq(target_index, freq=target_freq)

        self._source_index = source_index
        self._target_index = target_index
        self._source_freq = source_freq
        self._target_freq = target_freq
        self._silence_warnings = silence_warnings

        Configured.__init__(
            self,
            source_index=source_index,
            target_index=target_index,
            source_freq=source_freq,
            target_freq=target_freq,
            silence_warnings=silence_warnings,
            **kwargs,
        )

    @property
    def source_index(self) -> tp.Index:
        """Source index used for resampling.

        Returns:
            Index: Source index.
        """
        return self._source_index

    @property
    def target_index(self) -> tp.Index:
        """Target index produced by resampling.

        Returns:
            Index: Target index.
        """
        return self._target_index

    @property
    def source_freq(self) -> tp.AnyPandasFrequency:
        """Source index frequency or date offset.

        Returns:
            AnyPandasFrequency: Frequency of the source index.
        """
        return self._source_freq

    @property
    def target_freq(self) -> tp.AnyPandasFrequency:
        """Target index frequency or date offset.

        Returns:
            AnyPandasFrequency: Frequency of the target index.
        """
        return self._target_freq

    @property
    def silence_warnings(self) -> bool:
        """Flag indicating whether warnings are silenced.

        Returns:
            bool: True if warnings are silenced, False otherwise.

        !!! info
            For default settings, see `vectorbtpro._settings.resampling`.
        """
        from vectorbtpro._settings import settings

        resampling_cfg = settings["resampling"]

        silence_warnings = self._silence_warnings
        if silence_warnings is None:
            silence_warnings = resampling_cfg["silence_warnings"]
        return silence_warnings

    @classmethod
    def from_pd_resampler(
        cls: tp.Type[ResamplerT],
        pd_resampler: tp.PandasResampler,
        source_freq: tp.Optional[tp.FrequencyLike] = None,
        silence_warnings: bool = True,
    ) -> ResamplerT:
        """Create a `Resampler` instance from a `pandas.core.resample.Resampler` object.

        Args:
            pd_resampler (pandas.core.resample.Resampler): Pandas resampler object.
            source_freq (Optional[FrequencyLike]): Frequency of the source index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            silence_warnings (bool): Flag to suppress warning messages.

        Returns:
            Resampler: New `Resampler` instance.
        """
        target_index = pd_resampler.count().index
        return cls(
            source_index=pd_resampler.obj.index,
            target_index=target_index,
            source_freq=source_freq,
            target_freq=None,
            silence_warnings=silence_warnings,
        )

    @classmethod
    def from_pd_resample(
        cls: tp.Type[ResamplerT],
        source_index: tp.IndexLike,
        *args,
        source_freq: tp.Optional[tp.FrequencyLike] = None,
        silence_warnings: bool = True,
        **kwargs,
    ) -> ResamplerT:
        """Create a `Resampler` instance using the `pd.Series.resample` method.

        Args:
            source_index (IndexLike): Source index to be resampled.
            *args: Positional arguments for `pd.Series.resample`.
            source_freq (Optional[FrequencyLike]): Frequency of the source index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `pd.Series.resample`.

        Returns:
            Resampler: New `Resampler` instance.
        """
        pd_resampler = pd.Series(index=source_index, dtype=object).resample(*args, **kwargs)
        return cls.from_pd_resampler(
            pd_resampler, source_freq=source_freq, silence_warnings=silence_warnings
        )

    @classmethod
    def from_date_range(
        cls: tp.Type[ResamplerT],
        source_index: tp.IndexLike,
        *args,
        source_freq: tp.Optional[tp.FrequencyLike] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> ResamplerT:
        """Create a `Resampler` instance using `vectorbtpro.utils.datetime_.date_range`.

        Args:
            source_index (IndexLike): Source index to be resampled.
            *args: Positional arguments for `vectorbtpro.utils.datetime_.date_range`.
            source_freq (Optional[FrequencyLike]): Frequency of the source index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `vectorbtpro.utils.datetime_.date_range`.

        Returns:
            Resampler: New `Resampler` instance.
        """
        target_index = dt.date_range(*args, **kwargs)
        return cls(
            source_index=source_index,
            target_index=target_index,
            source_freq=source_freq,
            target_freq=None,
            silence_warnings=silence_warnings,
        )

    def get_np_source_freq(
        self, silence_warnings: tp.Optional[bool] = None
    ) -> tp.AnyPandasFrequency:
        """Convert the source index frequency to NumPy format.

        Args:
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            AnyPandasFrequency: Source index frequency in NumPy format.
        """
        if silence_warnings is None:
            silence_warnings = self.silence_warnings

        warned = False
        source_freq = self.source_freq
        if source_freq is not None:
            if not isinstance(source_freq, (int, float)):
                try:
                    source_freq = dt.to_timedelta64(source_freq)
                except ValueError:
                    if not silence_warnings:
                        warn(f"Cannot convert {source_freq} to np.timedelta64. Setting to None.")
                        warned = True
                    source_freq = None
        if source_freq is None:
            if not warned and not silence_warnings:
                warn("Using right bound of source index without frequency. Set source frequency.")
        return source_freq

    def get_np_target_freq(
        self, silence_warnings: tp.Optional[bool] = None
    ) -> tp.AnyPandasFrequency:
        """Convert the target index frequency to NumPy format.

        Args:
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            AnyPandasFrequency: Target index frequency in NumPy format.
        """
        if silence_warnings is None:
            silence_warnings = self.silence_warnings

        warned = False
        target_freq = self.target_freq
        if target_freq is not None:
            if not isinstance(target_freq, (int, float)):
                try:
                    target_freq = dt.to_timedelta64(target_freq)
                except ValueError:
                    if not silence_warnings:
                        warn(f"Cannot convert {target_freq} to np.timedelta64. Setting to None.")
                        warned = True
                    target_freq = None
        if target_freq is None:
            if not warned and not silence_warnings:
                warn("Using right bound of target index without frequency. Set target frequency.")
        return target_freq

    @classmethod
    def get_lbound_index(cls, index: tp.Index, freq: tp.AnyPandasFrequency = None) -> tp.Index:
        """Return the left bound of a datetime index.

        Args:
            index (Index): Datetime index.
            freq (AnyPandasFrequency): Pandas-friendly frequency used to shift the index.

        Returns:
            Index: Datetime index representing the calculated left bound.
        """
        index = dt.prepare_dt_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if freq is not None:
            return index.shift(-1, freq=freq) + pd.Timedelta(1, "ns")
        min_ts = pd.DatetimeIndex([pd.Timestamp.min.tz_localize(index.tz)])
        return (index[:-1] + pd.Timedelta(1, "ns")).append(min_ts)

    @classmethod
    def get_rbound_index(cls, index: tp.Index, freq: tp.AnyPandasFrequency = None) -> tp.Index:
        """Return the right bound of a datetime index.

        Args:
            index (Index): Datetime index.
            freq (AnyPandasFrequency): Pandas-friendly frequency used to shift the index.

        Returns:
            Index: Datetime index representing the calculated right bound.
        """
        index = dt.prepare_dt_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if freq is not None:
            return index.shift(1, freq=freq) - pd.Timedelta(1, "ns")
        max_ts = pd.DatetimeIndex([pd.Timestamp.max])
        if index.tz is not None:
            max_ts = max_ts.tz_localize("utc").tz_convert(index.tz)
        return (index[1:] - pd.Timedelta(1, "ns")).append(max_ts)

    @cached_property
    def source_lbound_index(self) -> tp.Index:
        """Left bound of the source datetime index.

        Returns:
            Index: Left bound of the source index.
        """
        return self.get_lbound_index(self.source_index, freq=self.source_freq)

    @cached_property
    def source_rbound_index(self) -> tp.Index:
        """Right bound of the source datetime index.

        Returns:
            Index: Right bound of the source index.
        """
        return self.get_rbound_index(self.source_index, freq=self.source_freq)

    @cached_property
    def target_lbound_index(self) -> tp.Index:
        """Left bound of the target datetime index.

        Returns:
            Index: Left bound of the target index.
        """
        return self.get_lbound_index(self.target_index, freq=self.target_freq)

    @cached_property
    def target_rbound_index(self) -> tp.Index:
        """Right bound of the target datetime index.

        Returns:
            Index: Right bound of the target index.
        """
        return self.get_rbound_index(self.target_index, freq=self.target_freq)

    def map_to_target_index(
        self,
        before: bool = False,
        raise_missing: bool = True,
        return_index: bool = True,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[tp.Array1d, tp.Index]:
        """Return the mapping from the source index to the target index.

        Args:
            before (bool): If True, include source indices preceding or equal to the target;
                otherwise, include those following or equal.
            raise_missing (bool): If True, raise an error when a source index cannot be mapped; otherwise, assign -1.
            return_index (bool): Return a Pandas Index if True; otherwise, return a NumPy array.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            Union[Array1d, Index]: Mapped index values.

        See:
            `vectorbtpro.base.resampling.nb.map_to_target_index_nb`
        """
        target_freq = self.get_np_target_freq(silence_warnings=silence_warnings)
        func = jit_reg.resolve_option(nb.map_to_target_index_nb, jitted)
        mapped_arr = func(
            self.source_index.values,
            self.target_index.values,
            target_freq=target_freq,
            before=before,
            raise_missing=raise_missing,
        )
        if return_index:
            nan_mask = mapped_arr == -1
            if nan_mask.any():
                mapped_index = self.source_index.to_series().copy()
                mapped_index[nan_mask] = np.nan
                mapped_index[~nan_mask] = self.target_index[mapped_arr]
                mapped_index = pd.Index(mapped_index)
            else:
                mapped_index = self.target_index[mapped_arr]
            return mapped_index
        return mapped_arr

    def index_difference(
        self,
        reverse: bool = False,
        return_index: bool = True,
        jitted: tp.JittedOption = None,
    ) -> tp.Union[tp.Array1d, tp.Index]:
        """Return the index difference mapping between the source and target indices.

        Args:
            reverse (bool): Reverse the order of indices for difference calculation if True.
            return_index (bool): Return a Pandas Index if True; otherwise, return a NumPy array.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Union[Array1d, Index]: Computed index difference mapping.

        See:
            `vectorbtpro.base.resampling.nb.index_difference_nb`
        """
        func = jit_reg.resolve_option(nb.index_difference_nb, jitted)
        if reverse:
            mapped_arr = func(self.target_index.values, self.source_index.values)
        else:
            mapped_arr = func(self.source_index.values, self.target_index.values)
        if return_index:
            return self.target_index[mapped_arr]
        return mapped_arr

    def map_index_to_source_ranges(
        self,
        before: bool = False,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Return the mapping of source index ranges corresponding to the target index.

        Args:
            before (bool): If True, include source indices preceding or equal to the target;
                otherwise, include those following or equal.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            Tuple[Array1d, Array1d]: Tuple with the start and end indices of the source ranges.

        See:
            `vectorbtpro.base.resampling.nb.map_index_to_source_ranges_nb`

        !!! note
            If `Resampler.target_freq` is a date offset, it is set to None and a warning is emitted.
            An additional warning is raised if `target_freq` is None.
        """
        target_freq = self.get_np_target_freq(silence_warnings=silence_warnings)
        func = jit_reg.resolve_option(nb.map_index_to_source_ranges_nb, jitted)
        return func(
            self.source_index.values,
            self.target_index.values,
            target_freq=target_freq,
            before=before,
        )

    @hybrid_method
    def map_bounds_to_source_ranges(
        cls_or_self,
        source_index: tp.Optional[tp.IndexLike] = None,
        target_lbound_index: tp.Optional[tp.IndexLike] = None,
        target_rbound_index: tp.Optional[tp.IndexLike] = None,
        closed_lbound: bool = True,
        closed_rbound: bool = False,
        skip_not_found: bool = False,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Return the mapping from target index bounds to source index ranges.

        Either `target_lbound_index` or `target_rbound_index` must be provided.

        Args:
            source_index (Optional[IndexLike]): Source datetime index.
            target_lbound_index (Optional[IndexLike]): Left bound of the target index.

                Set to "pandas" to use `Resampler.get_lbound_index`.
            target_rbound_index (Optional[IndexLike]): Right bound of the target index.

                Set to "pandas" to use `Resampler.get_rbound_index`.
            closed_lbound (bool): Indicates if the left bound is inclusive.
            closed_rbound (bool): Indicates if the right bound is inclusive.
            skip_not_found (bool): Whether to drop indices that are -1 (not found).
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Tuple[Array1d, Array1d]: Pair of arrays representing the mapping from target bounds to source ranges.

        See:
            `vectorbtpro.base.resampling.nb.map_bounds_to_source_ranges_nb`
        """
        if not isinstance(cls_or_self, type):
            if target_lbound_index is None and target_rbound_index is None:
                raise ValueError("Either target_lbound_index or target_rbound_index must be set")
            if target_lbound_index is not None:
                if isinstance(target_lbound_index, str) and target_lbound_index.lower() == "pandas":
                    target_lbound_index = cls_or_self.target_lbound_index
                else:
                    target_lbound_index = dt.prepare_dt_index(target_lbound_index)
                target_rbound_index = cls_or_self.target_index
            if target_rbound_index is not None:
                target_lbound_index = cls_or_self.target_index
                if isinstance(target_rbound_index, str) and target_rbound_index.lower() == "pandas":
                    target_rbound_index = cls_or_self.target_rbound_index
                else:
                    target_rbound_index = dt.prepare_dt_index(target_rbound_index)
            if len(target_lbound_index) == 1 and len(target_rbound_index) > 1:
                target_lbound_index = repeat_index(target_lbound_index, len(target_rbound_index))
            elif len(target_lbound_index) > 1 and len(target_rbound_index) == 1:
                target_rbound_index = repeat_index(target_rbound_index, len(target_lbound_index))
        else:
            source_index = dt.prepare_dt_index(source_index)
            target_lbound_index = dt.prepare_dt_index(target_lbound_index)
            target_rbound_index = dt.prepare_dt_index(target_rbound_index)

        checks.assert_len_equal(target_rbound_index, target_lbound_index)
        func = jit_reg.resolve_option(nb.map_bounds_to_source_ranges_nb, jitted)
        return func(
            source_index.values,
            target_lbound_index.values,
            target_rbound_index.values,
            closed_lbound=closed_lbound,
            closed_rbound=closed_rbound,
            skip_not_found=skip_not_found,
        )

    def resample_source_mask(
        self,
        source_mask: tp.ArrayLike,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Array1d:
        """Return a resampled mask for the source index.

        Args:
            source_mask (ArrayLike): Boolean mask for the source index.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            Array1d: Resampled array corresponding to the source mask.

        See:
            `vectorbtpro.base.resampling.nb.resample_source_mask_nb`
        """
        from vectorbtpro.base.reshaping import broadcast_array_to

        if silence_warnings is None:
            silence_warnings = self.silence_warnings
        source_mask = broadcast_array_to(source_mask, len(self.source_index))
        source_freq = self.get_np_source_freq(silence_warnings=silence_warnings)
        target_freq = self.get_np_target_freq(silence_warnings=silence_warnings)

        func = jit_reg.resolve_option(nb.resample_source_mask_nb, jitted)
        return func(
            source_mask,
            self.source_index.values,
            self.target_index.values,
            source_freq,
            target_freq,
        )

    def last_before_target_index(
        self,
        incl_source: bool = True,
        incl_target: bool = False,
        jitted: tp.JittedOption = None,
    ) -> tp.Array1d:
        """Return the index of the last element before each target index.

        Args:
            incl_source (bool): Whether to include the original source index in the result.
            incl_target (bool): Whether to include the target index if it matches a source index.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Array1d: Array of indices representing the last element before each target index.

        See:
            `vectorbtpro.base.resampling.nb.last_before_target_index_nb`
        """
        func = jit_reg.resolve_option(nb.last_before_target_index_nb, jitted)
        return func(
            self.source_index.values,
            self.target_index.values,
            incl_source=incl_source,
            incl_target=incl_target,
        )
