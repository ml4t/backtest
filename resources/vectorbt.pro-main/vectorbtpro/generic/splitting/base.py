# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base functionality for splitting."""

import inspect
import math

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.accessors import BaseIDXAccessor
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.indexes import combine_indexes, stack_indexes
from vectorbtpro.base.indexing import PandasIndexer, get_index_ranges, hslice
from vectorbtpro.base.merging import column_stack_merge, is_merge_func_from_config, row_stack_merge
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.reshaping import to_dict
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.splitting import nb
from vectorbtpro.generic.splitting.purged import BasePurgedCV, PurgedKFoldCV, PurgedWalkForwardCV
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.annotations import Annotatable, has_annotatables
from vectorbtpro.utils.array_ import is_range
from vectorbtpro.utils.attr_ import MISSING, DefineMixin, define
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import Config, HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.eval_ import Evaluable
from vectorbtpro.utils.execution import (
    NoResult,
    NoResultsException,
    Task,
    execute,
    filter_out_no_results,
)
from vectorbtpro.utils.merging import MergeFunc, parse_merge_func
from vectorbtpro.utils.parsing import (
    ann_args_to_args,
    annotate_args,
    flatten_ann_args,
    get_func_arg_names,
    unflatten_ann_args,
)
from vectorbtpro.utils.selection import LabelSel, PosSel
from vectorbtpro.utils.template import CustomTemplate, Rep, RepFunc, substitute_templates
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator as BaseCrossValidatorT
else:
    BaseCrossValidatorT = "sklearn.model_selection.BaseCrossValidator"

__all__ = [
    "FixRange",
    "RelRange",
    "Takeable",
    "Splitter",
]

__pdoc__ = {}

SplitterT = tp.TypeVar("SplitterT", bound="Splitter")


@define
class FixRange(DefineMixin):
    """Class representing a fixed range."""

    range_: tp.FixRangeLike = define.field()
    """Range."""


@define
class RelRange(DefineMixin):
    """Class representing a relative range."""

    offset: tp.Union[int, float, tp.TimedeltaLike] = define.field(default=0)
    """Offset value.

    Floating numbers between 0 and 1 are interpreted as relative.

    Can be negative.
    """

    offset_anchor: str = define.field(default="prev_end")
    """Anchor used for offset.

    Supported values:

    * 'start': Start of the range.
    * 'end': End of the range.
    * 'prev_start': Start of the previous range.
    * 'prev_end': End of the previous range.
    * 'next_start': Next start.
    * 'next_end': Next end.
    """

    offset_space: str = define.field(default="free")
    """Space used for applying the relative offset.

    Supported values:

    * 'all': Use the entire available space.
    * 'free': Use the remaining space after the offset anchor.
    * 'prev': Use the length of the previous range.

    Applied only when `RelRange.offset` is relative.
    """

    length: tp.Union[int, float, tp.TimedeltaLike] = define.field(default=1.0)
    """Range length value.

    Floating numbers between 0 and 1 are interpreted as relative.

    Can be negative.
    """

    length_space: str = define.field(default="free")
    """Space used for applying the relative length.

    Supported values:

    * 'all': Use the entire available space.
    * 'free': Use the remaining space after the offset.
    * 'free_or_prev': Use the remaining space after the offset or the size of the previous range,
        depending on which comes first in the direction of `RelRange.length`.

    Applied only when `RelRange.length` is relative.
    """

    out_of_bounds: str = define.field(default="warn")
    """Strategy for handling indices that are out of bounds.

    Supported values:

    * 'keep': Retain out-of-bounds values.
    * 'ignore': Silently ignore out-of-bound positions.
    * 'warn': Emit a warning if values are out of bounds.
    * 'raise': Raise an error for out-of-bound values.
    """

    is_gap: bool = define.field(default=False)
    """Indicates whether the range represents a gap."""

    def __attrs_post_init__(self):
        object.__setattr__(self, "offset_anchor", self.offset_anchor.lower())
        if self.offset_anchor not in (
            "start",
            "end",
            "prev_start",
            "prev_end",
            "next_start",
            "next_end",
        ):
            raise ValueError(f"Invalid offset_anchor: '{self.offset_anchor}'")
        object.__setattr__(self, "offset_space", self.offset_space.lower())
        if self.offset_space not in ("all", "free", "prev"):
            raise ValueError(f"Invalid offset_space: '{self.offset_space}'")
        object.__setattr__(self, "length_space", self.length_space.lower())
        if self.length_space not in ("all", "free", "free_or_prev"):
            raise ValueError(f"Invalid length_space: '{self.length_space}'")
        object.__setattr__(self, "out_of_bounds", self.out_of_bounds.lower())
        if self.out_of_bounds not in ("keep", "ignore", "warn", "raise"):
            raise ValueError(f"Invalid out_of_bounds: '{self.out_of_bounds}'")

    def to_slice(
        self,
        total_len: int,
        prev_start: int = 0,
        prev_end: int = 0,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> slice:
        """Convert the relative range to a slice.

        Args:
            total_len (int): Total number of indices.
            prev_start (int): Start index of the previous range.
            prev_end (int): End index of the previous range.
            index (Optional[IndexLike]): Index from which to derive datetime information.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            slice: Slice object computed based on the relative range parameters.
        """
        if index is not None:
            index = dt.prepare_dt_index(index)
            freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        offset_anchor = self.offset_anchor
        offset = self.offset
        length = self.length
        if not checks.is_number(offset) or not checks.is_number(length):
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError(f"Index must be of type DatetimeIndex, not {index.dtype}")

        if offset_anchor == "start":
            if checks.is_number(offset):
                offset_anchor = 0
            else:
                offset_anchor = index[0]
        elif offset_anchor == "end":
            if checks.is_number(offset):
                offset_anchor = total_len
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                offset_anchor = index[-1] + freq
        elif offset_anchor == "prev_start":
            if checks.is_number(offset):
                offset_anchor = prev_start
            else:
                offset_anchor = index[prev_start]
        else:
            if checks.is_number(offset):
                offset_anchor = prev_end
            else:
                if prev_end < total_len:
                    offset_anchor = index[prev_end]
                else:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset_anchor = index[-1] + freq

        if checks.is_float(offset) and 0 <= abs(offset) <= 1:
            if self.offset_space == "all":
                offset = offset_anchor + int(offset * total_len)
            elif self.offset_space == "free":
                if offset < 0:
                    offset = int((1 + offset) * offset_anchor)
                else:
                    offset = offset_anchor + int(offset * (total_len - offset_anchor))
            else:
                offset = offset_anchor + int(offset * (prev_end - prev_start))
        else:
            if checks.is_float(offset):
                if not offset.is_integer():
                    raise ValueError(
                        f"Floating number for offset ({offset}) must be between 0 and 1"
                    )
                offset = offset_anchor + int(offset)
            elif not checks.is_int(offset):
                offset = offset_anchor + dt.to_freq(offset)
                if index[0] <= offset <= index[-1]:
                    offset = index.get_indexer([offset], method="ffill")[0]
                elif offset < index[0]:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset = -int((index[0] - offset) / freq)
                else:
                    if freq is None:
                        raise ValueError("Must provide frequency")
                    offset = total_len - 1 + int((offset - index[-1]) / freq)
            else:
                offset = offset_anchor + offset

        if checks.is_float(length) and 0 <= abs(length) <= 1:
            if self.length_space == "all":
                length = int(length * total_len)
            elif self.length_space == "free":
                if length < 0:
                    length = int(length * offset)
                else:
                    length = int(length * (total_len - offset))
            else:
                if length < 0:
                    if offset > prev_end:
                        length = int(length * (offset - prev_end))
                    else:
                        length = int(length * offset)
                else:
                    if offset < prev_start:
                        length = int(length * (prev_start - offset))
                    else:
                        length = int(length * (total_len - offset))
        else:
            if checks.is_float(length):
                if not length.is_integer():
                    raise ValueError(
                        f"Floating number for length ({length}) must be between 0 and 1"
                    )
                length = int(length)
            elif not checks.is_int(length):
                length = dt.to_freq(length)

        start = offset
        if checks.is_int(length):
            if isinstance(length, int):
                stop = start + length
            else:
                stop = start + length
        else:
            if 0 <= start < total_len:
                stop = index[start] + length
            elif start < 0:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = index[0] + start * freq + length
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = index[-1] + (start - total_len + 1) * freq + length
            if stop <= index[-1]:
                stop = index.get_indexer([stop], method="bfill")[0]
            else:
                if freq is None:
                    raise ValueError("Must provide frequency")
                stop = total_len - 1 + int((stop - index[-1]) / freq)
        if checks.is_int(length):
            if length < 0:
                start, stop = stop, start
        else:
            if length < pd.Timedelta(0):
                start, stop = stop, start
        if start < 0:
            if self.out_of_bounds == "ignore":
                start = 0
            elif self.out_of_bounds == "warn":
                warn(f"Range start ({start}) is out of bounds")
                start = 0
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range start ({start}) is out of bounds")
        if stop > total_len:
            if self.out_of_bounds == "ignore":
                stop = total_len
            elif self.out_of_bounds == "warn":
                warn(f"Range stop ({stop}) is out of bounds")
                stop = total_len
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range stop ({stop}) is out of bounds")
        if stop - start < 0:
            raise ValueError("Range length is negative")
        if stop - start == 0:
            raise ZeroLengthError("Range has zero length")
        return slice(start, stop)


@define
class Takeable(Evaluable, Annotatable, DefineMixin):
    """Class representing an object from which a range can be taken."""

    obj: tp.Any = define.required_field()
    """Object from which the range is taken."""

    remap_to_obj: bool = define.optional_field()
    """Boolean indicating whether to remap `Splitter.index` to the index of `Takeable.obj`.

    If False, it is assumed that the object already has the same index.
    """

    index: tp.Optional[tp.IndexLike] = define.optional_field()
    """Index associated with the object.

    If not provided, `Splitter.get_obj_index` is used to retrieve it.
    """

    freq: tp.Optional[tp.FrequencyLike] = define.optional_field()
    """Frequency associated with `Takeable.index`."""

    point_wise: bool = define.optional_field()
    """Boolean indicating whether to select one range point at a time and return a tuple."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """Identifier(s) at which to evaluate this instance."""


class ZeroLengthError(ValueError):
    """Exception raised when a range has a zero length."""

    pass


class Splitter(Analyzable):
    """Base class for splitting.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        index (Index): Index used for splitting.
        splits_arr (SplitsArray): Two-dimensional array representing splits.

            The first axis represents splits and the second axis represents sets.
            Each element is a range defined as a slice, a sequence of indices, a mask,
            or a callable returning such.
        **kwargs: Keyword arguments for `vectorbtpro.generic.analyzable.Analyzable`.

    !!! info
        For default settings, see `vectorbtpro._settings.splitter`.
    """

    def __init__(
        self,
        wrapper: ArrayWrapper,
        index: tp.Index,
        splits_arr: tp.SplitsArray,
        **kwargs,
    ) -> None:
        if wrapper.grouper.is_grouped():
            raise ValueError("Splitter cannot be grouped")
        index = dt.prepare_dt_index(index)
        if splits_arr.shape[0] != wrapper.shape_2d[0]:
            raise ValueError("Number of splits must match wrapper index")
        if splits_arr.shape[1] != wrapper.shape_2d[1]:
            raise ValueError("Number of sets must match wrapper columns")

        Analyzable.__init__(
            self,
            wrapper,
            index=index,
            splits_arr=splits_arr,
            **kwargs,
        )

        self._index = index
        self._splits_arr = splits_arr

    @property
    def index(self) -> tp.Index:
        """Index used for splitting.

        Returns:
            Index: Index used for splitting.
        """
        return self._index

    @property
    def splits_arr(self) -> tp.SplitsArray:
        """Two-dimensional array representing splits.

        The first axis represents splits and the second axis represents sets.
        Each element is a range defined as a slice, a sequence of indices, a mask,
        or a callable returning such.

        Returns:
            SplitsArray: Two-dimensional array representing splits.
        """
        return self._splits_arr

    @classmethod
    def from_splits(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        splits: tp.Splits,
        squeeze: bool = False,
        fix_ranges: bool = True,
        wrap_with_fixrange: bool = False,
        split_range_kwargs: tp.KwargsLike = None,
        split_check_template: tp.Optional[tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from an iterable of splits.

        Args:
            index (IndexLike): Index used to align the splits.
            splits (Splits): Iterable of splits supporting both absolute and relative ranges.

                Enable `fix_ranges` to convert relative ranges to absolute ranges.
            squeeze (bool): Flag indicating whether to convert a single-column DataFrame to a Series.
            fix_ranges (bool): Whether to convert relative ranges (`RelRange`) into fixed (`FixRange`).
            wrap_with_fixrange (bool): Wrap fixed ranges with `FixRange`.

                If a range is an array, it is wrapped to avoid creating a 3D array.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            split_check_template (Optional[CustomTemplate]): Template to validate each split.

                The current split is passed as `split`; splits that evaluate to False are discarded.
            template_context (KwargsLike): Additional context for template substitution.
            split_labels (Optional[IndexLike]): Labels for the splits.

                Can be provided as a template.
            set_labels (Optional[IndexLike]): Labels for the sets.

                Can be provided as a template.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Splitter`.

        Returns:
            Splitter: New `Splitter` instance.
        """
        index = dt.prepare_dt_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}

        new_splits = []
        removed_indices = []
        for i, split in enumerate(splits):
            already_fixed = False
            if checks.is_number(split) or checks.is_td_like(split):
                split = cls.split_range(
                    slice(None),
                    split,
                    template_context=template_context,
                    index=index,
                    wrap_with_fixrange=False,
                    **split_range_kwargs,
                )
                already_fixed = True
                new_split = split
                ndim = 2
            elif cls.is_range_relative(split) or not checks.is_sequence(split) or isinstance(split, np.ndarray):
                new_split = [split]
                ndim = 1
            else:
                new_split = split
                ndim = 2
            if fix_ranges and not already_fixed:
                new_split = cls.split_range(
                    slice(None),
                    new_split,
                    template_context=template_context,
                    index=index,
                    wrap_with_fixrange=False,
                    **split_range_kwargs,
                )
            _new_split = []
            for range_ in new_split:
                if checks.is_number(range_) or checks.is_td_like(range_):
                    range_ = RelRange(length=range_)
                if not isinstance(range_, (FixRange, RelRange)):
                    if wrap_with_fixrange or checks.is_sequence(range_):
                        _new_split.append(FixRange(range_))
                    else:
                        _new_split.append(range_)
                else:
                    _new_split.append(range_)
            if split_check_template is not None:
                _template_context = merge_dicts(
                    dict(index=index, i=i, split=_new_split), template_context
                )
                split_ok = substitute_templates(
                    split_check_template, _template_context, eval_id="split_check_template"
                )
                if not split_ok:
                    removed_indices.append(i)
                    continue
            new_splits.append(_new_split)
        if len(new_splits) == 0:
            raise ValueError("Must provide at least one range")
        new_splits_arr = np.asarray(new_splits, dtype=object)
        if squeeze and new_splits_arr.shape[1] == 1:
            ndim = 1

        if split_labels is None:
            split_labels = pd.RangeIndex(stop=new_splits_arr.shape[0], name="split")
        else:
            if isinstance(split_labels, CustomTemplate):
                _template_context = merge_dicts(
                    dict(index=index, splits_arr=new_splits_arr), template_context
                )
                split_labels = substitute_templates(
                    split_labels, _template_context, eval_id=split_labels
                )
                if not isinstance(split_labels, pd.Index):
                    split_labels = pd.Index(split_labels, name="split")
            else:
                if not isinstance(split_labels, pd.Index):
                    split_labels = pd.Index(split_labels, name="split")
                if len(removed_indices) > 0:
                    split_labels = split_labels.delete(removed_indices)
        if set_labels is None:
            set_labels = pd.Index(
                ["set_%d" % i for i in range(new_splits_arr.shape[1])], name="set"
            )
        else:
            if isinstance(split_labels, CustomTemplate):
                _template_context = merge_dicts(
                    dict(index=index, splits_arr=new_splits_arr), template_context
                )
                set_labels = substitute_templates(set_labels, _template_context, eval_id=set_labels)
            if not isinstance(set_labels, pd.Index):
                set_labels = pd.Index(set_labels, name="set")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        wrapper = ArrayWrapper(index=split_labels, columns=set_labels, ndim=ndim, **wrapper_kwargs)
        return cls(wrapper, index, new_splits_arr, **kwargs)

    @classmethod
    def from_single(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        split: tp.Optional[tp.SplitLike],
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from a single split.

        Args:
            index (IndexLike): Index used for the split.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.
        """
        if split_range_kwargs is None:
            split_range_kwargs = {}
        new_split = cls.split_range(
            slice(None),
            split,
            template_context=template_context,
            index=index,
            **split_range_kwargs,
        )
        splits = [new_split]

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_rolling(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        length: tp.Union[int, float, tp.TimedeltaLike],
        offset: tp.Union[int, float, tp.TimedeltaLike] = 0,
        offset_anchor: str = "prev_end",
        offset_anchor_set: tp.Optional[int] = 0,
        offset_space: str = "prev",
        backwards: tp.Union[bool, str] = False,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from a rolling range of fixed length.

        Uses `Splitter.from_splits` to generate an array of splits and corresponding labels, and then
        construct the `Splitter` instance.

        Args:
            index (IndexLike): Index over which the rolling range is computed.
            length (Union[int, float, TimedeltaLike]): Desired length of the rolling range.
            offset (Union[int, float, TimedeltaLike]): Offset after the previous range's
                right boundary to determine the start of the next range.

                See `RelRange.offset`.
            offset_anchor (str): Anchor point used when applying the offset.

                See `RelRange.offset_anchor`.
            offset_anchor_set (Optional[int]): Index of the set from the previous range
                used as the offset anchor.

                If None, the entire previous split is used as a single anchor.
                By default, the first set is used.
            offset_space (str): Type of offset space.

                See `RelRange.offset_space`.
            backwards (Union[bool, str]): Determines whether rolling occurs in reverse order.

                If set to `'sorted'`, splits are sorted by their start index after rolling.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            range_bounds_kwargs (KwargsLike): Keyword arguments for getting range bounds.

                See `Splitter.get_range_bounds`.
            template_context (KwargsLike): Additional context for template substitution.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Divide a range into a set of non-overlapping ranges:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_rolling(index, 30)
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_rolling_1.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_rolling_1.dark.svg#only-dark){: .iimg loading=lazy }

            Divide a range into ranges, each split into 1/2:

            ```pycon
            >>> splitter = vbt.Splitter.from_rolling(
            ...     index,
            ...     60,
            ...     split=1/2,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_rolling_2.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_rolling_2.dark.svg#only-dark){: .iimg loading=lazy }

            Create non-overlapping ranges by using the right bound of the last set as an offset anchor:

            ```pycon
            >>> splitter = vbt.Splitter.from_rolling(
            ...     index,
            ...     60,
            ...     offset_anchor_set=-1,
            ...     split=1/2,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_rolling_3.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_rolling_3.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if isinstance(backwards, str):
            if backwards.lower() == "sorted":
                sort_backwards = True
            else:
                raise ValueError(f"Invalid backwards: '{backwards}'")
            backwards = True
        else:
            sort_backwards = False
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = RelRange(
                    length=-length if backwards else length,
                    offset_anchor="end" if backwards else "start",
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), index=index, freq=freq)
            else:
                if offset_anchor_set is None:
                    prev_start, prev_end = bounds[-1][0][0], bounds[-1][-1][1]
                else:
                    prev_start, prev_end = bounds[-1][offset_anchor_set]
                new_split = RelRange(
                    offset=offset,
                    offset_anchor=offset_anchor,
                    offset_space=offset_space,
                    length=-length if backwards else length,
                    length_space="all",
                    out_of_bounds="keep",
                ).to_slice(
                    total_len=len(index),
                    prev_start=prev_start,
                    prev_end=prev_end,
                    index=index,
                    freq=freq,
                )
                if backwards:
                    if new_split.stop >= bounds[-1][-1][1]:
                        raise ValueError("Infinite loop detected. Provide a positive offset.")
                else:
                    if new_split.start <= bounds[-1][0][0]:
                        raise ValueError("Infinite loop detected. Provide a positive offset.")
            if backwards:
                if new_split.start < 0:
                    break
                if new_split.stop > len(index):
                    raise ValueError("Range stop cannot exceed index length")
            else:
                if new_split.start < 0:
                    raise ValueError("Range start cannot be negative")
                if new_split.stop > len(index):
                    break
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                bounds.append(
                    tuple(
                        map(
                            lambda x: cls.get_range_bounds(
                                x,
                                template_context=template_context,
                                index=index,
                                **range_bounds_kwargs,
                            ),
                            new_split,
                        )
                    )
                )
            else:
                bounds.append(((new_split.start, new_split.stop),))
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits[::-1] if sort_backwards else splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_rolling(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        length: tp.Union[None, str, int, float, tp.TimedeltaLike] = None,
        optimize_anchor_set: int = 1,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a Splitter instance from a fixed number of rolling ranges with equal length.

        Args:
            index (IndexLike): Index used to generate rolling ranges.
            n (int): Number of rolling ranges to generate.
            length (Union[None, str, int, float, TimedeltaLike]): Length of each range.

                * If None, splits the index evenly into n non-overlapping ranges using `Splitter.from_rolling`.
                * If a numeric value, it defines either a fraction of the index length or an absolute length.
                * If "optimize", determines an optimal length to cover most of the index.
            optimize_anchor_set (int): Specifies which anchor set to optimize when using `length="optimize"`.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            template_context (KwargsLike): Additional context for template substitution.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_rolling` if `length` is None or "optimize",
                or `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Roll 10 ranges with 100 elements, and split it into 3/4:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_rolling(
            ...     index,
            ...     10,
            ...     length=100,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_n_rolling.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_n_rolling.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        if length is None:
            return cls.from_rolling(
                index,
                length=len(index) // n,
                offset=0,
                offset_anchor="prev_end",
                offset_anchor_set=None,
                split=split,
                split_range_kwargs=split_range_kwargs,
                template_context=template_context,
                **kwargs,
            )

        if isinstance(length, str) and length.lower() == "optimize":
            from scipy.optimize import minimize_scalar

            if split is not None and not checks.is_float(split):
                raise TypeError("Split must be a float when length='optimize'")
            checks.assert_in(optimize_anchor_set, (0, 1))

            if split is None:
                ratio = 1.0
            else:
                ratio = split

            def _empty_len_objective(length):
                length = math.ceil(length)
                first_len = int(ratio * length)
                second_len = length - first_len
                if split is None or optimize_anchor_set == 0:
                    empty_len = len(index) - (n * first_len + second_len)
                else:
                    empty_len = len(index) - (n * second_len + first_len)
                if empty_len >= 0:
                    return empty_len
                return len(index)

            length = math.ceil(minimize_scalar(_empty_len_objective).x)
            if split is None or optimize_anchor_set == 0:
                offset = int(ratio * length)
            else:
                offset = length - int(ratio * length)
            return cls.from_rolling(
                index,
                length=length,
                offset=offset,
                offset_anchor="prev_start",
                offset_anchor_set=None,
                split=split,
                split_range_kwargs=split_range_kwargs,
                template_context=template_context,
                **kwargs,
            )

        if checks.is_float(length):
            if 0 <= abs(length) <= 1:
                length = len(index) * length
            elif not length.is_integer():
                raise ValueError("Floating number for length must be between 0 and 1")
            length = int(length)
        if checks.is_int(length):
            if length < 1 or length > len(index):
                raise ValueError(f"Length must be within [{1}, {len(index)}]")
            offsets = np.arange(len(index))
            offsets = offsets[offsets + length <= len(index)]
        else:
            length = dt.to_freq(length)
            if freq is None:
                raise ValueError("Must provide freq")
            if length < freq or length > index[-1] + freq - index[0]:
                raise ValueError(f"Length must be within [{freq}, {index[-1] + freq - index[0]}]")
            offsets = index[index + length <= index[-1] + freq] - index[0]
        if n > len(offsets):
            n = len(offsets)
        rows = np.round(np.linspace(0, len(offsets) - 1, n)).astype(int)
        offsets = offsets[rows]

        splits = []
        for offset in offsets:
            new_split = RelRange(
                offset=offset,
                length=length,
            ).to_slice(len(index), index=index, freq=freq)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)
        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_expanding(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        min_length: tp.Union[int, float, tp.TimedeltaLike],
        offset: tp.Union[int, float, tp.TimedeltaLike],
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from an expanding range.

        Creates an expanding sequence of slices based on the provided index.
        The first slice uses a minimum length defined by min_length, and each subsequent
        slice begins after an offset from the previous slice's right boundary.

        Args:
            index (IndexLike): Index to split.
            min_length (Union[int, float, TimedeltaLike]): Minimum length for the first expanding range.
                If specified as a float between 0 and 1, it is interpreted relative to the length of the index.
            offset (Union[int, float, TimedeltaLike]): Offset after the previous range's
                right boundary to determine the start of the next range.

                It may also be provided as a float relative to the index length.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            range_bounds_kwargs (KwargsLike): Keyword arguments for getting range bounds.

                See `Splitter.get_range_bounds`.
            template_context (KwargsLike): Additional context for template substitution.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Roll an expanding range with a length of 10 and an offset of 10, and split it into 3/4:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_expanding(
            ...     index,
            ...     10,
            ...     10,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_expanding.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_expanding.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = RelRange(
                    length=min_length,
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), index=index, freq=freq)
            else:
                prev_end = bounds[-1][-1][-1]
                new_split = RelRange(
                    offset=offset,
                    offset_anchor="prev_end",
                    offset_space="all",
                    length=-1.0,
                    out_of_bounds="keep",
                ).to_slice(total_len=len(index), prev_end=prev_end, index=index, freq=freq)
                if new_split.stop <= prev_end:
                    raise ValueError("Infinite loop detected. Provide a positive offset.")
            if new_split.start < 0:
                raise ValueError("Range start cannot be negative")
            if new_split.stop > len(index):
                break
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                bounds.append(
                    tuple(
                        map(
                            lambda x: cls.get_range_bounds(
                                x,
                                template_context=template_context,
                                index=index,
                                **range_bounds_kwargs,
                            ),
                            new_split,
                        )
                    )
                )
            else:
                bounds.append(((new_split.start, new_split.stop),))
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_expanding(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        min_length: tp.Union[None, int, float, tp.TimedeltaLike] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from multiple expanding ranges.

        Selects n evenly spaced expanding ranges based on the given index. Each range uses
        a minimum length specified by min_length, which is automatically computed if not provided.
        An optional split configuration can be applied to transform each range.

        Args:
            index (IndexLike): Index to split.
            n (int): Number of expanding ranges to select.
            min_length (Union[None, int, float, TimedeltaLike]): Minimum length for each expanding range.

                If specified as a float between 0 and 1, it is interpreted relative to the length of the index.
                If None, it is determined based on the index length and the specified number of ranges.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            template_context (KwargsLike): Additional context for template substitution.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Roll 10 expanding ranges with a minimum length of 100, while reserving 50 elements for test:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_expanding(
            ...     index,
            ...     10,
            ...     min_length=100,
            ...     split=-50,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_n_expanding.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_n_expanding.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        if min_length is None:
            min_length = len(index) // n
        if checks.is_float(min_length):
            if 0 <= abs(min_length) <= 1:
                min_length = len(index) * min_length
            elif not min_length.is_integer():
                raise ValueError("Floating number for minimum length must be between 0 and 1")
        if checks.is_int(min_length):
            min_length = int(min_length)
            if min_length < 1 or min_length > len(index):
                raise ValueError(f"Minimum length must be within [{1}, {len(index)}]")
            lengths = np.arange(1, len(index) + 1)
            lengths = lengths[lengths >= min_length]
        else:
            min_length = dt.to_freq(min_length)
            if freq is None:
                raise ValueError("Must provide freq")
            if min_length < freq or min_length > index[-1] + freq - index[0]:
                raise ValueError(
                    f"Minimum length must be within [{freq}, {index[-1] + freq - index[0]}]"
                )
            lengths = index[1:].append(index[[-1]] + freq) - index[0]
            lengths = lengths[lengths >= min_length]
        if n > len(lengths):
            n = len(lengths)
        rows = np.round(np.linspace(0, len(lengths) - 1, n)).astype(int)
        lengths = lengths[rows]

        splits = []
        for length in lengths:
            new_split = RelRange(length=length).to_slice(len(index), index=index, freq=freq)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)
        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_ranges(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from ranges.

        Uses `vectorbtpro.base.indexing.get_index_ranges` to generate start and end indices
        for splitting the index. Keyword arguments relevant to index range generation are
        extracted from `**kwargs`, while the remaining ones are passed to `Splitter.from_splits`.

        Args:
            index (IndexLike): Index to be divided into ranges.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments distributed between `vectorbtpro.base.indexing.get_index_ranges`
                and `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Translate each quarter into a range:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_ranges(index, every="QS")
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_ranges_1.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_ranges_1.dark.svg#only-dark){: .iimg loading=lazy }

            In addition to the above, reserve the last month for testing purposes:

            ```pycon
            >>> splitter = vbt.Splitter.from_ranges(
            ...     index,
            ...     every="QS",
            ...     split=(1.0, lambda index: index.month == index.month[-1]),
            ...     split_range_kwargs=dict(backwards=True)
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_ranges_2.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_ranges_2.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        func_arg_names = get_func_arg_names(get_index_ranges)
        ranges_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in func_arg_names:
                ranges_kwargs[k] = kwargs.pop(k)

        start_idxs, stop_idxs = get_index_ranges(index, skip_not_found=True, **ranges_kwargs)
        splits = []
        for start, stop in zip(start_idxs, stop_idxs):
            new_split = slice(start, stop)
            if split is not None:
                try:
                    new_split = cls.split_range(
                        new_split,
                        split,
                        template_context=template_context,
                        index=index,
                        **split_range_kwargs,
                    )
                except ZeroLengthError:
                    continue
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_grouper(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        by: tp.AnyGroupByLike,
        groupby_kwargs: tp.KwargsLike = None,
        grouper_kwargs: tp.KwargsLike = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from a grouper.

        Uses `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper` to group the index and generate splits.
        Each group's indices may be adjusted using the provided `split` specification before being passed to
        `Splitter.from_splits` to build the instance.

        Args:
            index (IndexLike): Index to be grouped and split.
            by (AnyGroupByLike): Grouper-like specification.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            groupby_kwargs (KwargsLike): Keyword arguments for `pandas.Series.groupby` and
                `pandas.Series.resample` methods.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            grouper_kwargs (KwargsLike): Keyword arguments for constructing the grouper.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            template_context (KwargsLike): Additional context for template substitution.
            split_labels (Optional[IndexLike]): Labels for the splits.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Map each month into a range:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> def is_month_end(index, split):
            ...     last_range = split[-1]
            ...     return index[last_range][-1].is_month_end

            >>> splitter = vbt.Splitter.from_grouper(
            ...     index,
            ...     "M",
            ...     split_check_template=vbt.RepFunc(is_month_end)
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_grouper.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_grouper.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq
        if grouper_kwargs is None:
            grouper_kwargs = {}

        if isinstance(by, CustomTemplate):
            _template_context = merge_dicts(dict(index=index), template_context)
            by = substitute_templates(by, _template_context, eval_id="by")
        grouper = BaseIDXAccessor(index).get_grouper(
            by, groupby_kwargs=groupby_kwargs, **grouper_kwargs
        )
        splits = []
        indices = []
        for i, new_split in enumerate(grouper.iter_group_idxs()):
            if split is not None:
                try:
                    new_split = cls.split_range(
                        new_split,
                        split,
                        template_context=template_context,
                        index=index,
                        **split_range_kwargs,
                    )
                except ZeroLengthError:
                    continue
            else:
                new_split = [new_split]
            splits.append(new_split)
            indices.append(i)

        if split_labels is None:
            split_labels = grouper.get_index()[indices]
        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            split_labels=split_labels,
            **kwargs,
        )

    @classmethod
    def from_n_random(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        min_length: tp.Union[int, float, tp.TimedeltaLike],
        max_length: tp.Union[None, int, float, tp.TimedeltaLike] = None,
        min_start: tp.Union[None, int, float, tp.DatetimeLike] = None,
        max_end: tp.Union[None, int, float, tp.DatetimeLike] = None,
        length_choice_func: tp.Optional[tp.Callable] = None,
        start_choice_func: tp.Optional[tp.Callable] = None,
        length_p_func: tp.Optional[tp.Callable] = None,
        start_p_func: tp.Optional[tp.Callable] = None,
        seed: tp.Optional[int] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance with randomly generated ranges.

        Generate random ranges by selecting a range length and a start position.
        The range length is chosen between `min_length` and `max_length` (inclusive)
        using `length_choice_func`, which selects one value from the candidate lengths.
        Optionally, `length_p_func` returns probability weights for the length selection.

        The start position is selected from positions between `min_start` and `max_end`
        (adjusted to accommodate the chosen range length) using `start_choice_func`.
        Optionally, `start_p_func` returns probability weights for the start selection.

        Args:
            index (IndexLike): Index from which ranges are generated.
            n (int): Number of random ranges to generate.
            min_length (Union[int, float, TimedeltaLike]): Minimum length for each range.
            max_length (Union[None, int, float, TimedeltaLike]): Maximum length for each range.
                If not provided, it defaults to the same value as `min_length`.
            min_start (Union[None, int, float, DatetimeLike]): Minimum allowable start position for a range.
            max_end (Union[None, int, float, DatetimeLike]): Maximum allowable end position for a range.
            length_choice_func (Optional[Callable]): Function to select a range length
                from candidate values.

                It should accept the iteration index and an array of candidate lengths.
            start_choice_func (Optional[Callable]): Function to select a start position
                from candidate values.

                It should accept the iteration index and an array of candidate start positions.
            length_p_func (Optional[Callable]): Function that returns probability weights
                for the length selection.

                It should accept the iteration index and candidate lengths.
            start_p_func (Optional[Callable]): Function that returns probability weights
                for the start selection.

                It should accept the iteration index and candidate start positions.
            seed (Optional[int]): Random seed for deterministic output.
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            template_context (KwargsLike): Additional context for template substitution.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        !!! note
            Both choice functions must accept two arguments: the iteration index and the array of
            possible values.

        Examples:
            Generate 20 random ranges with a length from [40, 100], and split each into 3/4:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_random(
            ...     index,
            ...     20,
            ...     min_length=40,
            ...     max_length=100,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_n_random.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_n_random.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq

        if min_start is None:
            min_start = 0
        if min_start is not None:
            if checks.is_float(min_start):
                if 0 <= abs(min_start) <= 1:
                    min_start = len(index) * min_start
                elif not min_start.is_integer():
                    raise ValueError("Floating number for minimum start must be between 0 and 1")
            if checks.is_float(min_start):
                min_start = int(min_start)
            if checks.is_int(min_start):
                if min_start < 0 or min_start > len(index) - 1:
                    raise ValueError(f"Minimum start must be within [{0}, {len(index) - 1}]")
            else:
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(f"Index must be of type DatetimeIndex, not {index.dtype}")
                min_start = dt.try_align_dt_to_index(min_start, index)
                if not isinstance(min_start, pd.Timestamp):
                    raise ValueError(f"Minimum start ({min_start}) could not be parsed")
                if min_start < index[0] or min_start > index[-1]:
                    raise ValueError(f"Minimum start must be within [{index[0]}, {index[-1]}]")
                min_start = index.get_indexer([min_start], method="bfill")[0]
        if max_end is None:
            max_end = len(index)
        if checks.is_float(max_end):
            if 0 <= abs(max_end) <= 1:
                max_end = len(index) * max_end
            elif not max_end.is_integer():
                raise ValueError("Floating number for maximum end must be between 0 and 1")
        if checks.is_float(max_end):
            max_end = int(max_end)
        if checks.is_int(max_end):
            if max_end < 1 or max_end > len(index):
                raise ValueError(f"Maximum end must be within [{1}, {len(index)}]")
        else:
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError(f"Index must be of type DatetimeIndex, not {index.dtype}")
            max_end = dt.try_align_dt_to_index(max_end, index)
            if not isinstance(max_end, pd.Timestamp):
                raise ValueError(f"Maximum end ({max_end}) could not be parsed")
            if freq is None:
                raise ValueError("Must provide freq")
            if max_end < index[0] + freq or max_end > index[-1] + freq:
                raise ValueError(
                    f"Maximum end must be within [{index[0] + freq}, {index[-1] + freq}]"
                )
            if max_end > index[-1]:
                max_end = len(index)
            else:
                max_end = index.get_indexer([max_end], method="bfill")[0]
        space_len = max_end - min_start
        if not checks.is_number(min_length):
            index_min_start = index[min_start]
            if max_end < len(index):
                index_max_end = index[max_end]
            else:
                if freq is None:
                    raise ValueError("Must provide freq")
                index_max_end = index[-1] + freq
            index_space_len = index_max_end - index_min_start
        else:
            index_min_start = None
            index_max_end = None
            index_space_len = None

        if checks.is_float(min_length):
            if 0 <= abs(min_length) <= 1:
                min_length = space_len * min_length
            elif not min_length.is_integer():
                raise ValueError("Floating number for minimum length must be between 0 and 1")
            min_length = int(min_length)
        if checks.is_int(min_length):
            if min_length < 1 or min_length > space_len:
                raise ValueError(f"Minimum length must be within [{1}, {space_len}]")
        else:
            min_length = dt.to_freq(min_length)
            if freq is None:
                raise ValueError("Must provide freq")
            if min_length < freq or min_length > index_space_len:
                raise ValueError(f"Minimum length must be within [{freq}, {index_space_len}]")
        if max_length is not None:
            if checks.is_float(max_length):
                if 0 <= abs(max_length) <= 1:
                    max_length = space_len * max_length
                elif not max_length.is_integer():
                    raise ValueError("Floating number for maximum length must be between 0 and 1")
                max_length = int(max_length)
            if checks.is_int(max_length):
                if max_length < min_length or max_length > space_len:
                    raise ValueError(f"Maximum length must be within [{min_length}, {space_len}]")
            else:
                max_length = dt.to_freq(max_length)
                if freq is None:
                    raise ValueError("Must provide freq")
                if max_length < min_length or max_length > index_space_len:
                    raise ValueError(
                        f"Maximum length must be within [{min_length}, {index_space_len}]"
                    )
        else:
            max_length = min_length

        rng = np.random.default_rng(seed=seed)
        if length_p_func is None:
            length_p_func = lambda i, x: None
        if start_p_func is None:
            start_p_func = lambda i, x: None
        if length_choice_func is None:
            length_choice_func = lambda i, x: rng.choice(x, p=length_p_func(i, x))
        else:
            if seed is not None:
                np.random.seed(seed)
        if start_choice_func is None:
            start_choice_func = lambda i, x: rng.choice(x, p=start_p_func(i, x))
        else:
            if seed is not None:
                np.random.seed(seed)
        if checks.is_int(min_length):
            length_space = np.arange(min_length, max_length + 1)
        else:
            if freq is None:
                raise ValueError("Must provide freq")
            length_space = np.arange(min_length // freq, max_length // freq + 1) * freq
        index_space = np.arange(len(index))

        splits = []
        for i in range(n):
            length = length_choice_func(i, length_space)
            if checks.is_int(length):
                start = start_choice_func(i, index_space[min_start : max_end - length + 1])
            else:
                from_dt = index_min_start.to_datetime64()
                to_dt = index_max_end.to_datetime64() - length
                start = start_choice_func(
                    i, index_space[(index.values >= from_dt) & (index.values <= to_dt)]
                )
            new_split = RelRange(offset=start, length=length).to_slice(
                len(index), index=index, freq=freq
            )
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_sklearn(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        skl_splitter: BaseCrossValidatorT,
        groups: tp.Optional[tp.ArrayLike] = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance using a scikit-learn cross-validator.

        Args:
            index (IndexLike): Index representing the dataset.
            skl_splitter (BaseCrossValidator): Scikit-learn splitter instance.
            groups (Optional[ArrayLike]): Group labels for the splitting process.
            split_labels (Optional[IndexLike]): Labels for the splits.
            set_labels (Optional[IndexLike]): Labels for the training and testing sets.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.
        """
        from sklearn.model_selection import BaseCrossValidator

        index = dt.prepare_dt_index(index)
        checks.assert_instance_of(skl_splitter, BaseCrossValidator)
        if set_labels is None:
            set_labels = ["train", "test"]

        indices_generator = skl_splitter.split(np.arange(len(index))[:, None], groups=groups)
        return cls.from_splits(
            index,
            list(indices_generator),
            split_labels=split_labels,
            set_labels=set_labels,
            **kwargs,
        )

    @classmethod
    def from_purged(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        purged_splitter: BasePurgedCV,
        pred_times: tp.Union[None, tp.Index, tp.Series] = None,
        eval_times: tp.Union[None, tp.Index, tp.Series] = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance using a purged cross-validator.

        Args:
            index (IndexLike): Index representing the dataset.
            purged_splitter (BasePurgedCV): Purged cross-validation splitter instance
                from `vectorbtpro.generic.splitting.purged`.
            pred_times (Union[None, Index, Series]): Indices for prediction times.

                If None, the index of `X` is used.
            eval_times (Union[None, Index, Series]): Indices for evaluation times.

                If None, the index of `X` is used.
            split_labels (Optional[IndexLike]): Labels for the splits.
            set_labels (Optional[IndexLike]): Labels for the training and testing sets.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.
        """
        index = dt.prepare_dt_index(index)
        checks.assert_instance_of(purged_splitter, BasePurgedCV)
        if set_labels is None:
            set_labels = ["train", "test"]

        indices_generator = purged_splitter.split(
            pd.Series(np.arange(len(index)), index=index),
            pred_times=pred_times,
            eval_times=eval_times,
        )
        return cls.from_splits(
            index,
            list(indices_generator),
            split_labels=split_labels,
            set_labels=set_labels,
            **kwargs,
        )

    @classmethod
    def from_purged_walkforward(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n_folds: int = 10,
        n_test_folds: int = 1,
        min_train_folds: int = 2,
        max_train_folds: tp.Optional[int] = None,
        split_by_time: bool = False,
        purge_td: tp.TimedeltaLike = 0,
        pred_times: tp.Union[None, tp.Index, tp.Series] = None,
        eval_times: tp.Union[None, tp.Index, tp.Series] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance using a purged walk-forward cross-validator.

        Args:
            index (IndexLike): Index representing the dataset.
            n_folds (int): Total number of folds.
            n_test_folds (int): Total number of folds allocated for testing.
            min_train_folds (int): Minimum number of consecutive folds to use
                for training preceding the test set.
            max_train_folds (Optional[int]): Maximum number of consecutive folds to use
                for training preceding the test set.
            split_by_time (bool): Whether to partition folds based on equal time intervals using prediction times.
            purge_td (TimedeltaLike): Time delta used for purging between folds.
            pred_times (Union[None, Index, Series]): Indices for prediction times.

                If None, the index of `X` is used.
            eval_times (Union[None, Index, Series]): Indices for evaluation times.

                If None, the index of `X` is used.
            **kwargs: Keyword arguments for `Splitter.from_purged`.

        Returns:
            Splitter: New `Splitter` instance.
        """
        index = dt.prepare_dt_index(index)
        purged_splitter = PurgedWalkForwardCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            min_train_folds=min_train_folds,
            max_train_folds=max_train_folds,
            split_by_time=split_by_time,
            purge_td=purge_td,
        )
        return cls.from_purged(
            index,
            purged_splitter,
            pred_times=pred_times,
            eval_times=eval_times,
            **kwargs,
        )

    @classmethod
    def from_purged_kfold(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n_folds: int = 10,
        n_test_folds: int = 2,
        purge_td: tp.TimedeltaLike = 0,
        embargo_td: tp.TimedeltaLike = 0,
        pred_times: tp.Union[None, tp.Index, tp.Series] = None,
        eval_times: tp.Union[None, tp.Index, tp.Series] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance using a purged K-fold cross-validator.

        Args:
            index (IndexLike): Index representing the dataset.
            n_folds (int): Total number of folds.
            n_test_folds (int): Total number of folds allocated for testing.
            purge_td (TimedeltaLike): Time delta used for purging between splits.
            embargo_td (TimedeltaLike): Time interval defining the embargo period between
                test set evaluation times and training predictions.
            pred_times (Union[None, Index, Series]): Indices for prediction times.

                If None, the index of `X` is used.
            eval_times (Union[None, Index, Series]): Indices for evaluation times.

                If None, the index of `X` is used.
            **kwargs: Keyword arguments for `Splitter.from_purged`.

        Returns:
            Splitter: New `Splitter` instance.
        """
        index = dt.prepare_dt_index(index)
        purged_splitter = PurgedKFoldCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            purge_td=purge_td,
            embargo_td=embargo_td,
        )
        return cls.from_purged(
            index,
            purged_splitter,
            pred_times=pred_times,
            eval_times=eval_times,
            **kwargs,
        )

    @classmethod
    def from_split_func(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        split_func: tp.Callable,
        split_args: tp.ArgsLike = None,
        split_kwargs: tp.KwargsLike = None,
        fix_ranges: bool = True,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a `Splitter` instance from a custom split function.

        This method repeatedly calls `split_func` with substituted templates in `split_args`
        and `split_kwargs`. The function should return a split or a single range (if not iterable)
        or None to terminate the loop. When `fix_ranges` is True or if `split` is provided,
        the returned split is processed using `Splitter.split_range` and its bounds are
        determined via `Splitter.get_range_bounds`.

        Template substitutions have access to the following:

        * `split_idx`: Current split index, starting at 0.
        * `splits`: Nested list of splits generated so far.
        * `bounds`: Nested list of bounds generated so far.
        * `prev_start`: Left bound of the previous split.
        * `prev_end`: Right bound of the previous split.
        * All arguments for `Splitter.from_split_func`.

        Args:
            index (IndexLike): Index used for splitting.
            split_func (Callable): Function that returns a new split based on substituted arguments.
            split_args (ArgsLike): Positional arguments for `split_func`.
            split_kwargs (KwargsLike): Keyword arguments for `split_func`.
            fix_ranges (bool): Whether to convert relative ranges (`RelRange`) into fixed (`FixRange`).
            split (Optional[SplitLike]): Specification for further splitting of each range.

                If None, the entire range is treated as a single split;
                otherwise, `Splitter.split_range` is used.
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            range_bounds_kwargs (KwargsLike): Keyword arguments for getting range bounds.

                See `Splitter.get_range_bounds`.
            template_context (KwargsLike): Additional context for template substitution.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            **kwargs: Keyword arguments for `Splitter.from_splits`.

        Returns:
            Splitter: New `Splitter` instance.

        Examples:
            Rolling window of 30 elements, 20 for train and 10 for test:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> def split_func(splits, bounds, index):
            ...     if len(splits) == 0:
            ...         new_split = (slice(0, 20), slice(20, 30))
            ...     else:
            ...         # Previous split, first set, right bound
            ...         prev_end = bounds[-1][0][1]
            ...         new_split = (
            ...             slice(prev_end, prev_end + 20),
            ...             slice(prev_end + 20, prev_end + 30)
            ...         )
            ...     if new_split[-1].stop > len(index):
            ...         return None
            ...     return new_split

            >>> splitter = vbt.Splitter.from_split_func(
            ...     index,
            ...     split_func,
            ...     split_args=(
            ...         vbt.Rep("splits"),
            ...         vbt.Rep("bounds"),
            ...         vbt.Rep("index"),
            ...     ),
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/from_split_func.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_split_func.dark.svg#only-dark){: .iimg loading=lazy }
        """
        index = dt.prepare_dt_index(index)
        freq = BaseIDXAccessor(index, freq=freq).get_freq(allow_numeric=False)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if "freq" not in split_range_kwargs:
            split_range_kwargs = dict(split_range_kwargs)
            split_range_kwargs["freq"] = freq
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}
        if split_args is None:
            split_args = ()
        if split_kwargs is None:
            split_kwargs = {}

        splits = []
        bounds = []
        split_idx = 0
        n_sets = None
        while True:
            _template_context = merge_dicts(
                dict(
                    split_idx=split_idx,
                    splits=splits,
                    bounds=bounds,
                    prev_start=bounds[-1][0][0] if len(bounds) > 0 else None,
                    prev_end=bounds[-1][-1][1] if len(bounds) > 0 else None,
                    index=index,
                    freq=freq,
                    fix_ranges=fix_ranges,
                    split_args=split_args,
                    split_kwargs=split_kwargs,
                    split_range_kwargs=split_range_kwargs,
                    range_bounds_kwargs=range_bounds_kwargs,
                    **kwargs,
                ),
                template_context,
            )
            _split_func = substitute_templates(split_func, _template_context, eval_id="split_func")
            _split_args = substitute_templates(split_args, _template_context, eval_id="split_args")
            _split_kwargs = substitute_templates(
                split_kwargs, _template_context, eval_id="split_kwargs"
            )
            new_split = _split_func(*_split_args, **_split_kwargs)
            if new_split is None:
                break
            if not checks.is_iterable(new_split):
                new_split = (new_split,)
            if fix_ranges or split is not None:
                new_split = cls.split_range(
                    slice(None),
                    new_split,
                    template_context=_template_context,
                    index=index,
                    **split_range_kwargs,
                )
            if split is not None:
                if len(new_split) > 1:
                    raise ValueError(
                        "Split function must return only one range if split is already provided"
                    )
                new_split = cls.split_range(
                    new_split[0],
                    split,
                    template_context=_template_context,
                    index=index,
                    **split_range_kwargs,
                )
            if n_sets is None:
                n_sets = len(new_split)
            elif n_sets != len(new_split):
                raise ValueError("All splits must have the same number of sets")
            splits.append(new_split)
            if fix_ranges:
                split_bounds = tuple(
                    map(
                        lambda x: cls.get_range_bounds(
                            x,
                            template_context=_template_context,
                            index=index,
                            **range_bounds_kwargs,
                        ),
                        new_split,
                    )
                )
                bounds.append(split_bounds)
            split_idx += 1

        return cls.from_splits(
            index,
            splits,
            fix_ranges=fix_ranges,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def guess_method(cls, **kwargs) -> tp.Optional[str]:
        """Guess the appropriate factory method based on provided keyword arguments.

        This method inspects the keyword arguments and compares them against the required and optional
        arguments of factory methods (i.e., methods starting with `from_`) defined in the class. If multiple
        methods match, it selects the one with the fewest combined required and optional arguments,
        preferring `from_n_rolling` when available. Returns None if no suitable method is found.

        Args:
            **kwargs: Keyword arguments used to determine the factory method.

        Returns:
            Optional[str]: Name of the factory method if a unique match is found; otherwise, None.
        """
        if len(kwargs) == 0:
            return None
        keys = {"index"} | set(kwargs.keys())
        from_splits_arg_names = set(get_func_arg_names(cls.from_splits))
        from_splits_arg_names.remove("splits")
        matched_methods = []
        n_args = []
        for k in cls.__dict__:
            if k.startswith("from_") and inspect.ismethod(getattr(cls, k)):
                req_func_arg_names = set(get_func_arg_names(getattr(cls, k), req_only=True))
                if len(req_func_arg_names) > 0:
                    if not (req_func_arg_names <= keys):
                        continue
                opt_func_arg_names = set(get_func_arg_names(getattr(cls, k), opt_only=True))
                func_arg_names = from_splits_arg_names | req_func_arg_names | opt_func_arg_names
                if k == "from_ranges":
                    func_arg_names |= set(get_func_arg_names(get_index_ranges))
                if len(func_arg_names) > 0:
                    if not (keys <= func_arg_names):
                        continue
                matched_methods.append(k)
                n_args.append(len(req_func_arg_names) + len(opt_func_arg_names))
        if len(matched_methods) > 1:
            if "from_n_rolling" in matched_methods:
                return "from_n_rolling"
            return sorted(zip(matched_methods, n_args), key=lambda x: x[1])[0][0]
        if len(matched_methods) == 1:
            return matched_methods[0]
        return None

    @classmethod
    def split_and_take(
        cls,
        index: tp.IndexLike,
        obj: tp.Any,
        splitter: tp.Union[None, str, SplitterT, tp.Callable] = None,
        splitter_kwargs: tp.KwargsLike = None,
        take_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        _splitter_kwargs: tp.KwargsLike = None,
        _take_kwargs: tp.KwargsLike = None,
        **var_kwargs,
    ) -> tp.Any:
        """Split an index and take values from an object.

        Args:
            index (IndexLike): Index to be split.
            obj (Any): Object from which values are extracted.
            splitter (Union[None, str, Splitter, Callable]): Splitter instance, the name of a factory method
                (e.g. "from_n_rolling"), or the factory method itself.

                If None, the appropriate splitter is determined using `Splitter.guess_method`.
            splitter_kwargs (KwargsLike): Keyword arguments for `Splitter`.
            take_kwargs (KwargsLike): Keyword arguments for `Splitter.take`.
            template_context (KwargsLike): Additional context for template substitution.
            **var_kwargs: Keyword arguments to be distributed between `splitter_kwargs` and `take_kwargs`.

        Returns:
            Any: Result returned by `Splitter.take`.
        """
        if splitter_kwargs is None:
            splitter_kwargs = {}
        else:
            splitter_kwargs = dict(splitter_kwargs)
        if take_kwargs is None:
            take_kwargs = {}
        else:
            take_kwargs = dict(take_kwargs)
        if _splitter_kwargs is None:
            _splitter_kwargs = {}
        if _take_kwargs is None:
            _take_kwargs = {}

        if len(var_kwargs) > 0:
            var_splitter_kwargs = {}
            var_take_kwargs = {}
            if splitter is None or not isinstance(splitter, cls):
                take_arg_names = get_func_arg_names(cls.take)
                if splitter is not None:
                    if isinstance(splitter, str):
                        splitter_arg_names = get_func_arg_names(getattr(cls, splitter))
                    else:
                        splitter_arg_names = get_func_arg_names(splitter)
                    for k, v in var_kwargs.items():
                        assigned = False
                        if k in splitter_arg_names:
                            var_splitter_kwargs[k] = v
                            assigned = True
                        if k != "split" and k in take_arg_names:
                            var_take_kwargs[k] = v
                            assigned = True
                        if not assigned:
                            raise ValueError(f"Argument '{k}' couldn't be assigned")
                else:
                    for k, v in var_kwargs.items():
                        if k == "freq":
                            var_splitter_kwargs[k] = v
                            var_take_kwargs[k] = v
                        elif k == "split" or k not in take_arg_names:
                            var_splitter_kwargs[k] = v
                        else:
                            var_take_kwargs[k] = v
            else:
                var_take_kwargs = var_kwargs
            splitter_kwargs = merge_dicts(var_splitter_kwargs, splitter_kwargs)
            take_kwargs = merge_dicts(var_take_kwargs, take_kwargs)
        if len(splitter_kwargs) > 0:
            if splitter is None:
                splitter = cls.guess_method(**splitter_kwargs)
            if splitter is None:
                raise ValueError("Splitter method couldn't be guessed")
        else:
            if splitter is None:
                raise ValueError("Must provide splitter or splitter method")
        if not isinstance(splitter, cls):
            if isinstance(splitter, str):
                splitter = getattr(cls, splitter)
            for k, v in _splitter_kwargs.items():
                if k not in splitter_kwargs:
                    splitter_kwargs[k] = v
            splitter = splitter(index, template_context=template_context, **splitter_kwargs)
        for k, v in _take_kwargs.items():
            if k not in take_kwargs:
                take_kwargs[k] = v
        return splitter.take(obj, template_context=template_context, **take_kwargs)

    @classmethod
    def split_and_apply(
        cls,
        index: tp.IndexLike,
        apply_func: tp.Callable,
        *apply_args,
        splitter: tp.Union[None, str, SplitterT, tp.Callable] = None,
        splitter_kwargs: tp.KwargsLike = None,
        apply_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        _splitter_kwargs: tp.KwargsLike = None,
        _apply_kwargs: tp.KwargsLike = None,
        **var_kwargs,
    ) -> tp.Any:
        """Split an index and apply a function to each segment.

        Args:
            index (IndexLike): Index to be split.
            apply_func (Callable): Function to apply to each split segment.
            *apply_args: Positional arguments for `Splitter.apply`.
            splitter (Union[None, str, Splitter, Callable]): Splitter instance, the name of a factory method
                (e.g. "from_n_rolling"), or the factory method itself.

                If None, the appropriate splitter is determined using `Splitter.guess_method`.
            splitter_kwargs (KwargsLike): Keyword arguments for `Splitter`.
            apply_kwargs (KwargsLike): Keyword arguments for `Splitter.apply`.
            template_context (KwargsLike): Additional context for template substitution.
            **var_kwargs: Keyword arguments to be distributed between `splitter_kwargs` and `apply_kwargs`.

        Returns:
            Any: Result returned by `Splitter.apply`.
        """
        if splitter_kwargs is None:
            splitter_kwargs = {}
        else:
            splitter_kwargs = dict(splitter_kwargs)
        if apply_kwargs is None:
            apply_kwargs = {}
        else:
            apply_kwargs = dict(apply_kwargs)
        if _splitter_kwargs is None:
            _splitter_kwargs = {}
        if _apply_kwargs is None:
            _apply_kwargs = {}

        if len(var_kwargs) > 0:
            var_splitter_kwargs = {}
            var_apply_kwargs = {}
            if splitter is None or not isinstance(splitter, cls):
                apply_arg_names = get_func_arg_names(cls.apply)
                if splitter is not None:
                    if isinstance(splitter, str):
                        splitter_arg_names = get_func_arg_names(getattr(cls, splitter))
                    else:
                        splitter_arg_names = get_func_arg_names(splitter)
                    for k, v in var_kwargs.items():
                        assigned = False
                        if k in splitter_arg_names:
                            var_splitter_kwargs[k] = v
                            assigned = True
                        if k != "split" and k in apply_arg_names:
                            var_apply_kwargs[k] = v
                            assigned = True
                        if not assigned:
                            raise ValueError(f"Argument '{k}' couldn't be assigned")
                else:
                    for k, v in var_kwargs.items():
                        if k == "freq":
                            var_splitter_kwargs[k] = v
                            var_apply_kwargs[k] = v
                        elif k == "split" or k not in apply_arg_names:
                            var_splitter_kwargs[k] = v
                        else:
                            var_apply_kwargs[k] = v
            else:
                var_apply_kwargs = var_kwargs
            splitter_kwargs = merge_dicts(var_splitter_kwargs, splitter_kwargs)
            apply_kwargs = merge_dicts(var_apply_kwargs, apply_kwargs)
        if len(splitter_kwargs) > 0:
            if splitter is None:
                splitter = cls.guess_method(**splitter_kwargs)
            if splitter is None:
                raise ValueError("Splitter method couldn't be guessed")
        else:
            if splitter is None:
                raise ValueError("Must provide splitter or splitter method")
        if not isinstance(splitter, cls):
            if isinstance(splitter, str):
                splitter = getattr(cls, splitter)
            for k, v in _splitter_kwargs.items():
                if k not in splitter_kwargs:
                    splitter_kwargs[k] = v
            splitter = splitter(index, template_context=template_context, **splitter_kwargs)
        for k, v in _apply_kwargs.items():
            if k not in apply_kwargs:
                apply_kwargs[k] = v
        return splitter.apply(
            apply_func, *apply_args, template_context=template_context, **apply_kwargs
        )

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeSequence[SplitterT],
        **kwargs,
    ) -> tp.Kwargs:
        if "splits_arr" not in kwargs:
            kwargs["splits_arr"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.splits for obj in objs],
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeSequence[SplitterT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing a `Splitter` after stacking splits along columns.

        Args:
            *objs (MaybeSequence[Splitter]): Splitter instances whose `splits` arrays are to be stacked.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            **kwargs: Additional keyword arguments.

        Returns:
            Kwargs: Updated keyword arguments including a `splits_arr` key.
        """
        if "splits_arr" not in kwargs:
            kwargs["splits_arr"] = kwargs["wrapper"].column_stack_arrs(
                *[obj.splits for obj in objs],
                reindex_kwargs=reindex_kwargs,
                group_by=False,
                wrap=False,
            )
        return kwargs

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[SplitterT],
        *objs: tp.MaybeSequence[SplitterT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Stack multiple `Splitter` instances along rows.

        Stack multiple `Splitter` instances by stacking their wrappers along rows using
        `vectorbtpro.base.wrapping.ArrayWrapper.row_stack`.

        Args:
            *objs (MaybeSequence[Splitter]): (Additional) `Splitter` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Splitter` through
                `Splitter.resolve_row_stack_kwargs` and `Splitter.resolve_stack_kwargs`.

        Returns:
            Splitter: New `Splitter` instance with row-stacked wrappers.
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
            if not checks.is_instance_of(obj, Splitter):
                raise TypeError("Each object to be merged must be an instance of Splitter")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs],
                stack_columns=False,
                **wrapper_kwargs,
            )

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[SplitterT],
        *objs: tp.MaybeSequence[SplitterT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Stack multiple `Splitter` instances along columns.

        Stack multiple `Splitter` instances by stacking their wrappers along columns using
        `vectorbtpro.base.wrapping.ArrayWrapper.column_stack`.

        Args:
            *objs (MaybeSequence[Splitter]): (Additional) `Splitter` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Splitter` through
                `Splitter.resolve_column_stack_kwargs` and `Splitter.resolve_stack_kwargs`.

        Returns:
            Splitter: New `Splitter` instance with column-stacked wrappers.
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
            if not checks.is_instance_of(obj, Splitter):
                raise TypeError("Each object to be merged must be an instance of Splitter")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                union_index=False,
                **wrapper_kwargs,
            )

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on a `Splitter` instance and return metadata.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.

        Returns:
            dict: Dictionary with keys `wrapper_meta` and `new_splits_arr` representing
                the updated metadata and splits array.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        if wrapper_meta["rows_changed"] or wrapper_meta["columns_changed"]:
            new_splits_arr = ArrayWrapper.select_from_flex_array(
                self.splits_arr,
                row_idxs=wrapper_meta["row_idxs"],
                col_idxs=wrapper_meta["col_idxs"],
                rows_changed=wrapper_meta["rows_changed"],
                columns_changed=wrapper_meta["columns_changed"],
            )
        else:
            new_splits_arr = self.splits_arr
        return dict(
            wrapper_meta=wrapper_meta,
            new_splits_arr=new_splits_arr,
        )

    def indexing_func(
        self: SplitterT, *args, splitter_meta: tp.DictLike = None, **kwargs
    ) -> SplitterT:
        """Perform indexing on a `Splitter` instance.

        Args:
            *args: Positional arguments for `Splitter.indexing_func_meta`.
            splitter_meta (DictLike): Metadata for splitter indexing.
            **kwargs: Keyword arguments for `Splitter.indexing_func_meta`.

        Returns:
            Splitter: New `Splitter` instance reflecting the indexing operation.
        """
        if splitter_meta is None:
            splitter_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=splitter_meta["wrapper_meta"]["new_wrapper"],
            splits_arr=splitter_meta["new_splits_arr"],
        )

    @property
    def splits(self) -> tp.Frame:
        """Splits array as a DataFrame.

        Returns:
            Frame: DataFrame representing the splits.
        """
        return self.wrapper.wrap(self._splits_arr, group_by=False)

    @property
    def split_labels(self) -> tp.Index:
        """Labels for splits.

        Returns:
            Index: Labels for splits.
        """
        return self.wrapper.index

    @property
    def set_labels(self) -> tp.Index:
        """Labels for sets.

        Returns:
            Index: Labels for sets.
        """
        return self.wrapper.columns

    @property
    def n_splits(self) -> int:
        """Number of splits.

        Returns:
            int: Number of splits.
        """
        return self.splits_arr.shape[0]

    @property
    def n_sets(self) -> int:
        """Number of sets.

        Returns:
            int: Number of sets.
        """
        return self.splits_arr.shape[1]

    def get_split_grouper(self, split_group_by: tp.AnyGroupByLike = None) -> tp.Optional[Grouper]:
        """Return a grouper for splits based on the provided grouping parameter.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            Optional[Grouper]: Grouper for splits if applicable, otherwise None.
        """
        if split_group_by is None:
            return None
        if isinstance(split_group_by, Grouper):
            return split_group_by
        return BaseIDXAccessor(self.split_labels).get_grouper(
            split_group_by, def_lvl_name="split_group"
        )

    def get_set_grouper(self, set_group_by: tp.AnyGroupByLike = None) -> tp.Optional[Grouper]:
        """Return a grouper for sets based on the provided grouping parameter.

        Args:
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            Optional[Grouper]: Grouper for sets if applicable, otherwise None.
        """
        if set_group_by is None:
            return None
        if isinstance(set_group_by, Grouper):
            return set_group_by
        return BaseIDXAccessor(self.set_labels).get_grouper(set_group_by, def_lvl_name="set_group")

    def get_n_splits(self, split_group_by: tp.AnyGroupByLike = None) -> int:
        """Return the number of splits, optionally considering grouping.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            int: Count of splits after applying grouping.
        """
        if split_group_by is not None:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            return split_group_by.get_group_count()
        return self.n_splits

    def get_n_sets(self, set_group_by: tp.AnyGroupByLike = None) -> int:
        """Return the number of sets, optionally considering grouping.

        Args:
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            int: Count of sets after applying grouping.
        """
        if set_group_by is not None:
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            return set_group_by.get_group_count()
        return self.n_sets

    def get_split_labels(self, split_group_by: tp.AnyGroupByLike = None) -> tp.Index:
        """Return split labels, optionally modified by a grouper.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            Index: Split labels, potentially modified by the grouper.
        """
        if split_group_by is not None:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            return split_group_by.get_index()
        return self.split_labels

    def get_set_labels(self, set_group_by: tp.AnyGroupByLike = None) -> tp.Index:
        """Return set labels, optionally modified by a grouper.

        Args:
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            Index: Set labels, potentially modified by the grouper.
        """
        if set_group_by is not None:
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            return set_group_by.get_index()
        return self.set_labels

    # ############# Conversion ############# #

    def to_fixed(self: SplitterT, split_range_kwargs: tp.KwargsLike = None, **kwargs) -> SplitterT:
        """Convert relative ranges into fixed ranges and return a new `Splitter` instance.

        Args:
            split_range_kwargs (KwargsLike): Keyword arguments for range splitting.

                See `Splitter.split_range`.
            **kwargs: Keyword arguments for `Splitter.replace`.

        Returns:
            Splitter: New `Splitter` instance with fixed ranges.
        """
        if split_range_kwargs is None:
            split_range_kwargs = {}
        split_range_kwargs = dict(split_range_kwargs)
        wrap_with_fixrange = split_range_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        split_range_kwargs["wrap_with_fixrange"] = wrap_with_fixrange
        new_splits_arr = []
        for split in self.splits_arr:
            new_split = self.split_range(slice(None), split, **split_range_kwargs)
            new_splits_arr.append(new_split)
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)
        return self.replace(splits_arr=new_splits_arr, **kwargs)

    def to_grouped(
        self: SplitterT,
        split: tp.Optional[tp.Selection] = None,
        set_: tp.Optional[tp.Selection] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        merge_split_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Merge ranges within the same group.

        Merge ranges across both dimensions using group indices derived from the provided
        grouping parameters. A new `Splitter` instance is returned with its wrapper's index and
        columns replaced by the corresponding group labels and with a splits array containing
        the merged ranges.

        Args:
            split (Optional[Selection]): Selection criteria for splits.
            set_ (Optional[Selection]): Selection criteria for sets.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            merge_split_kwargs (KwargsLike): Keyword arguments for `Splitter.merge_split`.
            **kwargs: Keyword arguments for `Splitter.replace`.

        Returns:
            Splitter: New `Splitter` instance with merged ranges.
        """
        if merge_split_kwargs is None:
            merge_split_kwargs = {}
        merge_split_kwargs = dict(merge_split_kwargs)
        wrap_with_fixrange = merge_split_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        merge_split_kwargs["wrap_with_fixrange"] = wrap_with_fixrange
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        split_group_indices, set_group_indices, split_indices, set_indices = self.select_indices(
            split=split,
            set_=set_,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        )
        if split is not None:
            split_labels = split_labels[split_group_indices]
        if set_ is not None:
            set_labels = set_labels[set_group_indices]

        new_splits_arr = []
        for i in split_group_indices:
            new_splits_arr.append([])
            for j in set_group_indices:
                new_range = self.select_range(
                    split=PosSel(i),
                    set_=PosSel(j),
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    merge_split_kwargs=merge_split_kwargs,
                )
                new_splits_arr[-1].append(new_range)
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)

        if set_group_by is None or not set_group_by.is_grouped():
            ndim = self.wrapper.ndim
        else:
            ndim = 1 if new_splits_arr.shape[1] == 1 else 2
        wrapper = self.wrapper.replace(index=split_labels, columns=set_labels, ndim=ndim)
        return self.replace(wrapper=wrapper, splits_arr=new_splits_arr, **kwargs)

    # ############# Ranges ############# #

    @classmethod
    def is_range_relative(cls, range_: tp.RangeLike) -> bool:
        """Determine if the provided range is relative.

        A range is considered relative if it is a number, a time delta-like object, or an instance of `RelRange`.

        Args:
            range_ (RangeLike): Range object to evaluate.

        Returns:
            bool: True if the range is relative, otherwise False.
        """
        return checks.is_number(range_) or checks.is_td_like(range_) or isinstance(range_, RelRange)

    @hybrid_method
    def get_ready_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        allow_relative: bool = False,
        allow_zero_len: bool = False,
        range_format: str = "slice_or_any",
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        return_meta: bool = False,
    ) -> tp.Union[tp.RelRangeLike, tp.ReadyRangeLike, dict]:
        """Return a range directly usable for array indexing.

        This function converts an input range into a format suitable for array indexing.
        The converted range can be one of the following:
        a datetime-like or integer slice with an exclusive right bound, a 1D NumPy array of indices,
        or a 1D boolean mask matching the length of the index.

        Args:
            range_ (FixRangeLike): Initial range specification.

                This may be a callable, slice, fixed range, or array.
            allow_relative (bool): Allow relative ranges.

                Relative ranges must be fixed unless this is True.
            allow_zero_len (bool): Permit ranges with zero length.
            range_format (str): Format of the returned range.

                Accepted options are:

                * "any": Return any supported format.
                * "indices": Return integer indices.
                * "mask": Return a boolean mask matching the index length.
                * "slice": Return a slice.
                * "slice_or_indices": Return a slice, or indices if slice conversion fails.
                * "slice_or_mask": Return a slice, or a mask if slice conversion fails.
                * "slice_or_any": Return a slice, or any available format if slice conversion fails.
            template_context (KwargsLike): Additional context for template substitution.
            index (Optional[IndexLike]): Index used for aligning and validating the range.

                If not provided, the index is taken from `Splitter.index`.
            return_meta (bool): Return a metadata dictionary (which includes the converted range) if True.

        Returns:
            Union[RelRangeLike, ReadyRangeLike, dict]: Range converted to the specified format,
                or a metadata dictionary if `return_meta` is True.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        if range_format.lower() not in (
            "any",
            "indices",
            "mask",
            "slice",
            "slice_or_indices",
            "slice_or_mask",
            "slice_or_any",
        ):
            raise ValueError(f"Invalid range_format: '{range_format}'")

        meta = dict()
        meta["was_fixed"] = False
        meta["was_template"] = False
        meta["was_callable"] = False
        meta["was_relative"] = False
        meta["was_hslice"] = False
        meta["was_slice"] = False
        meta["was_neg_slice"] = False
        meta["was_datetime"] = False
        meta["was_mask"] = False
        meta["was_indices"] = False
        meta["is_constant"] = False
        meta["start"] = None
        meta["stop"] = None
        meta["length"] = None
        if isinstance(range_, FixRange):
            meta["was_fixed"] = True
            range_ = range_.range_
        if isinstance(range_, CustomTemplate):
            meta["was_template"] = True
            if template_context is None:
                template_context = {}
            if "index" not in template_context:
                template_context["index"] = index
            range_ = range_.substitute(context=template_context, eval_id="range")
        if callable(range_):
            meta["was_callable"] = True
            range_ = range_(index)
        if cls_or_self.is_range_relative(range_):
            meta["was_relative"] = True
            if allow_relative:
                if return_meta:
                    meta["range_"] = range_
                    return meta
                return range_
            raise TypeError("Relative ranges must be converted to fixed")
        if isinstance(range_, hslice):
            meta["was_hslice"] = True
            range_ = range_.to_slice()
        if isinstance(range_, slice):
            meta["was_slice"] = True
            meta["is_constant"] = True
            start = range_.start
            stop = range_.stop
            if range_.step is not None and range_.step > 1:
                raise ValueError("Step must be either None or 1")
            if start is not None and checks.is_int(start) and start < 0:
                if stop is not None and checks.is_int(stop) and stop > 0:
                    raise ValueError("Slices must be either strictly negative or positive")
                meta["was_neg_slice"] = True
                start = len(index) + start
                if stop is not None and checks.is_int(stop):
                    stop = len(index) + stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(index)
            if not checks.is_int(start):
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(f"Index must be of type DatetimeIndex, not {index.dtype}")
                start = dt.try_align_dt_to_index(start, index)
                if not isinstance(start, pd.Timestamp):
                    raise ValueError(f"Range start ({start}) could not be parsed")
                meta["was_datetime"] = True
            if not checks.is_int(stop):
                if not isinstance(index, pd.DatetimeIndex):
                    raise TypeError(f"Index must be of type DatetimeIndex, not {index.dtype}")
                stop = dt.try_align_dt_to_index(stop, index)
                if not isinstance(stop, pd.Timestamp):
                    raise ValueError(f"Range start ({stop}) could not be parsed")
                meta["was_datetime"] = True
            if checks.is_int(start):
                if start < 0:
                    start = 0
            else:
                if start < index[0]:
                    start = 0
                else:
                    start = index.get_indexer([start], method="bfill")[0]
                    if start == -1:
                        raise ValueError(f"Range start ({start}) is out of bounds")
            if checks.is_int(stop):
                if stop > len(index):
                    stop = len(index)
            else:
                if stop > index[-1]:
                    stop = len(index)
                else:
                    stop = index.get_indexer([stop], method="bfill")[0]
                    if stop == -1:
                        raise ValueError(f"Range stop ({stop}) is out of bounds")
            range_ = slice(start, stop)
            meta["start"] = start
            meta["stop"] = stop
            meta["length"] = stop - start
            if not allow_zero_len and meta["length"] == 0:
                raise ZeroLengthError("Range has zero length")
            if range_format.lower() == "indices":
                range_ = np.arange(*range_.indices(len(index)))
            elif range_format.lower() == "mask":
                mask = np.full(len(index), False)
                mask[range_] = True
                range_ = mask
        else:
            range_ = np.asarray(range_)
            if np.issubdtype(range_.dtype, np.bool_):
                if len(range_) != len(index):
                    raise ValueError("Mask must have the same length as index")
                meta["was_mask"] = True
                indices = np.flatnonzero(range_)
                if len(indices) == 0:
                    if not allow_zero_len:
                        raise ZeroLengthError("Range has zero length")
                    meta["is_constant"] = True
                    meta["start"] = 0
                    meta["stop"] = 0
                    meta["length"] = 0
                else:
                    meta["is_constant"] = is_range(indices)
                    meta["start"] = indices[0]
                    meta["stop"] = indices[-1] + 1
                    meta["length"] = len(indices)
                if range_format.lower() == "indices":
                    range_ = indices
                elif range_format.lower().startswith("slice"):
                    if not meta["is_constant"]:
                        if range_format.lower() == "slice":
                            raise ValueError("Cannot convert to slice: range is not constant")
                        if range_format.lower() == "slice_or_indices":
                            range_ = indices
                    else:
                        range_ = slice(meta["start"], meta["stop"])
            else:
                if not np.issubdtype(range_.dtype, np.integer):
                    range_ = dt.try_align_to_dt_index(range_, index)
                    if not isinstance(range_, pd.DatetimeIndex):
                        raise ValueError("Range array could not be parsed")
                    range_ = index.get_indexer(range_, method=None)
                    if -1 in range_:
                        raise ValueError("Range array has values that cannot be found in index")
                if np.issubdtype(range_.dtype, np.integer):
                    meta["was_indices"] = True
                    if len(range_) == 0:
                        if not allow_zero_len:
                            raise ZeroLengthError("Range has zero length")
                        meta["is_constant"] = True
                        meta["start"] = 0
                        meta["stop"] = 0
                        meta["length"] = 0
                    else:
                        meta["is_constant"] = is_range(range_)
                        if meta["is_constant"]:
                            meta["start"] = range_[0]
                            meta["stop"] = range_[-1] + 1
                        else:
                            meta["start"] = np.min(range_)
                            meta["stop"] = np.max(range_) + 1
                        meta["length"] = len(range_)
                    if range_format.lower() == "mask":
                        mask = np.full(len(index), False)
                        mask[range_] = True
                        range_ = mask
                    elif range_format.lower().startswith("slice"):
                        if not meta["is_constant"]:
                            if range_format.lower() == "slice":
                                raise ValueError("Cannot convert to slice: range is not constant")
                            if range_format.lower() == "slice_or_mask":
                                mask = np.full(len(index), False)
                                mask[range_] = True
                                range_ = mask
                        else:
                            range_ = slice(meta["start"], meta["stop"])
                else:
                    raise TypeError(f"Range array has invalid data type ({range_.dtype})")
        if meta["start"] != meta["stop"]:
            if meta["start"] > meta["stop"]:
                raise ValueError(
                    f"Range start ({meta['start']}) is higher than range stop ({meta['stop']})"
                )
            if meta["start"] < 0 or meta["start"] >= len(index):
                raise ValueError(f"Range start ({meta['start']}) is out of bounds")
            if meta["stop"] < 0 or meta["stop"] > len(index):
                raise ValueError(f"Range stop ({meta['stop']}) is out of bounds")
        if return_meta:
            meta["range_"] = range_
            return meta
        return range_

    @hybrid_method
    def split_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        new_split: tp.SplitLike,
        backwards: bool = False,
        allow_zero_len: bool = False,
        range_format: tp.Optional[str] = None,
        wrap_with_template: bool = False,
        wrap_with_fixrange: tp.Optional[bool] = False,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.FixSplit:
        """Split a fixed range into multiple fixed ranges.

        This method splits an input range into several sub-ranges based on the provided
        `new_split` specification. The input range (`range_`) may be defined as a template,
        callable, tuple (start and stop), slice, sequence of indices, or mask, and it is mapped
        onto the given index.

        Args:
            range_ (FixRangeLike): Input range specified as a template, callable, tuple (start, stop),
                slice, sequence of indices, or mask.

                It is mapped to the provided index.
            new_split (SplitLike): Specification for splitting `range_`.

                Each element can be one of:

                * Fixed or relative range (an instance of `RelRange`).
                * Number representing a length to create a relative range.
                * Integer or float indicating a length specifier, where the complementary
                    part stretches depending on `backwards`.
                * String. If set to 'by_gap' (case-insensitive), `range_` is split by gap using
                    `vectorbtpro.generic.splitting.nb.split_range_by_gap_nb`.
            backwards (bool): Whether to split the range in reverse order.

                When True, the order of the resulting ranges is reversed and length adjustments apply.
            allow_zero_len (bool): Permit ranges with zero length.
            range_format (Optional[str]): Format for the range.

                If not provided, the format is inferred from `range_`. See `Splitter.get_ready_range`.
            wrap_with_template (bool): Whether to wrap the resulting ranges with a template of type
                `vectorbtpro.utils.template.Rep`.
            wrap_with_fixrange (Optional[bool]): If True, wrap the merged range with `FixRange`.

                When None, the type is determined based on sequence checking.
            template_context (KwargsLike): Additional context for template substitution.
            index (Optional[IndexLike]): Index onto which `range_` is mapped.

                Must be provided when called on a class instance.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            FixSplit: Tuple of fixed ranges resulting from splitting `range_` relative to the provided index.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)

        # Prepare source range
        range_meta = cls_or_self.get_ready_range(
            range_,
            allow_zero_len=allow_zero_len,
            range_format="slice_or_indices",
            template_context=template_context,
            index=index,
            return_meta=True,
        )
        range_ = range_meta["range_"]
        range_was_hslice = range_meta["was_hslice"]
        range_was_indices = range_meta["was_indices"]
        range_was_mask = range_meta["was_mask"]
        range_length = range_meta["length"]
        if range_format is None:
            if range_was_indices:
                range_format = "slice_or_indices"
            elif range_was_mask:
                range_format = "slice_or_mask"
            else:
                range_format = "slice_or_any"

        # Substitute template
        if isinstance(new_split, CustomTemplate):
            _template_context = merge_dicts(dict(index=index[range_]), template_context)
            new_split = substitute_templates(new_split, _template_context, eval_id="new_split")

        # Split by gap
        if isinstance(new_split, str) and new_split.lower() == "by_gap":
            if isinstance(range_, np.ndarray) and np.issubdtype(range_.dtype, np.integer):
                range_arr = range_
            else:
                range_arr = np.arange(len(index))[range_]
            start_idxs, stop_idxs = nb.split_range_by_gap_nb(range_arr)
            new_split = list(map(lambda x: slice(x[0], x[1]), zip(start_idxs, stop_idxs)))

        # Prepare target ranges
        if checks.is_number(new_split):
            if new_split < 0:
                backwards = not backwards
                new_split = abs(new_split)
            if not backwards:
                new_split = (new_split, 1.0)
            else:
                new_split = (1.0, new_split)
        elif checks.is_td_like(new_split):
            new_split = dt.to_freq(new_split)
            if new_split < pd.Timedelta(0):
                backwards = not backwards
                new_split = abs(new_split)
            if not backwards:
                new_split = (new_split, 1.0)
            else:
                new_split = (1.0, new_split)
        elif not checks.is_iterable(new_split):
            new_split = (new_split,)

        # Perform split
        new_ranges = []
        if backwards:
            new_split = new_split[::-1]
            prev_start = range_length
            prev_end = range_length
        else:
            prev_start = 0
            prev_end = 0
        for new_range in new_split:
            # Resolve new range
            new_range_meta = cls_or_self.get_ready_range(
                new_range,
                allow_relative=True,
                allow_zero_len=allow_zero_len,
                range_format="slice_or_any",
                template_context=template_context,
                index=index[range_],
                return_meta=True,
            )
            new_range = new_range_meta["range_"]
            if checks.is_number(new_range) or checks.is_td_like(new_range):
                new_range = RelRange(length=new_range)
            if isinstance(new_range, RelRange):
                new_range_is_gap = new_range.is_gap
                new_range = new_range.to_slice(
                    range_length,
                    prev_start=range_length - prev_end if backwards else prev_start,
                    prev_end=range_length - prev_start if backwards else prev_end,
                    index=index,
                    freq=freq,
                )
                if backwards:
                    new_range = slice(range_length - new_range.stop, range_length - new_range.start)
            else:
                new_range_is_gap = False

            # Update previous bounds
            if isinstance(new_range, slice):
                prev_start = new_range.start
                prev_end = new_range.stop
            else:
                prev_start = new_range_meta["start"]
                prev_end = new_range_meta["stop"]

            # Remap new range to index
            if new_range_is_gap:
                continue
            if isinstance(range_, slice) and isinstance(new_range, slice):
                new_range = slice(
                    range_.start + new_range.start,
                    range_.start + new_range.stop,
                )
            else:
                if isinstance(range_, slice):
                    new_range = np.arange(range_.start, range_.stop)[new_range]
                else:
                    new_range = range_[new_range]
            new_range = cls_or_self.get_ready_range(
                new_range,
                allow_zero_len=allow_zero_len,
                range_format=range_format,
                index=index,
            )
            if isinstance(new_range, slice) and range_was_hslice:
                new_range = hslice.from_slice(new_range)
            if wrap_with_template:
                new_range = Rep("range_", context=dict(range_=new_range))
            if wrap_with_fixrange is None:
                _wrap_with_fixrange = checks.is_sequence(new_range)
            else:
                _wrap_with_fixrange = False
            if _wrap_with_fixrange:
                new_range = FixRange(new_range)
            new_ranges.append(new_range)

        if backwards:
            return tuple(new_ranges)[::-1]
        return tuple(new_ranges)

    @hybrid_method
    def merge_split(
        cls_or_self,
        split: tp.FixSplit,
        range_format: tp.Optional[str] = None,
        wrap_with_template: bool = False,
        wrap_with_fixrange: tp.Optional[bool] = False,
        wrap_with_hslice: tp.Optional[bool] = False,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.FixRangeLike:
        """Merge multiple fixed ranges from a split into a single fixed range.

        Create a single fixed range by merging individual ranges from the provided split.
        The function constructs a boolean mask marking True for elements within any range.
        If all input ranges are masks, the result is a mask; if all are slices, a slice is
        returned when possible; otherwise, integer indices are returned.

        Args:
            split (FixSplit): Collection of fixed ranges to merge.
            range_format (Optional[str]): Format for the range.

                See `Splitter.get_ready_range`.
            wrap_with_template (bool): Whether to wrap the resulting ranges with a template of type
                `vectorbtpro.utils.template.Rep`.
            wrap_with_fixrange (Optional[bool]): If True, wrap the merged range with `FixRange`.

                When None, the type is determined based on sequence checking.
            wrap_with_hslice (Optional[bool]): If True, and applicable, wrap a slice result with an `hslice`.
            template_context (KwargsLike): Additional context for template substitution.
            index (Optional[IndexLike]): Index used for alignment.

                If not provided, `Splitter.index` is used.

        Returns:
            FixRangeLike: Merged fixed range, represented as a mask, slice,
                or integer indices depending on input types.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        all_hslices = True
        all_masks = True
        new_ranges = []
        if len(split) == 1:
            raise ValueError("Two or more ranges are required to be merged")
        for range_ in split:
            range_meta = cls_or_self.get_ready_range(
                range_,
                allow_zero_len=True,
                range_format="any",
                template_context=template_context,
                index=index,
                return_meta=True,
            )
            if not range_meta["was_hslice"]:
                all_hslices = False
            if not range_meta["was_mask"]:
                all_masks = False
            new_ranges.append(range_meta["range_"])
        ranges = new_ranges
        if range_format is None:
            if all_masks:
                range_format = "slice_or_mask"
            else:
                range_format = "slice_or_indices"

        new_range = np.full(len(index), False)
        for range_ in ranges:
            new_range[range_] = True
        new_range = cls_or_self.get_ready_range(
            new_range,
            range_format=range_format,
            index=index,
        )
        if isinstance(new_range, slice) and all_hslices:
            if wrap_with_hslice is None:
                wrap_with_hslice = True
            if wrap_with_hslice:
                new_range = hslice.from_slice(new_range)
        if wrap_with_template:
            new_range = Rep("range_", context=dict(range_=new_range))
        if wrap_with_fixrange is None:
            _wrap_with_fixrange = checks.is_sequence(new_range)
        else:
            _wrap_with_fixrange = False
        if _wrap_with_fixrange:
            new_range = FixRange(new_range)
        return new_range

    # ############# Taking ############# #

    def select_indices(
        self,
        split: tp.Optional[tp.Selection] = None,
        set_: tp.Optional[tp.Selection] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
        """Retrieve indices corresponding to selected splits and sets.

        Interpret selections for splits and sets, which can be provided as integers, labels,
        or wrapped in `PosSel` or `LabelSel`. Multiple values are allowed, in which case
        the corresponding ranges are merged. When labels are of an integer data type, they are
        treated as labels unless the associated index or grouping indicates positions.

        If `split_group_by` and/or `set_group_by` is provided, grouper objects are created using
        `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper` so that the selections are interpreted
        relative to groups. If `split` or `set_` is not provided, all indices for that category are selected.

        Args:
            split (Optional[Selection]): Selection criteria for splits.
            set_ (Optional[Selection]): Selection criteria for sets.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            Tuple[Array1d, Array1d, Array1d, Array1d]: Tuple containing:

                * Split group indices.
                * Set group indices.
                * Split indices.
                * Set indices.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        if split is None:
            split_group_indices = np.arange(self.get_n_splits(split_group_by=split_group_by))
            split_indices = np.arange(self.n_splits)
        else:
            kind = None
            if isinstance(split, PosSel):
                split = split.value
                kind = "positions"
            elif isinstance(split, LabelSel):
                split = split.value
                kind = "labels"
            if checks.is_hashable(split):
                split = [split]
            if split_group_by is not None:
                split_group_indices = []
                groups, group_index = split_group_by.get_groups_and_index()
                mask = None
                for g in split:
                    if isinstance(g, PosSel):
                        g = g.value
                        kind = "positions"
                    elif isinstance(g, LabelSel):
                        g = g.value
                        kind = "labels"
                    if kind == "positions" or (
                        kind is None
                        and checks.is_int(g)
                        and not pd.api.types.is_integer_dtype(group_index)
                    ):
                        i = g
                    else:
                        i = group_index.get_indexer([g])[0]
                        if i == -1:
                            raise ValueError(f"Split group '{g}' not found")
                    if mask is None:
                        mask = groups == i
                    else:
                        mask |= groups == i
                    split_group_indices.append(i)
                split_group_indices = np.asarray(split_group_indices)
                split_indices = np.arange(self.n_splits)[mask]
            else:
                split_indices = []
                for s in split:
                    if isinstance(s, PosSel):
                        s = s.value
                        kind = "positions"
                    elif isinstance(s, LabelSel):
                        s = s.value
                        kind = "labels"
                    if kind == "positions" or (
                        kind is None
                        and checks.is_int(s)
                        and not pd.api.types.is_integer_dtype(self.split_labels)
                    ):
                        i = s
                    else:
                        i = self.split_labels.get_indexer([s])[0]
                        if i == -1:
                            raise ValueError(f"Split '{s}' not found")
                    split_indices.append(i)
                split_group_indices = split_indices = np.asarray(split_indices)
        if set_ is None:
            set_group_indices = np.arange(self.get_n_sets(set_group_by=set_group_by))
            set_indices = np.arange(self.n_sets)
        else:
            kind = None
            if isinstance(set_, PosSel):
                set_ = set_.value
                kind = "positions"
            elif isinstance(set_, LabelSel):
                set_ = set_.value
                kind = "labels"
            if checks.is_hashable(set_):
                set_ = [set_]
            if set_group_by is not None:
                set_group_indices = []
                groups, group_index = set_group_by.get_groups_and_index()
                mask = None
                for g in set_:
                    if isinstance(g, PosSel):
                        g = g.value
                        kind = "positions"
                    elif isinstance(g, LabelSel):
                        g = g.value
                        kind = "labels"
                    if kind == "positions" or (
                        kind is None
                        and checks.is_int(g)
                        and not pd.api.types.is_integer_dtype(group_index)
                    ):
                        i = g
                    else:
                        i = group_index.get_indexer([g])[0]
                        if i == -1:
                            raise ValueError(f"Set group '{g}' not found")
                    if mask is None:
                        mask = groups == i
                    else:
                        mask |= groups == i
                    set_group_indices.append(i)
                set_group_indices = np.asarray(set_group_indices)
                set_indices = np.arange(self.n_sets)[mask]
            else:
                set_indices = []
                for s in set_:
                    if isinstance(s, PosSel):
                        s = s.value
                        kind = "positions"
                    elif isinstance(s, LabelSel):
                        s = s.value
                        kind = "labels"
                    if kind == "positions" or (
                        kind is None
                        and checks.is_int(s)
                        and not pd.api.types.is_integer_dtype(self.set_labels)
                    ):
                        i = s
                    else:
                        i = self.set_labels.get_indexer([s])[0]
                        if i == -1:
                            raise ValueError(f"Set '{s}' not found")
                    set_indices.append(i)
                set_group_indices = set_indices = np.asarray(set_indices)
        return split_group_indices, set_group_indices, split_indices, set_indices

    def select_range(
        self, merge_split_kwargs: tp.KwargsLike = None, **select_indices_kwargs
    ) -> tp.RangeLike:
        """Select a range.

        Pass additional keyword arguments to `Splitter.select_indices` to obtain the indices
        for the selected splits and sets. If more than one range corresponds to these indices,
        merge them using `Splitter.merge_split`.

        Args:
            merge_split_kwargs (KwargsLike): Keyword arguments for `Splitter.merge_split`.
            **select_indices_kwargs: Keyword arguments for `Splitter.select_indices`.

        Returns:
            RangeLike: Selected range, or the merged range if multiple ranges are found.
        """
        _, _, split_indices, set_indices = self.select_indices(**select_indices_kwargs)
        ranges = []
        for i in split_indices:
            for j in set_indices:
                ranges.append(self.splits_arr[i, j])
        if len(ranges) == 1:
            return ranges[0]
        if merge_split_kwargs is None:
            merge_split_kwargs = {}
        return self.merge_split(ranges, **merge_split_kwargs)

    @hybrid_method
    def remap_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        target_index: tp.IndexLike,
        target_freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        silence_warnings: bool = False,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.FixRangeLike:
        """Remap a range to a target index.

        If the source `index` matches the `target_index`, return the original range.
        Otherwise, resample the range to align with the target index using
        `vectorbtpro.base.resampling.base.Resampler.resample_source_mask`.
        In such cases, both `freq` and `target_freq` must be provided.

        Args:
            range_ (FixRangeLike): Input range to be remapped.
            target_index (IndexLike): Target index to which the range is mapped.
            target_freq (Optional[FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            silence_warnings (bool): Flag to suppress warning messages.
            index (Optional[IndexLike]): Source index associated with the range.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            FixRangeLike: Remapped range corresponding to the target index.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        if target_index is None:
            raise ValueError("Must provide target index")
        target_index = dt.prepare_dt_index(target_index)
        if index.equals(target_index):
            return range_

        mask = cls_or_self.get_range_mask(range_, template_context=template_context, index=index)
        resampler = Resampler(
            source_index=index,
            target_index=target_index,
            source_freq=freq,
            target_freq=target_freq,
        )
        target_mask = resampler.resample_source_mask(
            mask, jitted=jitted, silence_warnings=silence_warnings
        )
        return target_mask

    @classmethod
    def get_obj_index(cls, obj: tp.Any) -> tp.Index:
        """Get the index from an object.

        Extract the index from an object that is either a Pandas Index or possesses
        an `index` or `wrapper.index` attribute.

        Args:
            obj (Any): Object with an associated index.

        Returns:
            Index: Extracted index.
        """
        if isinstance(obj, pd.Index):
            return obj
        if hasattr(obj, "index"):
            return obj.index
        if hasattr(obj, "wrapper"):
            return obj.wrapper.index
        raise ValueError("Must provide object index")

    @hybrid_method
    def get_ready_obj_range(
        cls_or_self,
        obj: tp.Any,
        range_: tp.FixRangeLike,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        silence_warnings: bool = False,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        return_obj_meta: bool = False,
        **ready_range_kwargs,
    ) -> tp.Any:
        """Get a ready-to-use range for indexing an array-like object.

        Determine and process a range that aligns with the object index. When the object
        is Pandas-like or an index is provided, obtain the index using `Splitter.get_obj_index`
        (if needed) and remap the range using `Splitter.remap_range`. Finally, convert the range
        into a form suitable for direct indexing using `Splitter.get_ready_range`.

        Args:
            obj (Any): Array-like object to be indexed.
            range_ (FixRangeLike): Input range to be processed.
            remap_to_obj (bool): Whether to remap the range to the object's index.
            obj_index (Optional[IndexLike]): Target index for remapping, if available.
            obj_freq (Optional[FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            template_context (KwargsLike): Additional context for template substitution.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            silence_warnings (bool): Flag to suppress warning messages.
            index (Optional[IndexLike]): Source index associated with the range.
            freq (Optional[FrequencyLike]): Frequency of the source index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            return_obj_meta (bool): Whether to return metadata about the object.
            **ready_range_kwargs: Keyword arguments for `Splitter.get_ready_range`.

                Allows zero-length ranges by default.

        Returns:
            Any: Processed range ready for indexing, or a tuple with object metadata and the range if requested.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        if remap_to_obj and (
            isinstance(obj, (pd.Index, pd.Series, pd.DataFrame, PandasIndexer))
            or obj_index is not None
        ):
            if obj_index is None:
                obj_index = cls_or_self.get_obj_index(obj)
            target_range = cls_or_self.remap_range(
                range_,
                target_index=obj_index,
                target_freq=obj_freq,
                template_context=template_context,
                jitted=jitted,
                silence_warnings=silence_warnings,
                index=index,
                freq=freq,
            )
        else:
            obj_index = index
            obj_freq = freq
            target_range = range_
        if "allow_zero_len" not in ready_range_kwargs:
            ready_range_kwargs["allow_zero_len"] = True
        ready_range_or_meta = cls_or_self.get_ready_range(
            target_range,
            template_context=template_context,
            index=obj_index,
            **ready_range_kwargs,
        )
        if return_obj_meta:
            obj_meta = dict(index=obj_index, freq=obj_freq)
            return obj_meta, ready_range_or_meta
        return ready_range_or_meta

    @classmethod
    def take_range(
        cls, obj: tp.Any, ready_range: tp.ReadyRangeLike, point_wise: bool = False
    ) -> tp.Any:
        """Take a ready range from an array-like object.

        Extract a segment from the object using the provided ready range.
        If `point_wise` is True, select one range point at a time and return a tuple.

        Args:
            obj (Any): Array-like object to index.
            ready_range (ReadyRangeLike): Preprocessed range used for indexing.
            point_wise (bool): Whether to perform point-wise range extraction.

        Returns:
            Any: Extracted segment of the object, or a tuple of elements if `point_wise` is True.
        """
        if isinstance(obj, (pd.Series, pd.DataFrame, PandasIndexer)):
            if point_wise:
                return tuple(obj.iloc[i] for i in np.arange(len(obj))[ready_range])
            return obj.iloc[ready_range]
        if point_wise:
            return tuple(obj[i] for i in np.arange(len(obj))[ready_range])
        return obj[ready_range]

    @hybrid_method
    def take_range_from_takeable(
        cls_or_self,
        takeable: Takeable,
        range_: tp.FixRangeLike,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        point_wise: bool = False,
        template_context: tp.KwargsLike = None,
        return_obj_meta: bool = False,
        return_meta: bool = False,
        **ready_obj_range_kwargs,
    ) -> tp.Any:
        """Take a range from a takeable object.

        Process the provided `range_` from a takeable object's field `obj` by ensuring it aligns
        with the object's index. If remapping is enabled (or an `obj_index` is provided),
        obtain the ready range using `get_ready_obj_range`. For objects of type `CustomTemplate`,
        substitute templates using a merged context; otherwise, extract the slice using `take_range`.

        Args:
            takeable (Takeable): Takeable object containing the data and configuration for range extraction.
            range_ (FixRangeLike): Original range to be processed.
            remap_to_obj (bool): Whether to remap the range to the object's index.
            obj_index (Optional[IndexLike]): Target index for remapping, if available.
            obj_freq (Optional[FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            point_wise (bool): Whether to perform point-wise range extraction.
            template_context (KwargsLike): Additional context for template substitution.
            return_obj_meta (bool): Whether to return metadata about the object.
            return_obj_meta (bool): Whether to return metadata about the object.
            **ready_obj_range_kwargs: Keyword arguments for `Splitter.get_ready_obj_range`.

        Returns:
            Any: Extracted range from the takeable object, or a tuple containing metadata
                and the range if requested.
        """
        takeable.assert_field_not_missing("obj")
        obj_meta, obj_range_meta = cls_or_self.get_ready_obj_range(
            takeable.obj,
            range_,
            remap_to_obj=takeable.remap_to_obj
            if takeable.remap_to_obj is not MISSING
            else remap_to_obj,
            obj_index=takeable.index if takeable.index is not MISSING else obj_index,
            obj_freq=takeable.freq if takeable.freq is not MISSING else obj_freq,
            template_context=template_context,
            return_obj_meta=True,
            return_meta=True,
            **ready_obj_range_kwargs,
        )
        if isinstance(takeable.obj, CustomTemplate):
            template_context = merge_dicts(
                dict(
                    range_=obj_range_meta["range_"],
                    range_meta=obj_range_meta,
                    point_wise=takeable.point_wise
                    if takeable.point_wise is not MISSING
                    else point_wise,
                ),
                template_context,
            )
            obj_slice = substitute_templates(takeable.obj, template_context, eval_id="take_range")
        else:
            obj_slice = cls_or_self.take_range(
                takeable.obj,
                obj_range_meta["range_"],
                point_wise=takeable.point_wise
                if takeable.point_wise is not MISSING
                else point_wise,
            )
        if return_obj_meta and return_meta:
            return obj_meta, obj_range_meta, obj_slice
        if return_obj_meta:
            return obj_meta, obj_slice
        if return_meta:
            return obj_range_meta, obj_slice
        return obj_slice

    def take(
        self,
        obj: tp.Any,
        split: tp.Optional[tp.Selection] = None,
        set_: tp.Optional[tp.Selection] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        into: tp.Optional[str] = None,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        range_format: str = "slice_or_any",
        point_wise: bool = False,
        attach_bounds: tp.Union[bool, str] = False,
        right_inclusive: bool = False,
        template_context: tp.KwargsLike = None,
        silence_warnings: bool = False,
        index_combine_kwargs: tp.KwargsLike = None,
        stack_axis: int = 1,
        stack_kwargs: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Any:
        """Take all ranges from an array-like object and optionally column-stack them.

        This method uses `Splitter.select_indices` to determine indices for selected splits and sets.
        Grouping is applied via `split_group_by` and `set_group_by` so that ranges within the
        same group are merged.

        For each split and set combination, the method:

        * Resolves the source range using `Splitter.select_range` and
            refines it with `Splitter.get_ready_range`.
        * Remaps the range to the object's index using `Splitter.get_ready_obj_range`
            and extracts the slice with `Splitter.take_range` (or substitutes using
            a custom template if the object is a `CustomTemplate`).
        * Merges the resulting slices by calling either `vectorbtpro.base.merging.column_stack_merge`
            (when `stack_axis` is 1) or `vectorbtpro.base.merging.row_stack_merge` (when `stack_axis` is 0).

        If `attach_bounds` is enabled, the method computes bounds for each range and attaches
        them as an additional level in the final index hierarchy. Supported options for `attach_bounds` are:

        * True, "index", "source", or "source_index": Attach bounds from the source index.
        * "target" or "target_index": Attach bounds from the target index.
        * False: Do not attach bounds.

        The `into` parameter controls the output format:

        * None: Returns a Series of range slices.
        * "stacked": Stacks all slices into a single object.
        * "stacked_by_split": Stacks set slices within each split and returns a Series of objects.
        * "stacked_by_set": Stacks split slices within each set and returns a Series of objects.
        * "split_major_meta": Returns a tuple with meta-information and a generator yielding
            range details in split-major order.
        * "set_major_meta": Returns a tuple with meta-information and a generator yielding
            range details in set-major order.

        Prepend any stacked option with "from_start_" (or "reset_") or "from_end_"
        to reset the index from the start or end.

        Args:
            obj (Any): Array-like object from which to extract ranges.
            split (Optional[Selection]): Selection criteria for splits.
            set_ (Optional[Selection]): Selection criteria for sets.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            into (Optional[str]): Specifies the output format.

                Options include "stacked", "stacked_by_split", "stacked_by_set",
                "split_major_meta", "set_major_meta", etc.
            remap_to_obj (bool): Whether to remap the range to the object's index.
            obj_index (Optional[IndexLike]): Target index for remapping, if available.
            obj_freq (Optional[FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            range_format (str): Format of the returned range.

                See `Splitter.get_ready_range`.
            point_wise (bool): Whether to perform point-wise range extraction.
            attach_bounds (Union[bool, str]): Specifies if and how to attach bounds to the result.

                Options include True, "index", "source", "target", etc.
            right_inclusive (bool): Whether the right bound is inclusive.
            template_context (KwargsLike): Additional context for template substitution.
            silence_warnings (bool): Flag to suppress warning messages.
            index_combine_kwargs (KwargsLike): Keyword arguments for combining indexes.

                See `vectorbtpro.base.indexes.combine_indexes`.
            stack_axis (int): Axis along which to stack slices (0 for rows, 1 for columns).
            stack_kwargs (KwargsLike): Keyword arguments for the stacking merge function.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            Any: Extracted range, which may be a single slice, a merged object,
                or a Pandas Series depending on the `into` parameter.

        Examples:
            Roll a window and stack it along columns by keeping the index:

            ```pycon
            >>> from vectorbtpro import *

            >>> data = vbt.YFData.pull(
            ...     "BTC-USD",
            ...     start="2020-01-01 UTC",
            ...     end="2021-01-01 UTC"
            ... )
            >>> splitter = vbt.Splitter.from_n_rolling(
            ...     data.wrapper.index,
            ...     3,
            ...     length=5
            ... )
            >>> splitter.take(data.close, into="stacked")
            split                                0            1             2
            Date
            2020-01-01 00:00:00+00:00  7200.174316          NaN           NaN
            2020-01-02 00:00:00+00:00  6985.470215          NaN           NaN
            2020-01-03 00:00:00+00:00  7344.884277          NaN           NaN
            2020-01-04 00:00:00+00:00  7410.656738          NaN           NaN
            2020-01-05 00:00:00+00:00  7411.317383          NaN           NaN
            2020-06-29 00:00:00+00:00          NaN  9190.854492           NaN
            2020-06-30 00:00:00+00:00          NaN  9137.993164           NaN
            2020-07-01 00:00:00+00:00          NaN  9228.325195           NaN
            2020-07-02 00:00:00+00:00          NaN  9123.410156           NaN
            2020-07-03 00:00:00+00:00          NaN  9087.303711           NaN
            2020-12-27 00:00:00+00:00          NaN          NaN  26272.294922
            2020-12-28 00:00:00+00:00          NaN          NaN  27084.808594
            2020-12-29 00:00:00+00:00          NaN          NaN  27362.437500
            2020-12-30 00:00:00+00:00          NaN          NaN  28840.953125
            2020-12-31 00:00:00+00:00          NaN          NaN  29001.720703
            ```

            Disregard the index and attach index bounds to the column hierarchy:

            ```pycon
            >>> splitter.take(
            ...     data.close,
            ...     into="reset_stacked",
            ...     attach_bounds="index"
            ... )
            split                         0                         1  \\
            start 2020-01-01 00:00:00+00:00 2020-06-29 00:00:00+00:00
            end   2020-01-06 00:00:00+00:00 2020-07-04 00:00:00+00:00
            0                   7200.174316               9190.854492
            1                   6985.470215               9137.993164
            2                   7344.884277               9228.325195
            3                   7410.656738               9123.410156
            4                   7411.317383               9087.303711

            split                         2
            start 2020-12-27 00:00:00+00:00
            end   2021-01-01 00:00:00+00:00
            0                  26272.294922
            1                  27084.808594
            2                  27362.437500
            3                  28840.953125
            4                  29001.720703
            ```
        """
        if isinstance(attach_bounds, bool):
            if attach_bounds:
                attach_bounds = "source"
            else:
                attach_bounds = None
        index_bounds = False
        if attach_bounds is not None:
            if attach_bounds.lower() == "index":
                attach_bounds = "source"
                index_bounds = True
            if attach_bounds.lower() in ("source_index", "target_index"):
                attach_bounds = attach_bounds.split("_")[0]
                index_bounds = True
            if attach_bounds.lower() not in ("source", "target"):
                raise ValueError(f"Invalid attach_bounds: '{attach_bounds}'")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        if stack_axis not in (0, 1):
            raise ValueError("Axis for stacking must be either 0 or 1")
        if stack_kwargs is None:
            stack_kwargs = {}

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        split_group_indices, set_group_indices, split_indices, set_indices = self.select_indices(
            split=split,
            set_=set_,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        )
        if split is not None:
            split_labels = split_labels[split_group_indices]
        if set_ is not None:
            set_labels = set_labels[set_group_indices]
        n_splits = len(split_group_indices)
        n_sets = len(set_group_indices)
        one_split = n_splits == 1 and squeeze_one_split
        one_set = n_sets == 1 and squeeze_one_set
        one_range = one_split and one_set

        def _get_bounds(range_meta, obj_meta, obj_range_meta):
            if attach_bounds is not None:
                if attach_bounds.lower() == "source":
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            range_meta["start"],
                            range_meta["stop"],
                            right_inclusive=right_inclusive,
                            freq=freq,
                        )
                    else:
                        if right_inclusive:
                            bounds = (range_meta["start"], range_meta["stop"] - 1)
                        else:
                            bounds = (range_meta["start"], range_meta["stop"])
                else:
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            obj_range_meta["start"],
                            obj_range_meta["stop"],
                            right_inclusive=right_inclusive,
                            index=obj_meta["index"],
                            freq=obj_meta["freq"],
                        )
                    else:
                        if right_inclusive:
                            bounds = (obj_range_meta["start"], obj_range_meta["stop"] - 1)
                        else:
                            bounds = (obj_range_meta["start"], obj_range_meta["stop"])
            else:
                bounds = (None, None)
            return bounds

        def _get_range_meta(i, j):
            split_idx = split_group_indices[i]
            set_idx = set_group_indices[j]
            range_ = self.select_range(
                split=PosSel(split_idx),
                set_=PosSel(set_idx),
                split_group_by=split_group_by,
                set_group_by=set_group_by,
                merge_split_kwargs=dict(template_context=template_context),
            )
            range_meta = self.get_ready_range(
                range_,
                range_format=range_format,
                template_context=template_context,
                return_meta=True,
            )
            obj_meta, obj_range_meta = self.get_ready_obj_range(
                obj,
                range_meta["range_"],
                remap_to_obj=remap_to_obj,
                obj_index=obj_index,
                obj_freq=obj_freq,
                range_format=range_format,
                template_context=template_context,
                silence_warnings=silence_warnings,
                freq=freq,
                return_obj_meta=True,
                return_meta=True,
            )
            if isinstance(obj, CustomTemplate):
                _template_context = merge_dicts(
                    dict(
                        split_idx=split_idx,
                        set_idx=set_idx,
                        range_=obj_range_meta["range_"],
                        range_meta=obj_range_meta,
                        point_wise=point_wise,
                    ),
                    template_context,
                )
                obj_slice = substitute_templates(obj, _template_context, eval_id="take_range")
            else:
                obj_slice = self.take_range(obj, obj_range_meta["range_"], point_wise=point_wise)
            bounds = _get_bounds(range_meta, obj_meta, obj_range_meta)
            return dict(
                split_idx=split_idx,
                set_idx=set_idx,
                range_meta=range_meta,
                obj_range_meta=obj_range_meta,
                obj_slice=obj_slice,
                bounds=bounds,
            )

        def _attach_bounds(keys, range_bounds):
            range_bounds = pd.MultiIndex.from_tuples(range_bounds, names=["start", "end"])
            if keys is None:
                return range_bounds
            clean_index_kwargs = dict(index_combine_kwargs)
            clean_index_kwargs.pop("ignore_ranges", None)
            return stack_indexes((keys, range_bounds), **clean_index_kwargs)

        if into is None:
            range_objs = []
            range_bounds = []
            for i in range(n_splits):
                for j in range(n_sets):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
            if one_range:
                return range_objs[0]
            if one_set:
                keys = split_labels
            elif one_split:
                keys = set_labels
            else:
                keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
            if attach_bounds is not None:
                keys = _attach_bounds(keys, range_bounds)
            return pd.Series(range_objs, index=keys, dtype=object)
        if isinstance(into, str) and into.lower().startswith("reset_"):
            if stack_axis == 0:
                raise ValueError("Cannot use reset_index with stack_axis=0")
            stack_kwargs["reset_index"] = "from_start"
            into = into.lower().replace("reset_", "")
        if isinstance(into, str) and into.lower().startswith("from_start_"):
            if stack_axis == 0:
                raise ValueError("Cannot use reset_index with stack_axis=0")
            stack_kwargs["reset_index"] = "from_start"
            into = into.lower().replace("from_start_", "")
        if isinstance(into, str) and into.lower().startswith("from_end_"):
            if stack_axis == 0:
                raise ValueError("Cannot use reset_index with stack_axis=0")
            stack_kwargs["reset_index"] = "from_end"
            into = into.lower().replace("from_end_", "")
        if isinstance(into, str) and into.lower() in ("split_major_meta", "set_major_meta"):
            meta = {
                "split_group_indices": split_group_indices,
                "set_group_indices": set_group_indices,
                "split_indices": split_indices,
                "set_indices": set_indices,
                "n_splits": n_splits,
                "n_sets": n_sets,
                "split_labels": split_labels,
                "set_labels": set_labels,
            }
            if isinstance(into, str) and into.lower() == "split_major_meta":

                def _get_generator():
                    for i in range(n_splits):
                        for j in range(n_sets):
                            yield _get_range_meta(i, j)

                return meta, _get_generator()
            if isinstance(into, str) and into.lower() == "set_major_meta":

                def _get_generator():
                    for j in range(n_sets):
                        for i in range(n_splits):
                            yield _get_range_meta(i, j)

                return meta, _get_generator()
        if isinstance(into, str) and into.lower() == "stacked":
            range_objs = []
            range_bounds = []
            for i in range(n_splits):
                for j in range(n_sets):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
            if one_range:
                return range_objs[0]
            if one_set:
                keys = split_labels
            elif one_split:
                keys = set_labels
            else:
                keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
            if attach_bounds is not None:
                keys = _attach_bounds(keys, range_bounds)
            _stack_kwargs = merge_dicts(dict(keys=keys), stack_kwargs)
            if stack_axis == 0:
                return row_stack_merge(range_objs, **_stack_kwargs)
            return column_stack_merge(range_objs, **_stack_kwargs)
        if isinstance(into, str) and into.lower() == "stacked_by_split":
            new_split_objs = []
            one_set_bounds = []
            for i in range(n_splits):
                range_objs = []
                range_bounds = []
                for j in range(n_sets):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
                if one_set and squeeze_one_set:
                    new_split_objs.append(range_objs[0])
                    one_set_bounds.append(range_bounds[0])
                else:
                    keys = set_labels
                    if attach_bounds is not None:
                        keys = _attach_bounds(keys, range_bounds)
                    _stack_kwargs = merge_dicts(dict(keys=keys), stack_kwargs)
                    if stack_axis == 0:
                        new_split_objs.append(row_stack_merge(range_objs, **_stack_kwargs))
                    else:
                        new_split_objs.append(column_stack_merge(range_objs, **_stack_kwargs))
            if one_split and squeeze_one_split:
                return new_split_objs[0]
            if one_set and squeeze_one_set:
                if attach_bounds is not None:
                    return pd.Series(
                        new_split_objs,
                        index=_attach_bounds(split_labels, one_set_bounds),
                        dtype=object,
                    )
            return pd.Series(new_split_objs, index=split_labels, dtype=object)
        if isinstance(into, str) and into.lower() == "stacked_by_set":
            new_set_objs = []
            one_split_bounds = []
            for j in range(n_sets):
                range_objs = []
                range_bounds = []
                for i in range(n_splits):
                    range_meta = _get_range_meta(i, j)
                    range_objs.append(range_meta["obj_slice"])
                    range_bounds.append(range_meta["bounds"])
                if one_split and squeeze_one_split:
                    new_set_objs.append(range_objs[0])
                    one_split_bounds.append(range_bounds[0])
                else:
                    keys = split_labels
                    if attach_bounds:
                        keys = _attach_bounds(keys, range_bounds)
                    _stack_kwargs = merge_dicts(dict(keys=keys), stack_kwargs)
                    if stack_axis == 0:
                        new_set_objs.append(row_stack_merge(range_objs, **_stack_kwargs))
                    else:
                        new_set_objs.append(column_stack_merge(range_objs, **_stack_kwargs))
            if one_set and squeeze_one_set:
                return new_set_objs[0]
            if one_split and squeeze_one_split:
                if attach_bounds is not None:
                    return pd.Series(
                        new_set_objs,
                        index=_attach_bounds(set_labels, one_split_bounds),
                        dtype=object,
                    )
            return pd.Series(new_set_objs, index=set_labels, dtype=object)
        raise ValueError(f"Invalid into: '{into}'")

    # ############# Applying ############# #

    @classmethod
    def parse_and_inject_takeables(
        cls,
        flat_ann_args: tp.FlatAnnArgs,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.FlatAnnArgs:
        """Parse `Takeable` instances in function annotations and inject their processed values
        into flattened annotated arguments.

        Args:
            flat_ann_args (FlatAnnArgs): Flattened annotated arguments.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            FlatAnnArgs: Dictionary with updated annotated arguments after processing `Takeable` instances.
        """
        new_flat_ann_args = dict()
        for k, v in flat_ann_args.items():
            new_flat_ann_args[k] = v = dict(v)
            if "annotation" in v:
                if isinstance(v["annotation"], type) and issubclass(v["annotation"], Takeable):
                    v["annotation"] = v["annotation"]()
                if isinstance(v["annotation"], Takeable) and v["annotation"].meets_eval_id(eval_id):
                    if "value" in v:
                        if not isinstance(v["value"], Takeable):
                            v["value"] = v["annotation"].replace(obj=v["value"])
                        else:
                            v["value"] = v["value"].merge_over(v["annotation"])
        return new_flat_ann_args

    def apply(
        self,
        apply_func: tp.Callable,
        *apply_args,
        split: tp.Optional[tp.Selection] = None,
        set_: tp.Optional[tp.Selection] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        remap_to_obj: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        range_format: str = "slice_or_any",
        point_wise: bool = False,
        attach_bounds: tp.Union[bool, str] = False,
        right_inclusive: bool = False,
        template_context: tp.KwargsLike = None,
        silence_warnings: bool = False,
        index_combine_kwargs: tp.KwargsLike = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        iteration: str = "split_wise",
        execute_kwargs: tp.KwargsLike = None,
        filter_results: bool = True,
        raise_no_results: bool = True,
        merge_func: tp.MergeFuncLike = None,
        merge_kwargs: tp.KwargsLike = None,
        merge_all: bool = True,
        wrap_results: bool = True,
        eval_id: tp.Optional[tp.Hashable] = None,
        **apply_kwargs,
    ) -> tp.Any:
        """Apply a function over each data range.

        Divides the index into ranges based on selected splits and sets, optionally grouping using
        `split_group_by` and `set_group_by`. For each combination of split and set, retrieves
        the corresponding range via `Splitter.select_range` and `Splitter.get_ready_range`.
        Positional and keyword arguments that are instances of `Takeable` are sliced based on
        these ranges using `Splitter.take_range`. Before slicing, the range into each object's index
        using `Splitter.get_ready_obj_range`. The function and its arguments are then
        template-substituted and scheduled for execution via `vectorbtpro.utils.execution.execute`.
        After execution, the results are optionally merged using `merge_func` and wrapped in
        a Pandas object if specified.

        Template substitution variables include:

        * `split/set_group_indices`: Indices for selected row/column groups.
        * `split/set_indices`: Indices for selected rows/columns.
        * `n_splits/sets`: Number of selected splits/sets.
        * `split/set_labels`: Labels for split or set groups.
        * `split/set_idx`: Index of the selected split or set.
        * `split/set_label`: Label of the selected split or set.
        * `range_`: Range used for indexing (see `Splitter.get_ready_range`).
        * `range_meta`: Metadata regarding the range.
        * `obj_range_meta`: Metadata for the ranges taken from each takeable argument.
            Positional arguments are denoted by position, keyword arguments are denoted by keys.
        * `args`: Positional arguments with selected ranges.
        * `kwargs`: Keyword arguments with selected ranges.
        * `bounds`: Tuple of range boundaries.
        * `template_context`: Template context provided for substitutions.

        Iteration over ranges is controlled by the `iteration` parameter:

        * `split_major`: Iterate over ranges in split-major order.
        * `set_major`: Iterate over ranges in set-major order.
        * `split_wise`: Process ranges sequentially within each split.
        * `set_wise`: Process ranges sequentially within each set.

        Args:
            apply_func (Callable): Function to apply over each range.
            *apply_args: Positional arguments for `apply_func`.
            split (Optional[Selection]): Selection criteria for splits.
            set_ (Optional[Selection]): Selection criteria for sets.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            remap_to_obj (bool): Whether to remap the range to the object's index.
            obj_index (Optional[IndexLike]): Target index for remapping, if available.
            obj_freq (Optional[FrequencyLike]): Frequency of the target index
                (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            range_format (str): Format of the returned range.

                See `Splitter.get_ready_range`.
            point_wise (bool): Whether to perform point-wise range extraction.
            attach_bounds (Union[bool, str]): Specifies if and how to attach bounds to the result.

                If True or "source", attaches the source bounds; other string options are supported.
            right_inclusive (bool): Whether the right bound is inclusive.
            template_context (KwargsLike): Additional context for template substitution.
            silence_warnings (bool): Flag to suppress warning messages.
            index_combine_kwargs (KwargsLike): Keyword arguments for combining indexes.

                See `vectorbtpro.base.indexes.combine_indexes`.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            iteration (str): Iteration mode over ranges.

                Options: "split_major", "set_major", "split_wise", "set_wise".
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
            raise_no_results (bool): Flag indicating whether to raise a
                `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
            merge_func (MergeFuncLike): Function to merge the results.

                See `vectorbtpro.utils.merging.MergeFunc`.
            merge_kwargs (KwargsLike): Keyword arguments for `merge_func`.
            merge_all (bool): Whether to merge all results across iterations regardless of the iteration mode.
            wrap_results (bool): Whether to wrap the final merged result in a Pandas object.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **apply_kwargs: Keyword arguments for `apply_func`.

        Returns:
            Any: Result of applying `apply_func` over each range, which may be a merged result,
                a Pandas Series, or a tuple of Pandas objects depending on the processing and
                output wrapping options.

        Examples:
            Get the return of each data range:

            ```pycon
            >>> from vectorbtpro import *

            >>> data = vbt.YFData.pull(
            ...     "BTC-USD",
            ...     start="2020-01-01 UTC",
            ...     end="2021-01-01 UTC"
            ... )
            >>> splitter = vbt.Splitter.from_n_rolling(data.wrapper.index, 5)

            >>> def apply_func(data):
            ...     return data.close.iloc[-1] - data.close.iloc[0]

            >>> splitter.apply(apply_func, vbt.Takeable(data))
            split
            0    -1636.467285
            1     3706.568359
            2     2944.720703
            3     -118.113281
            4    17098.916016
            dtype: float64
            ```

            The same but by indexing manually:

            ```pycon
            >>> def apply_func(range_, data):
            ...     data = data.iloc[range_]
            ...     return data.close.iloc[-1] - data.close.iloc[0]

            >>> splitter.apply(apply_func, vbt.Rep("range_"), data)
            split
            0    -1636.467285
            1     3706.568359
            2     2944.720703
            3     -118.113281
            4    17098.916016
            dtype: float64
            ```

            Divide into two windows, each consisting of 50% train and 50% test, compute SMA for
            each range, and row-stack the outputs of each set upon merging:

            ```pycon
            >>> splitter = vbt.Splitter.from_n_rolling(data.wrapper.index, 2, split=0.5)

            >>> def apply_func(data):
            ...     return data.run("SMA", 10).real

            >>> splitter.apply(
            ...     apply_func,
            ...     vbt.Takeable(data),
            ...     merge_func="row_stack"
            ... ).unstack("set").vbt.drop_levels("split", axis=0).vbt.plot().show()
            ```

            ![](/assets/images/api/Splitter_apply.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/Splitter_apply.dark.svg#only-dark){: .iimg loading=lazy }
        """
        if isinstance(attach_bounds, bool):
            if attach_bounds:
                attach_bounds = "source"
            else:
                attach_bounds = None
        index_bounds = False
        if attach_bounds is not None:
            if attach_bounds.lower() == "index":
                attach_bounds = "source"
                index_bounds = True
            if attach_bounds.lower() in ("source_index", "target_index"):
                attach_bounds = attach_bounds.split("_")[0]
                index_bounds = True
            if attach_bounds.lower() not in ("source", "target"):
                raise ValueError(f"Invalid attach_bounds: '{attach_bounds}'")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        if execute_kwargs is None:
            execute_kwargs = {}
        parsed_merge_func = parse_merge_func(apply_func, eval_id=eval_id)
        if parsed_merge_func is not None:
            if merge_func is not None:
                raise ValueError(
                    f"Two conflicting merge functions: {parsed_merge_func} (annotations) and {merge_func} (merge_func)"
                )
            merge_func = parsed_merge_func
        if merge_kwargs is None:
            merge_kwargs = {}

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        split_group_indices, set_group_indices, split_indices, set_indices = self.select_indices(
            split=split,
            set_=set_,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
        )
        if split is not None:
            split_labels = split_labels[split_group_indices]
        if set_ is not None:
            set_labels = set_labels[set_group_indices]
        n_splits = len(split_group_indices)
        n_sets = len(set_group_indices)
        one_split = n_splits == 1 and squeeze_one_split
        one_set = n_sets == 1 and squeeze_one_set
        one_range = one_split and one_set
        template_context = merge_dicts(
            {
                "splitter": self,
                "index": self.index,
                "split_group_indices": split_group_indices,
                "set_group_indices": set_group_indices,
                "split_indices": split_indices,
                "set_indices": set_indices,
                "n_splits": n_splits,
                "n_sets": n_sets,
                "split_labels": split_labels,
                "set_labels": set_labels,
                "one_split": one_split,
                "one_set": one_set,
                "one_range": one_range,
            },
            template_context,
        )
        template_context["eval_id"] = eval_id

        if has_annotatables(apply_func):
            ann_args = annotate_args(
                apply_func,
                apply_args,
                apply_kwargs,
                attach_annotations=True,
            )
            flat_ann_args = flatten_ann_args(ann_args)
            flat_ann_args = self.parse_and_inject_takeables(flat_ann_args, eval_id=eval_id)
            ann_args = unflatten_ann_args(flat_ann_args)
            apply_args, apply_kwargs = ann_args_to_args(ann_args)

        def _get_range_meta(i, j, _template_context):
            split_idx = split_group_indices[i]
            set_idx = set_group_indices[j]
            range_ = self.select_range(
                split=PosSel(split_idx),
                set_=PosSel(set_idx),
                split_group_by=split_group_by,
                set_group_by=set_group_by,
                merge_split_kwargs=dict(template_context=_template_context),
            )
            range_meta = self.get_ready_range(
                range_,
                range_format=range_format,
                template_context=_template_context,
                return_meta=True,
            )
            return range_meta

        def _take_args(args, range_, _template_context):
            obj_meta = {}
            obj_range_meta = {}
            new_args = ()
            if args is not None:
                for i, v in enumerate(args):
                    if isinstance(v, Takeable) and v.meets_eval_id(eval_id):
                        _obj_meta, _obj_range_meta, obj_slice = self.take_range_from_takeable(
                            v,
                            range_,
                            remap_to_obj=remap_to_obj,
                            obj_index=obj_index,
                            obj_freq=obj_freq,
                            range_format=range_format,
                            point_wise=point_wise,
                            template_context=_template_context,
                            silence_warnings=silence_warnings,
                            freq=freq,
                            return_obj_meta=True,
                            return_meta=True,
                        )
                        new_args += (obj_slice,)
                        obj_meta[i] = _obj_meta
                        obj_range_meta[i] = _obj_range_meta
                    else:
                        new_args += (v,)
            return obj_meta, obj_range_meta, new_args

        def _take_kwargs(kwargs, range_, _template_context):
            obj_meta = {}
            obj_range_meta = {}
            new_kwargs = {}
            if kwargs is not None:
                for k, v in kwargs.items():
                    if isinstance(v, Takeable) and v.meets_eval_id(eval_id):
                        _obj_meta, _obj_range_meta, obj_slice = self.take_range_from_takeable(
                            v,
                            range_,
                            remap_to_obj=remap_to_obj,
                            obj_index=obj_index,
                            obj_freq=obj_freq,
                            range_format=range_format,
                            point_wise=point_wise,
                            template_context=_template_context,
                            silence_warnings=silence_warnings,
                            freq=freq,
                            return_obj_meta=True,
                            return_meta=True,
                        )
                        new_kwargs[k] = obj_slice
                        obj_meta[k] = _obj_meta
                        obj_range_meta[k] = _obj_range_meta
                    else:
                        new_kwargs[k] = v
            return obj_meta, obj_range_meta, new_kwargs

        def _get_bounds(range_meta, _template_context):
            if attach_bounds is not None:
                if isinstance(attach_bounds, str) and attach_bounds.lower() == "source":
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            range_meta["start"],
                            range_meta["stop"],
                            right_inclusive=right_inclusive,
                            freq=freq,
                        )
                    else:
                        if right_inclusive:
                            bounds = (range_meta["start"], range_meta["stop"] - 1)
                        else:
                            bounds = (range_meta["start"], range_meta["stop"])
                else:
                    obj_meta, obj_range_meta = self.get_ready_obj_range(
                        self.index,
                        range_meta["range_"],
                        remap_to_obj=remap_to_obj,
                        obj_index=obj_index,
                        obj_freq=obj_freq,
                        range_format=range_format,
                        template_context=_template_context,
                        silence_warnings=silence_warnings,
                        freq=freq,
                        return_obj_meta=True,
                        return_meta=True,
                    )
                    if index_bounds:
                        bounds = self.map_bounds_to_index(
                            obj_range_meta["start"],
                            obj_range_meta["stop"],
                            right_inclusive=right_inclusive,
                            index=obj_meta["index"],
                            freq=obj_meta["freq"],
                        )
                    else:
                        if right_inclusive:
                            bounds = (
                                obj_range_meta["start"],
                                obj_range_meta["stop"] - 1,
                            )
                        else:
                            bounds = (
                                obj_range_meta["start"],
                                obj_range_meta["stop"],
                            )
            else:
                bounds = (None, None)
            return bounds

        bounds = {}

        def _get_task(i, j, _bounds=bounds):
            split_idx = split_group_indices[i]
            set_idx = set_group_indices[j]
            _template_context = merge_dicts(
                {
                    "split_idx": split_idx,
                    "split_label": split_labels[i],
                    "set_idx": set_idx,
                    "set_label": set_labels[j],
                },
                template_context,
            )
            range_meta = _get_range_meta(i, j, _template_context)
            _template_context = merge_dicts(
                dict(range_=range_meta["range_"], range_meta=range_meta),
                _template_context,
            )
            obj_meta1, obj_range_meta1, _apply_args = _take_args(
                apply_args, range_meta["range_"], _template_context
            )
            obj_meta2, obj_range_meta2, _apply_kwargs = _take_kwargs(
                apply_kwargs, range_meta["range_"], _template_context
            )
            obj_meta = {**obj_meta1, **obj_meta2}
            obj_range_meta = {**obj_range_meta1, **obj_range_meta2}
            _bounds[(i, j)] = _get_bounds(range_meta, _template_context)
            _template_context = merge_dicts(
                dict(
                    obj_meta=obj_meta,
                    obj_range_meta=obj_range_meta,
                    apply_args=_apply_args,
                    apply_kwargs=_apply_kwargs,
                    bounds=_bounds[(i, j)],
                ),
                _template_context,
            )
            _apply_func = substitute_templates(apply_func, _template_context, eval_id="apply_func")
            _apply_args = substitute_templates(_apply_args, _template_context, eval_id="apply_args")
            _apply_kwargs = substitute_templates(
                _apply_kwargs, _template_context, eval_id="apply_kwargs"
            )
            return Task(_apply_func, *_apply_args, **_apply_kwargs)

        def _attach_bounds(keys, range_bounds):
            range_bounds = pd.MultiIndex.from_tuples(range_bounds, names=["start", "end"])
            if keys is None:
                return range_bounds
            clean_index_kwargs = dict(index_combine_kwargs)
            clean_index_kwargs.pop("ignore_ranges", None)
            return stack_indexes((keys, range_bounds), **clean_index_kwargs)

        if iteration.lower() == "split_major":

            def _get_task_generator():
                for i in range(n_splits):
                    for j in range(n_sets):
                        yield _get_task(i, j)

            tasks = _get_task_generator()
            keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
            if eval_id is not None:
                new_keys = []
                for key in keys:
                    if isinstance(keys, pd.MultiIndex):
                        new_keys.append((MISSING, *key))
                    else:
                        new_keys.append((MISSING, key))
                keys = pd.MultiIndex.from_tuples(
                    new_keys, names=(f"eval_id={eval_id}", *keys.names)
                )
            execute_kwargs = merge_dicts(
                dict(show_progress=False if one_split and one_set else None), execute_kwargs
            )
            results = execute(tasks, size=n_splits * n_sets, keys=keys, **execute_kwargs)
        elif iteration.lower() == "set_major":

            def _get_task_generator():
                for j in range(n_sets):
                    for i in range(n_splits):
                        yield _get_task(i, j)

            tasks = _get_task_generator()
            keys = combine_indexes((set_labels, split_labels), **index_combine_kwargs)
            if eval_id is not None:
                new_keys = []
                for key in keys:
                    if isinstance(keys, pd.MultiIndex):
                        new_keys.append((MISSING, *key))
                    else:
                        new_keys.append((MISSING, key))
                keys = pd.MultiIndex.from_tuples(
                    new_keys, names=(f"eval_id={eval_id}", *keys.names)
                )
            execute_kwargs = merge_dicts(
                dict(show_progress=False if one_split and one_set else None), execute_kwargs
            )
            results = execute(tasks, size=n_splits * n_sets, keys=keys, **execute_kwargs)
        elif iteration.lower() == "split_wise":

            def _process_chunk_tasks(chunk_tasks):
                results = []
                for func, args, kwargs in chunk_tasks:
                    results.append(func(*args, **kwargs))
                return results

            def _get_task_generator():
                for i in range(n_splits):
                    chunk_tasks = []
                    for j in range(n_sets):
                        chunk_tasks.append(_get_task(i, j))
                    yield Task(_process_chunk_tasks, chunk_tasks)

            tasks = _get_task_generator()
            keys = split_labels
            if eval_id is not None:
                new_keys = []
                for key in keys:
                    if isinstance(keys, pd.MultiIndex):
                        new_keys.append((MISSING, *key))
                    else:
                        new_keys.append((MISSING, key))
                keys = pd.MultiIndex.from_tuples(
                    new_keys, names=(f"eval_id={eval_id}", *keys.names)
                )
            execute_kwargs = merge_dicts(
                dict(show_progress=False if one_split else None), execute_kwargs
            )
            results = execute(tasks, size=n_splits, keys=keys, **execute_kwargs)
        elif iteration.lower() == "set_wise":

            def _process_chunk_tasks(chunk_tasks):
                results = []
                for func, args, kwargs in chunk_tasks:
                    results.append(func(*args, **kwargs))
                return results

            def _get_task_generator():
                for j in range(n_sets):
                    chunk_tasks = []
                    for i in range(n_splits):
                        chunk_tasks.append(_get_task(i, j))
                    yield Task(_process_chunk_tasks, chunk_tasks)

            tasks = _get_task_generator()
            keys = set_labels
            if eval_id is not None:
                new_keys = []
                for key in keys:
                    if isinstance(keys, pd.MultiIndex):
                        new_keys.append((MISSING, *key))
                    else:
                        new_keys.append((MISSING, key))
                keys = pd.MultiIndex.from_tuples(
                    new_keys, names=(f"eval_id={eval_id}", *keys.names)
                )
            execute_kwargs = merge_dicts(
                dict(show_progress=False if one_set else None), execute_kwargs
            )
            results = execute(tasks, size=n_sets, keys=keys, **execute_kwargs)
        else:
            raise ValueError(f"Invalid iteration: '{iteration}'")

        if merge_all:
            if iteration.lower() in ("split_wise", "set_wise"):
                results = [result for _results in results for result in _results]
            if one_range:
                if results[0] is NoResult:
                    if raise_no_results:
                        raise NoResultsException
                    return NoResult
                return results[0]
            if iteration.lower() in ("split_major", "split_wise"):
                if one_set:
                    keys = split_labels
                elif one_split:
                    keys = set_labels
                else:
                    keys = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
                if attach_bounds is not None:
                    range_bounds = []
                    for i in range(n_splits):
                        for j in range(n_sets):
                            range_bounds.append(bounds[(i, j)])
                    keys = _attach_bounds(keys, range_bounds)
            else:
                if one_set:
                    keys = split_labels
                elif one_split:
                    keys = set_labels
                else:
                    keys = combine_indexes((set_labels, split_labels), **index_combine_kwargs)
                if attach_bounds is not None:
                    range_bounds = []
                    for j in range(n_sets):
                        for i in range(n_splits):
                            range_bounds.append(bounds[(i, j)])
                    keys = _attach_bounds(keys, range_bounds)
            if filter_results:
                try:
                    results, keys = filter_out_no_results(results, keys=keys)
                except NoResultsException as e:
                    if raise_no_results:
                        raise e
                    return NoResult
                no_results_filtered = True
            else:
                no_results_filtered = False

            def _wrap_output(_results):
                try:
                    return pd.Series(_results, index=keys)
                except Exception:
                    return pd.Series(_results, index=keys, dtype=object)

            if merge_func is not None:
                template_context["tasks"] = tasks
                template_context["keys"] = keys
                if is_merge_func_from_config(merge_func):
                    merge_kwargs = merge_dicts(
                        dict(
                            keys=keys,
                            filter_results=not no_results_filtered,
                            raise_no_results=raise_no_results,
                        ),
                        merge_kwargs,
                    )
                if isinstance(merge_func, MergeFunc):
                    merge_func = merge_func.replace(
                        merge_kwargs=merge_kwargs,
                        context=template_context,
                    )
                else:
                    merge_func = MergeFunc(
                        merge_func,
                        merge_kwargs=merge_kwargs,
                        context=template_context,
                    )
                return merge_func(results)
            if wrap_results:
                if isinstance(results[0], tuple):
                    return tuple(map(_wrap_output, zip(*results)))
                return _wrap_output(results)
            return results

        if iteration.lower() == "split_major":
            new_results = []
            for i in range(n_splits):
                new_results.append(results[i * n_sets : (i + 1) * n_sets])
            results = new_results
        elif iteration.lower() == "set_major":
            new_results = []
            for i in range(n_sets):
                new_results.append(results[i * n_splits : (i + 1) * n_splits])
            results = new_results
        if one_range:
            if results[0][0] is NoResult:
                if raise_no_results:
                    raise NoResultsException
                return NoResult
            return results[0][0]

        split_bounds = []
        if attach_bounds is not None:
            for i in range(n_splits):
                split_bounds.append([])
                for j in range(n_sets):
                    split_bounds[-1].append(bounds[(i, j)])
        set_bounds = []
        if attach_bounds is not None:
            for j in range(n_sets):
                set_bounds.append([])
                for i in range(n_splits):
                    set_bounds[-1].append(bounds[(i, j)])
        if iteration.lower() in ("split_major", "split_wise"):
            major_keys = split_labels
            minor_keys = set_labels
            major_bounds = split_bounds
            minor_bounds = set_bounds
            one_major = one_split
            one_minor = one_set
        else:
            major_keys = set_labels
            minor_keys = split_labels
            major_bounds = set_bounds
            minor_bounds = split_bounds
            one_major = one_set
            one_minor = one_split

        if merge_func is not None:
            merged_results = []
            keep_major_indices = []
            for i, _results in enumerate(results):
                if one_minor:
                    if _results[0] is not NoResult:
                        merged_results.append(_results[0])
                        keep_major_indices.append(i)
                else:
                    _template_context = dict(template_context)
                    _template_context["tasks"] = tasks
                    if attach_bounds is not None:
                        minor_keys_wbounds = _attach_bounds(minor_keys, major_bounds[i])
                    else:
                        minor_keys_wbounds = minor_keys
                    if filter_results:
                        _results, minor_keys_wbounds = filter_out_no_results(
                            _results,
                            keys=minor_keys_wbounds,
                            raise_error=False,
                        )
                        no_results_filtered = True
                    else:
                        no_results_filtered = False
                    if len(_results) > 0:
                        _template_context["keys"] = minor_keys_wbounds
                        if is_merge_func_from_config(merge_func):
                            _merge_kwargs = merge_dicts(
                                dict(
                                    keys=minor_keys_wbounds,
                                    filter_results=not no_results_filtered,
                                    raise_no_results=False,
                                ),
                                merge_kwargs,
                            )
                        else:
                            _merge_kwargs = merge_kwargs
                        if isinstance(merge_func, MergeFunc):
                            _merge_func = merge_func.replace(
                                merge_kwargs=_merge_kwargs,
                                context=_template_context,
                            )
                        else:
                            _merge_func = MergeFunc(
                                merge_func,
                                merge_kwargs=_merge_kwargs,
                                context=_template_context,
                            )
                        _result = _merge_func(_results)
                        if _result is not NoResult:
                            merged_results.append(_result)
                            keep_major_indices.append(i)
            if len(merged_results) == 0:
                if raise_no_results:
                    raise NoResultsException
                return NoResult
            if len(merged_results) < len(major_keys):
                major_keys = major_keys[keep_major_indices]

            if one_major:
                return merged_results[0]
            if wrap_results:

                def _wrap_output(_results):
                    try:
                        return pd.Series(_results, index=major_keys)
                    except Exception:
                        return pd.Series(_results, index=major_keys, dtype=object)

                if isinstance(merged_results[0], tuple):
                    return tuple(map(_wrap_output, zip(*merged_results)))
                return _wrap_output(merged_results)
            return merged_results

        if one_major:
            results = results[0]
        elif one_minor:
            results = [_results[0] for _results in results]
        if wrap_results:

            def _wrap_output(_results):
                if one_minor:
                    if attach_bounds is not None:
                        major_keys_wbounds = _attach_bounds(major_keys, minor_bounds[0])
                    else:
                        major_keys_wbounds = major_keys
                    if filter_results:
                        try:
                            _results, major_keys_wbounds = filter_out_no_results(
                                _results, keys=major_keys_wbounds
                            )
                        except NoResultsException as e:
                            if raise_no_results:
                                raise e
                            return NoResult
                    try:
                        return pd.Series(_results, index=major_keys_wbounds)
                    except Exception:
                        return pd.Series(_results, index=major_keys_wbounds, dtype=object)
                if one_major:
                    if attach_bounds is not None:
                        minor_keys_wbounds = _attach_bounds(minor_keys, major_bounds[0])
                    else:
                        minor_keys_wbounds = minor_keys
                    if filter_results:
                        try:
                            _results, major_keys_wbounds = filter_out_no_results(
                                _results, keys=minor_keys_wbounds
                            )
                        except NoResultsException as e:
                            if raise_no_results:
                                raise e
                            return NoResult
                    try:
                        return pd.Series(_results, index=minor_keys_wbounds)
                    except Exception:
                        return pd.Series(_results, index=minor_keys_wbounds, dtype=object)

                new_results = []
                keep_major_indices = []
                for i, r in enumerate(_results):
                    if attach_bounds is not None:
                        minor_keys_wbounds = _attach_bounds(minor_keys, major_bounds[i])
                    else:
                        minor_keys_wbounds = minor_keys
                    if filter_results:
                        r, minor_keys_wbounds = filter_out_no_results(
                            r,
                            keys=minor_keys_wbounds,
                            raise_error=False,
                        )
                    if len(r) > 0:
                        try:
                            new_r = pd.Series(r, index=minor_keys_wbounds)
                        except Exception:
                            new_r = pd.Series(r, index=minor_keys_wbounds, dtype=object)
                        new_results.append(new_r)
                        keep_major_indices.append(i)
                if len(new_results) == 0:
                    if raise_no_results:
                        raise NoResultsException
                    return NoResult
                if len(new_results) < len(major_keys):
                    _major_keys = major_keys[keep_major_indices]
                else:
                    _major_keys = major_keys
                try:
                    return pd.Series(new_results, index=_major_keys)
                except Exception:
                    return pd.Series(new_results, index=_major_keys, dtype=object)

            if one_major or one_minor:
                n_results = 1
                for r in results:
                    if isinstance(r, tuple):
                        n_results = len(r)
                        break
                if n_results > 1:
                    new_results = []
                    for k in range(n_results):
                        new_results.append([])
                        for i in range(len(results)):
                            if results[i] is NoResult:
                                new_results[-1].append(results[i])
                            else:
                                new_results[-1].append(results[i][k])
                    return tuple(map(_wrap_output, new_results))
            else:
                n_results = 1
                for r in results:
                    for _r in r:
                        if isinstance(_r, tuple):
                            n_results = len(_r)
                            break
                    if n_results > 1:
                        break
                if n_results > 1:
                    new_results = []
                    for k in range(n_results):
                        new_results.append([])
                        for i in range(len(results)):
                            new_results[-1].append([])
                            for j in range(len(results[0])):
                                if results[i][j] is NoResult:
                                    new_results[-1][-1].append(results[i][j])
                                else:
                                    new_results[-1][-1].append(results[i][j][k])
                    return tuple(map(_wrap_output, new_results))
            return _wrap_output(results)

        if filter_results:
            try:
                results = filter_out_no_results(results)
            except NoResultsException as e:
                if raise_no_results:
                    raise e
                return NoResult
        return results

    # ############# Splits ############# #

    def shuffle_splits(
        self: SplitterT,
        size: tp.Union[None, str, int] = None,
        replace: bool = False,
        p: tp.Optional[tp.Array1d] = None,
        seed: tp.Optional[int] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **init_kwargs,
    ) -> SplitterT:
        """Shuffle the splits by randomly selecting indices.

        Args:
            size (Union[None, str, int]): Number or specification of splits to select.

                If None, uses the total number of splits.
            replace (bool): Whether to sample with replacement.
            p (Optional[Array1d]): Probabilities associated with each split.
            seed (Optional[int]): Random seed for deterministic output.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **init_kwargs: Keyword arguments for replacing the splitter.

        Returns:
            Splitter: New splitter instance with the shuffled splits.
        """
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        rng = np.random.default_rng(seed=seed)
        if size is None:
            size = self.n_splits
        new_split_indices = rng.choice(np.arange(self.n_splits), size=size, replace=replace, p=p)
        new_splits_arr = self.splits_arr[new_split_indices]
        new_index = self.wrapper.index[new_split_indices]
        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = new_index
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    def break_up_splits(
        self: SplitterT,
        new_split: tp.SplitLike,
        sort: bool = False,
        template_context: tp.KwargsLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **split_range_kwargs,
    ) -> SplitterT:
        """Divide each split into multiple sub-splits using a new splitting specification.

        Args:
            new_split (SplitLike): Specification for splitting ranges.

                See `Splitter.split_range`.
            sort (bool): Whether to sort the resulting splits by their starting boundaries.
            template_context (KwargsLike): Additional context for template substitution.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            init_kwargs (KwargsLike): Keyword arguments for updating the splitter.
            **split_range_kwargs: Keyword arguments for `Splitter.split_range`.

        Returns:
            Splitter: New splitter instance with updated splits.

        !!! note
            Ensure that there is only one set before breaking up splits.
            Merge multiple sets into one if necessary.
        """
        if self.n_sets > 1:
            raise ValueError("Cannot break up splits with more than one set. Merge sets first.")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        split_range_kwargs = dict(split_range_kwargs)
        wrap_with_fixrange = split_range_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        split_range_kwargs["wrap_with_fixrange"] = wrap_with_fixrange

        new_splits_arr = []
        new_index = []
        range_starts = []
        for i, split in enumerate(self.splits_arr):
            new_ranges = self.split_range(
                split[0], new_split, template_context=template_context, **split_range_kwargs
            )
            for j, range_ in enumerate(new_ranges):
                if sort:
                    range_starts.append(
                        self.get_range_bounds(range_, template_context=template_context)[0]
                    )
                new_splits_arr.append([range_])
                if isinstance(self.split_labels, pd.MultiIndex):
                    new_index.append((*self.split_labels[i], j))
                else:
                    new_index.append((self.split_labels[i], j))
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)
        new_index = pd.MultiIndex.from_tuples(
            new_index, names=[*self.split_labels.names, "split_part"]
        )
        if sort:
            sorted_indices = np.argsort(range_starts)
            new_splits_arr = new_splits_arr[sorted_indices]
            new_index = new_index[sorted_indices]

        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = new_index
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    # ############# Sets ############# #

    def split_set(
        self: SplitterT,
        new_split: tp.SplitLike,
        column: tp.Optional[tp.Hashable] = None,
        new_set_labels: tp.Optional[tp.Sequence[tp.Hashable]] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **split_range_kwargs,
    ) -> SplitterT:
        """Split a set into multiple sets using a new splitting specification.

        This method applies `Splitter.split_range` to a specific column (or the only set)
        to generate new ranges.

        Args:
            new_split (SplitLike): Specification for splitting ranges.

                See `Splitter.split_range`.
            column (Optional[Hashable]): Identifier of the column to select.

                Required if multiple sets are present.
            new_set_labels (Optional[Sequence[Hashable]]): Labels to assign to the new sets.

                Must match the number of new ranges.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            init_kwargs (KwargsLike): Keyword arguments for updating the splitter.
            **split_range_kwargs: Keyword arguments for `Splitter.split_range`.

        Returns:
            Splitter: New splitter instance with the updated sets.

        !!! note
            The `column` parameter must be provided when multiple sets exist.
        """
        if self.n_sets == 0:
            raise ValueError("There are no sets to split")
        if self.n_sets > 1:
            if column is None:
                raise ValueError("Must provide column for multiple sets")
            if not isinstance(column, int):
                column = self.set_labels.get_indexer([column])[0]
                if column == -1:
                    raise ValueError(f"Column '{column}' not found")
        else:
            column = 0
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        split_range_kwargs = dict(split_range_kwargs)
        wrap_with_fixrange = split_range_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        split_range_kwargs["wrap_with_fixrange"] = wrap_with_fixrange

        new_splits_arr = []
        for split in self.splits_arr:
            new_ranges = self.split_range(split[column], new_split, **split_range_kwargs)
            new_splits_arr.append([*split[:column], *new_ranges, *split[column + 1 :]])
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)

        if "columns" not in wrapper_kwargs:
            wrapper_kwargs = dict(wrapper_kwargs)
            n_new_sets = new_splits_arr.shape[1] - self.n_sets + 1
            if new_set_labels is None:
                old_set_label = self.set_labels[column]
                if isinstance(old_set_label, str) and len(old_set_label.split("+")) == n_new_sets:
                    new_set_labels = old_set_label.split("+")
                else:
                    new_set_labels = [str(old_set_label) + "/" + str(i) for i in range(n_new_sets)]
            if len(new_set_labels) != n_new_sets:
                raise ValueError(
                    f"Argument new_set_labels must have length {n_new_sets}, not {len(new_set_labels)}"
                )
            new_columns = self.set_labels.copy()
            new_columns = new_columns.delete(column)
            new_columns = new_columns.insert(column, new_set_labels)
            wrapper_kwargs["columns"] = new_columns
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    def merge_sets(
        self: SplitterT,
        columns: tp.Optional[tp.Iterable[tp.Hashable]] = None,
        new_set_label: tp.Optional[tp.Hashable] = None,
        insert_at_last: bool = False,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **merge_split_kwargs,
    ) -> SplitterT:
        """Merge multiple sets (columns) into a single set (column).

        Args:
            columns (Optional[Iterable[Hashable]]): Columns to merge.

                If not provided, all columns are merged.
            new_set_label (Optional[Hashable]): Label for the new merged set.

                If not provided, a label is derived.
            insert_at_last (bool): If True, insert the merged set at the position of the last specified column.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            init_kwargs (KwargsLike): Keyword arguments for updating the splitter.
            **merge_split_kwargs: Keyword arguments for `Splitter.merge_split`.

        Returns:
            Splitter: New Splitter instance with the merged sets and updated splits.
        """
        if self.n_sets < 2:
            raise ValueError("There are no sets to merge")
        if columns is None:
            columns = range(len(self.set_labels))
        new_columns = []
        for column in columns:
            if not isinstance(column, int):
                column = self.set_labels.get_indexer([column])[0]
                if column == -1:
                    raise ValueError(f"Column '{column}' not found")
            new_columns.append(column)
        columns = sorted(new_columns)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}
        merge_split_kwargs = dict(merge_split_kwargs)
        wrap_with_fixrange = merge_split_kwargs.pop("wrap_with_fixrange", None)
        if isinstance(wrap_with_fixrange, bool) and not wrap_with_fixrange:
            raise ValueError("Argument wrap_with_fixrange must be True or None")
        merge_split_kwargs["wrap_with_fixrange"] = wrap_with_fixrange

        new_splits_arr = []
        for split in self.splits_arr:
            split_to_merge = []
            for j, range_ in enumerate(split):
                if j in columns:
                    split_to_merge.append(range_)
            new_range = self.merge_split(split_to_merge, **merge_split_kwargs)
            new_split = []
            for j in range(self.n_sets):
                if j not in columns:
                    new_split.append(split[j])
                else:
                    if insert_at_last:
                        if j == columns[-1]:
                            new_split.append(new_range)
                    else:
                        if j == columns[0]:
                            new_split.append(new_range)
            new_splits_arr.append(new_split)
        new_splits_arr = np.asarray(new_splits_arr, dtype=object)

        if "columns" not in wrapper_kwargs:
            wrapper_kwargs = dict(wrapper_kwargs)
            if new_set_label is None:
                old_set_labels = self.set_labels[columns]
                can_aggregate = True
                prefix = None
                suffix = None
                for i, old_set_label in enumerate(old_set_labels):
                    if not isinstance(old_set_label, str):
                        can_aggregate = False
                        break
                    _prefix = "/".join(old_set_label.split("/")[:-1])
                    _suffix = old_set_label.split("/")[-1]
                    if not _suffix.isdigit():
                        can_aggregate = False
                        break
                    _suffix = int(_suffix)
                    if prefix is None:
                        prefix = _prefix
                        suffix = _suffix
                        continue
                    if suffix != 0:
                        can_aggregate = False
                        break
                    if not _prefix == prefix or _suffix != i:
                        can_aggregate = False
                        break
                if can_aggregate and prefix + "/%d" % len(old_set_labels) not in self.set_labels:
                    new_set_label = prefix
                else:
                    new_set_label = "+".join(map(str, old_set_labels))
            new_columns = self.set_labels.copy()
            new_columns = new_columns.delete(columns)
            if insert_at_last:
                new_columns = new_columns.insert(columns[-1] - len(columns) + 1, new_set_label)
            else:
                new_columns = new_columns.insert(columns[0], new_set_label)
            wrapper_kwargs["columns"] = new_columns
        if "ndim" not in wrapper_kwargs:
            if len(wrapper_kwargs["columns"]) == 1:
                wrapper_kwargs["ndim"] = 1
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits_arr=new_splits_arr, **init_kwargs)

    # ############# Bounds ############# #

    @hybrid_method
    def map_bounds_to_index(
        cls_or_self,
        start: int,
        stop: int,
        right_inclusive: bool = False,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Tuple[tp.Any, tp.Any]:
        """Map bounds to corresponding index values.

        Args:
            start (int): Starting index for the bound.
            stop (int): Stopping index for the bound.
            right_inclusive (bool): Whether the right bound is inclusive.
            index (Optional[IndexLike]): Index to use for mapping bounds.

                If not provided, the instance index is used.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            Tuple[Any, Any]: Tuple with the mapped left and right bounds.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        if right_inclusive:
            return index[start], index[stop - 1]
        if stop == len(index):
            freq = BaseIDXAccessor(index, freq=freq).any_freq
            if freq is None:
                raise ValueError("Must provide freq")
            return index[start], index[stop - 1] + freq
        return index[start], index[stop]

    @hybrid_method
    def get_range_bounds(
        cls_or_self,
        range_: tp.FixRangeLike,
        index_bounds: bool = False,
        right_inclusive: bool = False,
        check_constant: bool = True,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Tuple[tp.Any, tp.Any]:
        """Get the inclusive left and exclusive right bounds of a range.

        Args:
            range_ (FixRangeLike): Range specification to process.
            index_bounds (bool): If True, map the bounds to the provided index.

                See `Splitter.map_bounds_to_index`.
            right_inclusive (bool): Whether the right bound is inclusive.
            check_constant (bool): If True, verify that the range is constant.
            template_context (KwargsLike): Additional context for template substitution.
            index (Optional[IndexLike]): Index used for mapping bounds.

                Required if called on a class.
            freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

                See `vectorbtpro.utils.datetime_.infer_index_freq`.

        Returns:
            Tuple[Any, Any]: Tuple with the calculated left and right bounds.

        !!! note
            Even when mapped to the index, the right bound remains exclusive.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        range_meta = cls_or_self.get_ready_range(
            range_,
            template_context=template_context,
            index=index,
            return_meta=True,
        )
        if check_constant and not range_meta["is_constant"]:
            raise ValueError("Range is not constant")
        if index_bounds:
            range_meta["start"], range_meta["stop"] = cls_or_self.map_bounds_to_index(
                range_meta["start"],
                range_meta["stop"],
                right_inclusive=right_inclusive,
                index=index,
                freq=freq,
            )
        else:
            if right_inclusive:
                range_meta["stop"] = range_meta["stop"] - 1
        return range_meta["start"], range_meta["stop"]

    def get_bounds_arr(
        self,
        index_bounds: bool = False,
        right_inclusive: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **range_bounds_kwargs,
    ) -> tp.BoundsArray:
        """Return a three-dimensional array of bounds.

        The array dimensions are:

        * First axis: splits.
        * Second axis: sets.
        * Third axis: bounds.

        Each range is selected using `Splitter.select_range` and processed using `Splitter.get_range_bounds`.
        Keyword arguments are passed to `Splitter.get_range_bounds`.

        Args:
            index_bounds (bool): If True, map the bounds to the provided index.

                See `Splitter.map_bounds_to_index`.
            right_inclusive (bool): Whether the right bound is inclusive.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            template_context (KwargsLike): Additional context for template substitution.
            **range_bounds_kwargs: Keyword arguments for `Splitter.get_range_bounds`.

        Returns:
            BoundsArray: Three-dimensional array containing the bounds.
        """
        if index_bounds:
            dtype = self.index.dtype
        else:
            dtype = int_
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)

        try:
            bounds = np.empty((n_splits, n_sets, 2), dtype=dtype)
        except TypeError:
            bounds = np.empty((n_splits, n_sets, 2), dtype=object)
        for i in range(n_splits):
            for j in range(n_sets):
                range_ = self.select_range(
                    split=PosSel(i),
                    set_=PosSel(j),
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    merge_split_kwargs=dict(template_context=template_context),
                )
                bounds[i, j, :] = self.get_range_bounds(
                    range_,
                    index_bounds=index_bounds,
                    right_inclusive=right_inclusive,
                    template_context=template_context,
                    **range_bounds_kwargs,
                )
        return bounds

    @property
    def bounds_arr(self) -> tp.BoundsArray:
        """Property returning the three-dimensional bounds array.

        This property obtains the bounds array by calling `Splitter.get_bounds_arr`
        with default parameters.

        Returns:
            BoundsArray: Three-dimensional array of bounds.
        """
        return self.get_bounds_arr()

    def get_bounds(
        self,
        index_bounds: bool = False,
        right_inclusive: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return a Series or DataFrame containing the start and end bounds.

        Args:
            index_bounds (bool): If True, map the bounds to the provided index.

                See `Splitter.map_bounds_to_index`.
            right_inclusive (bool): Whether the right bound is inclusive.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            index_combine_kwargs (KwargsLike): Keyword arguments for combining indexes.

                See `vectorbtpro.base.indexes.combine_indexes`.
            **kwargs: Keyword arguments for `Splitter.get_bounds_arr`.

        Returns:
            SeriesFrame: Pandas Series or DataFrame with index based on
                grouping and columns ['start', 'end'].
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        bounds_arr = self.get_bounds_arr(
            index_bounds=index_bounds,
            right_inclusive=right_inclusive,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        )
        out = bounds_arr.reshape((-1, 2))
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        new_columns = pd.Index(["start", "end"], name="bound")
        if one_split and one_set:
            return pd.Series(out[0], index=new_columns)
        if one_split:
            return pd.DataFrame(out, index=set_labels, columns=new_columns)
        if one_set:
            return pd.DataFrame(out, index=split_labels, columns=new_columns)
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        new_index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.DataFrame(out, index=new_index, columns=new_columns)

    @property
    def bounds(self) -> tp.Frame:
        """Bounds by calling `Splitter.get_bounds` with default arguments.

        Returns:
            Frame: Pandas DataFrame with the bounds.
        """
        return self.get_bounds()

    @property
    def index_bounds(self) -> tp.Frame:
        """Bounds computed using the index by calling `Splitter.get_bounds`
        with `index_bounds` set to True.

        Returns:
            Frame: Pandas DataFrame with the index bounds.
        """
        return self.get_bounds(index_bounds=True)

    def get_duration(self, **kwargs) -> tp.Series:
        """Return a Series representing the duration computed as the difference between
        the 'end' and 'start' bounds.

        Args:
            **kwargs: Keyword arguments for `Splitter.get_bounds`.

        Returns:
            Series: Pandas Series of durations.
        """
        bounds = self.get_bounds(right_inclusive=False, **kwargs)
        return (bounds["end"] - bounds["start"]).rename("duration")

    @property
    def duration(self) -> tp.Series:
        """Duration by calling `Splitter.get_duration` with default arguments.

        Returns:
            Series: Pandas Series of durations.
        """
        return self.get_duration()

    @property
    def index_duration(self) -> tp.Series:
        """Duration computed using index bounds by calling `Splitter.get_duration`
        with `index_bounds` set to True.

        Returns:
            Series: Pandas Series of durations.
        """
        return self.get_duration(index_bounds=True)

    # ############# Masks ############# #

    @hybrid_method
    def get_range_mask(
        cls_or_self,
        range_: tp.FixRangeLike,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Array1d:
        """Return a boolean mask array for the specified range.

        Args:
            range_ (FixRangeLike): Range specification to generate the mask.
            template_context (KwargsLike): Additional context for template substitution.
            index (Optional[IndexLike]): Index to apply the range on.

                If not provided, uses `Splitter.index`.

        Returns:
            Array1d: Boolean array mask where True indicates positions within the range.
        """
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        else:
            index = dt.prepare_dt_index(index)
        range_ = cls_or_self.get_ready_range(
            range_,
            allow_zero_len=True,
            template_context=template_context,
            index=index,
        )
        if isinstance(range_, np.ndarray) and range_.dtype == np.bool_:
            return range_
        mask = np.full(len(index), False)
        mask[range_] = True
        return mask

    def get_iter_split_mask_arrs(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Iterator[tp.Array2d]:
        """Yield two-dimensional boolean arrays for each split.

        Each array has rows representing sets and columns representing index positions.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `Splitter.get_range_mask`.

        Returns:
            Iterator[Array2d]: Iterator over two-dimensional boolean arrays.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)
        for i in range(n_splits):
            out = np.full((n_sets, len(self.index)), False)
            for j in range(n_sets):
                range_ = self.select_range(
                    split=PosSel(i),
                    set_=PosSel(j),
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    merge_split_kwargs=dict(template_context=template_context),
                )
                out[j, :] = self.get_range_mask(range_, template_context=template_context, **kwargs)
            yield out

    @property
    def iter_split_mask_arrs(self) -> tp.Iterator[tp.Array2d]:
        """Iterator over two-dimensional boolean arrays for splits by calling
        `Splitter.get_iter_split_mask_arrs` with default arguments.

        Returns:
            Iterator[Array2d]: Iterator over two-dimensional boolean arrays.
        """
        return self.get_iter_split_mask_arrs()

    def get_iter_set_mask_arrs(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Iterator[tp.Array2d]:
        """Yield two-dimensional boolean arrays for each set.

        Each array has rows representing splits and columns representing index positions.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `Splitter.get_range_mask`.

        Returns:
            Iterator[Array2d]: Iterator over two-dimensional boolean arrays.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)
        for j in range(n_sets):
            out = np.full((n_splits, len(self.index)), False)
            for i in range(n_splits):
                range_ = self.select_range(
                    split=PosSel(i),
                    set_=PosSel(j),
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    merge_split_kwargs=dict(template_context=template_context),
                )
                out[i, :] = self.get_range_mask(range_, template_context=template_context, **kwargs)
            yield out

    @property
    def iter_set_mask_arrs(self) -> tp.Iterator[tp.Array2d]:
        """Iterator over two-dimensional boolean arrays for sets by calling
        `Splitter.get_iter_set_mask_arrs` with default arguments.

        Returns:
            Iterator[Array2d]: Iterator over two-dimensional boolean arrays.
        """
        return self.get_iter_set_mask_arrs()

    def get_iter_split_masks(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Iterator[tp.Frame]:
        """Yield boolean DataFrames for each split.

        Each DataFrame is constructed by transposing the mask array and applying
        appropriate index and set labels.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            **kwargs: Keyword arguments for `Splitter.get_iter_split_mask_arrs`.

        Returns:
            Iterator[Frame]: Iterator over boolean DataFrames.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        for mask in self.get_iter_split_mask_arrs(
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        ):
            yield pd.DataFrame(np.moveaxis(mask, -1, 0), index=self.index, columns=set_labels)

    @property
    def iter_split_masks(self) -> tp.Iterator[tp.Frame]:
        """Iterator over boolean DataFrames for splits by calling
        `Splitter.get_iter_split_masks` with default arguments.

        Returns:
            Iterator[Frame]: Iterator over boolean DataFrames.
        """
        return self.get_iter_split_masks()

    def get_iter_set_masks(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Iterator[tp.Frame]:
        """Yield boolean DataFrames for each set.

        Each DataFrame is constructed by transposing the mask array and
        applying appropriate index and split labels.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            **kwargs: Keyword arguments for `Splitter.get_iter_set_mask_arrs`.

        Returns:
            Iterator[Frame]: Iterator over boolean DataFrames.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        for mask in self.get_iter_set_mask_arrs(
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        ):
            yield pd.DataFrame(np.moveaxis(mask, -1, 0), index=self.index, columns=split_labels)

    @property
    def iter_set_masks(self) -> tp.Iterator[tp.Frame]:
        """Iterator over boolean DataFrames for sets by calling
        `Splitter.get_iter_set_masks` with default arguments.

        Returns:
            Iterator[Frame]: Iterator over boolean DataFrames.
        """
        return self.get_iter_set_masks()

    def get_mask_arr(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SplitsMask:
        """Return a three-dimensional boolean array representing splits.

        The first dimension corresponds to splits, the second to sets, and the third to the index.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `Splitter.get_iter_split_mask_arrs`.

        Returns:
            SplitsMask: Three-dimensional boolean array representing the split mask.
        """
        return np.array(
            list(
                self.get_iter_split_mask_arrs(
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    template_context=template_context,
                    **kwargs,
                )
            )
        )

    @property
    def mask_arr(self) -> tp.SplitsMask:
        """Split mask array computed with default arguments from `Splitter.get_mask_arr`.

        Returns:
            SplitsMask: Three-dimensional boolean array representing the split mask.
        """
        return self.get_mask_arr()

    def get_mask(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Return a boolean Series or DataFrame representing the split mask.

        The returned object uses `Splitter.index` as the index and contains the splits as columns.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            index_combine_kwargs (KwargsLike): Keyword arguments for combining indexes.

                See `vectorbtpro.base.indexes.combine_indexes`.
            **kwargs: Keyword arguments for `Splitter.get_mask_arr`.

        Returns:
            SeriesFrame: Pandas Series or DataFrame representing the split mask.

        !!! warning
            Boolean arrays for a high number of splits may consume substantial memory.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(
            split_group_by=split_group_by, set_group_by=set_group_by, **kwargs
        )
        out = np.moveaxis(mask_arr, -1, 0).reshape((len(self.index), -1))
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        if one_split and one_set:
            return pd.Series(out[:, 0], index=self.index)
        if one_split:
            return pd.DataFrame(out, index=self.index, columns=set_labels)
        if one_set:
            return pd.DataFrame(out, index=self.index, columns=split_labels)
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        new_columns = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.DataFrame(out, index=self.index, columns=new_columns)

    @property
    def mask(self) -> tp.Frame:
        """Boolean mask computed with default parameters from `Splitter.get_mask`.

        Returns:
            Frame: Pandas DataFrame representing the split mask.
        """
        return self.get_mask()

    def get_split_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the coverage of each split mask.

        Coverage is calculated based on the provided parameters:

        * If `overlapping` is True, compute the count of overlapping True values
            between sets for each split.
        * If `normalize` is True, the count is normalized by the length of the index.
        * If both `normalize` and `relative` are True, compute the coverage relative
            to the total True count across all splits.

        Args:
            overlapping (bool): Whether to compute overlapping True values between sets.
            normalize (bool): Whether to normalize the coverage by the index length.
            relative (bool): When normalized, whether to compute coverage relative to the overall True count.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            **kwargs: Keyword arguments for `Splitter.get_mask_arr`.

        Returns:
            MaybeSeries: Coverage of each split, either as a scalar or as a Series indexed by split labels.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(
            split_group_by=split_group_by, set_group_by=set_group_by, **kwargs
        )
        if overlapping:
            if normalize:
                coverage = (mask_arr.sum(axis=1) > 1).sum(axis=1) / mask_arr.any(axis=1).sum(axis=1)
            else:
                coverage = (mask_arr.sum(axis=1) > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = mask_arr.any(axis=1).sum(axis=1) / mask_arr.any(axis=(0, 1)).sum()
                else:
                    coverage = mask_arr.any(axis=1).mean(axis=1)
            else:
                coverage = mask_arr.any(axis=1).sum(axis=1)
        one_split = len(split_labels) == 1 and squeeze_one_split
        if one_split:
            return coverage[0]
        return pd.Series(coverage, index=split_labels, name="split_coverage")

    @property
    def split_coverage(self) -> tp.Series:
        """Split coverage computed with default parameters from `Splitter.get_split_coverage`.

        Returns:
            Series: Pandas Series of split coverage.
        """
        return self.get_split_coverage()

    def get_set_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_set: bool = True,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Return the coverage of each set mask.

        Coverage is calculated based on the provided parameters:

        * If `overlapping` is True, compute the count of overlapping True values
            between splits for each set.
        * If `normalize` is True, the count is normalized by the length of the index.
        * If both `normalize` and `relative` are True, compute the coverage relative
            to the total True count across all sets.

        Args:
            overlapping (bool): Whether to compute overlapping True values between splits.
            normalize (bool): Whether to normalize the coverage by the index length.
            relative (bool): When normalized, whether to compute coverage relative to the overall True count.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            **kwargs: Keyword arguments for `Splitter.get_mask_arr`.

        Returns:
            MaybeSeries: Coverage for each set, either as a scalar or as a Series indexed by set labels.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(
            split_group_by=split_group_by, set_group_by=set_group_by, **kwargs
        )
        if overlapping:
            if normalize:
                coverage = (mask_arr.sum(axis=0) > 1).sum(axis=1) / mask_arr.any(axis=0).sum(axis=1)
            else:
                coverage = (mask_arr.sum(axis=0) > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = mask_arr.any(axis=0).sum(axis=1) / mask_arr.any(axis=(0, 1)).sum()
                else:
                    coverage = mask_arr.any(axis=0).mean(axis=1)
            else:
                coverage = mask_arr.any(axis=0).sum(axis=1)
        one_set = len(set_labels) == 1 and squeeze_one_set
        if one_set:
            return coverage[0]
        return pd.Series(coverage, index=set_labels, name="set_coverage")

    @property
    def set_coverage(self) -> tp.Series:
        """Set coverage computed with default parameters from `Splitter.get_set_coverage`.

        Returns:
            Series: Pandas Series of set coverage.
        """
        return self.get_set_coverage()

    def get_range_coverage(
        self,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get the coverage of each range mask.

        Args:
            normalize (bool): Flag to determine if coverage should be normalized
                relative to the index length.
            relative (bool): If True and normalization is enabled, compute coverage relative
                to the total True values in its split.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            index_combine_kwargs (KwargsLike): Keyword arguments for combining indexes.

                See `vectorbtpro.base.indexes.combine_indexes`.
            **kwargs: Keyword arguments for `Splitter.get_mask_arr`.

        Returns:
            MaybeSeries: Coverage values for each range mask, returned as a scalar or a Pandas Series.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(
            split_group_by=split_group_by, set_group_by=set_group_by, **kwargs
        )
        if normalize:
            if relative:
                coverage = (
                    mask_arr.sum(axis=2) / mask_arr.any(axis=1).sum(axis=1)[:, None]
                ).flatten()
            else:
                coverage = (mask_arr.sum(axis=2) / mask_arr.shape[2]).flatten()
        else:
            coverage = mask_arr.sum(axis=2).flatten()
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        if one_split and one_set:
            return coverage[0]
        if one_split:
            return pd.Series(coverage, index=set_labels, name="range_coverage")
        if one_set:
            return pd.Series(coverage, index=split_labels, name="range_coverage")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.Series(coverage, index=index, name="range_coverage")

    @property
    def range_coverage(self) -> tp.Series:
        """Range coverage computed using default parameters from `Splitter.get_range_coverage`.

        Returns:
            Series: Pandas Series of range coverage.
        """
        return self.get_range_coverage()

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> float:
        """Get the coverage of the entire mask.

        Args:
            overlapping (bool): Flag to compute overlapping coverage by counting overlapping True values.
            normalize (bool): Flag to normalize the coverage relative to the index length.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            **kwargs: Keyword arguments for `Splitter.get_mask_arr`.

        Returns:
            float: Coverage value computed based on the provided mask and parameters.
        """
        mask_arr = self.get_mask_arr(
            split_group_by=split_group_by, set_group_by=set_group_by, **kwargs
        )
        if overlapping:
            if normalize:
                return (mask_arr.sum(axis=(0, 1)) > 1).sum() / mask_arr.any(axis=(0, 1)).sum()
            return (mask_arr.sum(axis=(0, 1)) > 1).sum()
        if normalize:
            return mask_arr.any(axis=(0, 1)).mean()
        return mask_arr.any(axis=(0, 1)).sum()

    @property
    def coverage(self) -> float:
        """Coverage computed using default parameters from `Splitter.get_coverage`.

        Returns:
            float: Coverage value.
        """
        return self.get_coverage()

    def get_overlap_matrix(
        self,
        by: str = "split",
        normalize: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        jitted: tp.JittedOption = None,
        squeeze_one_split: bool = True,
        squeeze_one_set: bool = True,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Frame:
        """Get the overlap matrix between each pair of ranges.

        Args:
            by (str): Specifies which overlap matrix to compute; must be one of "split", "set", or "range".
            normalize (bool): Flag indicating whether to normalize overlaps relative to the
                total True values in both ranges.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            squeeze_one_split (bool): Whether to squeeze the output if only one split exists.
            squeeze_one_set (bool): Whether to squeeze the output if only one set exists.
            index_combine_kwargs (KwargsLike): Keyword arguments for combining indexes.

                See `vectorbtpro.base.indexes.combine_indexes`.
            **kwargs: Keyword arguments for `Splitter.get_mask_arr`.

        Returns:
            Frame: DataFrame representing the computed overlap matrix, or a scalar if the result is squeezed.

        See:
            * `vectorbtpro.generic.splitting.nb.norm_split_overlap_matrix_nb` for `by="split"` and `normalize=True`.
            * `vectorbtpro.generic.splitting.nb.split_overlap_matrix_nb` for `by="split"` and `normalize=False`.
            * `vectorbtpro.generic.splitting.nb.norm_set_overlap_matrix_nb` for `by="set"` and `normalize=True`.
            * `vectorbtpro.generic.splitting.nb.set_overlap_matrix_nb` for `by="set"` and `normalize=False`.
            * `vectorbtpro.generic.splitting.nb.norm_range_overlap_matrix_nb` for `by="range"` and `normalize=True`.
            * `vectorbtpro.generic.splitting.nb.range_overlap_matrix_nb` for `by="range"` and `normalize=False`.
        """
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(
            split_group_by=split_group_by, set_group_by=set_group_by, **kwargs
        )
        one_split = len(split_labels) == 1 and squeeze_one_split
        one_set = len(set_labels) == 1 and squeeze_one_set
        if by.lower() == "split":
            if normalize:
                func = jit_reg.resolve_option(nb.norm_split_overlap_matrix_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.split_overlap_matrix_nb, jitted)
            overlap_matrix = func(mask_arr)
            if one_split:
                return overlap_matrix[0, 0]
            index = split_labels
        elif by.lower() == "set":
            if normalize:
                func = jit_reg.resolve_option(nb.norm_set_overlap_matrix_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.set_overlap_matrix_nb, jitted)
            overlap_matrix = func(mask_arr)
            if one_set:
                return overlap_matrix[0, 0]
            index = set_labels
        elif by.lower() == "range":
            if normalize:
                func = jit_reg.resolve_option(nb.norm_range_overlap_matrix_nb, jitted)
            else:
                func = jit_reg.resolve_option(nb.range_overlap_matrix_nb, jitted)
            overlap_matrix = func(mask_arr)
            if one_split and one_set:
                return overlap_matrix[0, 0]
            if one_split:
                index = set_labels
            elif one_set:
                index = split_labels
            else:
                if index_combine_kwargs is None:
                    index_combine_kwargs = {}
                index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        else:
            raise ValueError(f"Invalid by: '{by}'")
        return pd.DataFrame(overlap_matrix, index=index, columns=index)

    @property
    def split_overlap_matrix(self) -> tp.Frame:
        """Overlap matrix computed with `get_overlap_matrix` using `by="split"`.

        Returns:
            Frame: DataFrame representing the split overlap matrix.
        """
        return self.get_overlap_matrix(by="split")

    @property
    def set_overlap_matrix(self) -> tp.Frame:
        """Overlap matrix computed with `get_overlap_matrix` using `by="set"`.

        Returns:
            Frame: DataFrame representing the set overlap matrix.
        """
        return self.get_overlap_matrix(by="set")

    @property
    def range_overlap_matrix(self) -> tp.Frame:
        """Overlap matrix computed with `get_overlap_matrix` using `by="range"`.

        Returns:
            Frame: DataFrame representing the range overlap matrix.
        """
        return self.get_overlap_matrix(by="range")

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `Splitter.stats`.

        Merges the defaults from `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.splitter`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        from vectorbtpro._settings import settings

        splitter_stats_cfg = settings["splitter"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), splitter_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Index Start",
                calc_func=lambda self: self.index[0],
                agg_func=None,
                tags=["splitter", "index"],
            ),
            end=dict(
                title="Index End",
                calc_func=lambda self: self.index[-1],
                agg_func=None,
                tags=["splitter", "index"],
            ),
            period=dict(
                title="Index Length",
                calc_func=lambda self: len(self.index),
                agg_func=None,
                tags=["splitter", "index"],
            ),
            split_count=dict(
                title="Splits",
                calc_func="n_splits",
                agg_func=None,
                tags=["splitter", "splits"],
            ),
            set_count=dict(
                title="Sets",
                calc_func="n_sets",
                agg_func=None,
                tags=["splitter", "splits"],
            ),
            coverage=dict(
                title=RepFunc(lambda normalize: "Coverage [%]" if normalize else "Coverage"),
                calc_func="coverage",
                overlapping=False,
                post_calc_func=lambda self, out, settings: out * 100
                if settings["normalize"]
                else out,
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_coverage=dict(
                title=RepFunc(lambda normalize: "Coverage [%]" if normalize else "Coverage"),
                check_has_multiple_sets=True,
                calc_func="set_coverage",
                overlapping=False,
                relative=False,
                post_calc_func=lambda self, out, settings: to_dict(
                    out * 100 if settings["normalize"] else out, orient="index_series"
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_mean_rel_coverage=dict(
                title="Mean Rel Coverage [%]",
                check_has_multiple_sets=True,
                check_normalize=True,
                calc_func="range_coverage",
                relative=True,
                post_calc_func=lambda self, out, settings: to_dict(
                    out.groupby(
                        self.get_set_labels(set_group_by=settings.get("set_group_by", None)).names
                    ).mean()[self.get_set_labels(set_group_by=settings.get("set_group_by", None))]
                    * 100,
                    orient="index_series",
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            overlap_coverage=dict(
                title=RepFunc(
                    lambda normalize: "Overlap Coverage [%]" if normalize else "Overlap Coverage"
                ),
                calc_func="coverage",
                overlapping=True,
                post_calc_func=lambda self, out, settings: out * 100
                if settings["normalize"]
                else out,
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_overlap_coverage=dict(
                title=RepFunc(
                    lambda normalize: "Overlap Coverage [%]" if normalize else "Overlap Coverage"
                ),
                check_has_multiple_sets=True,
                calc_func="set_coverage",
                overlapping=True,
                post_calc_func=lambda self, out, settings: to_dict(
                    out * 100 if settings["normalize"] else out, orient="index_series"
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        mask_kwargs: tp.KwargsLike = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot splits as rows with sets represented by distinct colors.

        Args:
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            mask_kwargs (KwargsLike): Keyword arguments for `Splitter.get_iter_set_masks`.
            trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Heatmap` for the mask.

                Can be a sequence, one per set.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure to which traces were added.

        Examples:
            Plot a scikit-learn splitter:

            ```pycon
            >>> from vectorbtpro import *
            >>> from sklearn.model_selection import TimeSeriesSplit

            >>> index = pd.date_range("2020", "2021", freq="D")
            >>> splitter = vbt.Splitter.from_sklearn(index, TimeSeriesSplit())
            >>> splitter.plot().show()
            ```

            ![](/assets/images/api/Splitter.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/Splitter.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.express as px

        from vectorbtpro.utils.figure import make_figure

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        if fig.layout.colorway is not None:
            colorway = fig.layout.colorway
        else:
            colorway = fig.layout.template.layout.colorway
        if len(set_labels) > len(colorway):
            colorway = px.colors.qualitative.Alphabet

        if self.get_n_splits(split_group_by=split_group_by) and self.get_n_sets(
            set_group_by=set_group_by
        ):
            if mask_kwargs is None:
                mask_kwargs = {}
            for i, mask in enumerate(
                self.get_iter_set_masks(
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    **mask_kwargs,
                )
            ):
                df = mask.vbt.wrapper.fill()
                df[mask] = i
                color = adjust_opacity(colorway[i % len(colorway)], 0.8)
                trace_name = str(set_labels[i])
                _trace_kwargs = merge_dicts(
                    dict(
                        showscale=False,
                        showlegend=True,
                        legendgroup=str(set_labels[i]),
                        name=trace_name,
                        colorscale=[color, color],
                        hovertemplate="%{x}<br>Split: %{y}<br>Set: " + trace_name,
                    ),
                    resolve_dict(trace_kwargs, i=i),
                )
                fig = df.vbt.ts_heatmap(
                    trace_kwargs=_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    is_y_category=True,
                    fig=fig,
                )
        return fig

    def plot_coverage(
        self,
        stacked: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        mask_kwargs: tp.KwargsLike = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot index coverage as rows and sets as lines.

        This method generates a plot where each index is represented as a row and
        each set is shown as a line. A stacked area plot is created if `stacked` is True;
        otherwise, a line plot is produced.

        Args:
            stacked (bool): Plot using a stacked area plot if True; otherwise, use a line plot.
            split_group_by (AnyGroupByLike): Grouping specification for defining splits.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (AnyGroupByLike): Grouping specification for defining sets.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            mask_kwargs (KwargsLike): Keyword arguments for `Splitter.get_iter_set_masks`.
            trace_kwargs (KwargsLikeSequence): Keyword arguments for `plotly.graph_objects.Scatter` for the mask.

                If provided as a sequence, each set uses its corresponding settings.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                for example, `dict(row=1, col=1)`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: Figure to which traces were added.

        Examples:
            Area plot:

            ```pycon
            >>> from vectorbtpro import *
            >>> from sklearn.model_selection import TimeSeriesSplit

            >>> index = pd.date_range("2020", "2021", freq="D")
            >>> splitter = vbt.Splitter.from_sklearn(index, TimeSeriesSplit())
            >>> splitter.plot_coverage().show()
            ```

            ![](/assets/images/api/Splitter_coverage_area.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/Splitter_coverage_area.dark.svg#only-dark){: .iimg loading=lazy }

            Line plot:

            ```pycon
            >>> splitter.plot_coverage(stacked=False).show()
            ```

            ![](/assets/images/api/Splitter_coverage_line.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/Splitter_coverage_line.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.express as px

        from vectorbtpro.utils.figure import make_figure

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        if fig.layout.colorway is not None:
            colorway = fig.layout.colorway
        else:
            colorway = fig.layout.template.layout.colorway
        if len(set_labels) > len(colorway):
            colorway = px.colors.qualitative.Alphabet

        if self.get_n_splits(split_group_by=split_group_by) > 0:
            if self.get_n_sets(set_group_by=set_group_by) > 0:
                if mask_kwargs is None:
                    mask_kwargs = {}
                for i, mask in enumerate(
                    self.get_iter_set_masks(
                        split_group_by=split_group_by,
                        set_group_by=set_group_by,
                        **mask_kwargs,
                    )
                ):
                    _trace_kwargs = merge_dicts(
                        dict(
                            stackgroup="coverage" if stacked else None,
                            legendgroup=str(set_labels[i]),
                            name=str(set_labels[i]),
                            line=dict(color=colorway[i % len(colorway)], shape="hv"),
                        ),
                        resolve_dict(trace_kwargs, i=i),
                    )
                    fig = mask.sum(axis=1).vbt.lineplot(
                        trace_kwargs=_trace_kwargs,
                        add_trace_kwargs=add_trace_kwargs,
                        fig=fig,
                    )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `Splitter.plots`.

        Merges the defaults from `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.splitter`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        from vectorbtpro._settings import settings

        splitter_plots_cfg = settings["splitter"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), splitter_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Splits",
                yaxis_kwargs=dict(title="Split"),
                plot_func="plot",
                tags="splitter",
            ),
            plot_coverage=dict(
                title="Coverage",
                yaxis_kwargs=dict(title="Count"),
                plot_func="plot_coverage",
                tags="splitter",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Splitter.override_metrics_doc(__pdoc__)
Splitter.override_subplots_doc(__pdoc__)
