# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a mixin class for simulation range operations."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.indexing import AutoIdxr
from vectorbtpro.base.reshaping import broadcast_array_to
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb
from vectorbtpro.utils import checks
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.decorators import hybrid_method

SimRangeMixinT = tp.TypeVar("SimRangeMixinT", bound="SimRangeMixin")


class SimRangeMixin(Base):
    """Mixin class for working with simulation ranges.

    Should be subclassed by a subclass of `vectorbtpro.base.wrapping.Wrapping`.

    Args:
        sim_start (Optional[ArrayLike]): Start index of the simulation range.
        sim_end (Optional[ArrayLike]): End index of the simulation range.
    """

    def __init__(
        self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
    ) -> None:
        sim_start = type(self).resolve_sim_start(
            sim_start=sim_start, wrapper=self.wrapper, group_by=False
        )
        sim_end = type(self).resolve_sim_end(sim_end=sim_end, wrapper=self.wrapper, group_by=False)

        self._sim_start = sim_start
        self._sim_end = sim_end

    @classmethod
    def row_stack_sim_start(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeSequence[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Row-stack simulation start.

        Args:
            new_wrapper (ArrayWrapper): New array wrapper.
            *objs (MaybeSequence[SimRangeMixin]): Simulation range mixin objects to merge.

                The first object's simulation start is used, and all subsequent
                objects must have it set to None.

        Returns:
            Optional[ArrayLike]: Computed simulation start array for row stacking, or None.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        if objs[0]._sim_start is not None:
            new_sim_start = broadcast_array_to(objs[0]._sim_start, len(new_wrapper.columns))
        else:
            new_sim_start = None
        for obj in objs[1:]:
            if obj._sim_start is not None:
                raise ValueError(
                    "Objects to be merged (except the first one) must have 'sim_start=None'"
                )
        return new_sim_start

    @classmethod
    def row_stack_sim_end(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeSequence[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Row-stack simulation end.

        Args:
            new_wrapper (ArrayWrapper): New array wrapper.
            *objs (MaybeSequence[SimRangeMixin]): Simulation range mixin objects to merge.

                The last object's simulation end is used, and all preceding objects must have it set to None.

        Returns:
            Optional[ArrayLike]: Computed simulation end array for row stacking, or None.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        if objs[-1]._sim_end is not None:
            new_sim_end = len(new_wrapper.index) - len(objs[-1].wrapper.index) + objs[-1]._sim_end
            new_sim_end = broadcast_array_to(new_sim_end, len(new_wrapper.columns))
        else:
            new_sim_end = None
        for obj in objs[:-1]:
            if obj._sim_end is not None:
                raise ValueError(
                    "Objects to be merged (except the last one) must have 'sim_end=None'"
                )
        return new_sim_end

    @classmethod
    def column_stack_sim_start(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeSequence[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Column-stack simulation start.

        Args:
            new_wrapper (ArrayWrapper): New array wrapper.
            *objs (MaybeSequence[SimRangeMixin]): Simulation range mixin objects to merge.

        Returns:
            Optional[ArrayLike]: Computed simulation start array for column stacking, or None.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        stack_sim_start_objs = False
        for obj in objs:
            if obj._sim_start is not None:
                stack_sim_start_objs = True
                break
        if stack_sim_start_objs:
            obj_sim_starts = []
            for obj in objs:
                obj_sim_start = np.empty(len(obj._sim_start), dtype=int_)
                for i in range(len(obj._sim_start)):
                    if obj._sim_start[i] == 0:
                        obj_sim_start[i] = 0
                    elif obj._sim_start[i] == len(obj.wrapper.index):
                        obj_sim_start[i] = len(new_wrapper.index)
                    else:
                        _obj_sim_start = new_wrapper.index.get_indexer(
                            [obj.wrapper.index[obj._sim_start[i]]]
                        )[0]
                        if _obj_sim_start == -1:
                            _obj_sim_start = 0
                        obj_sim_start[i] = _obj_sim_start
                obj_sim_starts.append(obj_sim_start)
            new_sim_start = new_wrapper.concat_arrs(*obj_sim_starts, wrap=False)
        else:
            new_sim_start = None
        return new_sim_start

    @classmethod
    def column_stack_sim_end(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeSequence[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Column-stack simulation end.

        Args:
            new_wrapper (ArrayWrapper): New array wrapper.
            *objs (MaybeSequence[SimRangeMixin]): Simulation range mixin objects to merge.

        Returns:
            Optional[ArrayLike]: Computed simulation end array for column stacking, or None.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        stack_sim_end_objs = False
        for obj in objs:
            if obj._sim_end is not None:
                stack_sim_end_objs = True
                break
        if stack_sim_end_objs:
            obj_sim_ends = []
            for obj in objs:
                obj_sim_end = np.empty(len(obj._sim_end), dtype=int_)
                for i in range(len(obj._sim_end)):
                    if obj._sim_end[i] == 0:
                        obj_sim_end[i] = 0
                    elif obj._sim_end[i] == len(obj.wrapper.index):
                        obj_sim_end[i] = len(new_wrapper.index)
                    else:
                        _obj_sim_end = new_wrapper.index.get_indexer(
                            [obj.wrapper.index[obj._sim_end[i]]]
                        )[0]
                        if _obj_sim_end == -1:
                            _obj_sim_end = 0
                        obj_sim_end[i] = _obj_sim_end
                obj_sim_ends.append(obj_sim_end)
            new_sim_end = new_wrapper.concat_arrs(*obj_sim_ends, wrap=False)
        else:
            new_sim_end = None
        return new_sim_end

    def sim_start_indexing_func(self, wrapper_meta: dict) -> tp.Optional[tp.ArrayLike]:
        """Index simulation start.

        Args:
            wrapper_meta (dict): Metadata from the indexing operation on the wrapper.

        Returns:
            Optional[ArrayLike]: Updated simulation start positions array,
                or None if simulation start is not defined.
        """
        if self._sim_start is None:
            new_sim_start = None
        elif not wrapper_meta["rows_changed"]:
            new_sim_start = self._sim_start
        else:
            if checks.is_int(wrapper_meta["row_idxs"]):
                new_sim_start = self._sim_start - wrapper_meta["row_idxs"]
            elif isinstance(wrapper_meta["row_idxs"], slice):
                new_sim_start = self._sim_start - wrapper_meta["row_idxs"].start
            else:
                new_sim_start = self._sim_start - wrapper_meta["row_idxs"][0]
            new_sim_start = np.clip(new_sim_start, 0, len(wrapper_meta["new_wrapper"].index))
        return new_sim_start

    def sim_end_indexing_func(self, wrapper_meta: dict) -> tp.Optional[tp.ArrayLike]:
        """Index simulation end.

        Args:
            wrapper_meta (dict): Metadata from the indexing operation on the wrapper.

        Returns:
            Optional[ArrayLike]: Updated simulation end positions array,
                or None if simulation end is not defined.
        """
        if self._sim_end is None:
            new_sim_end = None
        elif not wrapper_meta["rows_changed"]:
            new_sim_end = self._sim_end
        else:
            if checks.is_int(wrapper_meta["row_idxs"]):
                new_sim_end = self._sim_end - wrapper_meta["row_idxs"]
            elif isinstance(wrapper_meta["row_idxs"], slice):
                new_sim_end = self._sim_end - wrapper_meta["row_idxs"].start
            else:
                new_sim_end = self._sim_end - wrapper_meta["row_idxs"][0]
            new_sim_end = np.clip(new_sim_end, 0, len(wrapper_meta["new_wrapper"].index))
        return new_sim_end

    def resample_sim_start(self, new_wrapper: ArrayWrapper) -> tp.Optional[tp.ArrayLike]:
        """Resample simulation start indices based on a new array wrapper.

        Args:
            new_wrapper (ArrayWrapper): New array wrapper.

        Returns:
            Optional[ArrayLike]: Resampled simulation start array,
                or None if simulation start is not defined.
        """
        if self._sim_start is not None:
            new_sim_start = np.empty(len(self._sim_start), dtype=int_)
            for i in range(len(self._sim_start)):
                if self._sim_start[i] == 0:
                    new_sim_start[i] = 0
                elif self._sim_start[i] == len(self.wrapper.index):
                    new_sim_start[i] = len(new_wrapper.index)
                else:
                    _new_sim_start = new_wrapper.index.get_indexer(
                        [self.wrapper.index[self._sim_start[i]]],
                        method="ffill",
                    )[0]
                    if _new_sim_start == -1:
                        _new_sim_start = 0
                    new_sim_start[i] = _new_sim_start
        else:
            new_sim_start = None
        return new_sim_start

    def resample_sim_end(self, new_wrapper: ArrayWrapper) -> tp.Optional[tp.ArrayLike]:
        """Resample simulation end indices based on a new array wrapper.

        Args:
            new_wrapper (ArrayWrapper): New array wrapper.

        Returns:
            Optional[ArrayLike]: Resampled simulation end array,
                or None if simulation end is not defined.
        """
        if self._sim_end is not None:
            new_sim_end = np.empty(len(self._sim_end), dtype=int_)
            for i in range(len(self._sim_end)):
                if self._sim_end[i] == 0:
                    new_sim_end[i] = 0
                elif self._sim_end[i] == len(self.wrapper.index):
                    new_sim_end[i] = len(new_wrapper.index)
                else:
                    _new_sim_end = new_wrapper.index.get_indexer(
                        [self.wrapper.index[self._sim_end[i]]],
                        method="bfill",
                    )[0]
                    if _new_sim_end == -1:
                        _new_sim_end = len(new_wrapper.index)
                    new_sim_end[i] = _new_sim_end
        else:
            new_sim_end = None
        return new_sim_end

    @hybrid_method
    def resolve_sim_start_value(
        cls_or_self,
        value: tp.Scalar,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> int:
        """Return the resolved simulation start position for the provided scalar value.

        Args:
            value (Scalar): Scalar simulation start value to resolve.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

        Returns:
            int: Resolved position corresponding to the simulation start.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        auto_idxr = AutoIdxr(value, indexer_method="bfill", below_to_zero=True)
        return auto_idxr.get(wrapper.index, freq=wrapper.freq)

    @hybrid_method
    def resolve_sim_end_value(
        cls_or_self,
        value: tp.Scalar,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> int:
        """Return the resolved simulation end position for the provided scalar value.

        Args:
            value (Scalar): Scalar simulation end value to resolve.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

        Returns:
            int: Resolved position corresponding to the simulation end.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        auto_idxr = AutoIdxr(value, indexer_method="bfill", above_to_len=True)
        return auto_idxr.get(wrapper.index, freq=wrapper.freq)

    @hybrid_method
    def resolve_sim_start(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Optional[tp.ArrayLike]:
        """Resolve simulation start positions.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            allow_none (bool): Allow simulation positions to be None when applicable.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Optional[ArrayLike]: Resolved simulation start positions, or None if the input is treated as None.
        """
        already_resolved = False
        if not isinstance(cls_or_self, type):
            if sim_start is None:
                sim_start = cls_or_self._sim_start
                already_resolved = True
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if sim_start is False:
            sim_start = None
        if allow_none and sim_start is None:
            return None
        if not already_resolved and sim_start is not None:
            sim_start_arr = np.asarray(sim_start)
            if not np.issubdtype(sim_start_arr.dtype, np.integer):
                if sim_start_arr.ndim == 0:
                    sim_start = cls_or_self.resolve_sim_start_value(sim_start, wrapper=wrapper)
                else:
                    new_sim_start = np.empty(len(sim_start), dtype=int_)
                    for i in range(len(sim_start)):
                        new_sim_start[i] = cls_or_self.resolve_sim_start_value(
                            sim_start[i], wrapper=wrapper
                        )
                    sim_start = new_sim_start
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            sim_start = nb.resolve_grouped_sim_start_nb(
                wrapper.shape_2d,
                group_lens,
                sim_start=sim_start,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        elif not already_resolved and wrapper.grouper.is_grouped():
            group_lens = wrapper.grouper.get_group_lens()
            sim_start = nb.resolve_ungrouped_sim_start_nb(
                wrapper.shape_2d,
                group_lens,
                sim_start=sim_start,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        else:
            sim_start = nb.resolve_sim_start_nb(
                wrapper.shape_2d,
                sim_start=sim_start,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        return sim_start

    @hybrid_method
    def resolve_sim_end(
        cls_or_self,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Optional[tp.ArrayLike]:
        """Resolve simulation end positions.

        Args:
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            allow_none (bool): Allow simulation positions to be None when applicable.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Optional[ArrayLike]: Resolved simulation end positions, or None if the input is treated as None.
        """
        already_resolved = False
        if not isinstance(cls_or_self, type):
            if sim_end is None:
                sim_end = cls_or_self._sim_end
                already_resolved = True
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if sim_end is False:
            sim_end = None
        if allow_none and sim_end is None:
            return None
        if not already_resolved and sim_end is not None:
            sim_end_arr = np.asarray(sim_end)
            if not np.issubdtype(sim_end_arr.dtype, np.integer):
                if sim_end_arr.ndim == 0:
                    sim_end = cls_or_self.resolve_sim_end_value(sim_end, wrapper=wrapper)
                else:
                    new_sim_end = np.empty(len(sim_end), dtype=int_)
                    for i in range(len(sim_end)):
                        new_sim_end[i] = cls_or_self.resolve_sim_end_value(
                            sim_end[i], wrapper=wrapper
                        )
                    sim_end = new_sim_end
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            sim_end = nb.resolve_grouped_sim_end_nb(
                wrapper.shape_2d,
                group_lens,
                sim_end=sim_end,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        elif not already_resolved and wrapper.grouper.is_grouped():
            group_lens = wrapper.grouper.get_group_lens()
            sim_end = nb.resolve_ungrouped_sim_end_nb(
                wrapper.shape_2d,
                group_lens,
                sim_end=sim_end,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        else:
            sim_end = nb.resolve_sim_end_nb(
                wrapper.shape_2d,
                sim_end=sim_end,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        return sim_end

    @hybrid_method
    def get_sim_start(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        keep_flex: bool = False,
        allow_none: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, tp.Array1d, tp.Series]:
        """Return the simulation start positions after resolution and optional wrapping.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            keep_flex (bool): Whether to preserve the flexible array structure.
            allow_none (bool): Allow simulation positions to be None when applicable.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            Union[None, Array1d, Series]: Final simulation start result, either wrapped or raw.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.resolve_sim_start(
            sim_start=sim_start,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_start is None:
            return None
        if keep_flex:
            return sim_start
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_end"), wrap_kwargs)
        return wrapper.wrap_reduced(sim_start, group_by=group_by, **wrap_kwargs)

    @property
    def sim_start(self) -> tp.Series:
        """Simulation start value computed using default parameters from `SimRangeMixin.get_sim_start`.

        Returns:
            Series: Simulation start series.
        """
        return self.get_sim_start()

    @hybrid_method
    def get_sim_end(
        cls_or_self,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        keep_flex: bool = False,
        allow_none: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, tp.Array1d, tp.Series]:
        """Return the simulation end positions after resolution and optional wrapping.

        Args:
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            keep_flex (bool): Whether to preserve the flexible array structure.
            allow_none (bool): Allow simulation positions to be None when applicable.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            Union[None, Array1d, Series]: Final simulation end result, either wrapped or raw.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_end = cls_or_self.resolve_sim_end(
            sim_end=sim_end,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_end is None:
            return None
        if keep_flex:
            return sim_end
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_start"), wrap_kwargs)
        return wrapper.wrap_reduced(sim_end, group_by=group_by, **wrap_kwargs)

    @property
    def sim_end(self) -> tp.Series:
        """Simulation end value computed using default parameters from `SimRangeMixin.get_sim_end`.

        Returns:
            Series: Simulation end series.
        """
        return self.get_sim_end()

    @hybrid_method
    def get_sim_start_index(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.Series]:
        """Return the simulation start index.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            allow_none (bool): Allow simulation positions to be None when applicable.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            Optional[Series]: Resolved simulation start index, or None.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.resolve_sim_start(
            sim_start=sim_start,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_start is None:
            return None
        start_index = []
        for i in range(len(sim_start)):
            _sim_start = sim_start[i]
            if _sim_start == 0:
                start_index.append(wrapper.index[0])
            elif _sim_start == len(wrapper.index):
                if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                    start_index.append(wrapper.index[-1] + wrapper.freq)
                elif isinstance(wrapper.index, pd.RangeIndex):
                    start_index.append(wrapper.index[-1] + 1)
                else:
                    start_index.append(None)
            else:
                start_index.append(wrapper.index[_sim_start])
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_start_index"), wrap_kwargs)
        return wrapper.wrap_reduced(pd.Index(start_index), group_by=group_by, **wrap_kwargs)

    @property
    def sim_start_index(self) -> tp.Series:
        """Simulation start index using default arguments via `SimRangeMixin.get_sim_start_index`.

        Returns:
            Series: Simulation start index series.
        """
        return self.get_sim_start_index()

    @hybrid_method
    def get_sim_end_index(
        cls_or_self,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = False,
        inclusive: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.Series]:
        """Return the simulation end index.

        Args:
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            allow_none (bool): Allow simulation positions to be None when applicable.
            inclusive (bool): Determines if the simulation end should be treated as inclusive.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            Optional[Series]: Resolved simulation end index, or None.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_end = cls_or_self.resolve_sim_end(
            sim_end=sim_end,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_end is None:
            return None
        end_index = []
        for i in range(len(sim_end)):
            _sim_end = sim_end[i]
            if _sim_end == 0:
                if inclusive:
                    end_index.append(None)
                else:
                    end_index.append(wrapper.index[0])
            elif _sim_end == len(wrapper.index):
                if inclusive:
                    end_index.append(wrapper.index[-1])
                else:
                    if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                        end_index.append(wrapper.index[-1] + wrapper.freq)
                    elif isinstance(wrapper.index, pd.RangeIndex):
                        end_index.append(wrapper.index[-1] + 1)
                    else:
                        end_index.append(None)
            else:
                if inclusive:
                    end_index.append(wrapper.index[_sim_end - 1])
                else:
                    end_index.append(wrapper.index[_sim_end])
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_end_index"), wrap_kwargs)
        return wrapper.wrap_reduced(pd.Index(end_index), group_by=group_by, **wrap_kwargs)

    @property
    def sim_end_index(self) -> tp.Series:
        """Simulation end index using default arguments via `SimRangeMixin.get_sim_end_index`.

        Returns:
            Series: Simulation end index series.
        """
        return self.get_sim_end_index()

    @hybrid_method
    def get_sim_duration(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.Series]:
        """Return the duration of the simulation range.

        Args:
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

        Returns:
            Optional[Series]: Duration of the simulation range.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.resolve_sim_start(
            sim_start=sim_start,
            allow_none=False,
            wrapper=wrapper,
            group_by=group_by,
        )
        sim_end = cls_or_self.resolve_sim_end(
            sim_end=sim_end,
            allow_none=False,
            wrapper=wrapper,
            group_by=group_by,
        )
        total_duration = sim_end - sim_start
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_duration"), wrap_kwargs)
        return wrapper.wrap_reduced(total_duration, group_by=group_by, **wrap_kwargs)

    @property
    def sim_duration(self) -> tp.Series:
        """Simulation duration using default arguments via `SimRangeMixin.get_sim_duration`.

        Returns:
            Series: Simulation duration series.
        """
        return self.get_sim_duration()

    @hybrid_method
    def fit_fig_to_sim_range(
        cls_or_self,
        fig: tp.BaseFigure,
        column: tp.Optional[tp.Column] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        xref: tp.Optional[str] = None,
    ) -> tp.BaseFigure:
        """Adjust the figure's x-axis range to match the simulation range.

        Args:
            fig (BaseFigure): Figure to update.
            column (Optional[Column]): Identifier of the column or group to plot.
            sim_start (Optional[ArrayLike]): Start index of the simulation range.
            sim_end (Optional[ArrayLike]): End index of the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            xref (Optional[str]): Reference for the x-axis (e.g., "x", "x2").

                If None, it is inferred from the figure.

        Returns:
            BaseFigure: Updated figure with an adjusted x-axis range.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.get_sim_start(
            sim_start=sim_start,
            allow_none=True,
            wrapper=wrapper,
            group_by=group_by,
        )
        sim_end = cls_or_self.get_sim_end(
            sim_end=sim_end,
            allow_none=True,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_start is not None:
            sim_start = wrapper.select_col_from_obj(sim_start, column=column, group_by=group_by)
        if sim_end is not None:
            sim_end = wrapper.select_col_from_obj(sim_end, column=column, group_by=group_by)
        if sim_start is not None or sim_end is not None:
            if sim_start == len(wrapper.index) or sim_end == 0 or sim_start == sim_end:
                return fig
            if sim_start is None:
                sim_start = 0
            if sim_start > 0:
                sim_start_index = wrapper.index[sim_start - 1]
            else:
                if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                    sim_start_index = wrapper.index[0] - wrapper.freq
                elif isinstance(wrapper.index, pd.RangeIndex):
                    sim_start_index = wrapper.index[0] - 1
                else:
                    sim_start_index = wrapper.index[0]
            if sim_end is None:
                sim_end = len(wrapper.index)
            if sim_end < len(wrapper.index):
                sim_end_index = wrapper.index[sim_end]
            else:
                if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                    sim_end_index = wrapper.index[-1] + wrapper.freq
                elif isinstance(wrapper.index, pd.RangeIndex):
                    sim_end_index = wrapper.index[-1] + 1
                else:
                    sim_end_index = wrapper.index[-1]
            if xref is not None:
                xaxis = "xaxis" + xref[1:]
                fig.update_layout(**{xaxis: dict(range=[sim_start_index, sim_end_index])})
            else:
                fig.update_xaxes(range=[sim_start_index, sim_end_index])
        return fig
