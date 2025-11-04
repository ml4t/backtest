# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for wrapping NumPy arrays into Pandas Series and DataFrames.

!!! info
    For default settings, see `vectorbtpro._settings.wrapping`.
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import indexes, reshaping
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.indexes import IndexApplier, concat_indexes, stack_indexes
from vectorbtpro.base.indexing import (
    ExtPandasIndexer,
    IdxDict,
    IdxSetter,
    IdxSetterFactory,
    IndexingError,
    index_dict,
)
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.utils import checks
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.array_ import cast_to_max_precision, cast_to_min_precision, is_range
from vectorbtpro.utils.attr_ import AttrResolverMixin, AttrResolverMixinT
from vectorbtpro.utils.chunking import (
    ArraySelector,
    ArraySlicer,
    ChunkMeta,
    get_chunk_meta_key,
    iter_chunk_meta,
)
from vectorbtpro.utils.config import Configured, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import cached_method, cached_property, hybrid_method
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.params import ItemParamable
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from vectorbtpro.base.accessors import BaseIDXAccessor as BaseIDXAccessorT
    from vectorbtpro.generic.splitting.base import Splitter as SplitterT
else:
    BaseIDXAccessorT = "vectorbtpro.base.accessors.BaseIDXAccessor"
    SplitterT = "vectorbtpro.generic.splitting.base.Splitter"

__all__ = [
    "ArrayWrapper",
    "Wrapping",
]


HasWrapperT = tp.TypeVar("HasWrapperT", bound="HasWrapper")


class HasWrapper(ExtPandasIndexer, ItemParamable):
    """Abstract class for managing a wrapper that supports advanced indexing, grouping,
    and splitting operations on wrapped arrays."""

    @property
    def unwrapped(self) -> tp.Any:
        """Underlying unwrapped object.

        Returns:
            Any: Unwrapped object.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @hybrid_method
    def should_wrap(cls_or_self) -> bool:
        """Return whether wrapping should be applied.

        Returns:
            bool: True if wrapping is needed, False otherwise."""
        return True

    @property
    def wrapper(self) -> "ArrayWrapper":
        """Underlying array wrapper of type `ArrayWrapper` used for data manipulation.

        Returns:
            ArrayWrapper: Array wrapper instance.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def column_only_select(self) -> bool:
        """Indicates whether indexing is restricted to columns.

        Returns:
            bool: True if indexing is limited to columns, False otherwise.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def range_only_select(self) -> bool:
        """Indicates whether row indexing should be performed using slices only.

        Returns:
            bool: True if row indexing is limited to slices, False otherwise.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def group_select(self) -> bool:
        """Indicates whether indexing operations can be performed on groups.

        Returns:
            bool: True if indexing operations can be performed on groups, False otherwise.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    def regroup(self: HasWrapperT, group_by: tp.GroupByLike, **kwargs) -> HasWrapperT:
        """Regroup the instance based on the specified grouping criterion.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments passed for regrouping.

        Returns:
            HasWrapper: Regrouped instance.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def ungroup(self: HasWrapperT, **kwargs) -> HasWrapperT:
        """Ungroup the instance by removing any grouping.

        Args:
            **kwargs: Keyword arguments for `HasWrapper.regroup`

        Returns:
            HasWrapper: Ungrouped instance.
        """
        return self.regroup(False, **kwargs)

    # ############# Selection ############# #

    def select_col(
        self: HasWrapperT,
        column: tp.Any = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> HasWrapperT:
        """Select one column or group from the instance.

        Args:
            column (Any): Column identifier, which can be a label-based position or an integer position.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `HasWrapper.regroup`.

        Returns:
            HasWrapper: Instance narrowed down to a single column or group.
        """
        _self = self.regroup(group_by, **kwargs)

        def _check_out_dim(out: HasWrapperT) -> HasWrapperT:
            if out.wrapper.get_ndim() == 2:
                if out.wrapper.get_shape_2d()[1] == 1:
                    if out.column_only_select:
                        return out.iloc[0]
                    return out.iloc[:, 0]
                if _self.wrapper.grouper.is_grouped():
                    raise TypeError("Could not select one group: multiple groups returned")
                else:
                    raise TypeError("Could not select one column: multiple columns returned")
            return out

        if column is None:
            if _self.wrapper.get_ndim() == 2 and _self.wrapper.get_shape_2d()[1] == 1:
                column = 0
        if column is not None:
            if _self.wrapper.grouper.is_grouped():
                if _self.wrapper.grouped_ndim == 1:
                    raise TypeError("This instance already contains one group of data")
                if column not in _self.wrapper.get_columns():
                    if isinstance(column, int):
                        if _self.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Group '{column}' not found")
            else:
                if _self.wrapper.ndim == 1:
                    raise TypeError("This instance already contains one column of data")
                if column not in _self.wrapper.columns:
                    if isinstance(column, int):
                        if _self.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Column '{column}' not found")
            return _check_out_dim(_self[column])
        if _self.wrapper.grouper.is_grouped():
            if _self.wrapper.grouped_ndim == 1:
                return _self
            raise TypeError("Only one group is allowed. Use indexing or column argument.")
        if _self.wrapper.ndim == 1:
            return _self
        raise TypeError("Only one column is allowed. Use indexing or column argument.")

    @hybrid_method
    def select_col_from_obj(
        cls_or_self,
        obj: tp.Optional[tp.SeriesFrame],
        column: tp.Any = None,
        obj_ungrouped: bool = False,
        group_by: tp.GroupByLike = None,
        wrapper: tp.Optional["ArrayWrapper"] = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Select one column or group from a Pandas object.

        Args:
            obj (Optional[SeriesFrame]): Pandas object from which to select a column or group.
            column (Any): Column identifier, which can be a label-based position or an integer position.
            obj_ungrouped (bool): Flag indicating whether the Pandas object is ungrouped.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            **kwargs: Keyword arguments for `ArrayWrapper.regroup`.

        Returns:
            MaybeSeries: Selected column or group from the Pandas object.
        """
        if obj is None:
            return None
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")
        _wrapper = wrapper.regroup(group_by, **kwargs)

        def _check_out_dim(out: tp.SeriesFrame, from_df: bool) -> tp.Series:
            bad_shape = False
            if from_df and isinstance(out, pd.DataFrame):
                if len(out.columns) == 1:
                    return out.iloc[:, 0]
                bad_shape = True
            if not from_df and isinstance(out, pd.Series):
                if len(out) == 1:
                    return out.iloc[0]
                bad_shape = True
            if bad_shape:
                if _wrapper.grouper.is_grouped():
                    raise TypeError("Could not select one group: multiple groups returned")
                else:
                    raise TypeError("Could not select one column: multiple columns returned")
            return out

        if column is None:
            if _wrapper.get_ndim() == 2 and _wrapper.get_shape_2d()[1] == 1:
                column = 0
        if column is not None:
            if _wrapper.grouper.is_grouped():
                if _wrapper.grouped_ndim == 1:
                    raise TypeError("This instance already contains one group of data")
                if obj_ungrouped:
                    mask = _wrapper.grouper.group_by == column
                    if not mask.any():
                        raise KeyError(f"Group '{column}' not found")
                    if isinstance(obj, pd.DataFrame):
                        return obj.loc[:, mask]
                    return obj.loc[mask]
                else:
                    if column not in _wrapper.get_columns():
                        if isinstance(column, int):
                            if isinstance(obj, pd.DataFrame):
                                return _check_out_dim(obj.iloc[:, column], True)
                            return _check_out_dim(obj.iloc[column], False)
                        raise KeyError(f"Group '{column}' not found")
            else:
                if _wrapper.ndim == 1:
                    raise TypeError("This instance already contains one column of data")
                if column not in _wrapper.columns:
                    if isinstance(column, int):
                        if isinstance(obj, pd.DataFrame):
                            return _check_out_dim(obj.iloc[:, column], True)
                        return _check_out_dim(obj.iloc[column], False)
                    raise KeyError(f"Column '{column}' not found")
            if isinstance(obj, pd.DataFrame):
                return _check_out_dim(obj[column], True)
            return _check_out_dim(obj[column], False)
        if not _wrapper.grouper.is_grouped():
            if _wrapper.ndim == 1:
                return obj
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        if _wrapper.grouped_ndim == 1:
            return obj
        raise TypeError("Only one group is allowed. Use indexing or column argument.")

    # ############# Splitting ############# #

    def split(
        self,
        *args,
        splitter_cls: tp.Optional[tp.Type[SplitterT]] = None,
        wrap: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Split the instance using the specified splitter.

        Args:
            *args: Positional arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_take`.
            splitter_cls (Optional[Type[Splitter]]): Splitter class to use.

                Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
            wrap (Optional[bool]): Flag indicating whether the instance should be wrapped.
            **kwargs: Keyword arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_take`.

        Returns:
            Any: Result of the splitting operation.
        """
        from vectorbtpro.generic.splitting.base import Splitter

        if splitter_cls is None:
            splitter_cls = Splitter
        if wrap is None:
            wrap = self.should_wrap()
        wrapped_self = self if wrap else self.unwrapped
        return splitter_cls.split_and_take(self.wrapper.index, wrapped_self, *args, **kwargs)

    def split_apply(
        self,
        apply_func: tp.Union[str, tp.Callable],
        *args,
        splitter_cls: tp.Optional[tp.Type[SplitterT]] = None,
        wrap: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Split the instance and apply a given function to each split.

        Args:
            apply_func (Union[str, Callable]): Function or attribute name to apply to each split.
            *args: Positional arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_apply`.
            splitter_cls (Optional[Type[Splitter]]): Splitter class to use.

                Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
            wrap (Optional[bool]): Flag indicating whether the instance should be wrapped.
            **kwargs: Keyword arguments for `vectorbtpro.generic.splitting.base.Splitter.split_and_apply`.

        Returns:
            Any: Result of applying the function to each split.
        """
        from vectorbtpro.generic.splitting.base import Splitter, Takeable

        if isinstance(apply_func, str):
            from vectorbtpro.utils.attr_ import deep_getattr

            apply_func = deep_getattr(type(self), apply_func, call_last_attr=False)
        if splitter_cls is None:
            splitter_cls = Splitter
        if wrap is None:
            wrap = self.should_wrap()
        wrapped_self = self if wrap else self.unwrapped
        return splitter_cls.split_and_apply(
            self.wrapper.index, apply_func, Takeable(wrapped_self), *args, **kwargs
        )

    # ############# Chunking ############# #

    def chunk(
        self: HasWrapperT,
        axis: tp.Optional[int] = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Union[None, int, str] = None,
        chunk_len: tp.Union[None, int, str] = None,
        chunk_meta: tp.Optional[tp.Iterable[ChunkMeta]] = None,
        select: bool = False,
        wrap: tp.Optional[bool] = None,
        return_chunk_meta: bool = False,
    ) -> tp.Iterator[tp.Union[HasWrapperT, tp.Tuple[ChunkMeta, HasWrapperT]]]:
        """Divide the instance into chunks.

        This method splits the instance into smaller segments along a specified axis.
        If `axis` is not provided, it defaults to 0 for one-dimensional instances and 1
        for multi-dimensional instances.

        Args:
            axis (Optional[int]): Axis along which to split the instance.
            min_size (Optional[int]): Minimum number of elements to split.
            n_chunks (Union[None, int, str]): Specification for the number of chunks.
            chunk_len (Union[None, int, str]): Specification for the length of each chunk.
            chunk_meta (Optional[Iterable[ChunkMeta]]): Iterable containing metadata for each chunk.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.
            select (bool): Flag indicating whether to select chunks using `ArraySelector`.
            wrap (Optional[bool]): Flag to specify whether to wrap the result.
            return_chunk_meta (bool): Flag indicating whether to yield chunk metadata alongside each chunk.

        Yields:
            Union[HasWrapper, Tuple[ChunkMeta, HasWrapper]]: Chunk of the instance,
                or a tuple containing the chunk metadata and the chunk if `return_chunk_meta` is True.
        """
        if axis is None:
            axis = 0 if self.wrapper.ndim == 1 else 1
        if self.wrapper.ndim == 1 and axis == 1:
            raise TypeError("Axis 1 is not supported for one dimension")
        checks.assert_in(axis, (0, 1))
        size = self.wrapper.shape_2d[axis]
        if wrap is None:
            wrap = self.should_wrap()
        wrapped_self = self if wrap else self.unwrapped
        if chunk_meta is None:
            chunk_meta = iter_chunk_meta(
                size=size,
                min_size=min_size,
                n_chunks=n_chunks,
                chunk_len=chunk_len,
            )
        for _chunk_meta in chunk_meta:
            if select:
                array_taker = ArraySelector(axis=axis)
            else:
                array_taker = ArraySlicer(axis=axis)
            if return_chunk_meta:
                yield _chunk_meta, array_taker.take(wrapped_self, _chunk_meta)
            else:
                yield array_taker.take(wrapped_self, _chunk_meta)

    def chunk_apply(
        self: HasWrapperT,
        apply_func: tp.Union[str, tp.Callable],
        *args,
        chunk_kwargs: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MergeableResults:
        """Apply a function to each chunk of the instance.

        This method first divides the instance into chunks and then applies the specified
        function to each chunk. If `apply_func` is a string, it is treated as the name of
        the method to invoke on each chunk.

        Args:
            apply_func (Union[str, Callable]): Function or method name to apply to each chunk.
            *args: Positional arguments for `apply_func`.
            chunk_kwargs (KwargsLike): Keyword arguments for the chunking handler.

                See `HasWrapper.chunk`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            MergeableResults: Aggregated results obtained from applying the function to each chunk.
        """
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
            if isinstance(apply_func, str):
                from vectorbtpro.utils.attr_ import deep_getattr

                tasks.append(
                    Task(deep_getattr(chunk, apply_func, call_last_attr=False), *args, **kwargs)
                )
            else:
                tasks.append(Task(apply_func, chunk, *args, **kwargs))
            keys.append(get_chunk_meta_key(_chunk_meta))
        keys = pd.Index(keys, name="chunk_indices")
        return execute(tasks, size=len(tasks), keys=keys, **execute_kwargs)

    # ############# Iteration ############# #

    def get_item_keys(self, group_by: tp.GroupByLike = None) -> tp.Index:
        """Retrieve the keys for iterating over items.

        This method returns the key indices used for `Wrapping.items`. If the instance is grouped
        and group selection is enabled, the keys are obtained via `wrapper.get_columns`;
        otherwise, `wrapper.columns` is used.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Index: Index containing the keys for iterating over the items.
        """
        _self = self.regroup(group_by=group_by)
        if _self.group_select and _self.wrapper.grouper.is_grouped():
            return _self.wrapper.get_columns()
        return _self.wrapper.columns

    def items(
        self,
        group_by: tp.GroupByLike = None,
        apply_group_by: bool = False,
        keep_2d: bool = False,
        key_as_index: bool = False,
        wrap: tp.Optional[bool] = None,
    ) -> tp.Items:
        """Iterate over columns or groups of the instance.

        This method yields key-item pairs where keys represent column or group identifiers
        and items are the corresponding sub-instances. The iteration behavior is determined
        by the provided `group_by` and `apply_group_by` parameters, as well as whether the
        instance is grouped with group selection enabled.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            apply_group_by (bool): If True, applies the grouping to both iteration and the final output.

                If False, `group_by` is used solely as an iteration instruction.
            keep_2d (bool): Whether to maintain the output data in a two-dimensional format.
            key_as_index (bool): Whether to return the yielded key as an index.
            wrap (Optional[bool]): Flag indicating whether to wrap the yielded items
                with additional functionality.

        Yields:
            Items: Tuple where the first element is a key (column or group identifier)
                and the second element is the corresponding sub-instance.
        """
        if wrap is None:
            wrap = self.should_wrap()

        def _resolve_v(_self):
            return _self if wrap else _self.unwrapped

        if group_by is None or apply_group_by:
            _self = self.regroup(group_by=group_by)
            if _self.group_select and _self.wrapper.grouper.is_grouped():
                columns = _self.wrapper.get_columns()
                ndim = _self.wrapper.get_ndim()
            else:
                columns = _self.wrapper.columns
                ndim = _self.wrapper.ndim

            if ndim == 1:
                if key_as_index:
                    yield columns, _resolve_v(_self)
                else:
                    yield columns[0], _resolve_v(_self)
            else:
                for i in range(len(columns)):
                    if key_as_index:
                        key = columns[[i]]
                    else:
                        key = columns[i]
                    if _self.column_only_select:
                        if keep_2d:
                            yield key, _resolve_v(_self.iloc[i : i + 1])
                        else:
                            yield key, _resolve_v(_self.iloc[i])
                    else:
                        if keep_2d:
                            yield key, _resolve_v(_self.iloc[:, i : i + 1])
                        else:
                            yield key, _resolve_v(_self.iloc[:, i])
        else:
            if self.group_select and self.wrapper.grouper.is_grouped():
                raise ValueError("Cannot change grouping")
            wrapper = self.wrapper.regroup(group_by=group_by)
            if wrapper.get_ndim() == 1:
                if key_as_index:
                    yield wrapper.get_columns(), _resolve_v(self)
                else:
                    yield wrapper.get_columns()[0], _resolve_v(self)
            else:
                for group, group_idxs in wrapper.grouper.iter_groups(key_as_index=key_as_index):
                    if self.column_only_select:
                        if keep_2d or len(group_idxs) > 1:
                            yield group, _resolve_v(self.iloc[group_idxs])
                        else:
                            yield group, _resolve_v(self.iloc[group_idxs[0]])
                    else:
                        if keep_2d or len(group_idxs) > 1:
                            yield group, _resolve_v(self.iloc[:, group_idxs])
                        else:
                            yield group, _resolve_v(self.iloc[:, group_idxs[0]])


ArrayWrapperT = tp.TypeVar("ArrayWrapperT", bound="ArrayWrapper")


class ArrayWrapper(Configured, HasWrapper, IndexApplier):
    """Class for storing index, columns, and shape metadata for wrapping NumPy arrays,
    with integration with `vectorbtpro.base.grouping.base.Grouper` for grouping columns.

    If the underlying object is a Series, pass `[sr.name]` as `columns`.

    Args:
        index (IndexLike): Index to be associated with the array.

            It is processed using `vectorbtpro.utils.datetime_.prepare_dt_index`.
        columns (Optional[IndexLike]): Set of column labels.
        ndim (Optional[int]): Number of dimensions of the array.

            Deduced from the columns if not provided.
        freq (Optional[FrequencyLike]): Frequency of the index (e.g., "daily", "15 min", "index_mean").

            See `vectorbtpro.utils.datetime_.infer_index_freq`.
        parse_index (Optional[bool]): Flag to convert the index to a datetime index with `pd.to_datetime`.
        column_only_select (Optional[bool]): If True, index the wrapper as a Series of columns/groups.
        range_only_select (Optional[bool]): If True, allow row selection only by slicing.
        group_select (Optional[bool]): If True, enable group-based selection when grouping is active.
        grouped_ndim (Optional[int]): Number of dimensions after grouping columns.
        grouper (Optional[Grouper]): `vectorbtpro.base.grouping.base.Grouper` instance for grouping columns.
        **kwargs: Keyword arguments for `vectorbtpro.base.grouping.base.Grouper`
            and `vectorbtpro.utils.config.Configured`.

    !!! note
        This class is immutable. To modify attributes, use `ArrayWrapper.replace`.

        Use methods starting with `get_` for group-aware results.
    """

    def __init__(
        self,
        index: tp.IndexLike,
        columns: tp.Optional[tp.IndexLike] = None,
        ndim: tp.Optional[int] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        parse_index: tp.Optional[bool] = None,
        column_only_select: tp.Optional[bool] = None,
        range_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        grouped_ndim: tp.Optional[int] = None,
        grouper: tp.Optional[Grouper] = None,
        **kwargs,
    ) -> None:
        checks.assert_not_none(index, arg_name="index")
        index = dt.prepare_dt_index(index, parse_index=parse_index)
        if columns is None:
            columns = [None]
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if ndim is None:
            if len(columns) == 1 and not isinstance(columns, pd.MultiIndex):
                ndim = 1
            else:
                ndim = 2
        else:
            if len(columns) > 1:
                ndim = 2

        grouper_arg_names = get_func_arg_names(Grouper.__init__)
        grouper_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in grouper_arg_names:
                grouper_kwargs[k] = kwargs.pop(k)
        if grouper is None:
            grouper = Grouper(columns, **grouper_kwargs)
        elif not checks.is_index_equal(columns, grouper.index) or len(grouper_kwargs) > 0:
            grouper = grouper.replace(index=columns, **grouper_kwargs)

        HasWrapper.__init__(self)
        Configured.__init__(
            self,
            index=index,
            columns=columns,
            ndim=ndim,
            freq=freq,
            parse_index=parse_index,
            column_only_select=column_only_select,
            range_only_select=range_only_select,
            group_select=group_select,
            grouped_ndim=grouped_ndim,
            grouper=grouper,
            **kwargs,
        )

        self._index = index
        self._columns = columns
        self._ndim = ndim
        self._freq = freq
        self._parse_index = parse_index
        self._column_only_select = column_only_select
        self._range_only_select = range_only_select
        self._group_select = group_select
        self._grouper = grouper
        self._grouped_ndim = grouped_ndim

    @classmethod
    def from_obj(cls: tp.Type[ArrayWrapperT], obj: tp.ArrayLike, **kwargs) -> ArrayWrapperT:
        """Derive array wrapper metadata from the given object.

        Args:
            obj (ArrayLike): Input object from which to derive metadata.

                This may be an instance of `Data`, `Wrapping`, or `ArrayWrapper`.
            **kwargs: Keyword arguments for `ArrayWrapper`.

        Returns:
            ArrayWrapper: New instance with metadata derived from the input object.
        """
        from vectorbtpro.base.reshaping import to_pd_array
        from vectorbtpro.data.base import Data

        if isinstance(obj, Data):
            obj = obj.symbol_wrapper
        if isinstance(obj, Wrapping):
            obj = obj.wrapper
        if isinstance(obj, ArrayWrapper):
            return obj.replace(**kwargs)

        pd_obj = to_pd_array(obj)
        index = indexes.get_index(pd_obj, 0)
        columns = indexes.get_index(pd_obj, 1)
        ndim = pd_obj.ndim
        kwargs.pop("index", None)
        kwargs.pop("columns", None)
        kwargs.pop("ndim", None)
        return cls(index, columns, ndim, **kwargs)

    @classmethod
    def from_shape(
        cls: tp.Type[ArrayWrapperT],
        shape: tp.ShapeLike,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        ndim: tp.Optional[int] = None,
        *args,
        **kwargs,
    ) -> ArrayWrapperT:
        """Derive array wrapper metadata based on the given shape.

        Args:
            shape (ShapeLike): Shape representing the array dimensions.
            index (Optional[IndexLike]): Index labels.
            columns (Optional[IndexLike]): Column labels.
            ndim (Optional[int]): Number of dimensions.
            *args: Positional arguments for `ArrayWrapper`.
            **kwargs: Keyword arguments for `ArrayWrapper`.

        Returns:
            ArrayWrapper: New instance with metadata derived from the provided shape.
        """
        shape = reshaping.to_tuple_shape(shape)
        if index is None:
            index = pd.RangeIndex(stop=shape[0])
        if columns is None:
            columns = pd.RangeIndex(stop=shape[1] if len(shape) > 1 else 1)
        if ndim is None:
            ndim = len(shape)
        return cls(index, columns, ndim, *args, **kwargs)

    @staticmethod
    def extract_init_kwargs(**kwargs) -> tp.Tuple[tp.Kwargs, tp.Kwargs]:
        """Extract keyword arguments relevant for constructing an `ArrayWrapper` or
        `vectorbtpro.base.grouping.base.Grouper` instance.

        Args:
            **kwargs: Keyword arguments for `ArrayWrapper` or `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Tuple[Kwargs, Kwargs]: Tuple containing two dictionaries:

                * First dictionary comprises keyword arguments applicable to `ArrayWrapper` or
                    `vectorbtpro.base.grouping.base.Grouper`.
                * Second dictionary contains the remaining keyword arguments.
        """
        wrapper_arg_names = get_func_arg_names(ArrayWrapper.__init__)
        grouper_arg_names = get_func_arg_names(Grouper.__init__)
        init_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in wrapper_arg_names or k in grouper_arg_names:
                init_kwargs[k] = kwargs.pop(k)
        return init_kwargs, kwargs

    @classmethod
    def resolve_stack_kwargs(
        cls, *wrappers: tp.MaybeSequence[ArrayWrapperT], **kwargs
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `ArrayWrapper` after stacking.

        Args:
            *wrappers (MaybeSequence[ArrayWrapper]): `ArrayWrapper` instances to be stacked.
            **kwargs: Keyword arguments to override configuration parameters.

        Returns:
            Kwargs: Dictionary of resolved keyword arguments for `ArrayWrapper` initialization.
        """
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)

        common_keys = set()
        for wrapper in wrappers:
            common_keys = common_keys.union(set(wrapper.config.keys()))
            if "grouper" not in kwargs:
                common_keys = common_keys.union(set(wrapper.grouper.config.keys()))
        common_keys.remove("grouper")
        init_wrapper = wrappers[0]
        for i in range(1, len(wrappers)):
            wrapper = wrappers[i]
            for k in common_keys:
                if k not in kwargs:
                    same_k = True
                    try:
                        if k in wrapper.config:
                            if not checks.is_deep_equal(init_wrapper.config[k], wrapper.config[k]):
                                same_k = False
                        elif "grouper" not in kwargs and k in wrapper.grouper.config:
                            if not checks.is_deep_equal(
                                init_wrapper.grouper.config[k], wrapper.grouper.config[k]
                            ):
                                same_k = False
                        else:
                            same_k = False
                    except KeyError:
                        same_k = False
                    if not same_k:
                        raise ValueError(
                            f"Objects to be merged must have compatible '{k}'. Pass to override."
                        )
        for k in common_keys:
            if k not in kwargs:
                if k in init_wrapper.config:
                    kwargs[k] = init_wrapper.config[k]
                elif "grouper" not in kwargs and k in init_wrapper.grouper.config:
                    kwargs[k] = init_wrapper.grouper.config[k]
                else:
                    raise ValueError(
                        f"Objects to be merged must have compatible '{k}'. Pass to override."
                    )
        return kwargs

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[ArrayWrapperT],
        *wrappers: tp.MaybeSequence[ArrayWrapperT],
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        group_by: tp.GroupByLike = None,
        stack_columns: bool = True,
        index_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = "append",
        keys: tp.Optional[tp.IndexLike] = None,
        clean_index_kwargs: tp.KwargsLike = None,
        verify_integrity: bool = True,
        **kwargs,
    ) -> ArrayWrapperT:
        """Stack multiple `ArrayWrapper` instances along rows.

        Concatenates indexes using `vectorbtpro.base.indexes.concat_indexes`.
        All wrappers must share the same frequency unless a custom frequency is provided via `freq`.
        If the column levels differ among wrappers, they are stacked together if `stack_columns` is True;
        otherwise, an error is raised. When `group_by` is not provided, all wrappers must be uniformly
        grouped or ungrouped and have consistent group labels. Furthermore, all wrappers are expected
        to have compatible configuration values, except where explicitly overridden via `kwargs`.

        Args:
            *wrappers (MaybeSequence[ArrayWrapper]): (Additional) `ArrayWrapper` instances to stack.
            index (Optional[IndexLike]): Custom index for the stacked result.

                If not provided, indexes are concatenated.
            columns (Optional[IndexLike]): Custom columns for the stacked result.

                If not provided, columns are derived from the wrappers.
            freq (Optional[FrequencyLike]): Custom frequency for the stacked result
                (e.g., "daily", "15 min", "index_mean").

                If not provided, frequency is derived from the wrappers.
                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            stack_columns (bool): Whether to stack differing column levels from wrappers.
            index_concat_method (MaybeTuple[Union[str, Callable]]): Method used for concatenating indexes.
            keys (Optional[IndexLike]): Keys used during index concatenation.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            verify_integrity (bool): Flag to verify the integrity of the concatenated index.
            **kwargs: Keyword arguments for `ArrayWrapper`.

        Returns:
            ArrayWrapper: New `ArrayWrapper` instance representing the row-stacked result.
        """
        if not isinstance(cls_or_self, type):
            wrappers = (cls_or_self, *wrappers)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)
        for wrapper in wrappers:
            if not checks.is_instance_of(wrapper, ArrayWrapper):
                raise TypeError("Each object to be merged must be an instance of ArrayWrapper")

        if index is None:
            index = concat_indexes(
                [wrapper.index for wrapper in wrappers],
                index_concat_method=index_concat_method,
                keys=keys,
                clean_index_kwargs=clean_index_kwargs,
                verify_integrity=verify_integrity,
                axis=0,
            )
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)
        kwargs["index"] = index

        if freq is None:
            new_freq = None
            for wrapper in wrappers:
                if new_freq is None:
                    new_freq = wrapper.freq
                else:
                    if (
                        new_freq is not None
                        and wrapper.freq is not None
                        and new_freq != wrapper.freq
                    ):
                        raise ValueError("Objects to be merged must have the same frequency")
            freq = new_freq
        kwargs["freq"] = freq

        if columns is None:
            new_columns = None
            for wrapper in wrappers:
                if new_columns is None:
                    new_columns = wrapper.columns
                else:
                    if not checks.is_index_equal(new_columns, wrapper.columns):
                        if not stack_columns:
                            raise ValueError("Objects to be merged must have the same columns")
                        new_columns = stack_indexes(
                            (new_columns, wrapper.columns),
                            **resolve_dict(clean_index_kwargs),
                        )
            columns = new_columns
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        kwargs["columns"] = columns

        if "grouper" in kwargs:
            if not checks.is_index_equal(columns, kwargs["grouper"].index):
                raise ValueError("Columns and grouper index must match")
            if group_by is not None:
                kwargs["group_by"] = group_by
        else:
            if group_by is None:
                grouped = None
                for wrapper in wrappers:
                    wrapper_grouped = wrapper.grouper.is_grouped()
                    if grouped is None:
                        grouped = wrapper_grouped
                    else:
                        if grouped is not wrapper_grouped:
                            raise ValueError(
                                "Objects to be merged must have either all grouped or all ungrouped"
                            )
                if grouped:
                    new_group_by = None
                    for wrapper in wrappers:
                        wrapper_groups, wrapper_grouped_index = (
                            wrapper.grouper.get_groups_and_index()
                        )
                        wrapper_group_by = wrapper_grouped_index[wrapper_groups]
                        if new_group_by is None:
                            new_group_by = wrapper_group_by
                        else:
                            if not checks.is_index_equal(new_group_by, wrapper_group_by):
                                raise ValueError("Objects to be merged must have the same groups")
                    group_by = new_group_by
                else:
                    group_by = False
            kwargs["group_by"] = group_by

        if "ndim" not in kwargs:
            ndim = None
            for wrapper in wrappers:
                if ndim is None or wrapper.ndim > 1:
                    ndim = wrapper.ndim
            kwargs["ndim"] = ndim

        return cls(**ArrayWrapper.resolve_stack_kwargs(*wrappers, **kwargs))

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[ArrayWrapperT],
        *wrappers: tp.MaybeSequence[ArrayWrapperT],
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        group_by: tp.GroupByLike = None,
        union_index: bool = True,
        col_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = "append",
        group_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = (
            "append",
            "factorize_each",
        ),
        keys: tp.Optional[tp.IndexLike] = None,
        clean_index_kwargs: tp.KwargsLike = None,
        verify_integrity: bool = True,
        **kwargs,
    ) -> ArrayWrapperT:
        """Stack multiple `ArrayWrapper` instances along columns.

        Merge the column data and configurations from multiple `ArrayWrapper` instances.

        This function performs the following:

        * Determines the final index. If all wrappers share the same index, that index is used.

            Otherwise, if `union_index` is True, the union of the indexes is computed.
            The merged index must contain no duplicates, have a consistent data type,
            and be monotonically increasing. A custom index can be provided via the `index` parameter.
        * Verifies that frequency is consistent across wrappers, unless overridden by the `freq` parameter.
        * Concatenates columns and groups using `vectorbtpro.base.indexes.concat_indexes`.
        * Propagates settings such as `column_only_select` and `group_select` from the input wrappers.
        * Ensures that all wrappers have matching configurations, aside from parameters explicitly
            provided via `kwargs`.

        Args:
            *wrappers (MaybeSequence[ArrayWrapper]): (Additional) `ArrayWrapper` instances to stack.
            index (Optional[IndexLike]): Custom index for the resulting wrapper.

                If not provided, derived from the wrappers.
            columns (Optional[IndexLike]): Custom columns for the stacked result.

                If not provided, columns are derived from the wrappers.
            freq (Optional[FrequencyLike]): Custom frequency for the stacked result
                (e.g., "daily", "15 min", "index_mean").

                If not provided, frequency is derived from the wrappers.
                See `vectorbtpro.utils.datetime_.infer_index_freq`.
            group_by (GroupByLike): Grouping specification.

                If not provided, groups are concatenated if any wrapper is grouped, otherwise not applied.
                See `vectorbtpro.base.grouping.base.Grouper`.
            union_index (bool): Whether to merge differing indexes via a union operation.
            col_concat_method (MaybeTuple[Union[str, Callable]]): Method used to concatenate column indexes.
            group_concat_method (MaybeTuple[Union[str, Callable]]): Method used to concatenate group indexes.
            keys (Optional[IndexLike]): Keys used for concatenating indexes.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            verify_integrity (bool): Flag to verify the integrity of the concatenated index.
            **kwargs: Keyword arguments for `ArrayWrapper`.

        Returns:
            ArrayWrapper: New instance with combined array data and merged configuration.
        """
        if not isinstance(cls_or_self, type):
            wrappers = (cls_or_self, *wrappers)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)
        for wrapper in wrappers:
            if not checks.is_instance_of(wrapper, ArrayWrapper):
                raise TypeError("Each object to be merged must be an instance of ArrayWrapper")

        for wrapper in wrappers:
            if wrapper.index.has_duplicates:
                raise ValueError("Index of some objects to be merged contains duplicates")
        if index is None:
            new_index = None
            for wrapper in wrappers:
                if new_index is None:
                    new_index = wrapper.index
                else:
                    if not checks.is_index_equal(new_index, wrapper.index):
                        if not union_index:
                            raise ValueError(
                                "Objects to be merged must have the same index. "
                                "Use union_index=True to merge index as well."
                            )
                        else:
                            if new_index.dtype != wrapper.index.dtype:
                                raise ValueError(
                                    "Indexes to be merged must have the same data type"
                                )
                            new_index = new_index.union(wrapper.index)
            if not new_index.is_monotonic_increasing:
                raise ValueError("Merged index must be monotonically increasing")
            index = new_index
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)
        kwargs["index"] = index

        if freq is None:
            new_freq = None
            for wrapper in wrappers:
                if new_freq is None:
                    new_freq = wrapper.freq
                else:
                    if (
                        new_freq is not None
                        and wrapper.freq is not None
                        and new_freq != wrapper.freq
                    ):
                        raise ValueError("Objects to be merged must have the same frequency")
            freq = new_freq
        kwargs["freq"] = freq

        if columns is None:
            columns = concat_indexes(
                [wrapper.columns for wrapper in wrappers],
                index_concat_method=col_concat_method,
                keys=keys,
                clean_index_kwargs=clean_index_kwargs,
                verify_integrity=verify_integrity,
                axis=1,
            )
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        kwargs["columns"] = columns

        if "grouper" in kwargs:
            if not checks.is_index_equal(columns, kwargs["grouper"].index):
                raise ValueError("Columns and grouper index must match")
            if group_by is not None:
                kwargs["group_by"] = group_by
        else:
            if group_by is None:
                any_grouped = False
                for wrapper in wrappers:
                    if wrapper.grouper.is_grouped():
                        any_grouped = True
                        break
                if any_grouped:
                    group_by = concat_indexes(
                        [wrapper.grouper.get_stretched_index() for wrapper in wrappers],
                        index_concat_method=group_concat_method,
                        keys=keys,
                        clean_index_kwargs=clean_index_kwargs,
                        verify_integrity=verify_integrity,
                        axis=2,
                    )
                else:
                    group_by = False
            kwargs["group_by"] = group_by

        if "ndim" not in kwargs:
            kwargs["ndim"] = 2
        if "grouped_ndim" not in kwargs:
            kwargs["grouped_ndim"] = None
        if "column_only_select" not in kwargs:
            column_only_select = None
            for wrapper in wrappers:
                if column_only_select is None or wrapper.column_only_select:
                    column_only_select = wrapper.column_only_select
            kwargs["column_only_select"] = column_only_select
        if "range_only_select" not in kwargs:
            range_only_select = None
            for wrapper in wrappers:
                if range_only_select is None or wrapper.range_only_select:
                    range_only_select = wrapper.range_only_select
            kwargs["range_only_select"] = range_only_select
        if "group_select" not in kwargs:
            group_select = None
            for wrapper in wrappers:
                if group_select is None or not wrapper.group_select:
                    group_select = wrapper.group_select
            kwargs["group_select"] = group_select
        if "grouper" not in kwargs:
            if "allow_enable" not in kwargs:
                allow_enable = None
                for wrapper in wrappers:
                    if allow_enable is None or not wrapper.grouper.allow_enable:
                        allow_enable = wrapper.grouper.allow_enable
                kwargs["allow_enable"] = allow_enable
            if "allow_disable" not in kwargs:
                allow_disable = None
                for wrapper in wrappers:
                    if allow_disable is None or not wrapper.grouper.allow_disable:
                        allow_disable = wrapper.grouper.allow_disable
                kwargs["allow_disable"] = allow_disable
            if "allow_modify" not in kwargs:
                allow_modify = None
                for wrapper in wrappers:
                    if allow_modify is None or not wrapper.grouper.allow_modify:
                        allow_modify = wrapper.grouper.allow_modify
                kwargs["allow_modify"] = allow_modify

        return cls(**ArrayWrapper.resolve_stack_kwargs(*wrappers, **kwargs))

    def indexing_func_meta(
        self: ArrayWrapperT,
        pd_indexing_func: tp.PandasIndexingFunc,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        column_only_select: tp.Optional[bool] = None,
        range_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        return_slices: bool = True,
        return_none_slices: bool = True,
        return_scalars: bool = True,
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
    ) -> dict:
        """Perform indexing on an `ArrayWrapper` and return updated metadata.

        Indexing respects column grouping and various indexing options without flipping
        rows and columns. When indexing a Series, selecting one row always returns a Series;
        when indexing a DataFrame, it returns a DataFrame.

        Set `column_only_select` to True to treat the array wrapper as a Series of columns/groups,
        avoiding selection along the index (axis 0). Use `range_only_select` to restrict row
        selection to slices. If `group_select` is True and grouping is enabled, selection is
        performed based on groups; otherwise, indexing is applied to columns.

        Args:
            pd_indexing_func (PandasIndexingFunc): Function to perform Pandas-style indexing.
            index (Optional[IndexLike]): Index for selecting rows.
            columns (Optional[IndexLike]): Index for selecting columns.
            column_only_select (Optional[bool]): If True, index the wrapper as a Series of columns/groups.
            range_only_select (Optional[bool]): If True, allow row selection only by slicing.
            group_select (Optional[bool]): If True, enable group-based selection when grouping is active.
            return_slices (bool): If True, return indices as slice objects when they represent a continuous range.
            return_none_slices (bool): If True, return a slice `(None, None, None)` if an axis remains unchanged.
            return_scalars (bool): If True, return scalar values for single integer selections.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

        Returns:
            dict: Dictionary containing:

                * `new_wrapper`: Updated `ArrayWrapper` after applying indexing.
                * `row_idxs`: Selected row indices, possibly returned as a slice if identified as a range.
                * `rows_changed`: Boolean indicating whether the row axis was changed in any way.
                * `col_idxs`: Selected column indices, possibly returned as a slice if identified as a range.
                * `columns_changed`: Boolean indicating whether the column axis was changed in any way.
                * `group_idxs`: Selected group indices, or the same as column indices if grouping is disabled.
                * `groups_changed`: Boolean indicating whether the group axis was changed in any way.

        !!! note
            If `column_only_select` is True, ensure that the array wrapper is indexed as a Series
            of columns rather than a DataFrame. For example, use `.iloc[:2]` instead of `.iloc[:, :2]`.
            Operations are not allowed if the instance already contains a single column/group.
        """
        if column_only_select is None:
            column_only_select = self.column_only_select
        if range_only_select is None:
            range_only_select = self.range_only_select
        if group_select is None:
            group_select = self.group_select
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        _self = self.regroup(group_by)
        group_select = group_select and _self.grouper.is_grouped()
        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            if group_select:
                columns = _self.get_columns()
            else:
                columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if group_select:
            # Groups as columns
            i_wrapper = ArrayWrapper(index, columns, _self.get_ndim())
        else:
            # Columns as columns
            i_wrapper = ArrayWrapper(index, columns, _self.ndim)
        n_rows = len(index)
        n_cols = len(columns)

        def _resolve_arr(arr, n):
            if checks.is_np_array(arr) and is_range(arr):
                if arr[0] == 0 and arr[-1] == n - 1:
                    if return_none_slices:
                        return slice(None, None, None), False
                    return arr, False
                if return_slices:
                    return slice(arr[0], arr[-1] + 1, None), True
                return arr, True
            if isinstance(arr, np.integer):
                arr = arr.item()
            columns_changed = True
            if isinstance(arr, int):
                if arr == 0 and n == 1:
                    columns_changed = False
                if not return_scalars:
                    arr = np.array([arr])
            return arr, columns_changed

        if column_only_select:
            if i_wrapper.ndim == 1:
                raise IndexingError(
                    "Columns only: This instance already contains one column of data"
                )
            try:
                col_mapper = pd_indexing_func(
                    i_wrapper.wrap_reduced(np.arange(n_cols), columns=columns)
                )
            except pd.core.indexing.IndexingError as e:
                warn(
                    "Columns only: Make sure to treat this instance as a Series of columns rather than a DataFrame"
                )
                raise e
            if checks.is_series(col_mapper):
                new_columns = col_mapper.index
                col_idxs = col_mapper.values
                new_ndim = 2
            else:
                new_columns = columns[[col_mapper]]
                col_idxs = col_mapper
                new_ndim = 1
            new_index = index
            row_idxs = np.arange(len(index))
        else:
            init_row_mapper_values = reshaping.broadcast_array_to(
                np.arange(n_rows)[:, None], (n_rows, n_cols)
            )
            init_row_mapper = i_wrapper.wrap(init_row_mapper_values, index=index, columns=columns)
            row_mapper = pd_indexing_func(init_row_mapper)
            if i_wrapper.ndim == 1:
                if not checks.is_series(row_mapper):
                    row_idxs = np.array([row_mapper])
                    new_index = index[row_idxs]
                else:
                    row_idxs = row_mapper.values
                    new_index = indexes.get_index(row_mapper, 0)
                col_idxs = 0
                new_columns = columns
                new_ndim = 1
            else:
                init_col_mapper_values = reshaping.broadcast_array_to(
                    np.arange(n_cols)[None], (n_rows, n_cols)
                )
                init_col_mapper = i_wrapper.wrap(
                    init_col_mapper_values, index=index, columns=columns
                )
                col_mapper = pd_indexing_func(init_col_mapper)

                if checks.is_frame(col_mapper):
                    # Multiple rows and columns selected
                    row_idxs = row_mapper.values[:, 0]
                    col_idxs = col_mapper.values[0]
                    new_index = indexes.get_index(row_mapper, 0)
                    new_columns = indexes.get_index(col_mapper, 1)
                    new_ndim = 2
                elif checks.is_series(col_mapper):
                    multi_index = isinstance(index, pd.MultiIndex)
                    multi_columns = isinstance(columns, pd.MultiIndex)
                    multi_name = isinstance(col_mapper.name, tuple)
                    if multi_index and multi_name and col_mapper.name in index or not multi_index and not multi_name and col_mapper.name in index:
                        one_row = True
                    else:
                        one_row = False
                    if multi_columns and multi_name and col_mapper.name in columns or not multi_columns and not multi_name and col_mapper.name in columns:
                        one_col = True
                    else:
                        one_col = False
                    if (one_row and one_col) or (not one_row and not one_col):
                        one_row = np.all(row_mapper.values == row_mapper.values.item(0))
                        one_col = np.all(col_mapper.values == col_mapper.values.item(0))
                    if (one_row and one_col) or (not one_row and not one_col):
                        raise IndexingError("Could not parse indexing operation")
                    if one_row:
                        # One row selected
                        row_idxs = row_mapper.values[[0]]
                        col_idxs = col_mapper.values
                        new_index = index[row_idxs]
                        new_columns = indexes.get_index(col_mapper, 0)
                        new_ndim = 2
                    else:
                        # One column selected
                        row_idxs = row_mapper.values
                        col_idxs = col_mapper.values[0]
                        new_index = indexes.get_index(row_mapper, 0)
                        new_columns = columns[[col_idxs]]
                        new_ndim = 1
                else:
                    # One row and column selected
                    row_idxs = np.array([row_mapper])
                    col_idxs = col_mapper
                    new_index = index[row_idxs]
                    new_columns = columns[[col_idxs]]
                    new_ndim = 1

        if _self.grouper.is_grouped():
            # Grouping enabled
            if np.asarray(row_idxs).ndim == 0:
                raise IndexingError("Flipping index and columns is not allowed")

            if group_select:
                # Selection based on groups
                # Get indices of columns corresponding to selected groups
                group_idxs = col_idxs
                col_idxs, new_groups = _self.grouper.select_groups(group_idxs)
                ungrouped_columns = _self.columns[col_idxs]
                if new_ndim == 1 and len(ungrouped_columns) == 1:
                    ungrouped_ndim = 1
                    col_idxs = col_idxs[0]
                else:
                    ungrouped_ndim = 2

                row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
                if range_only_select and rows_changed:
                    if not isinstance(row_idxs, slice):
                        raise ValueError("Rows can be selected only by slicing")
                    if row_idxs.step not in (1, None):
                        raise ValueError("Slice for selecting rows must have a step of 1 or None")
                col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
                group_idxs, groups_changed = _resolve_arr(group_idxs, _self.get_shape_2d()[1])
                return dict(
                    new_wrapper=_self.replace(
                        **merge_dicts(
                            dict(
                                index=new_index,
                                columns=ungrouped_columns,
                                ndim=ungrouped_ndim,
                                grouped_ndim=new_ndim,
                                group_by=new_columns[new_groups],
                            ),
                            wrapper_kwargs,
                        )
                    ),
                    row_idxs=row_idxs,
                    rows_changed=rows_changed,
                    col_idxs=col_idxs,
                    columns_changed=columns_changed,
                    group_idxs=group_idxs,
                    groups_changed=groups_changed,
                )

            # Selection based on columns
            group_idxs = _self.grouper.get_groups()[col_idxs]
            new_group_by = _self.grouper.group_by[reshaping.to_1d_array(col_idxs)]
            row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
            if range_only_select and rows_changed:
                if not isinstance(row_idxs, slice):
                    raise ValueError("Rows can be selected only by slicing")
                if row_idxs.step not in (1, None):
                    raise ValueError("Slice for selecting rows must have a step of 1 or None")
            col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
            group_idxs, groups_changed = _resolve_arr(group_idxs, _self.get_shape_2d()[1])
            return dict(
                new_wrapper=_self.replace(
                    **merge_dicts(
                        dict(
                            index=new_index,
                            columns=new_columns,
                            ndim=new_ndim,
                            grouped_ndim=None,
                            group_by=new_group_by,
                        ),
                        wrapper_kwargs,
                    )
                ),
                row_idxs=row_idxs,
                rows_changed=rows_changed,
                col_idxs=col_idxs,
                columns_changed=columns_changed,
                group_idxs=group_idxs,
                groups_changed=groups_changed,
            )

        # Grouping disabled
        row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
        if range_only_select and rows_changed:
            if not isinstance(row_idxs, slice):
                raise ValueError("Rows can be selected only by slicing")
            if row_idxs.step not in (1, None):
                raise ValueError("Slice for selecting rows must have a step of 1 or None")
        col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
        return dict(
            new_wrapper=_self.replace(
                **merge_dicts(
                    dict(
                        index=new_index,
                        columns=new_columns,
                        ndim=new_ndim,
                        grouped_ndim=None,
                        group_by=None,
                    ),
                    wrapper_kwargs,
                )
            ),
            row_idxs=row_idxs,
            rows_changed=rows_changed,
            col_idxs=col_idxs,
            columns_changed=columns_changed,
            group_idxs=col_idxs,
            groups_changed=columns_changed,
        )

    def indexing_func(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform indexing on the `ArrayWrapper` instance by delegating to `ArrayWrapper.indexing_func_meta`.

        Args:
            *args: Positional arguments for `ArrayWrapper.indexing_func_meta`.
            **kwargs: Keyword arguments for `ArrayWrapper.indexing_func_meta`.

        Returns:
            ArrayWrapper: New `ArrayWrapper` instance produced by indexing.
        """
        return self.indexing_func_meta(*args, **kwargs)["new_wrapper"]

    @staticmethod
    def select_from_flex_array(
        arr: tp.ArrayLike,
        row_idxs: tp.Union[int, tp.Array1d, slice] = None,
        col_idxs: tp.Union[int, tp.Array1d, slice] = None,
        rows_changed: bool = True,
        columns_changed: bool = True,
        rotate_rows: bool = False,
        rotate_cols: bool = True,
    ) -> tp.Array2d:
        """Select specific rows and columns from a flexible array after converting it to a 2-dim array.

        Args:
            arr (ArrayLike): Input array to be indexed.
            row_idxs (Union[int, Array1d, slice]): Row indices or a slice specifying rows to select.
            col_idxs (Union[int, Array1d, slice]): Column indices or a slice specifying columns to select.
            rows_changed (bool): Whether to apply row selection.
            columns_changed (bool): Whether to apply column selection.
            rotate_rows (bool): Flag indicating whether to apply rotational indexing along rows.
            rotate_cols (bool): Flag indicating whether to apply rotational indexing along columns.

        Returns:
            Array2d: Two-dimensional NumPy array after applying the selection.
        """
        new_arr = arr_2d = reshaping.to_2d_array(arr)
        if row_idxs is not None and rows_changed:
            if arr_2d.shape[0] > 1:
                if isinstance(row_idxs, slice):
                    max_idx = row_idxs.stop - 1
                else:
                    row_idxs = reshaping.to_1d_array(row_idxs)
                    max_idx = np.max(row_idxs)
                if arr_2d.shape[0] <= max_idx:
                    if rotate_rows and not isinstance(row_idxs, slice):
                        new_arr = new_arr[row_idxs % arr_2d.shape[0], :]
                    else:
                        new_arr = new_arr[row_idxs, :]
                else:
                    new_arr = new_arr[row_idxs, :]
        if col_idxs is not None and columns_changed:
            if arr_2d.shape[1] > 1:
                if isinstance(col_idxs, slice):
                    max_idx = col_idxs.stop - 1
                else:
                    col_idxs = reshaping.to_1d_array(col_idxs)
                    max_idx = np.max(col_idxs)
                if arr_2d.shape[1] <= max_idx:
                    if rotate_cols and not isinstance(col_idxs, slice):
                        new_arr = new_arr[:, col_idxs % arr_2d.shape[1]]
                    else:
                        new_arr = new_arr[:, col_idxs]
                else:
                    new_arr = new_arr[:, col_idxs]
        return new_arr

    def get_resampler(self, *args, **kwargs) -> tp.Union[Resampler, tp.PandasResampler]:
        """Return a resampler by delegating to `vectorbtpro.base.accessors.BaseIDXAccessor.get_resampler`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_resampler`.
            **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_resampler`.

        Returns:
            Union[Resampler, PandasResampler]: Resampler object.
        """
        return self.index_acc.get_resampler(*args, **kwargs)

    def resample_meta(
        self: ArrayWrapperT, *args, wrapper_kwargs: tp.KwargsLike = None, **kwargs
    ) -> dict:
        """Perform resampling on the `ArrayWrapper` and return metadata.

        Args:
            *args: Positional arguments for `ArrayWrapper.get_resampler`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.
            **kwargs: Keyword arguments for `ArrayWrapper.get_resampler`.

        Returns:
            dict: Dictionary containing:

                * `resampler`: Resampler object.
                * `new_wrapper`: New `ArrayWrapper` instance after resampling.
        """
        resampler = self.get_resampler(*args, **kwargs)
        if isinstance(resampler, Resampler):
            _resampler = resampler
        else:
            _resampler = Resampler.from_pd_resampler(resampler)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = _resampler.target_index
        if "freq" not in wrapper_kwargs:
            wrapper_kwargs["freq"] = dt.infer_index_freq(
                wrapper_kwargs["index"], freq=_resampler.target_freq
            )
        new_wrapper = self.replace(**wrapper_kwargs)
        return dict(resampler=resampler, new_wrapper=new_wrapper)

    def resample(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform resampling on the `ArrayWrapper` instance using `ArrayWrapper.resample_meta`.

        Args:
            *args: Positional arguments for `ArrayWrapper.resample_meta`.
            **kwargs: Keyword arguments for `ArrayWrapper.resample_meta`.

        Returns:
            ArrayWrapper: New `ArrayWrapper` instance after resampling.
        """
        return self.resample_meta(*args, **kwargs)["new_wrapper"]

    @property
    def wrapper(self) -> "ArrayWrapper":
        return self

    @property
    def index(self) -> tp.Index:
        """Index associated with the wrapped array.

        Returns:
            Index: Index of the wrapped array.
        """
        return self._index

    @cached_property(whitelist=True)
    def index_acc(self) -> BaseIDXAccessorT:
        """Index accessor for the `ArrayWrapper`.

        Returns:
            BaseIDXAccessor: Instance of `vectorbtpro.base.accessors.BaseIDXAccessor`
                used for index operations.
        """
        from vectorbtpro.base.accessors import BaseIDXAccessor

        return BaseIDXAccessor(self.index, freq=self._freq)

    @property
    def ns_index(self) -> tp.Array1d:
        """Nanosecond index representation obtained from `vectorbtpro.base.accessors.BaseIDXAccessor.to_ns`.

        Returns:
            Array1d: Nanosecond index.
        """
        return self.index_acc.to_ns()

    def get_period_ns_index(self, *args, **kwargs) -> tp.Array1d:
        """Return a period-based nanosecond index computed via
        `vectorbtpro.base.accessors.BaseIDXAccessor.to_period_ns`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.to_period_ns`.
            **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.to_period_ns`.

        Returns:
            Array1d: Period-based nanosecond index.
        """
        return self.index_acc.to_period_ns(*args, **kwargs)

    @property
    def columns(self) -> tp.Index:
        """Columns associated with the wrapped array.

        Returns:
            Index: Columns of the wrapped array.
        """
        return self._columns

    def get_columns(self, group_by: tp.GroupByLike = None) -> tp.Index:
        """Return the group-aware columns index of the `ArrayWrapper`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Index: Columns index.
        """
        return self.resolve(group_by=group_by).columns

    @property
    def name(self) -> tp.Any:
        """Name of the `ArrayWrapper` when applicable.

        Returns:
            Any: Name derived from the columns if the instance is one-dimensional; otherwise, None.
        """
        if self.ndim == 1:
            if self.columns[0] == 0:
                return None
            return self.columns[0]
        return None

    def get_name(self, group_by: tp.GroupByLike = None) -> tp.Any:
        """Return the group-aware name of the `ArrayWrapper`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Any: Name for the group-aware `ArrayWrapper`.
        """
        return self.resolve(group_by=group_by).name

    @property
    def ndim(self) -> int:
        """Number of dimensions of the wrapped array.

        Returns:
            int: Number of dimensions.
        """
        return self._ndim

    def get_ndim(self, group_by: tp.GroupByLike = None) -> int:
        """Return the group-aware number of dimensions of the `ArrayWrapper`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            int: Number of dimensions.
        """
        return self.resolve(group_by=group_by).ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape of the `ArrayWrapper`.

        Returns:
            Shape: Tuple representing the dimensions of the instance.
        """
        if self.ndim == 1:
            return (len(self.index),)
        return len(self.index), len(self.columns)

    def get_shape(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Return the group-aware shape of the `ArrayWrapper`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Shape: Tuple representing the dimensions.
        """
        return self.resolve(group_by=group_by).shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape of the `ArrayWrapper` as a two-dimensional structure.

        Returns:
            Shape: Tuple representing the 2D dimensions.
        """
        if self.ndim == 1:
            return self.shape[0], 1
        return self.shape

    def get_shape_2d(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Return the group-aware two-dimensional shape of the `ArrayWrapper`.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Shape: Tuple representing the 2D dimensions.
        """
        return self.resolve(group_by=group_by).shape_2d

    def get_freq(self, *args, **kwargs) -> tp.Union[None, float, tp.PandasFrequency]:
        """Return the frequency determined by `vectorbtpro.base.accessors.BaseIDXAccessor.get_freq`.

        Args:
            *args: Positional arguments for `get_freq`.
            **kwargs: Keyword arguments for `get_freq`.

        Returns:
            Union[None, float, PandasFrequency]: Frequency information.
        """
        return self.index_acc.get_freq(*args, **kwargs)

    @property
    def freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Frequency associated with the `ArrayWrapper` as defined by
        `vectorbtpro.base.accessors.BaseIDXAccessor.freq`.

        Returns:
            Optional[PandasFrequency]: Frequency of the index.
        """
        return self.index_acc.freq

    @property
    def ns_freq(self) -> tp.Optional[int]:
        """Nanosecond frequency associated with the `ArrayWrapper` from
        `vectorbtpro.base.accessors.BaseIDXAccessor.ns_freq`.

        Returns:
            Optional[int]: Nanosecond frequency of the index.
        """
        return self.index_acc.ns_freq

    @property
    def any_freq(self) -> tp.Union[None, float, tp.PandasFrequency]:
        """Frequency value determined by `vectorbtpro.base.accessors.BaseIDXAccessor.any_freq`.

        Returns:
            Union[None, float, PandasFrequency]: Frequency of the index.
        """
        return self.index_acc.any_freq

    @property
    def periods(self) -> int:
        """Number of periods defined by `vectorbtpro.base.accessors.BaseIDXAccessor.periods`.

        Returns:
            int: Number of periods in the index.
        """
        return self.index_acc.periods

    @property
    def dt_periods(self) -> float:
        """Time-based periods derived from `vectorbtpro.base.accessors.BaseIDXAccessor.dt_periods`.

        Returns:
            float: Number of periods in the index as a float.
        """
        return self.index_acc.dt_periods

    def arr_to_timedelta(self, *args, **kwargs) -> tp.Union[pd.Index, tp.MaybeArray]:
        """Return a timedelta array converted from the index using
        `vectorbtpro.base.accessors.BaseIDXAccessor.arr_to_timedelta`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.arr_to_timedelta`.
            **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.arr_to_timedelta`.

        Returns:
            Union[pd.Index, MaybeArray]: Array representing time deltas.
        """
        return self.index_acc.arr_to_timedelta(*args, **kwargs)

    @property
    def parse_index(self) -> tp.Optional[bool]:
        """Flag indicating whether to convert the index to a datetime index.

        Applied during initialization via `vectorbtpro.utils.datetime_.prepare_dt_index`.

        Returns:
            Optional[bool]: True if the index should be parsed; otherwise, False.
        """
        return self._parse_index

    @property
    def column_only_select(self) -> bool:
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        column_only_select = self._column_only_select
        if column_only_select is None:
            column_only_select = wrapping_cfg["column_only_select"]
        return column_only_select

    @property
    def range_only_select(self) -> bool:
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        range_only_select = self._range_only_select
        if range_only_select is None:
            range_only_select = wrapping_cfg["range_only_select"]
        return range_only_select

    @property
    def group_select(self) -> bool:
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        group_select = self._group_select
        if group_select is None:
            group_select = wrapping_cfg["group_select"]
        return group_select

    @property
    def grouper(self) -> Grouper:
        """`vectorbtpro.base.grouping.base.Grouper` instance used for grouping columns.

        Returns:
            Grouper: Grouper instance.
        """
        return self._grouper

    @property
    def grouped_ndim(self) -> int:
        """Number of dimensions after applying column grouping.

        If not explicitly set, it is derived from the grouper's state.

        Returns:
            int: Number of dimensions after grouping.
        """
        if self._grouped_ndim is None:
            if self.grouper.is_grouped():
                return 2 if self.grouper.get_group_count() > 1 else 1
            return self.ndim
        return self._grouped_ndim

    @cached_method(whitelist=True)
    def regroup(self: ArrayWrapperT, group_by: tp.GroupByLike, **kwargs) -> ArrayWrapperT:
        if self.grouper.is_grouping_changed(group_by=group_by):
            self.grouper.check_group_by(group_by=group_by)
            grouped_ndim = None
            if self.grouper.is_grouped(group_by=group_by):
                if not self.grouper.is_group_count_changed(group_by=group_by):
                    grouped_ndim = self.grouped_ndim
            return self.replace(grouped_ndim=grouped_ndim, group_by=group_by, **kwargs)
        if len(kwargs) > 0:
            return self.replace(**kwargs)
        return self  # important for keeping cache

    def flip(self: ArrayWrapperT, **kwargs) -> ArrayWrapperT:
        """Swap the index and columns of the `ArrayWrapper`.

        Args:
            **kwargs: Keyword arguments for `ArrayWrapper.replace`.

        Returns:
            ArrayWrapper: New `ArrayWrapper` instance with flipped index and columns.
        """
        if "grouper" not in kwargs:
            kwargs["grouper"] = None
        return self.replace(index=self.columns, columns=self.index, **kwargs)

    @cached_method(whitelist=True)
    def resolve(self: ArrayWrapperT, group_by: tp.GroupByLike = None, **kwargs) -> ArrayWrapperT:
        """Resolve the instance by regrouping and updating metadata.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `ArrayWrapper.regroup`.

        Returns:
            ArrayWrapper: Resolved `ArrayWrapper` instance.

        !!! note
            If the grouper indicates a valid grouping, replaces the instance's columns and
            related attributes with the grouped configuration. Returns the updated instance
            while preserving cache integrity.
        """
        _self = self.regroup(group_by=group_by, **kwargs)
        if _self.grouper.is_grouped():
            return _self.replace(
                columns=_self.grouper.get_index(),
                ndim=_self.grouped_ndim,
                grouped_ndim=None,
                group_by=None,
            )
        return _self  # important for keeping cache

    def get_index_grouper(self, *args, **kwargs) -> Grouper:
        """Return the index grouper using `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Returns:
            Grouper: Index grouper instance.
        """
        return self.index_acc.get_grouper(*args, **kwargs)

    def wrap(
        self,
        arr: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        zero_to_none: tp.Optional[bool] = None,
        force_2d: bool = False,
        fillna: tp.Optional[tp.Scalar] = None,
        dtype: tp.Optional[tp.PandasDTypeLike] = None,
        min_precision: tp.Union[None, int, str] = None,
        max_precision: tp.Union[None, int, str] = None,
        prec_float_only: tp.Optional[bool] = None,
        prec_check_bounds: tp.Optional[bool] = None,
        prec_strict: tp.Optional[bool] = None,
        to_timedelta: bool = False,
        to_index: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SeriesFrame:
        """Wrap the input array using stored metadata and configuration.

        The function performs the following steps:

        * Convert the input to a NumPy array.
        * Replace NaN values if a fill value is provided.
        * Adjust the array shape to match the stored index and columns.
        * Apply minimum and maximum precision casting if configured.
        * Create a Pandas Series or DataFrame based on the array's dimensionality.
        * Optionally map output values to the original index.
        * Optionally convert data to timedelta using `ArrayWrapper.arr_to_timedelta`.

        Args:
            arr (ArrayLike): Array to be wrapped.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            index (Optional[IndexLike]): Index to assign to the wrapped object.

                Uses the stored index if not provided.
            columns (Optional[IndexLike]): Column labels for the wrapped object.

                May be adjusted if a single zero-valued column is detected.
            zero_to_none (Optional[bool]): If True, converts a zero column name to None
                when a single column is present.
            force_2d (bool): If True, force the output to be two-dimensional.
            fillna (Optional[Scalar]): Value to replace missing data (NaN).
            dtype (Optional[PandasDTypeLike]): Data type for converting the output.
            min_precision (Union[None, int, str]): Minimum precision for numerical conversion.
            max_precision (Union[None, int, str]): Maximum precision for numerical conversion.
            prec_float_only (Optional[bool]): If True, apply precision conversion only to floating point numbers.
            prec_check_bounds (Optional[bool]): If True, enforce bounds checking during precision conversion.
            prec_strict (Optional[bool]): If True, apply strict checking during precision conversion.
            to_timedelta (bool): Flag indicating whether to convert the output to timedelta format.
            to_index (bool): If True, map output values to the original index.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            SeriesFrame: Wrapped Pandas Series or DataFrame with applied metadata.

        !!! info
            For default settings, see `vectorbtpro._settings.wrapping`.
        """
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if zero_to_none is None:
            zero_to_none = wrapping_cfg["zero_to_none"]
        if min_precision is None:
            min_precision = wrapping_cfg["min_precision"]
        if max_precision is None:
            max_precision = wrapping_cfg["max_precision"]
        if prec_float_only is None:
            prec_float_only = wrapping_cfg["prec_float_only"]
        if prec_check_bounds is None:
            prec_check_bounds = wrapping_cfg["prec_check_bounds"]
        if prec_strict is None:
            prec_strict = wrapping_cfg["prec_strict"]
        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        _self = self.resolve(group_by=group_by)

        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if len(columns) == 1:
            name = columns[0]
            if zero_to_none and name == 0:  # was a Series before
                name = None
        else:
            name = None

        def _apply_dtype(obj):
            if dtype is None:
                return obj
            return obj.astype(dtype, errors="ignore")

        def _wrap(arr):
            orig_arr = arr
            arr = np.asarray(arr)
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            shape_2d = (arr.shape[0] if arr.ndim > 0 else 1, arr.shape[1] if arr.ndim > 1 else 1)
            target_shape_2d = (len(index), len(columns))
            if shape_2d != target_shape_2d:
                if isinstance(orig_arr, (pd.Series, pd.DataFrame)):
                    arr = reshaping.align_pd_arrays(
                        orig_arr, to_index=index, to_columns=columns
                    ).values
                arr = reshaping.broadcast_array_to(arr, target_shape_2d)
            arr = reshaping.soft_to_ndim(arr, self.ndim)
            if min_precision is not None:
                arr = cast_to_min_precision(
                    arr,
                    min_precision,
                    float_only=prec_float_only,
                )
            if max_precision is not None:
                arr = cast_to_max_precision(
                    arr,
                    max_precision,
                    float_only=prec_float_only,
                    check_bounds=prec_check_bounds,
                    strict=prec_strict,
                )
            if arr.ndim == 1:
                if force_2d:
                    return _apply_dtype(pd.DataFrame(arr[:, None], index=index, columns=columns))
                return _apply_dtype(pd.Series(arr, index=index, name=name))
            if arr.ndim == 2:
                if not force_2d and arr.shape[1] == 1 and _self.ndim == 1:
                    return _apply_dtype(pd.Series(arr[:, 0], index=index, name=name))
                return _apply_dtype(pd.DataFrame(arr, index=index, columns=columns))
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
        if to_timedelta:
            # Convert to timedelta
            out = self.arr_to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def wrap_reduced(
        self,
        arr: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        name_or_index: tp.NameIndex = None,
        columns: tp.Optional[tp.IndexLike] = None,
        force_1d: bool = False,
        fillna: tp.Optional[tp.Scalar] = None,
        dtype: tp.Optional[tp.PandasDTypeLike] = None,
        to_timedelta: bool = False,
        to_index: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.MaybeSeriesFrame:
        """Wrap the result of a reduction operation.

        Args:
            arr (ArrayLike): Input array to be wrapped.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            name_or_index (NameIndex): Name for a scalar reduction per column or index for an array reduction.
            columns (Optional[IndexLike]): Override for the object's default columns.
            force_1d (bool): Flag to force the input array to be treated as one-dimensional.
            fillna (Optional[Scalar]): Value to replace missing data (NaN).
            dtype (Optional[PandasDTypeLike]): Data type for converting the output.
            to_timedelta (bool): Flag indicating whether to convert the output to timedelta format.
            to_index (bool): Flag indicating whether to map scalar results to the object's index.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            MaybeSeriesFrame: Wrapped Series or DataFrame resulting from the reduction operation.

        !!! info
            For default settings, see `vectorbtpro._settings.wrapping`.

        !!! info
            See `ArrayWrapper.wrap` for details on the wrapping pipeline.
        """
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        _self = self.resolve(group_by=group_by)

        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        if to_index:
            if dtype is None:
                dtype = int_
            if fillna is None:
                fillna = -1

        def _apply_dtype(obj):
            if dtype is None:
                return obj
            return obj.astype(dtype, errors="ignore")

        def _wrap_reduced(arr):
            nonlocal name_or_index

            if isinstance(arr, dict):
                arr = reshaping.to_pd_array(arr)
            if isinstance(arr, pd.Series):
                if not checks.is_index_equal(arr.index, columns):
                    arr = arr.iloc[indexes.align_indexes(arr.index, columns)[0]]
            arr = np.asarray(arr)
            if force_1d and arr.ndim == 0:
                arr = arr[None]
            if fillna is not None:
                if arr.ndim == 0:
                    if pd.isnull(arr):
                        arr = fillna
                else:
                    arr[pd.isnull(arr)] = fillna
            if arr.ndim == 0:
                # Scalar per Series/DataFrame
                return _apply_dtype(pd.Series(arr[None]))[0]
            if arr.ndim == 1:
                if not force_1d and _self.ndim == 1:
                    if arr.shape[0] == 1:
                        # Scalar per Series/DataFrame with one column
                        return _apply_dtype(pd.Series(arr))[0]
                    # Array per Series
                    sr_name = columns[0]
                    if sr_name == 0:
                        sr_name = None
                    if isinstance(name_or_index, str):
                        name_or_index = None
                    return _apply_dtype(pd.Series(arr, index=name_or_index, name=sr_name))
                # Scalar per column in DataFrame
                if arr.shape[0] == 1 and len(columns) > 1:
                    arr = reshaping.broadcast_array_to(arr, len(columns))
                return _apply_dtype(pd.Series(arr, index=columns, name=name_or_index))
            if arr.ndim == 2:
                if arr.shape[1] == 1 and _self.ndim == 1:
                    arr = reshaping.soft_to_ndim(arr, 1)
                    # Array per Series
                    sr_name = columns[0]
                    if sr_name == 0:
                        sr_name = None
                    if isinstance(name_or_index, str):
                        name_or_index = None
                    return _apply_dtype(pd.Series(arr, index=name_or_index, name=sr_name))
                # Array per column in DataFrame
                if isinstance(name_or_index, str):
                    name_or_index = None
                if arr.shape[0] == 1 and len(columns) > 1:
                    arr = reshaping.broadcast_array_to(arr, (arr.shape[0], len(columns)))
                return _apply_dtype(pd.DataFrame(arr, index=name_or_index, columns=columns))
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap_reduced(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            elif checks.is_frame(out):
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = self.index[out] if out != -1 else np.nan
        if to_timedelta:
            # Convert to timedelta
            out = self.arr_to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def concat_arrs(
        self,
        *objs: tp.MaybeSequence[tp.ArrayLike],
        group_by: tp.GroupByLike = None,
        wrap: bool = True,
        **kwargs,
    ) -> tp.AnyArray1d:
        """Stack reduced arrays along columns and wrap the resulting object.

        Args:
            *objs (MaybeSequence[ArrayLike]): Arrays to concatenate.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap (bool): Flag indicating whether to apply wrapping to the concatenated result.
            **kwargs: Keyword arguments for `ArrayWrapper.wrap_reduced`.

        Returns:
            AnyArray1d: Stacked one-dimensional array after optional wrapping.
        """
        from vectorbtpro.base.merging import concat_arrays

        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        new_objs = []
        for obj in objs:
            new_objs.append(reshaping.to_1d_array(obj))

        stacked_obj = concat_arrays(new_objs)
        if wrap:
            _self = self.resolve(group_by=group_by)
            return _self.wrap_reduced(stacked_obj, **kwargs)
        return stacked_obj

    def row_stack_arrs(
        self,
        *objs: tp.MaybeSequence[tp.ArrayLike],
        group_by: tp.GroupByLike = None,
        wrap: bool = True,
        **kwargs,
    ) -> tp.AnyArray:
        """Stack arrays along rows and wrap the resulting object.

        Args:
            *objs (MaybeSequence[ArrayLike]): Arrays to stack.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap (bool): Flag indicating whether to apply wrapping to the stacked result.
            **kwargs: Keyword arguments for `ArrayWrapper.wrap`.

        Returns:
            AnyArray: Stacked array after optional wrapping.
        """
        from vectorbtpro.base.merging import row_stack_arrays

        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        new_objs = []
        for obj in objs:
            obj = reshaping.to_2d_array(obj)
            if obj.shape[1] != _self.shape_2d[1]:
                if obj.shape[1] != 1:
                    raise ValueError(
                        f"Cannot broadcast {obj.shape[1]} to {_self.shape_2d[1]} columns"
                    )
                obj = np.repeat(obj, _self.shape_2d[1], axis=1)
            new_objs.append(obj)

        stacked_obj = row_stack_arrays(new_objs)
        if wrap:
            return _self.wrap(stacked_obj, **kwargs)
        return stacked_obj

    def column_stack_arrs(
        self,
        *objs: tp.MaybeSequence[tp.ArrayLike],
        reindex_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        wrap: bool = True,
        **kwargs,
    ) -> tp.AnyArray2d:
        """Stack arrays along columns and wrap the resulting object.

        Args:
            *objs (MaybeSequence[ArrayLike]): Arrays to stack.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            wrap (bool): Flag indicating whether to apply wrapping to the concatenated result.
            **kwargs: Keyword arguments for `ArrayWrapper.wrap`.

        Returns:
            AnyArray2d: Concatenated two-dimensional array after optional wrapping.
        """
        from vectorbtpro.base.merging import column_stack_arrays

        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        new_objs = []
        for obj in objs:
            if not checks.is_index_equal(obj.index, _self.index, check_names=False):
                was_bool = (isinstance(obj, pd.Series) and obj.dtype == "bool") or (
                    isinstance(obj, pd.DataFrame) and (obj.dtypes == "bool").all()
                )
                obj = obj.reindex(_self.index, **resolve_dict(reindex_kwargs))
                is_object = (isinstance(obj, pd.Series) and obj.dtype == "object") or (
                    isinstance(obj, pd.DataFrame) and (obj.dtypes == "object").all()
                )
                if was_bool and is_object:
                    obj = obj.astype(None)
            new_objs.append(reshaping.to_2d_array(obj))

        stacked_obj = column_stack_arrays(new_objs)
        if wrap:
            return _self.wrap(stacked_obj, **kwargs)
        return stacked_obj

    def dummy(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Create a dummy Series or DataFrame with an empty array based on the internal shape.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Dummy Series or DataFrame.
        """
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.empty(_self.shape), **kwargs)

    def fill(
        self, fill_value: tp.Scalar = np.nan, group_by: tp.GroupByLike = None, **kwargs
    ) -> tp.SeriesFrame:
        """Fill a Series or DataFrame with a specified value.

        Args:
            fill_value (Scalar): Value used to fill the Series or DataFrame.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `ArrayWrapper.wrap`.

        Returns:
            SeriesFrame: Series or DataFrame with all elements filled with the specified value.
        """
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.full(_self.shape_2d, fill_value), **kwargs)

    def fill_reduced(
        self, fill_value: tp.Scalar = np.nan, group_by: tp.GroupByLike = None, **kwargs
    ) -> tp.SeriesFrame:
        """Fill a reduced Series or DataFrame with a specified value.

        Args:
            fill_value (Scalar): Value used to fill the reduced output.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `ArrayWrapper.wrap_reduced`.

        Returns:
            SeriesFrame: Reduced Series or DataFrame filled with the specified value.
        """
        _self = self.resolve(group_by=group_by)
        return _self.wrap_reduced(np.full(_self.shape_2d[1], fill_value), **kwargs)

    def apply_to_index(
        self: ArrayWrapperT,
        apply_func: tp.Callable,
        *args,
        axis: tp.Optional[int] = None,
        **kwargs,
    ) -> ArrayWrapperT:
        """Apply a function to the index of the `ArrayWrapper`.

        Args:
            apply_func (Callable): Function to apply to the index.
            *args: Positional arguments for `apply_func`.
            axis (Optional[int]): Axis to apply the function to.

                If None, defaults to 0 for one-dimensional arrays and 1 for two-dimensional arrays.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            ArrayWrapper: New `ArrayWrapper` instance with the modified index.
        """
        if axis is None:
            axis = 0 if self.ndim == 1 else 1
        if self.ndim == 1 and axis == 1:
            raise TypeError("Axis 1 is not supported for one dimension")
        checks.assert_in(axis, (0, 1))

        if axis == 1:
            return self.replace(columns=apply_func(self.columns, *args, **kwargs))
        return self.replace(index=apply_func(self.index, *args, **kwargs))

    def get_index_points(self, *args, **kwargs) -> tp.Array1d:
        """Return index points using `vectorbtpro.base.accessors.BaseIDXAccessor.get_points`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_points`.
            **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_points`.

        Returns:
            Array1d: Array representing the index points.
        """
        return self.index_acc.get_points(*args, **kwargs)

    def get_index_ranges(self, *args, **kwargs) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Return index ranges using `vectorbtpro.base.accessors.BaseIDXAccessor.get_ranges`.

        Args:
            *args: Positional arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_ranges`.
            **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseIDXAccessor.get_ranges`.

        Returns:
            Tuple[Array1d, Array1d]: Tuple containing two arrays that represent the index ranges.
        """
        return self.index_acc.get_ranges(*args, **kwargs)

    def fill_and_set(
        self,
        idx_setter: tp.Union[index_dict, IdxSetter, IdxSetterFactory],
        keep_flex: bool = False,
        fill_value: tp.Scalar = np.nan,
        **kwargs,
    ) -> tp.AnyArray:
        """Fill and set values in a new array based on an index object.

        Fill a new array using an index object that specifies positions and associated values.
        If `idx_setter` is not an instance of `vectorbtpro.base.indexing.IdxSetter`, it is wrapped accordingly.
        The method `vectorbtpro.base.indexing.IdxSetter.fill_and_set` is then called to update the array.

        Args:
            idx_setter (Union[index_dict, IdxSetter, IdxSetterFactory]): Index object indicating
                the positions and values to fill.

                If provided as a factory, it generates an `vectorbtpro.base.indexing.IdxSetter`.
                Otherwise, if given as an index object (e.g. `index_dict`), it is wrapped with
                    `vectorbtpro.base.indexing.IdxSetter`.
            keep_flex (bool): Whether to preserve the flexible array structure.

                If False, the resulting array is wrapped using `ArrayWrapper.wrap`.
            fill_value (Scalar): Default value used to fill positions not explicitly set.
            **kwargs: Keyword arguments for `vectorbtpro.base.indexing.IdxSetter.fill_and_set`.

        Returns:
            AnyArray: Resulting array with updated values, either wrapped or unwrapped
                depending on `keep_flex`.

        Examples:
            Set a single row:

            ```pycon
            >>> from vectorbtpro import *

            >>> index = pd.date_range("2020", periods=5)
            >>> columns = pd.Index(["a", "b", "c"])
            >>> wrapper = vbt.ArrayWrapper(index, columns)

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     1: 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     "2020-01-02": 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     "2020-01-02": [1, 2, 3]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN
            ```

            Set multiple rows:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     (1, 3): [2, 3]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  3.0  3.0  3.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [[1, 2, 3], [4, 5, 6]]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  4.0  5.0  6.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [[1, 2, 3]]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  1.0  2.0  3.0
            2020-01-05  NaN  NaN  NaN
            ```

            Set rows using slices:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.hslice(1, 3): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.hslice("2020-01-02", "2020-01-04"): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ((0, 2), (3, 5)): [[1], [2]]
            ... }))
                          a    b    c
            2020-01-01  1.0  1.0  1.0
            2020-01-02  1.0  1.0  1.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  2.0  2.0  2.0
            2020-01-05  2.0  2.0  2.0

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     ((0, 2), (3, 5)): [[1, 2, 3], [4, 5, 6]]
            ... }))
                          a    b    c
            2020-01-01  1.0  2.0  3.0
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  4.0  5.0  6.0
            2020-01-05  4.0  5.0  6.0
            ```

            Set rows using index points:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.pointidx(every="2D"): 2
            ... }))
                          a    b    c
            2020-01-01  2.0  2.0  2.0
            2020-01-02  NaN  NaN  NaN
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  2.0  2.0  2.0
            ```

            Set rows using index ranges:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.rangeidx(
            ...         start=("2020-01-01", "2020-01-03"),
            ...         end=("2020-01-02", "2020-01-05")
            ...     ): 2
            ... }))
                          a    b    c
            2020-01-01  2.0  2.0  2.0
            2020-01-02  NaN  NaN  NaN
            2020-01-03  2.0  2.0  2.0
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN
            ```

            Set column indices:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx("a"): 2
            ... }))
                          a   b   c
            2020-01-01  2.0 NaN NaN
            2020-01-02  2.0 NaN NaN
            2020-01-03  2.0 NaN NaN
            2020-01-04  2.0 NaN NaN
            2020-01-05  2.0 NaN NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx(("a", "b")): [1, 2]
            ... }))
                          a    b   c
            2020-01-01  1.0  2.0 NaN
            2020-01-02  1.0  2.0 NaN
            2020-01-03  1.0  2.0 NaN
            2020-01-04  1.0  2.0 NaN
            2020-01-05  1.0  2.0 NaN

            >>> multi_columns = pd.MultiIndex.from_arrays(
            ...     [["a", "a", "b", "b"], [1, 2, 1, 2]],
            ...     names=["c1", "c2"]
            ... )
            >>> multi_wrapper = vbt.ArrayWrapper(index, multi_columns)

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx(("a", 2)): 2
            ... }))
            c1           a        b
            c2           1    2   1   2
            2020-01-01 NaN  2.0 NaN NaN
            2020-01-02 NaN  2.0 NaN NaN
            2020-01-03 NaN  2.0 NaN NaN
            2020-01-04 NaN  2.0 NaN NaN
            2020-01-05 NaN  2.0 NaN NaN

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.colidx("b", level="c1"): [3, 4]
            ... }))
            c1           a        b
            c2           1   2    1    2
            2020-01-01 NaN NaN  3.0  4.0
            2020-01-02 NaN NaN  3.0  4.0
            2020-01-03 NaN NaN  3.0  4.0
            2020-01-04 NaN NaN  3.0  4.0
            2020-01-05 NaN NaN  3.0  4.0
            ```

            Set row and column indices:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(2, 2): 2
            ... }))
                         a   b    c
            2020-01-01 NaN NaN  NaN
            2020-01-02 NaN NaN  NaN
            2020-01-03 NaN NaN  2.0
            2020-01-04 NaN NaN  NaN
            2020-01-05 NaN NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(("2020-01-01", "2020-01-03"), 2): [1, 2]
            ... }))
                         a   b    c
            2020-01-01 NaN NaN  1.0
            2020-01-02 NaN NaN  NaN
            2020-01-03 NaN NaN  2.0
            2020-01-04 NaN NaN  NaN
            2020-01-05 NaN NaN  NaN

            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(("2020-01-01", "2020-01-03"), (0, 2)): [[1, 2], [3, 4]]
            ... }))
                          a   b    c
            2020-01-01  1.0 NaN  2.0
            2020-01-02  NaN NaN  NaN
            2020-01-03  3.0 NaN  4.0
            2020-01-04  NaN NaN  NaN
            2020-01-05  NaN NaN  NaN

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(
            ...         vbt.pointidx(every="2d"),
            ...         vbt.colidx(1, level="c2")
            ...     ): [[1, 2]]
            ... }))
            c1            a        b
            c2            1   2    1   2
            2020-01-01  1.0 NaN  2.0 NaN
            2020-01-02  NaN NaN  NaN NaN
            2020-01-03  1.0 NaN  2.0 NaN
            2020-01-04  NaN NaN  NaN NaN
            2020-01-05  1.0 NaN  2.0 NaN

            >>> multi_wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.idx(
            ...         vbt.pointidx(every="2d"),
            ...         vbt.colidx(1, level="c2")
            ...     ): [[1], [2], [3]]
            ... }))
            c1            a        b
            c2            1   2    1   2
            2020-01-01  1.0 NaN  1.0 NaN
            2020-01-02  NaN NaN  NaN NaN
            2020-01-03  2.0 NaN  2.0 NaN
            2020-01-04  NaN NaN  NaN NaN
            2020-01-05  3.0 NaN  3.0 NaN
            ```

            Set rows using a template:

            ```pycon
            >>> wrapper.fill_and_set(vbt.index_dict({
            ...     vbt.RepEval("index.day % 2 == 0"): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN
            ```
        """
        if isinstance(idx_setter, index_dict):
            idx_setter = IdxDict(idx_setter)
        if isinstance(idx_setter, IdxSetterFactory):
            idx_setter = idx_setter.get()
            if not isinstance(idx_setter, IdxSetter):
                raise ValueError("Index setter factory must return exactly one index setter")
        checks.assert_instance_of(idx_setter, IdxSetter)
        arr = idx_setter.fill_and_set(
            self.shape,
            keep_flex=keep_flex,
            fill_value=fill_value,
            index=self.index,
            columns=self.columns,
            freq=self.freq,
            **kwargs,
        )
        if not keep_flex:
            return self.wrap(arr, group_by=False)
        return arr


WrappingT = tp.TypeVar("WrappingT", bound="Wrapping")


class Wrapping(Configured, HasWrapper, IndexApplier, AttrResolverMixin):
    """Class for wrapping functionalities with a global `ArrayWrapper`.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.
    """

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        checks.assert_instance_of(wrapper, ArrayWrapper)
        self._wrapper = wrapper

        Configured.__init__(self, wrapper=wrapper, **kwargs)
        HasWrapper.__init__(self)
        AttrResolverMixin.__init__(self)

    @classmethod
    def resolve_row_stack_kwargs(
        cls, *wrappings: tp.MaybeSequence[WrappingT], **kwargs
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking along rows.

        Args:
            *wrappings (MaybeSequence[Wrapping]): Wrapping instances to be stacked.
            **kwargs: Additional keyword arguments.

        Returns:
            Kwargs: Resolved keyword arguments.
        """
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls, *wrappings: tp.MaybeSequence[WrappingT], **kwargs
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking along columns.

        Args:
            *wrappings (MaybeSequence[Wrapping]): Wrapping instances to be stacked.
            **kwargs: Additional keyword arguments.

        Returns:
            Kwargs: Resolved keyword arguments.
        """
        return kwargs

    @classmethod
    def resolve_stack_kwargs(cls, *wrappings: tp.MaybeSequence[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking.

        Args:
            *wrappings (MaybeSequence[Wrapping]): Wrapping instances to be stacked.
            **kwargs: Keyword arguments for `Wrapping.resolve_merge_kwargs`.

        Returns:
            Kwargs: Resolved keyword arguments.

        !!! note
            Should be called after `Wrapping.resolve_row_stack_kwargs` or `Wrapping.resolve_column_stack_kwargs`.
        """
        return cls.resolve_merge_kwargs(*[wrapping.config for wrapping in wrappings], **kwargs)

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[WrappingT],
        *objs: tp.MaybeSequence[WrappingT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> WrappingT:
        """Stack multiple `Wrapping` instances along rows.

        Args:
            *objs (MaybeSequence[Wrapping]): Wrapping instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `ArrayWrapper`.
            **kwargs: Additional keyword arguments.

        Returns:
            Wrapping: New wrapping instance resulting from stacking.

        !!! note
            Should use `ArrayWrapper.row_stack` for stacking.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[WrappingT],
        *objs: tp.MaybeSequence[WrappingT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> WrappingT:
        """Stack multiple `Wrapping` instances along columns.

        Args:
            *objs (MaybeSequence[Wrapping]): Wrapping instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `ArrayWrapper`.
            **kwargs: Additional keyword arguments.

        Returns:
            Wrapping: New wrapping instance resulting from stacking.

        !!! note
            Should use `ArrayWrapper.column_stack` for stacking.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def indexing_func(self: WrappingT, *args, **kwargs) -> WrappingT:
        """Perform indexing on `Wrapping` using the wrapper's indexing function.

        Args:
            *args: Positional arguments for `Wrapping.indexing_func`.
            **kwargs: Keyword arguments for `Wrapping.indexing_func`.

        Returns:
            Wrapping: New wrapping instance resulting from indexing.
        """
        new_wrapper = self.wrapper.indexing_func(
            *args,
            column_only_select=self.column_only_select,
            range_only_select=self.range_only_select,
            group_select=self.group_select,
            **kwargs,
        )
        return self.replace(wrapper=new_wrapper)

    def resample(self: WrappingT, *args, **kwargs) -> WrappingT:
        """Perform resampling on `Wrapping`.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Wrapping: New wrapping instance resulting from resampling.

        !!! note
            When overriding, pass `*args` and `**kwargs` to `ArrayWrapper.get_resampler` to create a resampler.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def wrapper(self) -> ArrayWrapper:
        return self._wrapper

    @property
    def column_only_select(self) -> bool:
        column_only_select = getattr(self, "_column_only_select", None)
        if column_only_select is None:
            return self.wrapper.column_only_select
        return column_only_select

    @property
    def range_only_select(self) -> bool:
        range_only_select = getattr(self, "_range_only_select", None)
        if range_only_select is None:
            return self.wrapper.range_only_select
        return range_only_select

    @property
    def group_select(self) -> bool:
        group_select = getattr(self, "_group_select", None)
        if group_select is None:
            return self.wrapper.group_select
        return group_select

    def apply_to_index(
        self: ArrayWrapperT,
        apply_func: tp.Callable,
        *args,
        axis: tp.Optional[int] = None,
        **kwargs,
    ) -> ArrayWrapperT:
        """Apply a function to the index of the `Wrapping`.

        Args:
            apply_func (Callable): Function to apply to the index.
            *args: Positional arguments for `apply_func`.
            axis (Optional[int]): Axis to apply the function to.

                If None, defaults to 0 for one-dimensional arrays and 1 for two-dimensional arrays.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            ArrayWrapper: New `Wrapping` instance with the modified index.
        """
        if axis is None:
            axis = 0 if self.wrapper.ndim == 1 else 1
        if self.wrapper.ndim == 1 and axis == 1:
            raise TypeError("Axis 1 is not supported for one dimension")
        checks.assert_in(axis, (0, 1))

        if axis == 1:
            new_wrapper = self.wrapper.replace(
                columns=apply_func(self.wrapper.columns, *args, **kwargs)
            )
        else:
            new_wrapper = self.wrapper.replace(
                index=apply_func(self.wrapper.index, *args, **kwargs)
            )
        return self.replace(wrapper=new_wrapper)

    def regroup(self: WrappingT, group_by: tp.GroupByLike, **kwargs) -> WrappingT:
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.replace(wrapper=self.wrapper.regroup(group_by, **kwargs))
        return self  # important for keeping cache

    def resolve_self(
        self: AttrResolverMixinT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.Optional[tp.Set[str]] = None,
        impacts_caching: bool = True,
        silence_warnings: tp.Optional[bool] = None,
    ) -> AttrResolverMixinT:
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()
        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        if "freq" in cond_kwargs:
            wrapper_copy = self.wrapper.replace(freq=cond_kwargs["freq"])

            if wrapper_copy.freq != self.wrapper.freq:
                if not silence_warnings:
                    warn(
                        "Changing the frequency will create a copy of this instance. "
                        "Consider setting it upon instantiation to re-use existing cache."
                    )
                self_copy = self.replace(wrapper=wrapper_copy)
                for alias in self.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs["freq"] = self_copy.wrapper.freq
                if impacts_caching:
                    cond_kwargs["use_caching"] = False
                return self_copy
        return self
