# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module for mapping column arrays."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.grouping import nb as grouping_nb
from vectorbtpro.base.reshaping import to_1d_array
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.records import nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.decorators import cached_method, cached_property, hybrid_method

__all__ = [
    "ColumnMapper",
]

ColumnMapperT = tp.TypeVar("ColumnMapperT", bound="ColumnMapper")


class ColumnMapper(Wrapping):
    """Class for mapping column arrays.

    Used by `vectorbtpro.records.base.Records` and `vectorbtpro.records.mapped_array.MappedArray`
    classes to make use of column and group metadata.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        col_arr (Array1d): Array of column indices.
        **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.Wrapping`.
    """

    def __init__(self, wrapper: ArrayWrapper, col_arr: tp.Array1d, **kwargs) -> None:
        Wrapping.__init__(self, wrapper, col_arr=col_arr, **kwargs)

        self._col_arr = col_arr

        # Cannot select rows
        self._column_only_select = True

    @property
    def col_arr(self) -> tp.Array1d:
        """Column array.

        Returns:
            Array1d: Column array.
        """
        return self._col_arr

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[ColumnMapperT],
        *objs: tp.MaybeSequence[ColumnMapperT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ColumnMapperT:
        """Stack multiple `ColumnMapper` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        Args:
            *objs (MaybeSequence[ColumnMapper]): (Additional) `ColumnMapper` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `ColumnMapper` through
                `ColumnMapper.resolve_row_stack_kwargs` and `ColumnMapper.resolve_stack_kwargs`.

        Returns:
            ColumnMapper: New column mapper instance with row-stacked wrappers and updated column metadata.

        !!! note
            Will produce a column-sorted array.
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
            if not checks.is_instance_of(obj, ColumnMapper):
                raise TypeError("Each object to be merged must be an instance of ColumnMapper")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs], **wrapper_kwargs
            )

        if "col_arr" not in kwargs:
            col_arrs = []
            for col in range(kwargs["wrapper"].shape_2d[1]):
                for obj in objs:
                    col_idxs, col_lens = obj.col_map
                    if len(col_idxs) > 0:
                        if col > 0 and obj.wrapper.shape_2d[1] == 1:
                            col_arrs.append(np.full(col_lens[0], col))
                        elif col_lens[col] > 0:
                            col_arrs.append(np.full(col_lens[col], col))
            kwargs["col_arr"] = np.concatenate(col_arrs)
        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[ColumnMapperT],
        *objs: tp.MaybeSequence[ColumnMapperT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ColumnMapperT:
        """Stack multiple `ColumnMapper` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        Args:
            *objs (MaybeSequence[ColumnMapper]): (Additional) `ColumnMapper` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `ColumnMapper` through
                `ColumnMapper.resolve_column_stack_kwargs` and `ColumnMapper.resolve_stack_kwargs`.

        Returns:
            ColumnMapper: New column mapper instance with column-stacked wrappers and updated column metadata.

        !!! note
            Will produce a column-sorted array.
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
            if not checks.is_instance_of(obj, ColumnMapper):
                raise TypeError("Each object to be merged must be an instance of ColumnMapper")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        if "col_arr" not in kwargs:
            col_arrs = []
            col_sum = 0
            for obj in objs:
                col_idxs, col_lens = obj.col_map
                if len(col_idxs) > 0:
                    col_arrs.append(obj.col_arr[col_idxs] + col_sum)
                col_sum += obj.wrapper.shape_2d[1]
            kwargs["col_arr"] = np.concatenate(col_arrs)
        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    def select_cols(
        self,
        col_idxs: tp.MaybeIndexArray,
        jitted: tp.JittedOption = None,
    ) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Select columns.

        Automatically chooses between using column lengths or column map based on sorted status.

        Args:
            col_idxs (MaybeIndexArray): Column indices or slice to select.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            Tuple[Array1d, Array1d]: Tuple containing the new indices and the updated column array.

        See:
            * `vectorbtpro.base.grouping.nb.group_lens_select_nb` if `ColumnMapper.is_sorted` returns True.
            * `vectorbtpro.base.grouping.nb.group_map_select_nb` if `ColumnMapper.is_sorted` returns False.
        """
        if len(self.col_arr) == 0:
            return np.arange(len(self.col_arr)), self.col_arr
        if isinstance(col_idxs, slice):
            if col_idxs.start is None and col_idxs.stop is None:
                return np.arange(len(self.col_arr)), self.col_arr
            col_idxs = np.arange(col_idxs.start, col_idxs.stop)
        if self.is_sorted():
            func = jit_reg.resolve_option(grouping_nb.group_lens_select_nb, jitted)
            new_indices, new_col_arr = func(self.col_lens, to_1d_array(col_idxs))  # faster
        else:
            func = jit_reg.resolve_option(grouping_nb.group_map_select_nb, jitted)
            new_indices, new_col_arr = func(self.col_map, to_1d_array(col_idxs))  # more flexible
        return new_indices, new_col_arr

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `ColumnMapper` and return metadata.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.

        Returns:
            dict: Dictionary with the following keys:

                * `wrapper_meta`: Metadata from the wrapper's indexing function.
                * `new_indices`: Indices after selecting columns.
                * `new_col_arr`: Updated column array.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(
                *args,
                column_only_select=self.column_only_select,
                group_select=self.group_select,
                **kwargs,
            )
        new_indices, new_col_arr = self.select_cols(wrapper_meta["col_idxs"])
        return dict(
            wrapper_meta=wrapper_meta,
            new_indices=new_indices,
            new_col_arr=new_col_arr,
        )

    def indexing_func(
        self: ColumnMapperT, *args, col_mapper_meta: tp.DictLike = None, **kwargs
    ) -> ColumnMapperT:
        """Perform indexing on `ColumnMapper`.

        Args:
            *args: Positional arguments for `ColumnMapper.indexing_func_meta`.
            col_mapper_meta (DictLike): Optional precomputed metadata for column mapping.

                If not provided, it is derived from `ColumnMapper.indexing_func_meta`.
            **kwargs: Keyword arguments for `ColumnMapper.indexing_func_meta`.

        Returns:
            ColumnMapper: New column mapper instance with indexing applied.
        """
        if col_mapper_meta is None:
            col_mapper_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=col_mapper_meta["wrapper_meta"]["new_wrapper"],
            col_arr=col_mapper_meta["new_col_arr"],
        )

    @cached_method(whitelist=True)
    def get_col_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """Get group-aware column array.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Array1d: Column array adjusted for grouping.
        """
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        return col_arr

    @cached_property(whitelist=True)
    def col_lens(self) -> tp.GroupLens:
        """Column lengths.

        Faster than `ColumnMapper.col_map` but only compatible with sorted columns.

        Returns:
            GroupLens: Column lengths.

        See:
            `vectorbtpro.records.nb.col_lens_nb`
        """
        func = jit_reg.resolve_option(nb.col_lens_nb, None)
        return func(self.col_arr, len(self.wrapper.columns))

    @cached_method(whitelist=True)
    def get_col_lens(
        self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None
    ) -> tp.GroupLens:
        """Get group-aware column lengths.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            GroupLens: Group-aware column lengths.

        See:
            `vectorbtpro.records.nb.col_lens_nb`
        """
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_lens
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_reg.resolve_option(nb.col_lens_nb, jitted)
        return func(col_arr, len(columns))

    @cached_property(whitelist=True)
    def col_map(self) -> tp.GroupMap:
        """Column map.

        More flexible than `ColumnMapper.col_lens` and more suited for mapped arrays.

        Returns:
            GroupMap: Column mapping.

        See:
            `vectorbtpro.records.nb.col_map_nb`
        """
        func = jit_reg.resolve_option(nb.col_map_nb, None)
        return func(self.col_arr, len(self.wrapper.columns))

    @cached_method(whitelist=True)
    def get_col_map(
        self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None
    ) -> tp.GroupMap:
        """Get group-aware column map.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            GroupMap: Group-aware column mapping.

        See:
            `vectorbtpro.records.nb.col_map_nb`
        """
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_map
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_reg.resolve_option(nb.col_map_nb, jitted)
        return func(col_arr, len(columns))

    @cached_method(whitelist=True)
    def is_sorted(self, jitted: tp.JittedOption = None) -> bool:
        """Check whether the column array is sorted.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        Returns:
            bool: True if the column array is sorted, otherwise False.

        See:
            `vectorbtpro.records.nb.is_col_sorted_nb`
        """
        func = jit_reg.resolve_option(nb.is_col_sorted_nb, jitted)
        return func(self.col_arr)

    @cached_property(whitelist=True)
    def new_id_arr(self) -> tp.Array1d:
        """New ID array derived from the column array and the wrapper's 2D shape.

        Returns:
            Array1d: New ID array.

        See:
            `vectorbtpro.records.nb.generate_ids_nb`
        """
        func = jit_reg.resolve_option(nb.generate_ids_nb, None)
        return func(self.col_arr, self.wrapper.shape_2d[1])

    @cached_method(whitelist=True)
    def get_new_id_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """Generate a new group-aware id array.

        Computes a group-aware id array by applying a JIT-compiled function to the column array.
        If a valid grouping specification is provided, `ColumnMapper.col_arr` is mapped using
        the grouping before generating the id array.

        Args:
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.

        Returns:
            Array1d: New group-aware id array.

        See:
            `vectorbtpro.records.nb.generate_ids_nb`
        """
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_reg.resolve_option(nb.generate_ids_nb, None)
        return func(col_arr, len(columns))
