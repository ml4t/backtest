# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing extensions for chunking of base operations."""

import uuid

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.chunking import (
    ArgGetter,
    ArgSizer,
    ArraySelector,
    ArraySizer,
    ArraySlicer,
    Chunked,
    ChunkMapper,
    ChunkMeta,
    ChunkSlicer,
    ShapeSlicer,
)
from vectorbtpro.utils.parsing import Regex

__all__ = [
    "GroupLensSizer",
    "GroupLensSlicer",
    "ChunkedGroupLens",
    "GroupLensMapper",
    "GroupMapSlicer",
    "ChunkedGroupMap",
    "GroupIdxsMapper",
    "FlexArraySizer",
    "FlexArraySelector",
    "FlexArraySlicer",
    "ChunkedFlexArray",
    "shape_gl_slicer",
    "flex_1d_array_gl_slicer",
    "flex_array_gl_slicer",
    "array_gl_slicer",
]


class GroupLensSizer(ArgSizer):
    """Class for determining the size of group lengths.

    The input argument may be a group map tuple or a group lengths array.
    """

    @classmethod
    def get_obj_size(
        cls, obj: tp.Union[tp.GroupLens, tp.GroupMap], single_type: tp.Optional[type] = None
    ) -> int:
        """Get the size of the provided object based on group lengths.

        Args:
            obj (Union[GroupLens, GroupMap]): Group lengths array or a group map tuple.
            single_type (Optional[type]): Type of value that is considered single.

        Returns:
            int: Size computed from the object.
        """
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        if isinstance(obj, tuple):
            return len(obj[1])
        return len(obj)

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), single_type=self.single_type)


class GroupLensSlicer(ChunkSlicer):
    """Class for slicing a subset of group lengths or a group map based on chunk metadata."""

    def get_size(self, obj: tp.Union[tp.GroupLens, tp.GroupMap], **kwargs) -> int:
        return GroupLensSizer.get_obj_size(obj, single_type=self.single_type)

    def take(
        self, obj: tp.Union[tp.GroupLens, tp.GroupMap], chunk_meta: ChunkMeta, **kwargs
    ) -> tp.GroupMap:
        if isinstance(obj, tuple):
            return obj[1][chunk_meta.start : chunk_meta.end]
        return obj[chunk_meta.start : chunk_meta.end]


class ChunkedGroupLens(Chunked):
    """Class representing group lengths that support chunking operations."""

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.select:
                raise ValueError("Selection is not supported")
            return GroupLensSlicer
        return self.take_spec


def get_group_lens_slice(group_lens: tp.GroupLens, chunk_meta: ChunkMeta) -> slice:
    """Compute a slice object for group lengths using the provided chunk metadata.

    Args:
        group_lens (GroupLens): Array defining the number of columns in each group.
        chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.

    Returns:
        slice: Slice object representing the group lengths segment.
    """
    group_lens_cumsum = np.cumsum(group_lens[: chunk_meta.end])
    start = group_lens_cumsum[chunk_meta.start] - group_lens[chunk_meta.start]
    end = group_lens_cumsum[-1]
    return slice(start, end)


@define
class GroupLensMapper(ChunkMapper, ArgGetter, DefineMixin):
    """Class for mapping chunk metadata to corresponding per-group column lengths.

    The provided argument may be a group map tuple or an array of group lengths.
    """

    def map(
        self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs
    ) -> ChunkMeta:
        group_lens = self.get_arg(ann_args)
        if isinstance(group_lens, tuple):
            group_lens = group_lens[1]
        group_lens_slice = get_group_lens_slice(group_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=group_lens_slice.start,
            end=group_lens_slice.stop,
            indices=None,
        )


group_lens_mapper = GroupLensMapper(arg_query=Regex(r"(group_lens|group_map)"))
"""Default instance of `GroupLensMapper`."""


class GroupMapSlicer(ChunkSlicer):
    """Class for slicing a subset of a group map based on provided chunk metadata."""

    def get_size(self, obj: tp.GroupMap, **kwargs) -> int:
        return GroupLensSizer.get_obj_size(obj, single_type=self.single_type)

    def take(self, obj: tp.GroupMap, chunk_meta: ChunkMeta, **kwargs) -> tp.GroupMap:
        group_idxs, group_lens = obj
        group_lens = group_lens[chunk_meta.start : chunk_meta.end]
        return np.arange(np.sum(group_lens)), group_lens


class ChunkedGroupMap(Chunked):
    """Class representing a group map that supports chunking operations."""

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.select:
                raise ValueError("Selection is not supported")
            return GroupMapSlicer
        return self.take_spec


@define
class GroupIdxsMapper(ChunkMapper, ArgGetter, DefineMixin):
    """Class for mapping chunk metadata to per-group column indices using a group map tuple."""

    def map(
        self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs
    ) -> ChunkMeta:
        group_map = self.get_arg(ann_args)
        group_idxs, group_lens = group_map
        group_lens_slice = get_group_lens_slice(group_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=None,
            end=None,
            indices=group_idxs[group_lens_slice],
        )


group_idxs_mapper = GroupIdxsMapper(arg_query="group_map")
"""Default instance of `GroupIdxsMapper`."""


class FlexArraySizer(ArraySizer):
    """Class for determining the size of a flexible array along a specified axis."""

    @classmethod
    def get_obj_size(
        cls, obj: tp.AnyArray, axis: int, single_type: tp.Optional[type] = None
    ) -> int:
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        obj = np.asarray(obj)
        if len(obj.shape) == 0:
            return 1
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        checks.assert_in(axis, (0, 1), arg_name="axis")
        if len(obj.shape) == 1:
            if axis == 1:
                return 1
            return obj.shape[0]
        if len(obj.shape) == 2:
            if axis == 1:
                return obj.shape[1]
            return obj.shape[0]
        raise ValueError(f"FlexArraySizer supports max 2 dimensions, not {len(obj.shape)}")


@define
class FlexArraySelector(ArraySelector, DefineMixin):
    """Class for flexibly selecting a single element from a NumPy array along a specified axis
    based on a chunk index.

    This selector is intended for use with `vectorbtpro.base.flex_indexing.flex_select_1d_nb` and
    `vectorbtpro.base.flex_indexing.flex_select_nb`.
    """

    def get_size(self, obj: tp.ArrayLike, **kwargs) -> int:
        return FlexArraySizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def suggest_size(self, obj: tp.ArrayLike, **kwargs) -> tp.Optional[int]:
        return None

    def take(
        self,
        obj: tp.ArrayLike,
        chunk_meta: ChunkMeta,
        ann_args: tp.Optional[tp.AnnArgs] = None,
        **kwargs,
    ) -> tp.ArrayLike:
        if np.isscalar(obj):
            return obj
        obj = np.asarray(obj)
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        checks.assert_in(axis, (0, 1), arg_name="axis")
        if len(obj.shape) == 1:
            if axis == 1 or obj.shape[0] == 1:
                return obj
            if self.keep_dims:
                return obj[chunk_meta.idx : chunk_meta.idx + 1]
            return obj[chunk_meta.idx]
        if len(obj.shape) == 2:
            if axis == 1:
                if obj.shape[1] == 1:
                    return obj
                if self.keep_dims:
                    return obj[:, chunk_meta.idx : chunk_meta.idx + 1]
                return obj[:, chunk_meta.idx]
            if obj.shape[0] == 1:
                return obj
            if self.keep_dims:
                return obj[chunk_meta.idx : chunk_meta.idx + 1, :]
            return obj[chunk_meta.idx, :]
        raise ValueError(f"FlexArraySelector supports max 2 dimensions, not {len(obj.shape)}")


@define
class FlexArraySlicer(ArraySlicer, DefineMixin):
    """Class for flexibly slicing a NumPy array along a specified axis using chunk indices.

    This slicer is intended for use with `vectorbtpro.base.flex_indexing.flex_select_1d_nb` and
    `vectorbtpro.base.flex_indexing.flex_select_nb`.
    """

    def get_size(self, obj: tp.ArrayLike, **kwargs) -> int:
        return FlexArraySizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def suggest_size(self, obj: tp.ArrayLike, **kwargs) -> tp.Optional[int]:
        return None

    def take(
        self,
        obj: tp.ArrayLike,
        chunk_meta: ChunkMeta,
        ann_args: tp.Optional[tp.AnnArgs] = None,
        **kwargs,
    ) -> tp.ArrayLike:
        if np.isscalar(obj):
            return obj
        obj = np.asarray(obj)
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        checks.assert_in(axis, (0, 1), arg_name="axis")
        if len(obj.shape) == 1:
            if axis == 1 or obj.shape[0] == 1:
                return obj
            return obj[chunk_meta.start : chunk_meta.end]
        if len(obj.shape) == 2:
            if axis == 1:
                if obj.shape[1] == 1:
                    return obj
                return obj[:, chunk_meta.start : chunk_meta.end]
            if obj.shape[0] == 1:
                return obj
            return obj[chunk_meta.start : chunk_meta.end, :]
        raise ValueError(f"FlexArraySlicer supports max 2 dimensions, not {len(obj.shape)}")


class ChunkedFlexArray(Chunked):
    """Class for a chunkable flexible array that dynamically resolves its take specification.

    This class determines the appropriate take specification based on its configuration.
    """

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.select:
                return FlexArraySelector
            return FlexArraySlicer
        return self.take_spec


shape_gl_slicer = ShapeSlicer(axis=1, mapper=group_lens_mapper)
"""Flexible 2-dimensional shape slicer along the column axis based on group lengths."""

flex_1d_array_gl_slicer = FlexArraySlicer(mapper=group_lens_mapper)
"""Flexible 1-dimensional array slicer along the column axis based on group lengths."""

flex_array_gl_slicer = FlexArraySlicer(axis=1, mapper=group_lens_mapper)
"""Flexible 2-dimensional array slicer along the column axis based on group lengths."""

array_gl_slicer = ArraySlicer(axis=1, mapper=group_lens_mapper)
"""2-dimensional array slicer along the column axis based on group lengths."""
