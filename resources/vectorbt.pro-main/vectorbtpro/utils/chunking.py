# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for chunking.

!!! info
    For default settings, see `vectorbtpro._settings.chunking`.
"""

import inspect
import multiprocessing
import uuid
from functools import wraps

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.annotations import Annotatable, Union, flatten_annotations, get_annotations
from vectorbtpro.utils.attr_ import MISSING, DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import Configured, FrozenConfig, merge_dicts
from vectorbtpro.utils.eval_ import Evaluable
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.merging import MergeFunc, parse_merge_func
from vectorbtpro.utils.parsing import (
    Regex,
    ann_args_to_args,
    annotate_args,
    get_func_arg_names,
    match_ann_arg,
)
from vectorbtpro.utils.template import Rep, substitute_templates
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "ChunkMeta",
    "ArgChunkMeta",
    "LenChunkMeta",
    "iter_chunk_meta",
    "Sizer",
    "ArgSizer",
    "CountSizer",
    "LenSizer",
    "ShapeSizer",
    "ArraySizer",
    "ChunkMapper",
    "NotChunked",
    "ChunkTaker",
    "ChunkSelector",
    "ChunkSlicer",
    "CountAdapter",
    "ShapeSelector",
    "ShapeSlicer",
    "ArraySelector",
    "ArraySlicer",
    "ContainerTaker",
    "SequenceTaker",
    "MappingTaker",
    "ArgsTaker",
    "KwargsTaker",
    "Chunkable",
    "Chunked",
    "ChunkedCount",
    "ChunkedShape",
    "ChunkedArray",
    "Chunker",
    "chunked",
]

__pdoc__ = {}


# ############# Universal ############# #


@define
class ArgGetter(DefineMixin):
    """Class for retrieving an argument from annotated arguments using a specified query."""

    arg_query: tp.Optional[tp.AnnArgQuery] = define.field(default=None)
    """Query for the annotated argument from which to derive size."""

    def get_arg(self, ann_args: tp.AnnArgs) -> tp.Any:
        """Retrieve argument using `vectorbtpro.utils.parsing.match_ann_arg`.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            Any: Argument value.
        """
        if self.arg_query is None:
            raise ValueError("Please provide arg_query")
        return match_ann_arg(ann_args, self.arg_query)


@define
class AxisSpecifier(DefineMixin):
    """Class for specifying an axis."""

    axis: tp.Optional[int] = define.field(default=None)
    """Specifies the axis from which to extract data."""


@define
class DimRetainer(DefineMixin):
    """Class for retaining dimensions in an output."""

    keep_dims: bool = define.field(default=False)
    """Flag indicating whether to retain dimensions."""


# ############# Chunk sizing ############# #


@define
class Sizer(Evaluable, Annotatable, DefineMixin):
    """Abstract base class for determining size from annotated arguments.

    !!! note
        Use `Sizer.apply` instead of calling `Sizer.get_size` directly.
    """

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """Identifier or sequence of identifiers used for evaluating this instance."""

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        """Retrieve the size based on the provided annotated arguments.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            int: Retrieved size.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def apply(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        """Apply the sizer to compute the size from annotated arguments.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            int: Retrieved size.
        """
        return self.get_size(ann_args, **kwargs)


@define
class ArgSizer(Sizer, ArgGetter, DefineMixin):
    """Class for determining size based on an argument extracted from annotated arguments."""

    single_type: tp.Optional[tp.TypeLike] = define.field(default=None)
    """Type or tuple of types considered as representing a single value."""

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_arg(ann_args)

    def apply(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        arg = self.get_arg(ann_args)
        if self.single_type is not None:
            if checks.is_instance_of(arg, self.single_type):
                return 1
        return self.get_size(ann_args, **kwargs)


class CountSizer(ArgSizer):
    """Class for determining size based on a count value."""

    @classmethod
    def get_obj_size(cls, obj: int, single_type: tp.Optional[type] = None) -> int:
        """Compute size from a count.

        If `single_type` is provided and the object matches it, returns 1;
        otherwise, returns the count itself.

        Args:
            obj (int): Object.
            single_type (Optional[type]): Type of value that is considered single.

        Returns:
            int: Computed size of the object.
        """
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        return obj

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), single_type=self.single_type)


class LenSizer(ArgSizer):
    """Class for determining size based on the length of an argument."""

    @classmethod
    def get_obj_size(cls, obj: tp.Sequence, single_type: tp.Optional[type] = None) -> int:
        """Compute size as the length of a sequence, returning 1 if the object matches
        `single_type`; otherwise, return its length.

        Args:
            obj (int): Object.
            single_type (Optional[type]): Type of value that is considered single.

        Returns:
            int: Computed size of the object.
        """
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        return len(obj)

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), single_type=self.single_type)


@define
class ShapeSizer(ArgSizer, AxisSpecifier, DefineMixin):
    """Class for determining size from a specified axis in a shape-like object."""

    @classmethod
    def get_obj_size(
        cls, obj: tp.ShapeLike, axis: int, single_type: tp.Optional[type] = None
    ) -> int:
        """Compute size along a given axis from a shape-like object.

        If `single_type` is provided and the object matches it, returns 1.
        Converts an integer input to a tuple, defaults to axis 0 if unspecified for
        single-dimensional objects, and returns 0 if the axis is out of bounds.

        Args:
            obj (int): Object.
            axis (int): Axis of the object.
            single_type (Optional[type]): Type of value that is considered single.

        Returns:
            int: Computed size of the object's axis.
        """
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        if checks.is_int(obj):
            obj = (obj,)
        if len(obj) == 0:
            return 0
        if axis is None:
            if len(obj) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis <= len(obj) - 1:
            return obj[axis]
        return 0

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), self.axis, single_type=self.single_type)


class ArraySizer(ShapeSizer):
    """Class for determining size along a specified axis in an array."""

    @classmethod
    def get_obj_size(
        cls, obj: tp.AnyArray, axis: int, single_type: tp.Optional[type] = None
    ) -> int:
        from vectorbtpro.base.wrapping import Wrapping

        if isinstance(obj, Wrapping):
            shape = obj.wrapper.shape
        else:
            shape = obj.shape
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        if len(shape) == 0:
            return 0
        if axis is None:
            if len(shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis <= len(shape) - 1:
            return shape[axis]
        return 0

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), self.axis, single_type=self.single_type)


# ############# Chunk generation ############# #


@define
class ChunkMeta(DefineMixin):
    """Class representing metadata for a chunk."""

    uuid: str = define.field()
    """Unique identifier for the chunk, used for caching."""

    idx: int = define.field()
    """Index of the chunk."""

    start: tp.Optional[int] = define.field()
    """Starting index of the chunk range (inclusive); may be None."""

    end: tp.Optional[int] = define.field()
    """Ending index of the chunk range (exclusive); may be None."""

    indices: tp.Optional[tp.Sequence[int]] = define.field()
    """Sequence of indices included in the chunk; takes precedence over `start` and `end`
    if provided, and may be None."""


class ChunkMetaGenerator(Base):
    """Abstract base class for generating chunk metadata based on annotated arguments."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs, **kwargs) -> tp.Iterable[ChunkMeta]:
        """Generate an iterable of chunk metadata from the provided annotated arguments.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            Iterable[ChunkMeta]: Argument value.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class ArgChunkMeta(ChunkMetaGenerator, ArgGetter):
    """Class for generating chunk metadata directly from an argument extracted from annotated arguments."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs, **kwargs) -> tp.Iterable[ChunkMeta]:
        return self.get_arg(ann_args)


class LenChunkMeta(ArgChunkMeta):
    """Class for generating chunk metadata based on a sequence of chunk lengths."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs, **kwargs) -> tp.Iterable[ChunkMeta]:
        arg = self.get_arg(ann_args)
        start = 0
        end = 0
        for i, chunk_len in enumerate(arg):
            end += chunk_len
            yield ChunkMeta(uuid=str(uuid.uuid4()), idx=i, start=start, end=end, indices=None)
            start = end


def iter_chunk_meta(
    size: tp.Optional[int] = None,
    min_size: tp.Optional[int] = None,
    n_chunks: tp.Union[None, int, str] = None,
    chunk_len: tp.Union[None, int, str] = None,
) -> tp.Iterator[ChunkMeta]:
    """Yield chunk metadata for successive chunks from a sequence of elements.

    If `size`, `n_chunks`, and `chunk_len` are all None after resolving settings,
    a single chunk is returned. If only `n_chunks` and `chunk_len` are None, `n_chunks` is set to "auto".

    Args:
        size (Optional[int]): Total number of elements to split.
        min_size (Optional[int]): Minimum number of elements to split.

            If `size` is less than this value, a single chunk is returned.
        n_chunks (Union[None, int, str]): Specification for the number of chunks.

            If "auto", the number of CPU cores is used.
        chunk_len (Union[None, int, str]): Specification for the length of each chunk.

            If "auto", the number of CPU cores is used.

    Yields:
        ChunkMeta: Chunk metadata.
    """
    if size is not None and min_size is not None and size < min_size:
        yield ChunkMeta(uuid=str(uuid.uuid4()), idx=0, start=0, end=size, indices=None)
    else:
        if n_chunks is None and chunk_len is None and size is None:
            n_chunks = 1
        if n_chunks is None and chunk_len is None:
            n_chunks = "auto"
        if n_chunks is not None and chunk_len is not None:
            raise ValueError("Must provide either n_chunks or chunk_len, not both")
        if n_chunks is not None:
            if isinstance(n_chunks, str):
                if n_chunks.lower() == "auto":
                    n_chunks = multiprocessing.cpu_count()
                else:
                    raise ValueError(f"Invalid n_chunks: '{n_chunks}'")
            if n_chunks == 0:
                raise ValueError("Chunk count cannot be zero")
            if size is not None:
                if n_chunks > size:
                    n_chunks = size
                d, r = divmod(size, n_chunks)
                for i in range(n_chunks):
                    si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                    yield ChunkMeta(
                        uuid=str(uuid.uuid4()),
                        idx=i,
                        start=si,
                        end=si + (d + 1 if i < r else d),
                        indices=None,
                    )
            else:
                for i in range(n_chunks):
                    yield ChunkMeta(
                        uuid=str(uuid.uuid4()), idx=i, start=None, end=None, indices=None
                    )
        if chunk_len is not None:
            checks.assert_not_none(size, arg_name="size")
            if isinstance(chunk_len, str):
                if chunk_len.lower() == "auto":
                    chunk_len = multiprocessing.cpu_count()
                else:
                    raise ValueError(f"Invalid chunk_len: '{chunk_len}'")
            if chunk_len == 0:
                raise ValueError("Chunk length cannot be zero")
            for chunk_i, i in enumerate(range(0, size, chunk_len)):
                yield ChunkMeta(
                    uuid=str(uuid.uuid4()),
                    idx=chunk_i,
                    start=i,
                    end=min(i + chunk_len, size),
                    indices=None,
                )


def get_chunk_meta_key(chunk_meta: ChunkMeta) -> tp.Any:
    """Return a key representing the given `ChunkMeta`.

    If `chunk_meta.indices` is provided, returns a string in the format "first..last".
    If `chunk_meta.start` and `chunk_meta.end` are provided and indicate a single element,
    returns the start value; otherwise, returns a range string in the format "start..(end - 1)".
    Returns `MISSING` if no valid key can be determined.

    Args:
        chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.

    Returns:
        Any: Key representing the chunk metadata.
    """
    if chunk_meta.indices is not None:
        return f"{chunk_meta.indices[0]}..{chunk_meta.indices[-1]}"
    if chunk_meta.start is not None and chunk_meta.end is not None:
        if chunk_meta.start == chunk_meta.end - 1:
            return chunk_meta.start
        return f"{chunk_meta.start}..{chunk_meta.end - 1}"
    return MISSING


# ############# Chunk mapping ############# #


@define
class ChunkMapper(DefineMixin):
    """Abstract class for mapping chunk metadata.

    Implements the abstract `ChunkMapper.map` method and supports caching of mapped
    `ChunkMeta` instances.

    !!! note
        Use `ChunkMapper.apply` instead of `ChunkMapper.map`.
    """

    should_cache: bool = define.field(default=True)
    """Indicates whether to cache mapped `ChunkMeta` results."""

    chunk_meta_cache: tp.Dict[str, ChunkMeta] = define.field(factory=dict)
    """Cache for mapped `ChunkMeta` instances, keyed by the UUID of the input metadata."""

    def apply(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Apply the chunk mapper to the given `ChunkMeta`.

        Args:
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            **kwargs: Keyword arguments for `ChunkMapper.map`.

        Returns:
            ChunkMeta: Mapped chunk metadata, possibly retrieved from cache.
        """
        if not self.should_cache:
            return self.map(chunk_meta, **kwargs)
        if chunk_meta.uuid not in self.chunk_meta_cache:
            new_chunk_meta = self.map(chunk_meta, **kwargs)
            self.chunk_meta_cache[chunk_meta.uuid] = new_chunk_meta
            return new_chunk_meta
        return self.chunk_meta_cache[chunk_meta.uuid]

    def map(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Map the input `ChunkMeta` to a new `ChunkMeta`.

        Args:
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            **kwargs: Additional keyword arguments.

        Returns:
            ChunkMeta: Mapped chunk metadata.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


# ############# Chunk taking ############# #


@define
class NotChunked(Evaluable, Annotatable, DefineMixin):
    """Class representing an argument that should not be chunked."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """Identifier(s) at which to evaluate this instance."""


@define
class ChunkTaker(Evaluable, Annotatable, DefineMixin):
    """Abstract class for extracting elements from a collection based on chunk index or range.

    !!! note
        Use `ChunkTaker.apply` instead of `ChunkTaker.take`.
    """

    single_type: tp.Optional[tp.TypeLike] = define.field(default=None)
    """Type or tuple of types that should be treated as a single value."""

    ignore_none: bool = define.field(default=True)
    """Indicates whether None values should be ignored."""

    mapper: tp.Optional[ChunkMapper] = define.field(default=None)
    """Optional chunk mapper (`ChunkMapper`) to process chunk metadata."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """Identifier(s) at which to evaluate this instance."""

    def get_size(self, obj: tp.Any, **kwargs) -> int:
        """Return the actual size of the given object.

        Args:
            obj (Any): Input object.
            **kwargs: Additional keyword arguments.

        Returns:
            int: Size of the object.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def suggest_size(self, obj: tp.Any, **kwargs) -> tp.Optional[int]:
        """Return a suggested global size derived from the given object.

        Args:
            obj (Any): Input object.
            **kwargs: Keyword arguments for `ChunkTaker.get_size`.

        Returns:
            Optional[int]: Suggested size of the object, or None if a mapper is configured.
        """
        if self.mapper is not None:
            return None
        return self.get_size(obj, **kwargs)

    def should_take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> bool:
        """Determine whether to extract a chunk from the given object based on the chunk metadata.

        Args:
            obj (Any): Input object.
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            **kwargs: Additional keyword arguments.

        Returns:
            bool:
                * Returns False if the object is None and `ignore_none` is True.
                * Returns False if the object is an instance of `single_type`.
                * Otherwise, returns True.
        """
        if self.ignore_none and obj is None:
            return False
        if self.single_type is not None:
            if checks.is_instance_of(obj, self.single_type):
                return False
        return True

    def apply(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Apply the chunk taker to the given object using the specified chunk metadata.

        If a mapper is configured, the chunk metadata is first processed by the mapper.
        If criteria in `should_take` are not met, returns the original object;
        otherwise, extracts the chunk using `take`.

        Args:
            obj (Any): Input object.
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            **kwargs: Keyword arguments for `ChunkTaker.should_take` or `ChunkTaker.take`.

        Returns:
            Any: Resulting object after chunk extraction.
        """
        if self.mapper is not None:
            chunk_meta = self.mapper.apply(chunk_meta, **kwargs)
        if not self.should_take(obj, chunk_meta, **kwargs):
            return obj
        return self.take(obj, chunk_meta, **kwargs)

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Extract a subset of data from the given object using the provided chunk metadata.

        Args:
            obj (Any): Input object from which to extract data.
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Extracted subset of data.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


@define
class ChunkSelector(ChunkTaker, DimRetainer, DefineMixin):
    """Class for selecting a single element from a sequence based on the chunk index."""

    def get_size(self, obj: tp.Sequence, **kwargs) -> int:
        return LenSizer.get_obj_size(obj, single_type=self.single_type)

    def suggest_size(self, obj: tp.Sequence, **kwargs) -> tp.Optional[int]:
        return None

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        if self.keep_dims:
            return obj[chunk_meta.idx : chunk_meta.idx + 1]
        return obj[chunk_meta.idx]


class ChunkSlicer(ChunkTaker):
    """Class for slicing multiple elements based on a specified chunk range."""

    def get_size(self, obj: tp.Sequence, **kwargs) -> int:
        return LenSizer.get_obj_size(obj, single_type=self.single_type)

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Sequence:
        if chunk_meta.indices is not None:
            return obj[chunk_meta.indices]
        return obj[chunk_meta.start : chunk_meta.end]


class CountAdapter(ChunkSlicer):
    """Class for adapting a count using a specified chunk range."""

    def get_size(self, obj: int, **kwargs) -> int:
        return CountSizer.get_obj_size(obj, single_type=self.single_type)

    def take(self, obj: int, chunk_meta: ChunkMeta, **kwargs) -> int:
        checks.assert_instance_of(obj, int)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj):
                raise IndexError("Positional indexers are out-of-bounds")
            return len(indices)
        if chunk_meta.start >= obj:
            return 0
        return min(obj, chunk_meta.end) - chunk_meta.start


@define
class ShapeSelector(ChunkSelector, AxisSpecifier, DefineMixin):
    """Class for selecting a single element from a shape's axis based on a chunk index."""

    def get_size(self, obj: tp.ShapeLike, **kwargs) -> int:
        return ShapeSizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.ShapeLike, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        if checks.is_int(obj):
            obj = (obj,)
        checks.assert_instance_of(obj, tuple)
        if len(obj) == 0:
            return ()
        axis = self.axis
        if axis is None:
            if len(obj) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {axis} were indexed")
        if chunk_meta.idx >= obj[axis]:
            raise IndexError(
                f"Index {chunk_meta.idx} is out of bounds for axis {axis} with size {obj[axis]}"
            )
        obj = list(obj)
        if self.keep_dims:
            obj[axis] = 1
        else:
            del obj[axis]
        return tuple(obj)


@define
class ShapeSlicer(ChunkSlicer, AxisSpecifier, DefineMixin):
    """Class for slicing multiple elements from a shape's axis using a specified chunk range."""

    def get_size(self, obj: tp.ShapeLike, **kwargs) -> int:
        return ShapeSizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.ShapeLike, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        if checks.is_int(obj):
            obj = (obj,)
        checks.assert_instance_of(obj, tuple)
        if len(obj) == 0:
            return ()
        axis = self.axis
        if axis is None:
            if len(obj) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {axis} were indexed")
        obj = list(obj)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj[axis]):
                raise IndexError("Positional indexers are out-of-bounds")
            obj[axis] = len(indices)
        else:
            if chunk_meta.start >= obj[axis]:
                del obj[axis]
            else:
                obj[axis] = min(obj[axis], chunk_meta.end) - chunk_meta.start
        return tuple(obj)


class ArraySelector(ShapeSelector):
    """Class for selecting a single element from a specified axis of an array based on a chunk index."""

    def get_size(self, obj: tp.AnyArray, **kwargs) -> int:
        return ArraySizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.ArrayLike:
        from vectorbtpro.base.indexing import PandasIndexer
        from vectorbtpro.base.wrapping import Wrapping

        if isinstance(obj, Wrapping):
            shape = obj.wrapper.shape
        else:
            shape = obj.shape
        if len(shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(shape):
            raise IndexError(f"Array is {len(shape)}-dimensional, but {axis} were indexed")
        slc = [slice(None)] * len(shape)
        if self.keep_dims:
            slc[axis] = slice(chunk_meta.idx, chunk_meta.idx + 1)
        else:
            slc[axis] = chunk_meta.idx
        if isinstance(obj, (pd.Series, pd.DataFrame, PandasIndexer)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


class ArraySlicer(ShapeSlicer):
    """Class for slicing multiple elements from a specified axis of an array using a chunk range."""

    def get_size(self, obj: tp.AnyArray, **kwargs) -> int:
        return ArraySizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.AnyArray:
        from vectorbtpro.base.indexing import PandasIndexer
        from vectorbtpro.base.wrapping import Wrapping

        if isinstance(obj, Wrapping):
            shape = obj.wrapper.shape
        else:
            shape = obj.shape
        if len(shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(shape):
            raise IndexError(f"Array is {len(shape)}-dimensional, but {axis} were indexed")
        slc = [slice(None)] * len(shape)
        if chunk_meta.indices is not None:
            slc[axis] = np.asarray(chunk_meta.indices)
        else:
            slc[axis] = slice(chunk_meta.start, chunk_meta.end)
        if isinstance(obj, (pd.Series, pd.DataFrame, PandasIndexer)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


@define
class ContainerTaker(ChunkTaker, DefineMixin):
    """Class for taking elements from a container using other chunk takers.

    Accepts a container take specification.

    Args:
        cont_take_spec (Optional[ContainerTakeSpec]): Specification for taking elements from the container.
        single_type (Optional[TypeLike]): Type or tuple of types that should be treated as a single value.
        ignore_none (bool): Indicates whether None values should be ignored.
        mapper (Optional[ChunkMapper]): Optional chunk mapper (`ChunkMapper`) to process chunk metadata.
        eval_id (Optional[MaybeSequence[Hashable]]): Identifier(s) used for evaluation.
    """

    cont_take_spec: tp.Optional[tp.ContainerTakeSpec] = define.field(default=None)
    """Specification of the container."""

    def __init__(
        self,
        cont_take_spec: tp.Optional[tp.ContainerTakeSpec] = None,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
        eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
    ) -> None:
        ChunkTaker.__init__(
            self,
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            eval_id=eval_id,
            cont_take_spec=cont_take_spec,
        )

    def get_size(self, obj: tp.Sequence, **kwargs) -> int:
        raise NotImplementedError

    def check_cont_take_spec(self) -> None:
        """Check that `ContainerTaker.cont_take_spec` is provided.

        Returns:
            None
        """
        if self.cont_take_spec is None:
            raise ValueError("Please provide cont_take_spec")

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        raise NotImplementedError


class SequenceTaker(ContainerTaker):
    """Class for taking items from a sequence container.

    Calls `Chunker.take_from_arg` on each element.
    """

    def adapt_cont_take_spec(self, obj: tp.Sequence) -> tp.ContainerTakeSpec:
        """Adapt the container take specification for the given sequence.

        If the last element is an ellipsis, replace it by repeating the preceding specification
        to match the sequence length.

        Args:
            obj (Sequence): Sequence object.

        Returns:
            ContainerTakeSpec: Adapted container take specification.
        """
        cont_take_spec = list(self.cont_take_spec)
        if len(cont_take_spec) >= 2:
            if isinstance(cont_take_spec[-1], type(...)):
                if len(obj) >= len(cont_take_spec):
                    cont_take_spec = cont_take_spec[:-1]
                    cont_take_spec.extend([cont_take_spec[-1]] * (len(obj) - len(cont_take_spec)))
        return cont_take_spec

    def suggest_size(
        self, obj: tp.Sequence, chunker: tp.Optional["Chunker"] = None, **kwargs
    ) -> tp.Optional[int]:
        if self.mapper is not None:
            return None
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        size_i = None
        size = None
        for i, v in enumerate(obj):
            if i < len(cont_take_spec) and cont_take_spec[i] is not MISSING:
                take_spec = chunker.resolve_take_spec(cont_take_spec[i])
                if isinstance(take_spec, ChunkTaker):
                    try:
                        new_size = take_spec.suggest_size(v)
                        if new_size is not None:
                            if size is None:
                                size_i = i
                                size = new_size
                            elif size != new_size:
                                warn(
                                    f"Arguments at indices {size_i} and {i} have conflicting sizes "
                                    f"{size} and {new_size}. Setting size to None."
                                )
                                return None
                    except NotImplementedError:
                        pass
        return size

    def take(
        self,
        obj: tp.Sequence,
        chunk_meta: ChunkMeta,
        chunker: tp.Optional["Chunker"] = None,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Sequence:
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        new_obj = []
        for i, v in enumerate(obj):
            if i < len(cont_take_spec) and cont_take_spec[i] is not MISSING:
                take_spec = cont_take_spec[i]
            else:
                if not silence_warnings:
                    warn(
                        f"Argument at index {i} not found in SequenceTaker.cont_take_spec. "
                        "Setting its specification to None."
                    )
                take_spec = None
            new_obj.append(
                chunker.take_from_arg(
                    v,
                    take_spec,
                    chunk_meta,
                    chunker=chunker,
                    silence_warnings=silence_warnings,
                    **kwargs,
                )
            )
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)


class MappingTaker(ContainerTaker):
    """Class for taking items from a mapping container.

    Calls `Chunker.take_from_arg` on each element.
    """

    def adapt_cont_take_spec(self, obj: tp.Mapping) -> tp.ContainerTakeSpec:
        """Adapt the container take specification for the given mapping.

        If an ellipsis key is present, assign its corresponding specification to any missing keys.

        Args:
            obj (Mapping): Mapping object.

        Returns:
            ContainerTakeSpec: Adapted container take specification.
        """
        cont_take_spec = dict(self.cont_take_spec)
        ellipsis_take_spec = None
        ellipsis_found = False
        for k in cont_take_spec:
            if isinstance(k, type(...)):
                ellipsis_take_spec = cont_take_spec[k]
                ellipsis_found = True
        if ellipsis_found:
            for k, v in dict(obj).items():
                if k not in cont_take_spec:
                    cont_take_spec[k] = ellipsis_take_spec
        return cont_take_spec

    def suggest_size(
        self, obj: tp.Mapping, chunker: tp.Optional["Chunker"] = None, **kwargs
    ) -> tp.Optional[int]:
        if self.mapper is not None:
            return None
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        size_k = None
        size = None
        for k, v in dict(obj).items():
            if k in cont_take_spec and cont_take_spec[k] is not MISSING:
                take_spec = chunker.resolve_take_spec(cont_take_spec[k])
                if isinstance(take_spec, ChunkTaker):
                    try:
                        new_size = take_spec.suggest_size(v)
                        if new_size is not None:
                            if size is None:
                                size_k = k
                                size = new_size
                            elif size != new_size:
                                warn(
                                    f"Arguments with keys '{size_k}' and '{k}' have conflicting sizes "
                                    f"{size} and {new_size}. Setting size to None."
                                )
                                return None
                    except NotImplementedError:
                        pass
        return size

    def take(
        self,
        obj: tp.Mapping,
        chunk_meta: ChunkMeta,
        chunker: tp.Optional["Chunker"] = None,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Mapping:
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        new_obj = {}
        for k, v in dict(obj).items():
            if k in cont_take_spec and cont_take_spec[k] is not MISSING:
                take_spec = cont_take_spec[k]
            else:
                if not silence_warnings:
                    warn(
                        f"Argument with key '{k}' not found in MappingTaker.cont_take_spec. "
                        "Setting its specification to None."
                    )
                take_spec = None
            new_obj[k] = chunker.take_from_arg(
                v,
                take_spec,
                chunk_meta,
                chunker=chunker,
                silence_warnings=silence_warnings,
                **kwargs,
            )
        return type(obj)(new_obj)


class ArgsTaker(SequenceTaker):
    """Class for taking items from a variable-length positional arguments container.

    Args:
        *args: Positional arguments to be used as `ContainerTaker.cont_take_spec`.
        single_type (Optional[TypeLike]): Type or tuple of types that should be treated as a single value.
        ignore_none (bool): Indicates whether None values should be ignored.
        mapper (Optional[ChunkMapper]): Optional chunk mapper (`ChunkMapper`) to process chunk metadata.
    """

    def __init__(
        self,
        *args,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
    ) -> None:
        SequenceTaker.__init__(
            self,
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=args,
        )


class KwargsTaker(MappingTaker):
    """Class for taking items from a variable-length keyword arguments container.

    Args:
        single_type (Optional[TypeLike]): Type or tuple of types that should be treated as a single value.
        ignore_none (bool): Indicates whether None values should be ignored.
        mapper (Optional[ChunkMapper]): Optional chunk mapper (`ChunkMapper`) to process chunk metadata.
        **kwargs: Keyword arguments to be used as `ContainerTaker.cont_take_spec`.
    """

    def __init__(
        self,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
        **kwargs,
    ) -> None:
        MappingTaker.__init__(
            self,
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=kwargs,
        )


# ############# Chunkables ############# #


class Chunkable(Evaluable, Annotatable):
    """Abstract class representing a value with an associated chunk-taking specification."""

    def get_value(self) -> tp.Any:
        """Return the encapsulated value.

        Returns:
            Any: Encapsulated value.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_take_spec(self) -> tp.TakeSpec:
        """Return the associated chunk-taking specification.

        Returns:
            TakeSpec: Chunk-taking specification.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


@define
class Chunked(Chunkable, DefineMixin):
    """Class representing a chunkable value.

    This class encapsulates a value and associated chunking behavior.
    It accepts additional keyword arguments that configure the chunk-taking specification.

    Args:
        value (Any): Value to be chunked.
        take_spec (TakeSpec): Specification for taking chunks.
        take_spec_kwargs (KwargsLike): Keyword arguments for the `ChunkTaker` subclass.

            If `take_spec` is an instance rather than a class, these arguments update its configuration.
        select (bool): Indicates whether chunking is performed by selection.
        eval_id (Optional[MaybeSequence[Hashable]]): Identifier(s) used for evaluation.
        **kwargs: Keyword arguments acting as `take_spec_kwargs`.
    """

    value: tp.Any = define.required_field()
    """Value to be chunked."""

    take_spec: tp.TakeSpec = define.optional_field()
    """Specification for taking chunks."""

    take_spec_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for the `ChunkTaker` subclass.

    If `take_spec` is an instance rather than a class, these arguments update its configuration.
    """

    select: bool = define.field(default=False)
    """Indicates whether chunking is performed by selection."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """Identifier(s) used for evaluation."""

    def __init__(self, *args, **kwargs) -> None:
        attr_names = [a.name for a in self.fields]
        if attr_names.index("take_spec_kwargs") < len(args):
            new_args = list(args)
            take_spec_kwargs = new_args[attr_names.index("take_spec_kwargs")]
            if take_spec_kwargs is None:
                take_spec_kwargs = {}
            else:
                take_spec_kwargs = dict(take_spec_kwargs)
            take_spec_kwargs.update(
                {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names}
            )
            new_args[attr_names.index("take_spec_kwargs")] = take_spec_kwargs
            args = tuple(new_args)
        else:
            take_spec_kwargs = kwargs.pop("take_spec_kwargs", None)
            if take_spec_kwargs is None:
                take_spec_kwargs = {}
            else:
                take_spec_kwargs = dict(take_spec_kwargs)
            take_spec_kwargs.update(
                {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names}
            )
            kwargs["take_spec_kwargs"] = take_spec_kwargs

        DefineMixin.__init__(self, *args, **kwargs)

    def get_value(self) -> tp.Any:
        self.assert_field_not_missing("value")
        return self.value

    @property
    def take_spec_missing(self) -> bool:
        """Boolean flag indicating whether `take_spec` is missing.

        Returns:
            bool: True if `take_spec` is missing, False otherwise.
        """
        return self.take_spec is MISSING

    def resolve_take_spec(self) -> tp.TakeSpec:
        """Return the resolved chunk-taking specification.

        Returns:
            TakeSpec: Chunk-taking strategy determined by the instance configuration.

        !!! note
            If `take_spec` is missing, returns `ChunkSelector` when `select` is True,
            otherwise returns `ChunkSlicer`.
        """
        if self.take_spec_missing:
            if self.select:
                return ChunkSelector
            return ChunkSlicer
        return self.take_spec

    def get_take_spec(self) -> tp.TakeSpec:
        take_spec = self.resolve_take_spec()
        take_spec_kwargs = self.take_spec_kwargs
        if take_spec_kwargs is None:
            take_spec_kwargs = {}
        else:
            take_spec_kwargs = dict(take_spec_kwargs)
        if "eval_id" not in take_spec_kwargs:
            take_spec_kwargs["eval_id"] = self.eval_id
        if isinstance(take_spec, type) and issubclass(take_spec, ChunkTaker):
            take_spec = take_spec(**take_spec_kwargs)
        elif isinstance(take_spec, ChunkTaker):
            take_spec = take_spec.replace(**take_spec_kwargs)
        return take_spec


class ChunkedCount(Chunked):
    """Class representing a chunkable count value.

    Inherits all initialization parameters from `Chunked`.
    """

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            return CountAdapter
        return self.take_spec


class ChunkedShape(Chunked):
    """Class representing a chunkable shape value.

    Inherits all initialization parameters from `Chunked`.
    """

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.select:
                return ShapeSelector
            return ShapeSlicer
        return self.take_spec


@define
class ChunkedArray(Chunked, DefineMixin):
    """Class representing a chunkable array.

    Inherits all initialization parameters from `Chunked` and adds array flexibility configuration.
    """

    flex: bool = define.field(default=False)
    """Indicates whether the array is flexible."""

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.flex:
                if self.select:
                    from vectorbtpro.base.chunking import FlexArraySelector

                    return FlexArraySelector
                from vectorbtpro.base.chunking import FlexArraySlicer

                return FlexArraySlicer
            if self.select:
                return ArraySelector
            return ArraySlicer
        return self.take_spec


# ############# Chunker ############# #


class Chunker(Configured):
    """Class responsible for chunking arguments of a function and running the function.

    Generates chunk metadata, splits function arguments into chunks, executes chunks,
    and optionally merges results.

    It performs the following steps:

    1. Generates chunk metadata by passing `n_chunks`, `size`, `min_size`, `chunk_len`, and
        `chunk_meta` to `Chunker.get_chunk_meta_from_args`.
    2. Splits arguments and keyword arguments by passing chunk metadata, `arg_take_spec`,
        and `template_context` to `Chunker.iter_tasks`, which yields one chunk at a time.
    3. Executes all chunks by passing `**execute_kwargs` to `vectorbtpro.utils.execution.execute`.
    4. Optionally, post-processes and merges the results by passing them and `**merge_kwargs` to `merge_func`.

    Args:
        size (Optional[int]): Chunk size used for metadata generation.

            See `Chunker.get_chunk_meta_from_args`.
        min_size (Optional[int]): Minimum number of elements to split.

            See `Chunker.get_chunk_meta_from_args`.
        n_chunks (Optional[SizeLike]): Desired number of chunks used in metadata generation.

            See `Chunker.get_chunk_meta_from_args`.
        chunk_len (Optional[SizeLike]): Length of each chunk used in metadata generation.

            See `Chunker.get_chunk_meta_from_args`.
        chunk_meta (Optional[ChunkMetaLike]): Custom chunk metadata for argument chunking.

            See `Chunker.get_chunk_meta_from_args`.
        prepend_chunk_meta (Optional[bool]): Determines whether to prepend a `ChunkMeta` instance to the arguments.

            If set to None, prepending occurs automatically when the first argument is named `chunk_meta`.
        skip_single_chunk (Optional[bool]): Specifies whether to bypass chunking and execute the function
            directly when only one chunk is present.
        arg_take_spec (Optional[ArgTakeSpecLike]): Specification for selecting function arguments during chunking.
        template_context (KwargsLike): Additional context for template substitution.
        merge_func (MergeFuncLike): Function to merge the results.

            See `vectorbtpro.utils.merging.MergeFunc`.
        merge_kwargs (KwargsLike): Keyword arguments for `merge_func`.
        return_raw_chunks (Optional[bool]): Determines whether to return raw chunk data instead
            of post-processed results.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        forward_kwargs_as (KwargsLike): Mapping for renaming keyword arguments when forwarding them.

            Variables from the context of `Chunker.run` may be included.
        execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

            See `vectorbtpro.utils.execution.execute`.
        disable (Optional[bool]): Flag to disable chunking.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.chunking`.
    """

    _settings_path: tp.SettingsPath = "chunking"

    def __init__(
        self,
        size: tp.Optional[int] = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Optional[tp.SizeLike] = None,
        chunk_len: tp.Optional[tp.SizeLike] = None,
        chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
        prepend_chunk_meta: tp.Optional[bool] = None,
        skip_single_chunk: tp.Optional[bool] = None,
        arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
        template_context: tp.KwargsLike = None,
        merge_func: tp.MergeFuncLike = None,
        merge_kwargs: tp.KwargsLike = None,
        return_raw_chunks: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        forward_kwargs_as: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        disable: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            size=size,
            min_size=min_size,
            n_chunks=n_chunks,
            chunk_len=chunk_len,
            chunk_meta=chunk_meta,
            prepend_chunk_meta=prepend_chunk_meta,
            skip_single_chunk=skip_single_chunk,
            arg_take_spec=arg_take_spec,
            template_context=template_context,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            return_raw_chunks=return_raw_chunks,
            silence_warnings=silence_warnings,
            forward_kwargs_as=forward_kwargs_as,
            execute_kwargs=execute_kwargs,
            disable=disable,
            **kwargs,
        )

        self._size = self.resolve_setting(size, "size")
        self._min_size = self.resolve_setting(min_size, "min_size")
        self._n_chunks = self.resolve_setting(n_chunks, "n_chunks")
        self._chunk_len = self.resolve_setting(chunk_len, "chunk_len")
        self._chunk_meta = self.resolve_setting(chunk_meta, "chunk_meta")
        self._prepend_chunk_meta = self.resolve_setting(prepend_chunk_meta, "prepend_chunk_meta")
        self._skip_single_chunk = self.resolve_setting(skip_single_chunk, "skip_single_chunk")
        self._arg_take_spec = self.resolve_setting(arg_take_spec, "arg_take_spec")
        self._template_context = self.resolve_setting(
            template_context, "template_context", merge=True
        )
        self._merge_func = self.resolve_setting(merge_func, "merge_func")
        self._merge_kwargs = self.resolve_setting(merge_kwargs, "merge_kwargs", merge=True)
        self._return_raw_chunks = self.resolve_setting(return_raw_chunks, "return_raw_chunks")
        self._silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")
        self._forward_kwargs_as = self.resolve_setting(
            forward_kwargs_as, "forward_kwargs_as", merge=True
        )
        self._execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)
        self._disable = self.resolve_setting(disable, "disable")

    @property
    def size(self) -> tp.Optional[int]:
        """Chunk size used for metadata generation.

        See `Chunker.get_chunk_meta_from_args`.

        Returns:
            Optional[int]: Configured chunk size.
        """
        return self._size

    @property
    def min_size(self) -> tp.Optional[int]:
        """Minimum chunk size used in metadata generation.

        See `Chunker.get_chunk_meta_from_args`.

        Returns:
            Optional[int]: Minimum allowable chunk size.
        """
        return self._min_size

    @property
    def n_chunks(self) -> tp.Optional[tp.SizeLike]:
        """Desired number of chunks used in metadata generation.

        See `Chunker.get_chunk_meta_from_args`.

        Returns:
            Optional[SizeLike]: Target number of chunks.
        """
        return self._n_chunks

    @property
    def chunk_len(self) -> tp.Optional[tp.SizeLike]:
        """Length of each chunk used in metadata generation.

        See `Chunker.get_chunk_meta_from_args`.

        Returns:
            Optional[SizeLike]: Length assigned to each chunk.
        """
        return self._chunk_len

    @property
    def chunk_meta(self) -> tp.Optional[tp.ChunkMetaLike]:
        """Custom chunk metadata for argument chunking.

        See `Chunker.get_chunk_meta_from_args`.

        Returns:
            Optional[ChunkMetaLike]: Custom chunk metadata configuration.
        """
        return self._chunk_meta

    @property
    def prepend_chunk_meta(self) -> tp.Optional[bool]:
        """Determines whether to prepend a `ChunkMeta` instance to the function arguments.

        If set to None, prepending occurs automatically when the first argument is named `chunk_meta`.

        Returns:
            Optional[bool]: True if chunk metadata should be prepended; otherwise False.
        """
        return self._prepend_chunk_meta

    @property
    def skip_single_chunk(self) -> bool:
        """Specifies whether to bypass chunking and execute the function directly when only one chunk is present.

        Returns:
            bool: True if single chunk execution should skip chunk processing; otherwise False.
        """
        return self._skip_single_chunk

    @property
    def arg_take_spec(self) -> tp.Optional[tp.ArgTakeSpecLike]:
        """Specification for selecting function arguments during chunking.

        Returns:
            Optional[ArgTakeSpecLike]: Specification dict or object for argument extraction.

        See:
            `Chunker.iter_tasks`
        """
        return self._arg_take_spec

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def merge_func(self) -> tp.Optional[tp.MergeFuncLike]:
        """Function to merge the results.

        See `vectorbtpro.utils.merging.MergeFunc`.

        Returns:
            Optional[MergeFuncLike]: Merging function or merge function configuration.
        """
        return self._merge_func

    @property
    def merge_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Chunker.merge_func`.

        Returns:
            Kwargs: Dictionary of keyword arguments for merging results.
        """
        return self._merge_kwargs

    @property
    def return_raw_chunks(self) -> bool:
        """Determines whether to return raw chunk data instead of post-processed results.

        Returns:
            bool: True if the raw chunk data should be returned; otherwise False.
        """
        return self._return_raw_chunks

    @property
    def silence_warnings(self) -> bool:
        """Indicates whether to suppress warnings during chunk processing.

        Returns:
            bool: True if warnings should be suppressed; otherwise False.
        """
        return self._silence_warnings

    @property
    def forward_kwargs_as(self) -> tp.Kwargs:
        """Mapping for renaming keyword arguments.

        Variables from the context of `Chunker.run` may be included.

        Returns:
            Kwargs: Mapping of keyword arguments for renaming.
        """
        return self._forward_kwargs_as

    @property
    def execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the execution handler.

        See `vectorbtpro.utils.execution.execute`.

        Returns:
            Kwargs: Dictionary of execution keyword arguments.
        """
        return self._execute_kwargs

    @property
    def disable(self) -> bool:
        """Specifies whether chunking is disabled.

        Returns:
            bool: True if chunking is disabled; otherwise False.
        """
        return self._disable

    @classmethod
    def get_chunk_meta_from_args(
        cls,
        ann_args: tp.AnnArgs,
        size: tp.Optional[tp.SizeLike] = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Optional[tp.SizeLike] = None,
        chunk_len: tp.Optional[tp.SizeLike] = None,
        chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
        **kwargs,
    ) -> tp.Iterable[ChunkMeta]:
        """Generate chunk metadata from annotated arguments.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            size (Optional[SizeLike]): Chunk size used for metadata generation.

                It can be an integer, an instance of `Sizer`, or a callable that receives the annotated
                arguments and returns a value.
            min_size (Optional[int]): Minimum number of elements to split.
            n_chunks (Optional[SizeLike]): Desired number of chunks.

                It can be provided as an integer, a string, an instance of `Sizer`, or a callable that
                accepts annotated arguments and keyword arguments.
            chunk_len (Optional[SizeLike]): Length of each chunk.

                It can be provided as an integer, a string, an instance of `Sizer`, or a callable that
                receives the annotated arguments and returns a value.
            chunk_meta (Optional[ChunkMetaLike]): Custom chunk metadata.

                It can be an iterable of `ChunkMeta` instances, a `ChunkMeta` generator, or a callable
                returning chunk metadata.
            **kwargs: Keyword arguments for the metadata generation process.

        Returns:
            Iterable[ChunkMeta]: Iterable of chunk metadata.
        """
        if chunk_meta is None:
            if size is not None:
                if isinstance(size, Sizer):
                    size = size.apply(ann_args, **kwargs)
                elif callable(size):
                    size = size(ann_args, **kwargs)
                elif not isinstance(size, int):
                    raise TypeError(f"Type {type(size)} for size is not supported")
            if n_chunks is not None:
                if isinstance(n_chunks, Sizer):
                    n_chunks = n_chunks.apply(ann_args, **kwargs)
                elif callable(n_chunks):
                    n_chunks = n_chunks(ann_args, **kwargs)
                elif not isinstance(n_chunks, (int, str)):
                    raise TypeError(f"Type {type(n_chunks)} for n_chunks is not supported")
            if chunk_len is not None:
                if isinstance(chunk_len, Sizer):
                    chunk_len = chunk_len.apply(ann_args, **kwargs)
                elif callable(chunk_len):
                    chunk_len = chunk_len(ann_args, **kwargs)
                elif not isinstance(chunk_len, (int, str)):
                    raise TypeError(f"Type {type(chunk_len)} for chunk_len is not supported")
            return iter_chunk_meta(
                size=size, min_size=min_size, n_chunks=n_chunks, chunk_len=chunk_len
            )
        if isinstance(chunk_meta, ChunkMetaGenerator):
            return chunk_meta.get_chunk_meta(ann_args, **kwargs)
        if callable(chunk_meta):
            return chunk_meta(ann_args, **kwargs)
        return chunk_meta

    @classmethod
    def resolve_take_spec(cls, take_spec: tp.TakeSpec) -> tp.TakeSpec:
        """Resolve the chunk-taking specification.

        Args:
            take_spec (TakeSpec): Specification for taking chunks.

        Returns:
            TakeSpec: Resolved chunk-taking specification.
        """
        if isinstance(take_spec, type) and issubclass(take_spec, Chunked):
            take_spec = take_spec()
        if isinstance(take_spec, Chunkable):
            take_spec = take_spec.get_take_spec()
        if isinstance(take_spec, type) and issubclass(take_spec, (NotChunked, ChunkTaker)):
            take_spec = take_spec()
        return take_spec

    @classmethod
    def take_from_arg(
        cls,
        arg: tp.Any,
        take_spec: tp.TakeSpec,
        chunk_meta: ChunkMeta,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Any:
        """Extract a chunk from the given argument based on the provided specification.

        Args:
            arg (Any): Input argument.
            take_spec (TakeSpec): Specification for taking chunks.

                If None or a `NotChunked` instance, the original argument is returned.
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **kwargs: Keyword arguments for `ChunkTaker.apply`.

        Returns:
            Any: Result after applying the chunk-taking specification.
        """
        if take_spec is None:
            return arg
        take_spec = cls.resolve_take_spec(take_spec)
        if isinstance(take_spec, NotChunked):
            return arg
        if isinstance(take_spec, ChunkTaker):
            if not take_spec.meets_eval_id(eval_id):
                return arg
            return take_spec.apply(arg, chunk_meta, **kwargs)
        raise TypeError(f"Specification of type {type(take_spec)} is not supported")

    @classmethod
    def find_take_spec(
        cls,
        i: int,
        ann_arg_name: str,
        ann_arg: tp.Kwargs,
        arg_take_spec: tp.ArgTakeSpec,
    ) -> tp.TakeSpec:
        """Resolve the chunk-taking specification for a given argument.

        Args:
            i (int): Index of the argument.
            ann_arg_name (str): Name of the annotated argument.
            ann_arg (Kwargs): Details of the annotated argument.
            arg_take_spec (ArgTakeSpec): Mapping specifying the extraction rules for each argument.

        Returns:
            TakeSpec: Resolved specification for the argument, or `MISSING` if not found.
        """
        take_spec_found = False
        found_take_spec = None
        for k, v in arg_take_spec.items():
            if isinstance(k, int):
                if k == i:
                    take_spec_found = True
                    found_take_spec = v
                    break
            elif isinstance(k, Regex):
                if k.matches(ann_arg_name):
                    take_spec_found = True
                    found_take_spec = v
                    break
            elif isinstance(v, Regex):
                if v.matches(k):
                    take_spec_found = True
                    found_take_spec = v
                    break
            else:
                if k == ann_arg_name:
                    take_spec_found = True
                    found_take_spec = v
                    break
        if take_spec_found:
            found_take_spec = cls.resolve_take_spec(found_take_spec)
            if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
                if not isinstance(found_take_spec, ContainerTaker):
                    if checks.is_sequence(found_take_spec):
                        found_take_spec = SequenceTaker(found_take_spec)
                    else:
                        found_take_spec = SequenceTaker([found_take_spec, ...])
            elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
                if not isinstance(found_take_spec, ContainerTaker):
                    if checks.is_mapping(found_take_spec):
                        found_take_spec = MappingTaker(found_take_spec)
                    else:
                        found_take_spec = MappingTaker({...: found_take_spec})
            return found_take_spec
        return MISSING

    @classmethod
    def take_from_args(
        cls,
        ann_args: tp.AnnArgs,
        arg_take_spec: tp.ArgTakeSpec,
        chunk_meta: ChunkMeta,
        silence_warnings: bool = False,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Tuple[tp.Args, tp.Kwargs]:
        """Extract chunks from the annotated arguments based on the provided taking specification.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            arg_take_spec (ArgTakeSpec): Mapping specifying the extraction rules for each argument.
            chunk_meta (ChunkMeta): Metadata specifying the chunk boundaries.
            silence_warnings (bool): Flag to suppress warning messages.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **kwargs: Keyword arguments for `Chunker.take_from_arg`.

        Returns:
            Tuple[tuple, dict]: Tuple containing the new positional arguments and keyword
                arguments for function execution.
        """
        new_args = ()
        new_kwargs = dict()
        for i, (k, v) in enumerate(ann_args.items()):
            take_spec = cls.find_take_spec(i, k, v, arg_take_spec)
            if take_spec is MISSING:
                take_spec = None
                if not silence_warnings:
                    warn(
                        f"Argument '{k}' not found in arg_take_spec. Setting its specification to None."
                    )
            result = cls.take_from_arg(
                v["value"],
                take_spec,
                chunk_meta,
                ann_args=ann_args,
                arg_take_spec=arg_take_spec,
                silence_warnings=silence_warnings,
                eval_id=eval_id,
                **kwargs,
            )
            if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                for new_arg in result:
                    new_args += (new_arg,)
            elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                for new_kwarg_name, new_kwarg in result.items():
                    new_kwargs[new_kwarg_name] = new_kwarg
            elif v["kind"] == inspect.Parameter.KEYWORD_ONLY:
                new_kwargs[k] = result
            else:
                new_args += (result,)
        return new_args, new_kwargs

    @classmethod
    def iter_tasks(
        cls,
        func: tp.Callable,
        ann_args: tp.AnnArgs,
        chunk_meta: tp.Iterable[ChunkMeta],
        arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Iterator[Task]:
        """Split annotated arguments into chunks and yield each chunk as a task.

        Args:
            func (Callable): Callable to execute as a task.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            chunk_meta (Iterable[ChunkMeta]): Iterable containing metadata for each chunk.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.
            arg_take_spec (Optional[tp.ArgTakeSpecLike]): Specification for chunk-taking.

                It can be a mapping, a sequence (which will be converted into a mapping),
                a callable, or a `CustomTemplate`.

                !!! note
                    If a callable, it must accept the same arguments as `Chunker.take_from_args`
                    except for `arg_take_spec`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `Chunker.take_from_args` or to
                `arg_take_spec` if it is callable.

        Yields:
            Task: Each task containing a chunk of arguments.
        """
        if arg_take_spec is None:
            arg_take_spec = {}
        if template_context is None:
            template_context = {}

        for _chunk_meta in chunk_meta:
            _template_context = dict(template_context)
            _template_context["ann_args"] = ann_args
            _template_context["chunk_meta"] = _chunk_meta
            chunk_ann_args = substitute_templates(
                ann_args, _template_context, eval_id="chunk_ann_args"
            )
            _template_context["chunk_ann_args"] = chunk_ann_args
            chunk_arg_take_spec = substitute_templates(
                arg_take_spec, _template_context, eval_id="chunk_arg_take_spec"
            )
            _template_context["chunk_arg_take_spec"] = chunk_arg_take_spec

            if callable(chunk_arg_take_spec):
                chunk_args, chunk_kwargs = chunk_arg_take_spec(
                    chunk_ann_args,
                    _chunk_meta,
                    template_context=_template_context,
                    **kwargs,
                )
            else:
                if not checks.is_mapping(chunk_arg_take_spec):
                    chunk_arg_take_spec = dict(
                        zip(range(len(chunk_arg_take_spec)), chunk_arg_take_spec)
                    )
                chunk_args, chunk_kwargs = cls.take_from_args(
                    chunk_ann_args,
                    chunk_arg_take_spec,
                    _chunk_meta,
                    template_context=_template_context,
                    **kwargs,
                )
            yield Task(func, *chunk_args, **chunk_kwargs)

    @classmethod
    def parse_sizer_from_func(
        cls,
        func: tp.Callable,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Optional[Sizer]:
        """Parse and return the sizer extracted from a function's annotations.

        Args:
            func (Callable): Function to parse for sizer annotations.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            Optional[Sizer]: Sizer instance that meets the evaluation criteria, or None if not found.
        """
        annotations = flatten_annotations(get_annotations(func))
        sizer = None
        for k, v in annotations.items():
            if not isinstance(v, Union):
                v = Union(v)
            for annotation in v.annotations:
                if isinstance(annotation, type) and issubclass(annotation, Sizer):
                    annotation = annotation()
                if isinstance(annotation, Sizer) and annotation.meets_eval_id(eval_id):
                    if isinstance(annotation, ArgGetter):
                        if annotation.arg_query is None:
                            annotation = annotation.replace(arg_query=k)
                    if sizer is not None:
                        raise ValueError(
                            f"Two sizers found in annotations: {sizer} and {annotation}"
                        )
                    sizer = annotation
        return sizer

    @classmethod
    def parse_spec_from_annotations(
        cls,
        annotations: tp.Annotations,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.ArgTakeSpec:
        """Parse and return the chunk-taking specification extracted from provided annotations.

        Args:
            annotations (Annotations): Mapping of annotation names to annotation values.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            ArgTakeSpec: Dictionary mapping parameter names to their chunk-taking specifications.
        """
        arg_take_spec = {}
        for k, v in annotations.items():
            if not isinstance(v, Union):
                v = Union(v)
            for annotation in v.annotations:
                annotation = cls.resolve_take_spec(annotation)
                if isinstance(annotation, ChunkTaker) and annotation.meets_eval_id(eval_id):
                    if isinstance(annotation, ArgGetter):
                        if annotation.arg_query is None:
                            annotation = annotation.replace(arg_query=k)
                    if k in arg_take_spec:
                        raise ValueError(
                            f"Two specifications found in annotations for the key '{k}': "
                            f"{arg_take_spec[k]} and {annotation}"
                        )
                    arg_take_spec[k] = annotation
        return arg_take_spec

    @classmethod
    def parse_spec_from_func(
        cls,
        func: tp.Callable,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.ArgTakeSpec:
        """Parse and return the chunk-taking specification extracted from a function's annotations,
        including handling for variable arguments.

        Args:
            func (Callable): Function to parse.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            ArgTakeSpec: Dictionary mapping parameter names to chunk-taking specifications.
        """
        annotations = get_annotations(func)
        arg_take_spec = cls.parse_spec_from_annotations(annotations, eval_id=eval_id)
        flat_annotations, var_args_map, var_kwargs_map = flatten_annotations(
            annotations,
            only_var_args=True,
            return_var_arg_maps=True,
        )
        if len(flat_annotations) > 0:
            flat_arg_take_spec = cls.parse_spec_from_annotations(flat_annotations, eval_id=eval_id)
            if len(var_args_map) > 0:
                var_args_name = None
                var_args_specs = []
                for k in var_args_map:
                    if k in flat_arg_take_spec:
                        if var_args_map[k] in arg_take_spec:
                            raise ValueError(
                                "Two specifications found in annotations: "
                                f"{arg_take_spec[var_args_map[k]]} ('*{var_args_map[k]}') and "
                                f"{flat_arg_take_spec[k]} ('{k}')"
                            )
                        if var_args_name is None:
                            var_args_name = var_args_map[k]
                        i = int(k.split("_")[-1])
                        if i > len(var_args_specs):
                            var_args_specs.extend([MISSING] * (i - len(var_args_specs)))
                        var_args_specs.append(flat_arg_take_spec[k])
                if len(var_args_specs) > 0:
                    arg_take_spec[var_args_name] = ArgsTaker(*var_args_specs)
            if len(var_kwargs_map) > 0:
                var_kwargs_name = None
                var_kwargs_specs = dict()
                for k in var_kwargs_map:
                    if k in flat_arg_take_spec:
                        if var_kwargs_map[k] in arg_take_spec:
                            raise ValueError(
                                "Two specifications found in annotations: "
                                f"{arg_take_spec[var_kwargs_map[k]]} ('**{var_kwargs_map[k]}') and "
                                f"{flat_arg_take_spec[k]} ('{k}')"
                            )
                        if var_kwargs_name is None:
                            var_kwargs_name = var_kwargs_map[k]
                        var_kwargs_specs[k] = flat_arg_take_spec[k]
                if len(var_kwargs_specs) > 0:
                    arg_take_spec[var_kwargs_name] = KwargsTaker(**var_kwargs_specs)
        return arg_take_spec

    @classmethod
    def parse_spec_from_args(
        cls,
        ann_args: tp.AnnArgs,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.ArgTakeSpec:
        """Parse and return the chunk-taking specification derived from annotated arguments.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            ArgTakeSpec: Dictionary mapping argument names to chunk-taking specifications.
        """
        arg_take_spec = {}
        for k, v in ann_args.items():
            if isinstance(v["value"], Chunkable) and v["value"].meets_eval_id(eval_id):
                arg_take_spec[k] = v["value"].get_take_spec()
            elif v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                chunkable_found = False
                for v2 in v["value"]:
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        chunkable_found = True
                        break
                if chunkable_found:
                    take_spec = []
                    for v2 in v["value"]:
                        if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                            take_spec.append(v2.get_take_spec())
                        else:
                            take_spec.append(MISSING)
                    arg_take_spec[k] = ArgsTaker(*take_spec)
            elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                chunkable_found = False
                for v2 in v["value"].values():
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        chunkable_found = True
                        break
                if chunkable_found:
                    take_spec = {}
                    for k2, v2 in v["value"].items():
                        if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                            take_spec[k2] = v2.get_take_spec()
                        else:
                            take_spec[k2] = MISSING
                    arg_take_spec[k] = KwargsTaker(**take_spec)
        return arg_take_spec

    @classmethod
    def fill_arg_take_spec(
        cls, arg_take_spec: tp.ArgTakeSpec, ann_args: tp.AnnArgs
    ) -> tp.ArgTakeSpec:
        """Fill and return the chunk-taking specification with missing keys set to None to avoid warnings.

        Args:
            arg_take_spec (ArgTakeSpec): Mapping specifying the extraction rules for each argument.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            ArgTakeSpec: Updated chunk-taking specification with missing keys filled with None.
        """
        arg_take_spec = dict(arg_take_spec)
        for k, v in ann_args.items():
            if k not in arg_take_spec:
                arg_take_spec[k] = None
        return arg_take_spec

    @classmethod
    def adapt_ann_args(
        cls, ann_args: tp.AnnArgs, eval_id: tp.Optional[tp.Hashable] = None
    ) -> tp.AnnArgs:
        """Adapt and return annotated arguments by replacing Chunkable objects with their evaluated values.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            AnnArgs: New dictionary of annotated arguments with updated values.
        """
        new_ann_args = {}
        for k, v in ann_args.items():
            new_ann_args[k] = v = dict(v)
            if isinstance(v["value"], Chunkable) and v["value"].meets_eval_id(eval_id):
                v["value"] = v["value"].get_value()
            elif v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                new_value = []
                for v2 in v["value"]:
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        new_value.append(v2.get_value())
                    else:
                        new_value.append(v2)
                v["value"] = tuple(new_value)
            elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                new_value = {}
                for k2, v2 in v["value"].items():
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        new_value[k2] = v2.get_value()
                    else:
                        new_value[k2] = v2
                v["value"] = new_value
        return new_ann_args

    @classmethod
    def suggest_size(
        cls,
        ann_args: tp.AnnArgs,
        arg_take_spec: tp.ArgTakeSpec,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Optional[int]:
        """Suggest a global size based on annotated arguments and a chunk-taking specification.

        Args:
            cls: Class reference.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            arg_take_spec (ArgTakeSpec): Mapping specifying the extraction rules for each argument.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **kwargs: Keyword arguments for `ChunkTaker.suggest_size`.

        Returns:
            Optional[int]: Determined global size if found; otherwise, None.
        """
        size_k = None
        size = None
        for i, (k, v) in enumerate(ann_args.items()):
            take_spec = cls.find_take_spec(i, k, v, arg_take_spec)
            if isinstance(take_spec, ChunkTaker) and take_spec.meets_eval_id(eval_id):
                try:
                    new_size = take_spec.suggest_size(v["value"], **kwargs)
                    if new_size is not None:
                        if size is None:
                            size_k = k
                            size = new_size
                        elif size != new_size:
                            warn(
                                f"Arguments '{size_k}' and '{k}' have conflicting sizes "
                                f"{size} and {new_size}. Setting size to None."
                            )
                            return None
                except NotImplementedError:
                    pass
        return size

    def run(
        self, func: tp.Callable, *args, eval_id: tp.Optional[tp.Hashable] = None, **kwargs
    ) -> tp.Any:
        """Chunk the arguments and execute the function.

        Args:
            func (Callable): Function to execute.
            *args: Positional arguments for `func`.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **kwargs: Keyword arguments for `func`.

        Returns:
            Any: Result of executing `func`, either directly or after processing chunks.
        """
        size = self.size
        min_size = self.min_size
        n_chunks = self.n_chunks
        chunk_len = self.chunk_len
        chunk_meta = self.chunk_meta
        prepend_chunk_meta = self.prepend_chunk_meta
        skip_single_chunk = self.skip_single_chunk
        arg_take_spec = self.arg_take_spec
        template_context = self.template_context
        merge_func = self.merge_func
        merge_kwargs = self.merge_kwargs
        return_raw_chunks = self.return_raw_chunks
        silence_warnings = self.silence_warnings
        forward_kwargs_as = self.forward_kwargs_as
        execute_kwargs = self.execute_kwargs
        disable = self.disable

        template_context["eval_id"] = eval_id

        if arg_take_spec is None:
            arg_take_spec = {}
        if checks.is_mapping(arg_take_spec):
            main_arg_take_spec = dict(arg_take_spec)
            arg_take_spec = dict(arg_take_spec)
            if "chunk_meta" not in arg_take_spec:
                arg_take_spec["chunk_meta"] = None
        else:
            main_arg_take_spec = None

        if forward_kwargs_as is None:
            forward_kwargs_as = {}
        if len(forward_kwargs_as) > 0:
            new_kwargs = dict()
            for k, v in kwargs.items():
                if k in forward_kwargs_as:
                    new_kwargs[forward_kwargs_as.pop(k)] = v
                else:
                    new_kwargs[k] = v
            kwargs = new_kwargs
        if len(forward_kwargs_as) > 0:
            for k, v in forward_kwargs_as.items():
                kwargs[v] = locals()[k]

        if disable:
            return func(*args, **kwargs)

        if prepend_chunk_meta is None:
            prepend_chunk_meta = False
            func_arg_names = get_func_arg_names(func)
            if len(func_arg_names) > 0:
                if func_arg_names[0] == "chunk_meta":
                    prepend_chunk_meta = True
        if prepend_chunk_meta:
            args = (Rep("chunk_meta"), *args)

        parsed_sizer = self.parse_sizer_from_func(func, eval_id=eval_id)
        if parsed_sizer is not None:
            if size is not None:
                raise ValueError(
                    f"Two conflicting sizers: {parsed_sizer} (annotations) and {size} (size)"
                )
            size = parsed_sizer
        parsed_arg_take_spec = self.parse_spec_from_func(func, eval_id=eval_id)
        if len(parsed_arg_take_spec) > 0:
            if (
                not isinstance(arg_take_spec, dict)
                or parsed_arg_take_spec.keys() & arg_take_spec.keys()
            ):
                raise ValueError(
                    f"Two conflicting specifications: {parsed_arg_take_spec} (annotations) "
                    f"and {arg_take_spec} (arg_take_spec)"
                )
            arg_take_spec = {**parsed_arg_take_spec, **arg_take_spec}
        parsed_merge_func = parse_merge_func(func, eval_id=eval_id)
        if parsed_merge_func is not None:
            if merge_func is not None:
                raise ValueError(
                    f"Two conflicting merge functions: {parsed_merge_func} (annotations) and {merge_func} (merge_func)"
                )
            merge_func = parsed_merge_func
        ann_args = annotate_args(func, args, kwargs)
        parsed_arg_take_spec = self.parse_spec_from_args(ann_args, eval_id=eval_id)
        if len(parsed_arg_take_spec) > 0:
            if (
                not isinstance(arg_take_spec, dict)
                or parsed_arg_take_spec.keys() & arg_take_spec.keys()
            ):
                raise ValueError(
                    f"Two conflicting specifications: {parsed_arg_take_spec} (arguments) "
                    f"and {arg_take_spec} (arg_take_spec & annotations)"
                )
            arg_take_spec = {**parsed_arg_take_spec, **arg_take_spec}
        if (
            main_arg_take_spec is not None
            and len(main_arg_take_spec) == 0
            and len(arg_take_spec) > 0
        ):
            arg_take_spec = self.fill_arg_take_spec(arg_take_spec, ann_args)
        ann_args = self.adapt_ann_args(ann_args, eval_id=eval_id)
        args, kwargs = ann_args_to_args(ann_args)
        template_context["chunker"] = self
        template_context["arg_take_spec"] = arg_take_spec
        template_context["ann_args"] = ann_args

        if size is None and isinstance(arg_take_spec, dict):
            size = self.suggest_size(
                ann_args,
                arg_take_spec,
                template_context=template_context,
                silence_warnings=silence_warnings,
                chunker=self,
                eval_id=eval_id,
            )
        template_context["size"] = size
        chunk_meta = list(
            self.get_chunk_meta_from_args(
                ann_args,
                size=size,
                min_size=min_size,
                n_chunks=n_chunks,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta,
                template_context=template_context,
                silence_warnings=silence_warnings,
                chunker=self,
                eval_id=eval_id,
            )
        )
        template_context["chunk_meta"] = chunk_meta
        if len(chunk_meta) < 2 and skip_single_chunk:
            return func(*args, **kwargs)
        tasks = self.iter_tasks(
            func,
            ann_args,
            chunk_meta,
            arg_take_spec=arg_take_spec,
            template_context=template_context,
            silence_warnings=silence_warnings,
            chunker=self,
            eval_id=eval_id,
        )
        if return_raw_chunks:
            return chunk_meta, tasks
        execute_kwargs = substitute_templates(
            execute_kwargs, template_context, eval_id="execute_kwargs"
        )
        execute_kwargs = merge_dicts(
            dict(show_progress=False if len(chunk_meta) == 1 else None), execute_kwargs
        )
        keys = []
        for _chunk_meta in chunk_meta:
            key = get_chunk_meta_key(_chunk_meta)
            if eval_id is not None:
                keys.append((MISSING, key))
            else:
                keys.append(key)
        if eval_id is not None:
            keys = pd.MultiIndex.from_tuples(keys, names=(f"eval_id={eval_id}", "chunk_indices"))
        else:
            keys = pd.Index(keys, name="chunk_indices")
        results = execute(tasks, size=len(chunk_meta), keys=keys, **execute_kwargs)
        if merge_func is not None:
            template_context["tasks"] = tasks
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
        return results


def chunked(
    *args,
    chunker: tp.Optional[tp.Type[Chunker]] = None,
    replace_chunker: tp.Optional[bool] = None,
    merge_to_execute_kwargs: tp.Optional[bool] = None,
    prepend_chunk_meta: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Callable:
    """Decorate a function to process its inputs in chunks using `Chunker`.

    This decorator splits the input arguments of a function into chunks, dispatches each chunk
    for processing via an engine, and optionally merges the results. The returned function
    preserves the signature of the original function.

    Each option can be updated at any time by modifying the `options` attribute of the wrapper
    or by passing a keyword argument prefixed with an underscore.

    Chunking can be disabled by using the `disable` argument, and the entire wrapping mechanism
    can be bypassed with the global setting `disable_wrapping` (which returns the original function).

    If keyword arguments are not recognized by `Chunker` or `execute_kwargs`, they are merged into
    `execute_kwargs` when `merge_to_execute_kwargs` is True; otherwise, they are passed directly to
    `Chunker`. Additionally, if a chunker instance is provided and `replace_chunker` is True, a new
    `Chunker` instance is created by replacing any arguments that are not None.

    Args:
        func (Callable): Function to be decorated.
        chunker (Optional[Chunker]): `Chunker` type used for splitting the inputs.
        replace_chunker (Optional[bool]): If True, create a new `Chunker` instance by replacing provided attributes.
        merge_to_execute_kwargs (Optional[bool]): Flag that determines whether to merge unspecified
            keyword arguments into `execute_kwargs`.
        prepend_chunk_meta (Optional[bool]): Determines whether to prepend a `ChunkMeta` instance to the arguments.

            If set to None, prepending occurs automatically when the first argument is named `chunk_meta`.
        **kwargs: Keyword arguments for `Chunker` or the decorated function.

    Returns:
        Callable: Decorated function with chunking capability.

    !!! info
        For default settings, see `vectorbtpro._settings.chunking`.

    Examples:
        For testing purposes, let's divide the input array into 2 chunks and compute
        the mean in a sequential manner:

        ```pycon
        >>> from vectorbtpro import *

        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer())
        ... )
        ... def f(a):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        Same can be done using annotations:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.LenSizer() | vbt.ChunkSlicer()):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        Sizer can be omitted most of the time:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.ChunkSlicer()):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        Another way is by using specialized `Chunker` subclasses that depend on the type of the argument:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.ChunkedArray()):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        ```

        Also, instead of specifying the chunk-taking specification beforehand, it can be passed
        dynamically by wrapping each value to be chunked with `Chunked` or any of its subclasses:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a):
        ...     return np.mean(a)

        >>> f(vbt.ChunkedArray(np.arange(10)))
        [2.0, 7.0]
        ```

        The `chunked` function is a decorator that takes `f` and creates a function that splits
        passed arguments, runs each chunk using an engine, and optionally, merges the results.
        It has the same signature as the original function:

        ```pycon
        >>> f
        <function __main__.f(a)>
        ```

        We can change any option at any time:

        ```pycon
        >>> # Change the option directly on the function
        >>> f.options.n_chunks = 3

        >>> f(np.arange(10))
        [1.5, 5.0, 8.0]

        >>> # Pass a new option with a leading underscore
        >>> f(np.arange(10), _n_chunks=4)
        [1.0, 4.0, 6.5, 8.5]
        ```

        When we run the decorated function, it first generates a list of chunk metadata of type `ChunkMeta`.
        Chunk metadata contains the chunk index that can be used to split any input:

        ```pycon
        >>> list(vbt.iter_chunk_meta(n_chunks=2))
        [ChunkMeta(uuid='84d64eed-fbac-41e7-ad61-c917e809b3b8', idx=0, start=None, end=None, indices=None),
         ChunkMeta(uuid='577817c4-fdee-4ceb-ab38-dcd663d9ab11', idx=1, start=None, end=None, indices=None)]
        ```

        Additionally, it may contain the start and end index of the space we want to split.
        The space can be defined by the length of an input array, for example. In our case:

        ```pycon
        >>> list(vbt.iter_chunk_meta(n_chunks=2, size=10))
        [ChunkMeta(uuid='c1593842-dc31-474c-a089-e47200baa2be', idx=0, start=0, end=5, indices=None),
         ChunkMeta(uuid='6d0265e7-1204-497f-bc2c-c7b7800ec57d', idx=1, start=5, end=10, indices=None)]
        ```

        If we know the size of the space in advance, we can pass it as an integer constant.
        Otherwise, we need to instruct `chunked` to derive the size from the inputs dynamically
        by passing any subclass of `Sizer`. In the example above, the decorated function derives
        the size from the length of the input array `a`.

        Once all chunks are generated, the decorated function attempts to split inputs into chunks.
        The specification for this operation can be provided via the `arg_take_spec` argument, which
        in most cases is a dictionary of `ChunkTaker` instances keyed by the input name.
        Here's an example of a complex specification:

        ```pycon
        >>> arg_take_spec = dict(
        ...     a=vbt.ChunkSelector(),
        ...     args=vbt.ArgsTaker(
        ...         None,
        ...         vbt.ChunkSelector()
        ...     ),
        ...     b=vbt.SequenceTaker([
        ...         None,
        ...         vbt.ChunkSelector()
        ...     ]),
        ...     kwargs=vbt.KwargsTaker(
        ...         c=vbt.MappingTaker(dict(
        ...             d=vbt.ChunkSelector(),
        ...             e=None
        ...         ))
        ...     )
        ... )

        >>> @vbt.chunked(
        ...     n_chunks=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=arg_take_spec
        ... )
        ... def f(a, *args, b=None, **kwargs):
        ...     return a + sum(args) + sum(b) + sum(kwargs['c'].values())

        >>> f([1, 2, 3], 10, [1, 2, 3], b=(100, [1, 2, 3]), c=dict(d=[1, 2, 3], e=1000))
        [1114, 1118, 1122]
        ```

        After splitting all inputs into chunks, the decorated function forwards them to an engine.
        The engine can be specified either as the name of a supported engine or as a callable.
        Once the engine completes its tasks and returns a list of results, they can be merged back using `merge_func`:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
        ...     merge_func="concat"
        ... )
        ... def f(a):
        ...     return a

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        The same using annotations:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.ChunkSlicer()) -> vbt.MergeFunc("concat"):
        ...     return a

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        Instead of (or in addition to) specifying `arg_take_spec`, define the function with the first argument
        as `chunk_meta` to manage input splitting during execution. The `chunked` decorator will recognize
        and replace it with the actual `ChunkMeta` object:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=None),
        ...     merge_func="concat"
        ... )
        ... def f(chunk_meta, a):
        ...     return a[chunk_meta.start:chunk_meta.end]

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        This may be a good idea in multi-threading, but a bad idea in multi-processing.

        The same can be accomplished by using templates (here we tell `chunked` to not replace
        the first argument by setting `prepend_chunk_meta` to False):

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=None),
        ...     merge_func="concat",
        ...     prepend_chunk_meta=False
        ... )
        ... def f(chunk_meta, a):
        ...     return a[chunk_meta.start:chunk_meta.end]

        >>> f(vbt.Rep('chunk_meta'), np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        Templates in arguments are substituted just before processing each chunk.

        Keyword arguments for the engine can be provided using `execute_kwargs`:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
        ...     show_progress=True
        ... )
        ... def f(a):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        100% |█████████████████████████████████| 2/2 [00:00<00:00, 81.11it/s]
        [2.0, 7.0]
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        nonlocal prepend_chunk_meta

        from vectorbtpro._settings import settings

        chunking_cfg = settings["chunking"]

        if chunking_cfg["disable_wrapping"]:
            return func

        if prepend_chunk_meta is None:
            prepend_chunk_meta = False
            func_arg_names = get_func_arg_names(func)
            if len(func_arg_names) > 0:
                if func_arg_names[0] == "chunk_meta":
                    prepend_chunk_meta = True

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            chunker = kwargs.get("_chunker")
            if chunker is None:
                chunker = wrapper.options["chunker"]
            if chunker is None:
                chunker = chunking_cfg["chunker"]
            if chunker is None:
                chunker = Chunker

            arg_names_set = set(chunker._expected_keys)
            kwargs_options = {}
            for k in list(kwargs.keys()):
                if k.startswith("_"):
                    if k[1:] in wrapper.options or k[1:] in arg_names_set:
                        kwargs_options[k[1:]] = kwargs.pop(k)
            chunker_kwargs = merge_dicts(wrapper.options, kwargs_options)
            _ = chunker_kwargs.pop("chunker")
            replace_chunker = chunker_kwargs.pop("replace_chunker")
            merge_to_execute_kwargs = chunker_kwargs.pop("merge_to_execute_kwargs")
            eval_id = chunker_kwargs.pop("eval_id", None)

            if merge_to_execute_kwargs is None:
                merge_to_execute_kwargs = chunking_cfg["merge_to_execute_kwargs"]
            if merge_to_execute_kwargs and len(chunker_kwargs) > 0:
                arg_names_set = set(chunker._expected_keys)
                execute_kwargs = chunker_kwargs.pop("execute_kwargs", None)
                if execute_kwargs is None:
                    _execute_kwargs = {}
                else:
                    _execute_kwargs = dict(execute_kwargs)
                execute_kwargs_changed = False
                for k in list(chunker_kwargs.keys()):
                    if k not in arg_names_set and k not in _execute_kwargs:
                        _execute_kwargs[k] = chunker_kwargs.pop(k)
                        execute_kwargs_changed = True
                if execute_kwargs_changed:
                    chunker_kwargs["execute_kwargs"] = _execute_kwargs
                else:
                    chunker_kwargs["execute_kwargs"] = execute_kwargs
            if isinstance(chunker, type):
                checks.assert_subclass_of(chunker, Chunker, arg_name="chunker")
                chunker = chunker(**chunker_kwargs)
            else:
                checks.assert_instance_of(chunker, Chunker, arg_name="chunker")
                if replace_chunker is None:
                    replace_chunker = chunking_cfg["replace_chunker"]
                if replace_chunker and len(chunker_kwargs) > 0:
                    chunker = chunker.replace(**chunker_kwargs)
            return chunker.run(func, *args, eval_id=eval_id, **kwargs)

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_chunked = True
        wrapper.options = FrozenConfig(
            chunker=chunker,
            replace_chunker=replace_chunker,
            merge_to_execute_kwargs=merge_to_execute_kwargs,
            prepend_chunk_meta=prepend_chunk_meta,
            **kwargs,
        )

        if prepend_chunk_meta:
            signature = inspect.signature(wrapper)
            wrapper.__signature__ = signature.replace(
                parameters=tuple(signature.parameters.values())[1:]
            )

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


# ############# Chunking option ############# #


def resolve_chunked_option(option: tp.ChunkedOption = None) -> tp.KwargsLike:
    """Return keyword arguments for `chunked` based on a given option.

    Args:
        option (ChunkedOption): Option to control chunked processing.

            * True: Use default chunking settings.
            * None or False: Disable chunking.
            * str: Specify the name of an execution engine (see `vectorbtpro.utils.execution.execute`).
            * dict: Provide keyword arguments to be passed to `chunked`.

    Returns:
        KwargsLike: Dictionary of keyword arguments for chunking configuration, or None if chunking is disabled.

    !!! info
        For default settings, see `vectorbtpro._settings.chunking`.
    """
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    if option is None:
        option = chunking_cfg["option"]

    if isinstance(option, bool):
        if not option:
            return None
        return dict()
    if isinstance(option, dict):
        return option
    elif isinstance(option, str):
        return dict(engine=option)
    raise TypeError(f"Type {type(option)} is invalid for a chunking option")


def specialize_chunked_option(option: tp.ChunkedOption = None, **kwargs) -> tp.KwargsLike:
    """Resolve the provided chunking option and merge it with additional keyword arguments.

    Args:
        option (ChunkedOption): Option to control chunked processing.

            See `resolve_chunked_option`.
        **kwargs: Keyword arguments to be merged with the resolved chunking option.

    Returns:
        KwargsLike: Dictionary of merged chunking configuration options, or None if chunking is disabled.
    """
    chunked_kwargs = resolve_chunked_option(option)
    if chunked_kwargs is not None:
        return merge_dicts(kwargs, chunked_kwargs)
    return chunked_kwargs


def resolve_chunked(func: tp.Callable, option: tp.ChunkedOption = None, **kwargs) -> tp.Callable:
    """Decorate a function with chunked processing according to a given option.

    Args:
        func (Callable): Function to decorate.
        option (ChunkedOption): Option to control chunked processing.

            See `resolve_chunked_option`.
        **kwargs: Keyword arguments for `chunked`.

            These are merged with the default chunking settings.

    Returns:
        Callable: Decorated function with chunked processing applied if enabled;
            otherwise, the original function.

    !!! info
        For default settings, see `vectorbtpro._settings.chunking`.
    """
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    chunked_kwargs = resolve_chunked_option(option)
    if chunked_kwargs is not None:
        if isinstance(chunking_cfg["option"], dict):
            chunked_kwargs = merge_dicts(chunking_cfg["option"], kwargs, chunked_kwargs)
        else:
            chunked_kwargs = merge_dicts(kwargs, chunked_kwargs)
        return chunked(func, **chunked_kwargs)
    return func
