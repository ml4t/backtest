# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functions for reshaping arrays.

Reshape functions transform a Pandas object/NumPy array in some way.
"""

import functools
import itertools

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, indexing, wrapping
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, resolve_dict
from vectorbtpro.utils.params import Param, combine_params
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.template import CustomTemplate

__all__ = [
    "to_1d_shape",
    "to_2d_shape",
    "repeat_shape",
    "tile_shape",
    "to_1d_array",
    "to_2d_array",
    "to_2d_pr_array",
    "to_2d_pc_array",
    "to_1d_array_nb",
    "to_2d_array_nb",
    "to_2d_pr_array_nb",
    "to_2d_pc_array_nb",
    "broadcast_shapes",
    "broadcast_array_to",
    "broadcast_arrays",
    "repeat",
    "tile",
    "align_pd_arrays",
    "BCO",
    "Default",
    "Ref",
    "broadcast",
    "broadcast_to",
]


def to_tuple_shape(shape: tp.ShapeLike) -> tp.Shape:
    """Convert a shape-like object to a tuple.

    Args:
        shape (ShapeLike): Shape-like object that can be an integer or iterable.

    Returns:
        Shape: Tuple representation of the input shape.
    """
    if checks.is_int(shape):
        return (int(shape),)
    return tuple(shape)


def to_1d_shape(shape: tp.ShapeLike) -> tp.Shape:
    """Convert a shape-like object to a one-dimensional shape tuple.

    Args:
        shape (ShapeLike): Shape-like object; can be an integer or iterable.

    Returns:
        Shape: One-dimensional shape tuple.
    """
    shape = to_tuple_shape(shape)
    if len(shape) == 0:
        return (1,)
    if len(shape) == 1:
        return shape
    if len(shape) == 2 and shape[1] == 1:
        return (shape[0],)
    raise ValueError(f"Cannot reshape a {len(shape)}-dimensional shape to 1 dimension")


def to_2d_shape(shape: tp.ShapeLike, expand_axis: int = 1) -> tp.Shape:
    """Convert a shape-like object to a two-dimensional shape tuple.

    Args:
        shape (ShapeLike): Shape-like object to convert.
        expand_axis (int): Axis along which to expand the array if necessary.

    Returns:
        Shape: Two-dimensional shape tuple.
    """
    shape = to_tuple_shape(shape)
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        if expand_axis == 1:
            return shape[0], 1
        else:
            return shape[0], 0
    if len(shape) == 2:
        return shape
    raise ValueError(f"Cannot reshape a {len(shape)}-dimensional shape to 2 dimensions")


def repeat_shape(shape: tp.ShapeLike, n: int, axis: int = 1) -> tp.Shape:
    """Repeat a shape along a specified axis.

    Args:
        shape (ShapeLike): Shape-like object representing the original shape.
        n (int): Number of repetitions along the specified axis.
        axis (int): Axis along which the shape is repeated.

    Returns:
        Shape: Resulting shape after repetition.
    """
    shape = to_tuple_shape(shape)
    if len(shape) <= axis:
        shape = tuple([shape[i] if i < len(shape) else 1 for i in range(axis + 1)])
    return *shape[:axis], shape[axis] * n, *shape[axis + 1 :]


def tile_shape(shape: tp.ShapeLike, n: int, axis: int = 1) -> tp.Shape:
    """Tile a shape along a specified axis.

    This function is identical to `repeat_shape` and exists solely for naming consistency.

    Args:
        shape (ShapeLike): Shape-like object representing the original shape.
        n (int): Number of tiles along the specified axis.
        axis (int): Axis along which the shape is tiled.

    Returns:
        Shape: Resulting tiled shape.
    """
    return repeat_shape(shape, n, axis=axis)


def index_to_series(obj: tp.Index, reset_index: bool = False) -> tp.Series:
    """Convert an Index to a Series.

    Args:
        obj (Index): Index to convert.
        reset_index (bool): Whether to reset the index in the resulting Series.

    Returns:
        Series: Pandas Series created from the index.
    """
    if reset_index:
        return obj.to_series(index=pd.RangeIndex(stop=len(obj)))
    return obj.to_series()


def index_to_frame(obj: tp.Index, reset_index: bool = False) -> tp.Frame:
    """Convert an Index to a DataFrame.

    Args:
        obj (Index): Index to convert.
        reset_index (bool): Whether to reset the index in the resulting DataFrame.

    Returns:
        DataFrame: Pandas DataFrame representation of the index.
    """
    if not isinstance(obj, pd.MultiIndex):
        return index_to_series(obj, reset_index=reset_index).to_frame()
    return obj.to_frame(index=not reset_index)


def mapping_to_series(obj: tp.MappingLike) -> tp.Series:
    """Convert a mapping-like object to a Series.

    If the input is a namedtuple, it is first converted to a dictionary.

    Args:
        obj (MappingLike): Mapping-like object to convert.

    Returns:
        Series: Pandas Series constructed from the mapping.
    """
    if checks.is_namedtuple(obj):
        obj = obj._asdict()
    return pd.Series(obj)


def to_any_array(obj: tp.ArrayLike, raw: bool = False, convert_index: bool = True) -> tp.AnyArray:
    """Convert any array-like object to an array.

    For Pandas objects, the original object is returned unless `raw` is True.
    Mapping-like objects are converted to a Series, and index objects may be converted
    if `convert_index` is True.

    Args:
        obj (ArrayLike): Array-like object to convert.
        raw (bool): If True, return a raw NumPy array.
        convert_index (bool): If True, convert index objects to a Series.

    Returns:
        AnyArray: Converted array.
    """
    from vectorbtpro.indicators.factory import IndicatorBase

    if isinstance(obj, IndicatorBase):
        obj = obj.main_output
    if not raw:
        if checks.is_any_array(obj):
            if convert_index and checks.is_index(obj):
                return index_to_series(obj)
            return obj
        if checks.is_mapping_like(obj):
            return mapping_to_series(obj)
    return np.asarray(obj)


def to_pd_array(obj: tp.ArrayLike, convert_index: bool = True) -> tp.PandasArray:
    """Convert an array-like object to a Pandas object.

    Pandas objects are returned as-is or converted to a Series based on the index conversion.
    Mapping-like objects are converted to a Series. If the object is a NumPy array, it is
    converted to a Series or DataFrame based on its dimensions.

    Args:
        obj (ArrayLike): Array-like object to convert.
        convert_index (bool): If True, convert index objects to a Series.

    Returns:
        PandasArray: Pandas Series or DataFrame representing the array.
    """
    from vectorbtpro.indicators.factory import IndicatorBase

    if isinstance(obj, IndicatorBase):
        obj = obj.main_output
    if checks.is_pandas(obj):
        if convert_index and checks.is_index(obj):
            return index_to_series(obj)
        return obj
    if checks.is_mapping_like(obj):
        return mapping_to_series(obj)

    obj = np.asarray(obj)
    if obj.ndim == 0:
        obj = obj[None]
    if obj.ndim == 1:
        return pd.Series(obj)
    if obj.ndim == 2:
        return pd.DataFrame(obj)
    raise ValueError("Wrong number of dimensions: cannot convert to Series or DataFrame")


def soft_to_ndim(obj: tp.ArrayLike, ndim: int, raw: bool = False) -> tp.AnyArray:
    """Adjust the dimensions of an array-like object to the specified number of dimensions (up to 2).

    If `ndim` is 1 and the object is two-dimensional with a single column, the object is collapsed
    to one dimension. If `ndim` is 2 and the object is one-dimensional, it is expanded to two dimensions.

    Args:
        obj (ArrayLike): Array-like object to reshape.
        ndim (int): Target number of dimensions (maximum 2).
        raw (bool): If True, return a raw NumPy array.

    Returns:
        AnyArray: Reshaped array.
    """
    obj = to_any_array(obj, raw=raw)
    if ndim == 1:
        if obj.ndim == 2:
            if obj.shape[1] == 1:
                if checks.is_frame(obj):
                    return obj.iloc[:, 0]
                return obj[:, 0]  # downgrade
    if ndim == 2:
        if obj.ndim == 1:
            if checks.is_series(obj):
                return obj.to_frame()
            return obj[:, None]  # upgrade
    return obj  # do nothing


def to_1d(obj: tp.ArrayLike, raw: bool = False) -> tp.AnyArray1d:
    """Reshape an array-like object to one dimension.

    If `raw` is True, the output is a raw NumPy array. For two-dimensional inputs with a single column,
    the column is collapsed into a one-dimensional array.

    Args:
        obj (ArrayLike): Array-like object to reshape.
        raw (bool): If True, return a raw NumPy array.

    Returns:
        AnyArray1d: Reshaped one-dimensional array.
    """
    obj = to_any_array(obj, raw=raw)
    if obj.ndim == 2:
        if obj.shape[1] == 1:
            if checks.is_frame(obj):
                return obj.iloc[:, 0]
            return obj[:, 0]
    if obj.ndim == 1:
        return obj
    elif obj.ndim == 0:
        return obj.reshape((1,))
    raise ValueError(f"Cannot reshape a {obj.ndim}-dimensional array to 1 dimension")


to_1d_array = functools.partial(to_1d, raw=True)
"""Partial version of `to_1d` with `raw` set to True."""


def to_2d(obj: tp.ArrayLike, raw: bool = False, expand_axis: int = 1) -> tp.AnyArray2d:
    """Reshape an array-like object to two dimensions.

    If `raw` is True, the output is a raw NumPy array. For one-dimensional inputs,
    the array is expanded along the specified axis.

    Args:
        obj (ArrayLike): Array-like object to reshape.
        raw (bool): If True, return a raw NumPy array.
        expand_axis (int): Axis along which to expand the array if necessary.

    Returns:
        AnyArray2d: Reshaped two-dimensional array.
    """
    obj = to_any_array(obj, raw=raw)
    if obj.ndim == 2:
        return obj
    elif obj.ndim == 1:
        if checks.is_series(obj):
            if expand_axis == 0:
                return pd.DataFrame(obj.values[None, :], columns=obj.index)
            elif expand_axis == 1:
                return obj.to_frame()
        return np.expand_dims(obj, expand_axis)
    elif obj.ndim == 0:
        return obj.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {obj.ndim}-dimensional array to 2 dimensions")


to_2d_array = functools.partial(to_2d, raw=True)
"""Partial version of `to_2d` with `raw` set to True."""

to_2d_pr_array = functools.partial(to_2d_array, expand_axis=1)
"""Partial version of `to_2d_array` with `expand_axis` set to 1."""

to_2d_pc_array = functools.partial(to_2d_array, expand_axis=0)
"""Partial version of `to_2d_array` with `expand_axis` set to 0."""


@register_jitted(cache=True)
def to_1d_array_nb(obj: tp.Array) -> tp.Array1d:
    """Resize an array to one dimension.

    Args:
        obj (Array): Input array to reshape.

    Returns:
        Array1d: One-dimensional array.
    """
    if obj.ndim == 0:
        return np.expand_dims(obj, axis=0)
    if obj.ndim == 1:
        return obj
    if obj.ndim == 2 and obj.shape[1] == 1:
        return obj[:, 0]
    raise ValueError("Array cannot be resized to one dimension")


@register_jitted(cache=True)
def to_2d_array_nb(obj: tp.Array, expand_axis: int = 1) -> tp.Array2d:
    """Resize an array to two dimensions.

    If the input array is scalar (0-D) or one-dimensional, a new dimension is added
    according to the specified axis.

    Args:
        obj (Array): Input array.
        expand_axis (int): Axis along which to expand the array if necessary.

    Returns:
        Array2d: Two-dimensional array.
    """
    if obj.ndim == 0:
        return np.expand_dims(np.expand_dims(obj, axis=0), axis=0)
    if obj.ndim == 1:
        return np.expand_dims(obj, axis=expand_axis)
    if obj.ndim == 2:
        return obj
    raise ValueError("Array cannot be resized to two dimensions")


@register_jitted(cache=True)
def to_2d_pr_array_nb(obj: tp.Array) -> tp.Array2d:
    """Return a two-dimensional array by applying `to_2d_array_nb` with `expand_axis` set to 1.

    Args:
        obj (Array): Input array.

    Returns:
        Array2d: Two-dimensional array.
    """
    return to_2d_array_nb(obj, expand_axis=1)


@register_jitted(cache=True)
def to_2d_pc_array_nb(obj: tp.Array) -> tp.Array2d:
    """Return a two-dimensional array by applying `to_2d_array_nb` with `expand_axis` set to 0.

    Args:
        obj (Array): Input array.

    Returns:
        Array2d: Two-dimensional array.
    """
    return to_2d_array_nb(obj, expand_axis=0)


def to_dict(obj: tp.ArrayLike, orient: str = "dict") -> dict:
    """Convert an array-like object to a dictionary.

    Args:
        obj (ArrayLike): Array-like object.
        orient (str): Orientation for conversion.

            Use "index_series" to create a dictionary mapping index to series.

    Returns:
        dict: Converted dictionary.
    """
    obj = to_pd_array(obj)
    if orient == "index_series":
        return {obj.index[i]: obj.iloc[i] for i in range(len(obj.index))}
    return obj.to_dict(orient)


def repeat(
    obj: tp.ArrayLike,
    n: int,
    axis: int = 1,
    raw: bool = False,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Repeat an array-like object multiple times along a specified axis.

    Args:
        obj (ArrayLike): Input array-like object.
        n (int): Number of repetitions.
        axis (int): Axis along which to repeat.
        raw (bool): If True, return a raw NumPy array.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.

    Returns:
        AnyArray: Array repeated along the specified axis.
    """
    obj = to_any_array(obj, raw=raw)
    if axis == 0:
        if checks.is_pandas(obj):
            new_index = indexes.repeat_index(obj.index, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(obj).wrap(
                np.repeat(obj.values, n, axis=0), index=new_index
            )
        return np.repeat(obj, n, axis=0)
    elif axis == 1:
        obj = to_2d(obj)
        if checks.is_pandas(obj):
            new_columns = indexes.repeat_index(obj.columns, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(obj).wrap(
                np.repeat(obj.values, n, axis=1), columns=new_columns
            )
        return np.repeat(obj, n, axis=1)
    else:
        raise ValueError(f"Only axes 0 and 1 are supported, not {axis}")


def tile(
    obj: tp.ArrayLike,
    n: int,
    axis: int = 1,
    raw: bool = False,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Tile an array-like object by repeating its elements along a specified axis.

    Args:
        obj (ArrayLike): Input array-like object.
        n (int): Number of times to tile the array.
        axis (int): Axis along which to tile.
        raw (bool): If True, return a raw NumPy array.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.

    Returns:
        AnyArray: Tiled array.
    """
    obj = to_any_array(obj, raw=raw)
    if axis == 0:
        if obj.ndim == 2:
            if checks.is_pandas(obj):
                new_index = indexes.tile_index(obj.index, n, ignore_ranges=ignore_ranges)
                return wrapping.ArrayWrapper.from_obj(obj).wrap(
                    np.tile(obj.values, (n, 1)), index=new_index
                )
            return np.tile(obj, (n, 1))
        if checks.is_pandas(obj):
            new_index = indexes.tile_index(obj.index, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(obj).wrap(np.tile(obj.values, n), index=new_index)
        return np.tile(obj, n)
    elif axis == 1:
        obj = to_2d(obj)
        if checks.is_pandas(obj):
            new_columns = indexes.tile_index(obj.columns, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(obj).wrap(
                np.tile(obj.values, (1, n)), columns=new_columns
            )
        return np.tile(obj, (1, n))
    else:
        raise ValueError(f"Only axes 0 and 1 are supported, not {axis}")


def broadcast_shapes(
    *shapes: tp.ArrayLike,
    axis: tp.Optional[tp.MaybeSequence[int]] = None,
    expand_axis: tp.Optional[tp.MaybeSequence[int]] = None,
) -> tp.Tuple[tp.Shape, ...]:
    """Broadcast shape-like objects following vectorbtpro's broadcasting rules.

    Args:
        *shapes (ArrayLike): Shape-like objects to broadcast.
        axis (Optional[MaybeSequence[int]]): Axis to broadcast along; allows for
            different axes for each shape-like object.
        expand_axis (Optional[MaybeSequence[int]]): Axis to expand along; allows for
            different axes for each shape-like object.

    Returns:
        Tuple[Shape, ...]: Tuple of broadcasted shapes.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if expand_axis is None:
        expand_axis = broadcasting_cfg["expand_axis"]

    is_2d = False
    for i, shape in enumerate(shapes):
        shape = to_tuple_shape(shape)
        if len(shape) == 2:
            is_2d = True
            break

    new_shapes = []
    for i, shape in enumerate(shapes):
        shape = to_tuple_shape(shape)
        if is_2d:
            if checks.is_sequence(expand_axis):
                _expand_axis = expand_axis[i]
            else:
                _expand_axis = expand_axis
            new_shape = to_2d_shape(shape, expand_axis=_expand_axis)
        else:
            new_shape = to_1d_shape(shape)
        if axis is not None:
            if checks.is_sequence(axis):
                _axis = axis[i]
            else:
                _axis = axis
            if _axis is not None:
                if _axis == 0:
                    if is_2d:
                        new_shape = (new_shape[0], 1)
                    else:
                        new_shape = (new_shape[0],)
                elif _axis == 1:
                    if is_2d:
                        new_shape = (1, new_shape[1])
                    else:
                        new_shape = (1,)
                else:
                    raise ValueError(f"Only axes 0 and 1 are supported, not {_axis}")
        new_shapes.append(new_shape)
    return tuple(np.broadcast_shapes(*new_shapes))


def broadcast_array_to(
    arr: tp.ArrayLike,
    target_shape: tp.ShapeLike,
    axis: tp.Optional[int] = None,
    expand_axis: tp.Optional[int] = None,
) -> tp.Array:
    """Broadcast an array-like object to a specified target shape following
    vectorbtpro's broadcasting rules.

    Args:
        arr (ArrayLike): Input array-like object.
        target_shape (ShapeLike): Desired target shape, which must have one or two dimensions.
        axis (Optional[int]): Axis index for adjusting the target shape.
        expand_axis (Optional[int]): Axis used for array expansion.

    Returns:
        Array: Broadcasted array.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if expand_axis is None:
        expand_axis = broadcasting_cfg["expand_axis"]

    arr = np.asarray(arr)
    target_shape = to_tuple_shape(target_shape)
    if len(target_shape) not in (1, 2):
        raise ValueError(
            f"Target shape must have either 1 or 2 dimensions, not {len(target_shape)}"
        )
    if len(target_shape) == 2:
        new_arr = to_2d_array(arr, expand_axis=expand_axis)
    else:
        new_arr = to_1d_array(arr)
    if axis is not None:
        if axis == 0:
            if len(target_shape) == 2:
                target_shape = (target_shape[0], new_arr.shape[1])
            else:
                target_shape = (target_shape[0],)
        elif axis == 1:
            target_shape = (new_arr.shape[0], target_shape[1])
        else:
            raise ValueError(f"Only axes 0 and 1 are supported, not {axis}")
    return np.broadcast_to(new_arr, target_shape)


def broadcast_arrays(
    *arrs: tp.ArrayLike,
    target_shape: tp.Optional[tp.ShapeLike] = None,
    axis: tp.Optional[tp.MaybeSequence[int]] = None,
    expand_axis: tp.Optional[tp.MaybeSequence[int]] = None,
) -> tp.Tuple[tp.Array, ...]:
    """Broadcast multiple array-like objects following vectorbtpro's broadcasting rules.

    Args:
        *arrs (ArrayLike): Array-like objects to broadcast.
        target_shape (Optional[ShapeLike]): Target shape to which arrays should be broadcasted.
        axis (Optional[MaybeSequence[int]]): Axis to broadcast along; allows for
            different axes for each array-like object.
        expand_axis (Optional[MaybeSequence[int]]): Axis to expand along; allows for
            different axes for each array-like object.

    Returns:
        Tuple[Array, ...]: Tuple containing the broadcasted arrays.
    """
    if target_shape is None:
        shapes = []
        for arr in arrs:
            shapes.append(np.asarray(arr).shape)
        target_shape = broadcast_shapes(*shapes, axis=axis, expand_axis=expand_axis)
    new_arrs = []
    for i, arr in enumerate(arrs):
        if axis is not None:
            if checks.is_sequence(axis):
                _axis = axis[i]
            else:
                _axis = axis
        else:
            _axis = None
        if expand_axis is not None:
            if checks.is_sequence(expand_axis):
                _expand_axis = expand_axis[i]
            else:
                _expand_axis = expand_axis
        else:
            _expand_axis = None
        new_arr = broadcast_array_to(arr, target_shape, axis=_axis, expand_axis=_expand_axis)
        new_arrs.append(new_arr)
    return tuple(new_arrs)


def broadcast_index(
    objs: tp.Sequence[tp.AnyArray],
    to_shape: tp.Shape,
    index_from: tp.IndexFromLike = None,
    axis: int = 0,
    ignore_sr_names: tp.Optional[bool] = None,
    ignore_ranges: tp.Optional[bool] = None,
    check_index_names: tp.Optional[bool] = None,
    **clean_index_kwargs,
) -> tp.Optional[tp.Index]:
    """Return a broadcasted index or columns based on the provided array-like objects.

    Args:
        objs (Sequence[AnyArray]): Array-like objects.
        to_shape (tuple[int]): Target shape.
        index_from (Union[None, str, int, Any]): Broadcasting strategy for the index or columns.

            Accepts the following values:

            * "keep" or None: Retain the original index/columns of `objs`.
            * "stack": Stack indexes/columns using `vectorbtpro.base.indexes.stack_indexes`.
            * "strict": Require identical indexes/columns across all Pandas objects.
            * "reset": Reset to a default range index.
            * integer: Use the index/columns from the i-th object in `objs`.
            * Any other value will be converted to a `pd.Index`.
        axis (int): Axis along which to broadcast, where 0 indicates index and 1 indicates columns.
        ignore_sr_names (Optional[bool]): Whether to ignore conflicting Series names.

            Conflicting Series names are those that differ and are not None.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.
        check_index_names (Optional[bool]): Whether to validate index/columns names equality
            (see `vectorbtpro.utils.checks.is_index_equal`).
        **clean_index_kwargs: Keyword arguments for `vectorbtpro.base.indexes.clean_index`.

    Returns:
        Optional[Index]: Broadcasted index/columns, or None if original index/columns should be retained.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.

    !!! note
        Series names are interpreted as columns with a single element but without a name.
        If a column level without a name loses its meaning, consider converting Series to DataFrames
        with a single column.Alternatively, if the Series name is insignificant, set it to None.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_sr_names is None:
        ignore_sr_names = broadcasting_cfg["ignore_sr_names"]
    if check_index_names is None:
        check_index_names = broadcasting_cfg["check_index_names"]

    index_str = "columns" if axis == 1 else "index"
    to_shape_2d = (to_shape[0], 1) if len(to_shape) == 1 else to_shape
    maxlen = to_shape_2d[1] if axis == 1 else to_shape_2d[0]
    new_index = None
    objs = list(objs)

    if index_from is None or (isinstance(index_from, str) and index_from.lower() == "keep"):
        return None
    if isinstance(index_from, int):
        if not checks.is_pandas(objs[index_from]):
            raise TypeError(f"Argument under index {index_from} must be a Pandas object")
        new_index = indexes.get_index(objs[index_from], axis)
    elif isinstance(index_from, str):
        if index_from.lower() == "reset":
            new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
        elif index_from.lower() in ("stack", "strict"):
            last_index = None
            index_conflict = False
            for obj in objs:
                if checks.is_pandas(obj):
                    index = indexes.get_index(obj, axis)
                    if last_index is not None:
                        if not checks.is_index_equal(
                            index, last_index, check_names=check_index_names
                        ):
                            index_conflict = True
                    last_index = index
                    continue
            if not index_conflict:
                new_index = last_index
            else:
                for obj in objs:
                    if checks.is_pandas(obj):
                        index = indexes.get_index(obj, axis)
                        if axis == 1 and checks.is_series(obj) and ignore_sr_names:
                            continue
                        if checks.is_default_index(index):
                            continue
                        if new_index is None:
                            new_index = index
                        else:
                            if checks.is_index_equal(
                                index, new_index, check_names=check_index_names
                            ):
                                continue
                            if index_from.lower() == "strict":
                                raise ValueError(
                                    f"Arrays have different index. Broadcasting {index_str} "
                                    f"is not allowed when {index_str}_from=strict"
                                )
                            if len(index) != len(new_index):
                                if len(index) > 1 and len(new_index) > 1:
                                    raise ValueError("Indexes could not be broadcast together")
                                if len(index) > len(new_index):
                                    new_index = indexes.repeat_index(
                                        new_index, len(index), ignore_ranges=ignore_ranges
                                    )
                                elif len(index) < len(new_index):
                                    index = indexes.repeat_index(
                                        index, len(new_index), ignore_ranges=ignore_ranges
                                    )
                            new_index = indexes.stack_indexes(
                                [new_index, index], **clean_index_kwargs
                            )
        else:
            raise ValueError(
                f"Invalid value '{index_from}' for {'columns' if axis == 1 else 'index'}_from"
            )
    else:
        if not isinstance(index_from, pd.Index):
            index_from = pd.Index(index_from)
        new_index = index_from
    if new_index is not None:
        if maxlen > len(new_index):
            if isinstance(index_from, str) and index_from.lower() == "strict":
                raise ValueError(
                    f"Broadcasting {index_str} is not allowed when {index_str}_from=strict"
                )
            if maxlen > 1 and len(new_index) > 1:
                raise ValueError("Indexes could not be broadcast together")
            new_index = indexes.repeat_index(new_index, maxlen, ignore_ranges=ignore_ranges)
    else:
        new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
    return new_index


def wrap_broadcasted(
    new_obj: tp.Array,
    old_obj: tp.Optional[tp.AnyArray] = None,
    axis: tp.Optional[int] = None,
    is_pd: bool = False,
    new_index: tp.Optional[tp.Index] = None,
    new_columns: tp.Optional[tp.Index] = None,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Return a Pandas object with updated index and columns if the broadcasted array
    originated from a Pandas object.

    Args:
        new_obj (AnyArray): Broadcasted array.
        old_obj (Optional[AnyArray]): Original object from which broadcasting
            parameters may be inferred.
        axis (Optional[int]): Axis along which broadcasting was performed.
        is_pd (bool): Indicates whether the original object was a Pandas object.
        new_index (Optional[Index]): New index derived from broadcasting.
        new_columns (Optional[Index]): New columns derived from broadcasting.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.

    Returns:
        AnyArray: Pandas DataFrame or Series if `is_pd` is True;
            otherwise, the unmodified broadcasted array.
    """
    if is_pd:
        if axis == 0:
            new_columns = None
        elif axis == 1:
            new_index = None
        if old_obj is not None and checks.is_pandas(old_obj):
            if new_index is None:
                old_index = indexes.get_index(old_obj, 0)
                if old_obj.shape[0] == new_obj.shape[0]:
                    new_index = old_index
                else:
                    new_index = indexes.repeat_index(
                        old_index, new_obj.shape[0], ignore_ranges=ignore_ranges
                    )
            if new_columns is None:
                old_columns = indexes.get_index(old_obj, 1)
                new_ncols = new_obj.shape[1] if new_obj.ndim == 2 else 1
                if len(old_columns) == new_ncols:
                    new_columns = old_columns
                else:
                    new_columns = indexes.repeat_index(
                        old_columns, new_ncols, ignore_ranges=ignore_ranges
                    )
        if new_obj.ndim == 2:
            return pd.DataFrame(new_obj, index=new_index, columns=new_columns)
        if new_columns is not None and len(new_columns) == 1:
            name = new_columns[0]
            if name == 0:
                name = None
        else:
            name = None
        return pd.Series(new_obj, index=new_index, name=name)
    return new_obj


def align_pd_arrays(
    *objs: tp.AnyArray,
    align_index: bool = True,
    align_columns: bool = True,
    to_index: tp.Optional[tp.Index] = None,
    to_columns: tp.Optional[tp.Index] = None,
    axis: tp.Optional[tp.MaybeSequence[int]] = None,
    reindex_kwargs: tp.KwargsLikeSequence = None,
) -> tp.MaybeTuple[tp.ArrayLike]:
    """Align Pandas arrays by reindexing their index and/or columns to a common level.

    Args:
        *objs (AnyArray): Pandas arrays to align.
        align_index (bool): Whether to align array indices.
        align_columns (bool): Whether to align DataFrame columns.
        to_index (Optional[Index]): Target index for alignment.
        to_columns (Optional[Index]): Target columns for alignment.
        axis (Optional[MaybeSequence[int]]): Axis to align along; allows for
            different axes for each Pandas array.
        reindex_kwargs (KwargsLikeSequence): Keyword arguments for `pd.DataFrame.reindex`.

    Returns:
        MaybeTuple[ArrayLike]: Aligned array if a single input is provided,
            otherwise a tuple of aligned arrays.
    """
    objs = list(objs)
    if align_index:
        indexes_to_align = []
        for i in range(len(objs)):
            if axis is not None:
                if checks.is_sequence(axis):
                    _axis = axis[i]
                else:
                    _axis = axis
            else:
                _axis = None
            if _axis in (None, 0):
                if checks.is_pandas(objs[i]):
                    if not checks.is_default_index(objs[i].index):
                        indexes_to_align.append(i)
        if (len(indexes_to_align) > 0 and to_index is not None) or len(indexes_to_align) > 1:
            if to_index is None:
                new_index = None
                index_changed = False
                for i in indexes_to_align:
                    arg_index = objs[i].index
                    if new_index is None:
                        new_index = arg_index
                    else:
                        if not checks.is_index_equal(new_index, arg_index):
                            if new_index.dtype != arg_index.dtype:
                                raise ValueError(
                                    "Indexes to be aligned must have the same data type"
                                )
                            new_index = new_index.union(arg_index)
                            index_changed = True
            else:
                new_index = to_index
                index_changed = True
            if index_changed:
                for i in indexes_to_align:
                    if to_index is None or not checks.is_index_equal(objs[i].index, to_index):
                        if objs[i].index.has_duplicates:
                            raise ValueError(f"Index at position {i} contains duplicates")
                        if not objs[i].index.is_monotonic_increasing:
                            raise ValueError(
                                f"Index at position {i} is not monotonically increasing"
                            )
                        _reindex_kwargs = resolve_dict(reindex_kwargs, i=i)
                        was_bool = (isinstance(objs[i], pd.Series) and objs[i].dtype == "bool") or (
                            isinstance(objs[i], pd.DataFrame) and (objs[i].dtypes == "bool").all()
                        )
                        objs[i] = objs[i].reindex(new_index, **_reindex_kwargs)
                        is_object = (
                            isinstance(objs[i], pd.Series) and objs[i].dtype == "object"
                        ) or (
                            isinstance(objs[i], pd.DataFrame) and (objs[i].dtypes == "object").all()
                        )
                        if was_bool and is_object:
                            objs[i] = objs[i].astype(None)
    if align_columns:
        columns_to_align = []
        for i in range(len(objs)):
            if axis is not None:
                if checks.is_sequence(axis):
                    _axis = axis[i]
                else:
                    _axis = axis
            else:
                _axis = None
            if _axis in (None, 1):
                if checks.is_frame(objs[i]) and len(objs[i].columns) > 1:
                    if not checks.is_default_index(objs[i].columns):
                        columns_to_align.append(i)
        if (len(columns_to_align) > 0 and to_columns is not None) or len(columns_to_align) > 1:
            indexes_ = [objs[i].columns for i in columns_to_align]
            if to_columns is not None:
                indexes_.append(to_columns)
            if len(set(map(len, indexes_))) > 1:
                col_indices = indexes.align_indexes(*indexes_)
                for i in columns_to_align:
                    objs[i] = objs[i].iloc[:, col_indices[columns_to_align.index(i)]]
    if len(objs) == 1:
        return objs[0]
    return tuple(objs)


@define
class BCO(DefineMixin):
    """Class representing a broadcast configuration object for `broadcast`.

    If any attribute value is None, it defaults to the corresponding global value provided to `broadcast`.
    """

    value: tp.Any = define.field()
    """Value to be broadcast."""

    axis: tp.Optional[int] = define.field(default=None)
    """Axis used for broadcasting.

    Set to None to broadcast across all axes.
    """

    to_pd: tp.Optional[bool] = define.field(default=None)
    """Determines whether the output array should be converted to a Pandas object.
    """

    keep_flex: tp.Optional[bool] = define.field(default=None)
    """Indicates whether to retain the raw output array for flexible indexing.

    This ensures the array can be broadcast to the target shape.
    """

    min_ndim: tp.Optional[int] = define.field(default=None)
    """Specifies the minimum number of dimensions required.
    """

    expand_axis: tp.Optional[int] = define.field(default=None)
    """Specifies the axis along which to expand a 1-dimensional array when a 2-dimensional shape is expected.
    """

    post_func: tp.Optional[tp.Callable] = define.field(default=None)
    """Function to post-process the output array.
    """

    require_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for `np.require`.
    """

    reindex_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for `pd.DataFrame.reindex`.
    """

    merge_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for `vectorbtpro.base.merging.column_stack_merge`.
    """

    context: tp.KwargsLike = define.field(default=None)
    """Context for evaluating templates.

    Will be merged with `template_context`.
    """


@define
class Default(DefineMixin):
    """Class representing a wrapped default value."""

    value: tp.Any = define.field()
    """Default value."""


@define
class Ref(DefineMixin):
    """Class representing a reference to another value."""

    key: tp.Hashable = define.field()
    """Hashable key that references another value."""


def resolve_ref(
    dct: dict, k: tp.Hashable, inside_bco: bool = False, keep_wrap_default: bool = False
) -> tp.Any:
    """Recursively resolve a reference in a dictionary.

    If the value associated with the given key is an instance of `Default` or `Ref`,
    its underlying value is returned. When `inside_bco` is True, nested references within
    `BCO` instances are also resolved.

    Args:
        dct (dict): Dictionary containing values and reference wrappers.
        k (Hashable): Key whose value is to be resolved.
        inside_bco (bool): Whether to resolve nested references within a `BCO` context.
        keep_wrap_default (bool): Whether to wrap the resolved default value with a `Default` instance.

    Returns:
        Any: Fully resolved value.
    """
    v = dct[k]
    is_default = False
    if isinstance(v, Default):
        v = v.value
        is_default = True
    if isinstance(v, Ref):
        new_v = resolve_ref(dct, v.key, inside_bco=inside_bco)
        if keep_wrap_default and is_default:
            return Default(new_v)
        return new_v
    if isinstance(v, BCO) and inside_bco:
        v = v.value
        is_default = False
        if isinstance(v, Default):
            v = v.value
            is_default = True
        if isinstance(v, Ref):
            new_v = resolve_ref(dct, v.key, inside_bco=inside_bco)
            if keep_wrap_default and is_default:
                return Default(new_v)
            return new_v
    return v


def broadcast(
    *objs,
    to_shape: tp.Optional[tp.ShapeLike] = None,
    align_index: tp.Optional[bool] = None,
    align_columns: tp.Optional[bool] = None,
    index_from: tp.Optional[tp.IndexFromLike] = None,
    columns_from: tp.Optional[tp.IndexFromLike] = None,
    to_frame: tp.Optional[bool] = None,
    axis: tp.Optional[tp.MaybeMappingSequence[int]] = None,
    to_pd: tp.Optional[tp.MaybeMappingSequence[bool]] = None,
    keep_flex: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    min_ndim: tp.MaybeMappingSequence[tp.Optional[int]] = None,
    expand_axis: tp.MaybeMappingSequence[tp.Optional[int]] = None,
    post_func: tp.MaybeMappingSequence[tp.Optional[tp.Callable]] = None,
    require_kwargs: tp.MaybeMappingSequence[tp.KwargsLike] = None,
    reindex_kwargs: tp.MaybeMappingSequence[tp.KwargsLike] = None,
    merge_kwargs: tp.MaybeMappingSequence[tp.KwargsLike] = None,
    tile: tp.Union[None, int, tp.IndexLike] = None,
    random_subset: tp.Optional[int] = None,
    seed: tp.Optional[int] = None,
    keep_wrap_default: tp.Optional[bool] = None,
    return_wrapper: bool = False,
    wrapper_kwargs: tp.KwargsLike = None,
    ignore_sr_names: tp.Optional[bool] = None,
    ignore_ranges: tp.Optional[bool] = None,
    check_index_names: tp.Optional[bool] = None,
    clean_index_kwargs: tp.KwargsLike = None,
    template_context: tp.KwargsLike = None,
) -> tp.Any:
    """Broadcast any array-like object to a common shape using NumPy-like broadcasting.

    See [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

    Supports broadcasting of Pandas objects by aligning their index and columns using `broadcast_index`.

    Keyword arguments for object-specific parameters can be provided as follows:

    * Constant applied to all objects.
    * Sequence with a value for each object.
    * Mapping with values for each object and a special key `_def` for the default.

    Additionally, any object can be wrapped with `BCO` to override the corresponding
    global arguments if its attributes are not None.

    Args:
        *objs (Any): Objects to broadcast.

            If a single mapping is provided, the function returns a dict.

            Supported types include `BCO`, `Ref`, `Default`, `vectorbtpro.utils.params.Param`,
            `vectorbtpro.base.indexing.index_dict`, `vectorbtpro.base.indexing.IdxSetter`,
            `vectorbtpro.base.indexing.IdxSetterFactory`, and templates.
            For an index dictionary, broadcasting fills using `vectorbtpro.base.wrapping.ArrayWrapper.fill_and_set`.
        to_shape (Optional[ShapeLike]): Target shape.

            When provided, each object in `objs` is broadcast to this shape.
        align_index (Optional[bool]): Whether to align the index of Pandas objects using the union of indexes.

            Pass None to use the default.
        align_columns (Optional[bool]): Whether to align the columns of Pandas objects using multi-index alignment.

            Pass None to use the default.
        index_from (Optional[IndexFromLike]): Rule for selecting the index during broadcasting.

            Pass None to use the default.
        columns_from (Optional[IndexFromLike]): Rule for selecting columns during broadcasting.

            Pass None to use the default.
        to_frame (Optional[bool]): Whether to convert all Series to DataFrames.
        axis (Optional[MaybeMappingSequence[int]]): See `BCO.axis`.
        to_pd (Optional[MaybeMappingSequence[bool]]): See `BCO.to_pd`.

            If not provided, conversion is performed only if at least one object is a Pandas object.
        keep_flex (MaybeMappingSequence[Optional[bool]]): See `BCO.keep_flex`.
        min_ndim (MaybeMappingSequence[Optional[int]]): See `BCO.min_ndim`.

            If not provided, becomes 2 when `keep_flex` is True, otherwise 1.
        expand_axis (MaybeMappingSequence[Optional[int]]): See `BCO.expand_axis`.
        post_func (MaybeMappingSequence[Optional[Callable]]): See `BCO.post_func`.

            Applied only when `keep_flex` is False.
        require_kwargs (MaybeMappingSequence[KwargsLike]): See `BCO.require_kwargs`.

            The provided values are merged with any argument-specific configuration.

            If the mapping contains all keys in `np.require`, it applies to all objects.
        reindex_kwargs (MaybeMappingSequence[KwargsLike]): See `BCO.reindex_kwargs`.

            The provided values are merged with any argument-specific configuration.

            If the mapping contains all keys in `pd.DataFrame.reindex`, it applies to all objects.
        merge_kwargs (MaybeMappingSequence[KwargsLike]): See `BCO.merge_kwargs`.

            The provided values are merged with any argument-specific configuration.

            If the mapping contains all keys in `pd.DataFrame.merge`, it applies to all objects.
        tile (Union[None, int, IndexLike]): Tile the final object by the specified number or index.
        random_subset (Optional[int]): Select a random subset of parameter combinations.

            Set the seed for reproducibility.
        seed (Optional[int]): Random seed for deterministic output.
        keep_wrap_default (Optional[bool]): Whether to retain wrapping with
            `vectorbtpro.base.reshaping.Default` for default values.
        return_wrapper (bool): Whether to also return the associated wrapper.
        wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        ignore_sr_names (Optional[bool]): Whether to ignore conflicting Series names.

            Conflicting Series names are those that differ and are not None.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.
        check_index_names (Optional[bool]): Whether to validate index/columns names equality
            (see `vectorbtpro.utils.checks.is_index_equal`).
        clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        template_context (KwargsLike): Additional context for template substitution.

    Returns:
        Any: Broadcasted object(s) and the associated wrapper if `return_wrapper` is True.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.

    !!! important
        The major difference to NumPy is that one-dimensional arrays always broadcast against the row axis!

    Examples:
        Without broadcasting index and columns:

        ```pycon
        >>> from vectorbtpro import *

        >>> v = 0
        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([1, 2, 3], index=pd.Index(['x', 'y', 'z']), name='a')
        >>> df = pd.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...     index=pd.Index(['x2', 'y2', 'z2']),
        ...     columns=pd.Index(['a2', 'b2', 'c2']),
        ... )

        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from='keep',
        ...     columns_from='keep',
        ...     align_index=False
        ... ): print(i)
           0  1  2
        0  0  0  0
        1  0  0  0
        2  0  0  0
           0  1  2
        0  1  2  3
        1  1  2  3
        2  1  2  3
           a  a  a
        x  1  1  1
        y  2  2  2
        z  3  3  3
            a2  b2  c2
        x2   1   2   3
        y2   4   5   6
        z2   7   8   9
        ```

        Take index and columns from the argument at specific position:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from=2,
        ...     columns_from=3,
        ...     align_index=False
        ... ): print(i)
           a2  b2  c2
        x   0   0   0
        y   0   0   0
        z   0   0   0
           a2  b2  c2
        x   1   2   3
        y   1   2   3
        z   1   2   3
           a2  b2  c2
        x   1   1   1
        y   2   2   2
        z   3   3   3
           a2  b2  c2
        x   1   2   3
        y   4   5   6
        z   7   8   9
        ```

        Broadcast index and columns through stacking:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from='stack',
        ...     columns_from='stack',
        ...     align_index=False
        ... ): print(i)
              a2  b2  c2
        x x2   0   0   0
        y y2   0   0   0
        z z2   0   0   0
              a2  b2  c2
        x x2   1   2   3
        y y2   1   2   3
        z z2   1   2   3
              a2  b2  c2
        x x2   1   1   1
        y y2   2   2   2
        z z2   3   3   3
              a2  b2  c2
        x x2   1   2   3
        y y2   4   5   6
        z z2   7   8   9
        ```

        Set index and columns manually:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from=['a', 'b', 'c'],
        ...     columns_from=['d', 'e', 'f'],
        ...     align_index=False
        ... ): print(i)
           d  e  f
        a  0  0  0
        b  0  0  0
        c  0  0  0
           d  e  f
        a  1  2  3
        b  1  2  3
        c  1  2  3
           d  e  f
        a  1  1  1
        b  2  2  2
        c  3  3  3
           d  e  f
        a  1  2  3
        b  4  5  6
        c  7  8  9
        ```

        Pass arguments as a mapping returns a mapping:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df),
        ...     index_from='stack',
        ...     align_index=False
        ... )
        {'v':       a2  b2  c2
              x x2   0   0   0
              y y2   0   0   0
              z z2   0   0   0,
         'a':       a2  b2  c2
              x x2   1   2   3
              y y2   1   2   3
              z z2   1   2   3,
         'sr':       a2  b2  c2
               x x2   1   1   1
               y y2   2   2   2
               z z2   3   3   3,
         'df':       a2  b2  c2
               x x2   1   2   3
               y y2   4   5   6
               z z2   7   8   9}
        ```

        Keep all results in a format suitable for flexible indexing apart from one:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df),
        ...     index_from='stack',
        ...     keep_flex=dict(_def=True, df=False),
        ...     require_kwargs=dict(df=dict(dtype=float)),
        ...     align_index=False
        ... )
        {'v': array([[0]]),
         'a': array([[1],
                     [2],
                     [3]]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df':        a2   b2   c2
               x x2  1.0  2.0  3.0
               y y2  4.0  5.0  6.0
               z z2  7.0  8.0  9.0}
        ```

        Specify arguments per object using `BCO`:

        ```pycon
        >>> df_bco = vbt.BCO(df, keep_flex=False, require_kwargs=dict(dtype=float))
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df_bco),
        ...     index_from='stack',
        ...     keep_flex=True,
        ...     align_index=False
        ... )
        {'v': array([[0]]),
         'a': array([[1],
                     [2],
                     [3]]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df':        a2   b2   c2
               x x2  1.0  2.0  3.0
               y y2  4.0  5.0  6.0
               z z2  7.0  8.0  9.0}
        ```

        Introduce a parameter that should build a Cartesian product of its values and other objects:

        ```pycon
        >>> df_bco = vbt.BCO(df, keep_flex=False, require_kwargs=dict(dtype=float))
        >>> p_bco = vbt.BCO(vbt.Param([1, 2, 3], name='my_p'))
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df_bco, p=p_bco),
        ...     index_from='stack',
        ...     keep_flex=True,
        ...     align_index=False
        ... )
        {'v': array([[0]]),
         'a': array([[1],
                     [2],
                     [3]]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df': my_p             1              2              3
                      a2   b2   c2   a2   b2   c2   a2   b2   c2
               x x2  1.0  2.0  3.0  1.0  2.0  3.0  1.0  2.0  3.0
               y y2  4.0  5.0  6.0  4.0  5.0  6.0  4.0  5.0  6.0
               z z2  7.0  8.0  9.0  7.0  8.0  9.0  7.0  8.0  9.0,
         'p': array([[1, 1, 1, 2, 2, 2, 3, 3, 3]])}
        ```

        Build a Cartesian product of all parameters:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.Param([1, 2, 3]),
        ...         b=vbt.Param(['x', 'y']),
        ...         c=vbt.Param([False, True])
        ...     )
        ... )
        {'a': array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]]),
         'b': array([['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x', 'y', 'y']], dtype='<U1'),
         'c': array([[False, True, False, True, False, True, False, True, False, True, False, True]])}
        ```

        Build a Cartesian product of two groups of parameters - (a, d) and (b, c):

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.Param([1, 2, 3], level=0),
        ...         b=vbt.Param(['x', 'y'], level=1),
        ...         d=vbt.Param([100., 200., 300.], level=0),
        ...         c=vbt.Param([False, True], level=1)
        ...     )
        ... )
        {'a': array([[1, 1, 2, 2, 3, 3]]),
         'b': array([['x', 'y', 'x', 'y', 'x', 'y']], dtype='<U1'),
         'd': array([[100., 100., 200., 200., 300., 300.]]),
         'c': array([[False,  True, False,  True, False,  True]])}
        ```

        Select a random subset of parameter combinations:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.Param([1, 2, 3]),
        ...         b=vbt.Param(['x', 'y']),
        ...         c=vbt.Param([False, True])
        ...     ),
        ...     random_subset=5,
        ...     seed=42
        ... )
        {'a': array([[1, 2, 3, 3, 3]]),
         'b': array([['x', 'x', 'x', 'x', 'y']], dtype='<U1'),
         'c': array([[False,  True, False,  True, False]])}
        ```
    """
    # Get defaults
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if align_index is None:
        align_index = broadcasting_cfg["align_index"]
    if align_columns is None:
        align_columns = broadcasting_cfg["align_columns"]
    if index_from is None:
        index_from = broadcasting_cfg["index_from"]
    if columns_from is None:
        columns_from = broadcasting_cfg["columns_from"]
    if keep_wrap_default is None:
        keep_wrap_default = broadcasting_cfg["keep_wrap_default"]
    require_kwargs_per_obj = True
    if require_kwargs is not None and checks.is_mapping(require_kwargs):
        require_arg_names = get_func_arg_names(np.require)
        if set(require_kwargs) <= set(require_arg_names):
            require_kwargs_per_obj = False
    reindex_kwargs_per_obj = True
    if reindex_kwargs is not None and checks.is_mapping(reindex_kwargs):
        reindex_arg_names = get_func_arg_names(pd.DataFrame.reindex)
        if set(reindex_kwargs) <= set(reindex_arg_names):
            reindex_kwargs_per_obj = False
    merge_kwargs_per_obj = True
    if merge_kwargs is not None and checks.is_mapping(merge_kwargs):
        merge_arg_names = get_func_arg_names(pd.DataFrame.merge)
        if set(merge_kwargs) <= set(merge_arg_names):
            merge_kwargs_per_obj = False
    if clean_index_kwargs is None:
        clean_index_kwargs = {}
    if checks.is_mapping(objs[0]) and not isinstance(objs[0], indexing.index_dict):
        if len(objs) > 1:
            raise ValueError("Only one argument is allowed when passing a mapping")
        all_keys = list(dict(objs[0]).keys())
        objs = list(objs[0].values())
        return_dict = True
    else:
        objs = list(objs)
        all_keys = list(range(len(objs)))
        return_dict = False

    def _resolve_arg(
        obj: tp.Any, arg_name: str, global_value: tp.Any, default_value: tp.Any
    ) -> tp.Any:
        if isinstance(obj, BCO) and getattr(obj, arg_name) is not None:
            return getattr(obj, arg_name)
        if checks.is_mapping(global_value):
            return global_value.get(k, global_value.get("_def", default_value))
        if checks.is_sequence(global_value):
            return global_value[i]
        return global_value

    # Build BCO instances
    none_keys = set()
    default_keys = set()
    param_keys = set()
    special_keys = set()
    bco_instances = {}
    pool = dict(zip(all_keys, objs))
    for i, k in enumerate(all_keys):
        obj = objs[i]

        if isinstance(obj, Default):
            obj = obj.value
            default_keys.add(k)
        if isinstance(obj, Ref):
            obj = resolve_ref(pool, k)
        if isinstance(obj, BCO):
            value = obj.value
        else:
            value = obj
        if isinstance(value, Default):
            value = value.value
            default_keys.add(k)
        if isinstance(value, Ref):
            value = resolve_ref(pool, k, inside_bco=True)
        if value is None:
            none_keys.add(k)
            continue

        _axis = _resolve_arg(obj, "axis", axis, None)
        _to_pd = _resolve_arg(obj, "to_pd", to_pd, None)

        _keep_flex = _resolve_arg(obj, "keep_flex", keep_flex, None)
        if _keep_flex is None:
            _keep_flex = broadcasting_cfg["keep_flex"]

        _min_ndim = _resolve_arg(obj, "min_ndim", min_ndim, None)
        if _min_ndim is None:
            _min_ndim = broadcasting_cfg["min_ndim"]

        _expand_axis = _resolve_arg(obj, "expand_axis", expand_axis, None)
        if _expand_axis is None:
            _expand_axis = broadcasting_cfg["expand_axis"]

        _post_func = _resolve_arg(obj, "post_func", post_func, None)

        if isinstance(obj, BCO) and obj.require_kwargs is not None:
            _require_kwargs = obj.require_kwargs
        else:
            _require_kwargs = None
        if checks.is_mapping(require_kwargs) and require_kwargs_per_obj:
            _require_kwargs = merge_dicts(
                require_kwargs.get("_def", None),
                require_kwargs.get(k, None),
                _require_kwargs,
            )
        elif checks.is_sequence(require_kwargs) and require_kwargs_per_obj:
            _require_kwargs = merge_dicts(require_kwargs[i], _require_kwargs)
        else:
            _require_kwargs = merge_dicts(require_kwargs, _require_kwargs)

        if isinstance(obj, BCO) and obj.reindex_kwargs is not None:
            _reindex_kwargs = obj.reindex_kwargs
        else:
            _reindex_kwargs = None
        if checks.is_mapping(reindex_kwargs) and reindex_kwargs_per_obj:
            _reindex_kwargs = merge_dicts(
                reindex_kwargs.get("_def", None),
                reindex_kwargs.get(k, None),
                _reindex_kwargs,
            )
        elif checks.is_sequence(reindex_kwargs) and reindex_kwargs_per_obj:
            _reindex_kwargs = merge_dicts(reindex_kwargs[i], _reindex_kwargs)
        else:
            _reindex_kwargs = merge_dicts(reindex_kwargs, _reindex_kwargs)

        if isinstance(obj, BCO) and obj.merge_kwargs is not None:
            _merge_kwargs = obj.merge_kwargs
        else:
            _merge_kwargs = None
        if checks.is_mapping(merge_kwargs) and merge_kwargs_per_obj:
            _merge_kwargs = merge_dicts(
                merge_kwargs.get("_def", None),
                merge_kwargs.get(k, None),
                _merge_kwargs,
            )
        elif checks.is_sequence(merge_kwargs) and merge_kwargs_per_obj:
            _merge_kwargs = merge_dicts(merge_kwargs[i], _merge_kwargs)
        else:
            _merge_kwargs = merge_dicts(merge_kwargs, _merge_kwargs)

        if isinstance(obj, BCO):
            _context = merge_dicts(template_context, obj.context)
        else:
            _context = template_context

        if isinstance(value, Param):
            param_keys.add(k)
        elif isinstance(
            value,
            (indexing.index_dict, indexing.IdxSetter, indexing.IdxSetterFactory, CustomTemplate),
        ):
            special_keys.add(k)
        else:
            value = to_any_array(value)

        bco_instances[k] = BCO(
            value,
            axis=_axis,
            to_pd=_to_pd,
            keep_flex=_keep_flex,
            min_ndim=_min_ndim,
            expand_axis=_expand_axis,
            post_func=_post_func,
            require_kwargs=_require_kwargs,
            reindex_kwargs=_reindex_kwargs,
            merge_kwargs=_merge_kwargs,
            context=_context,
        )

    # Check whether we should broadcast Pandas metadata and work on 2-dim data
    is_pd = False
    is_2d = False

    old_objs = {}
    obj_axis = {}
    obj_reindex_kwargs = {}
    for k, bco_obj in bco_instances.items():
        if k in none_keys or k in param_keys or k in special_keys:
            continue

        obj = bco_obj.value
        if obj.ndim > 1:
            is_2d = True
        if checks.is_pandas(obj):
            is_pd = True
        if bco_obj.to_pd is not None and bco_obj.to_pd:
            is_pd = True
        old_objs[k] = obj
        obj_axis[k] = bco_obj.axis
        obj_reindex_kwargs[k] = bco_obj.reindex_kwargs

    if to_shape is not None:
        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        if len(to_shape) > 1:
            is_2d = True

    if to_frame is not None:
        is_2d = to_frame

    if to_pd is not None:
        is_pd = to_pd or (return_wrapper and is_pd)

    # Align Pandas arrays
    if index_from is not None and not isinstance(index_from, (int, str, pd.Index)):
        index_from = pd.Index(index_from)
    if columns_from is not None and not isinstance(columns_from, (int, str, pd.Index)):
        columns_from = pd.Index(columns_from)
    aligned_objs = align_pd_arrays(
        *old_objs.values(),
        align_index=align_index,
        align_columns=align_columns,
        to_index=index_from if isinstance(index_from, pd.Index) else None,
        to_columns=columns_from if isinstance(columns_from, pd.Index) else None,
        axis=list(obj_axis.values()),
        reindex_kwargs=list(obj_reindex_kwargs.values()),
    )
    if not isinstance(aligned_objs, tuple):
        aligned_objs = (aligned_objs,)
    aligned_objs = dict(zip(old_objs.keys(), aligned_objs))

    # Convert to NumPy
    ready_objs = {}
    for k, obj in aligned_objs.items():
        _expand_axis = bco_instances[k].expand_axis

        new_obj = np.asarray(obj)
        if is_2d and new_obj.ndim == 1:
            if isinstance(obj, pd.Series):
                new_obj = new_obj[:, None]
            else:
                new_obj = np.expand_dims(new_obj, _expand_axis)
        ready_objs[k] = new_obj

    # Get final shape
    if to_shape is None:
        try:
            to_shape = broadcast_shapes(
                *map(lambda x: x.shape, ready_objs.values()),
                axis=list(obj_axis.values()),
            )
        except ValueError:
            arr_shapes = {}
            for i, k in enumerate(bco_instances):
                if k in none_keys or k in param_keys or k in special_keys:
                    continue

                if len(ready_objs[k].shape) > 0:
                    arr_shapes[k] = ready_objs[k].shape
            raise ValueError("Could not broadcast shapes: %s" % str(arr_shapes))
    if not isinstance(to_shape, tuple):
        to_shape = (to_shape,)
    if len(to_shape) == 0:
        to_shape = (1,)
    to_shape_2d = to_shape if len(to_shape) > 1 else (*to_shape, 1)

    if is_pd:
        # Decide on index and columns
        # NOTE: Important to pass aligned_objs, not ready_objs, to preserve original shape info
        new_index = broadcast_index(
            [v for k, v in aligned_objs.items() if obj_axis[k] in (None, 0)],
            to_shape,
            index_from=index_from,
            axis=0,
            ignore_sr_names=ignore_sr_names,
            ignore_ranges=ignore_ranges,
            check_index_names=check_index_names,
            **clean_index_kwargs,
        )
        new_columns = broadcast_index(
            [v for k, v in aligned_objs.items() if obj_axis[k] in (None, 1)],
            to_shape,
            index_from=columns_from,
            axis=1,
            ignore_sr_names=ignore_sr_names,
            ignore_ranges=ignore_ranges,
            check_index_names=check_index_names,
            **clean_index_kwargs,
        )
    else:
        new_index = pd.RangeIndex(stop=to_shape_2d[0])
        new_columns = pd.RangeIndex(stop=to_shape_2d[1])

    # Build a product
    param_product = None
    param_columns = None
    n_params = 0
    if len(param_keys) > 0:
        # Combine parameters
        param_dct = {}
        for k, bco_obj in bco_instances.items():
            if k not in param_keys:
                continue
            param_dct[k] = bco_obj.value
        param_product, param_columns = combine_params(
            param_dct,
            random_subset=random_subset,
            seed=seed,
            clean_index_kwargs=clean_index_kwargs,
        )
        n_params = len(param_columns)

        # Combine parameter columns with new columns
        if param_columns is not None and new_columns is not None:
            new_columns = indexes.combine_indexes(
                [param_columns, new_columns], **clean_index_kwargs
            )

    # Tile
    if tile is not None:
        if isinstance(tile, int):
            if new_columns is not None:
                new_columns = indexes.tile_index(new_columns, tile)
        else:
            if new_columns is not None:
                new_columns = indexes.combine_indexes([tile, new_columns], **clean_index_kwargs)
            tile = len(tile)
        n_params = max(n_params, 1) * tile

    # Build wrapper
    if n_params == 0:
        new_shape = to_shape
    else:
        new_shape = (to_shape_2d[0], to_shape_2d[1] * n_params)
    wrapper = wrapping.ArrayWrapper.from_shape(
        new_shape,
        **merge_dicts(
            dict(
                index=new_index,
                columns=new_columns,
            ),
            wrapper_kwargs,
        ),
    )

    def _adjust_dims(new_obj, _keep_flex, _min_ndim, _expand_axis):
        if _min_ndim is None:
            if _keep_flex:
                _min_ndim = 2
            else:
                _min_ndim = 1
        if _min_ndim not in (1, 2):
            raise ValueError("Argument min_ndim must be either 1 or 2")
        if _min_ndim in (1, 2) and new_obj.ndim == 0:
            new_obj = new_obj[None]
        if _min_ndim == 2 and new_obj.ndim == 1:
            if len(to_shape) == 1:
                new_obj = new_obj[:, None]
            else:
                new_obj = np.expand_dims(new_obj, _expand_axis)
        return new_obj

    # Perform broadcasting
    aligned_objs2 = {}
    new_objs = {}
    for i, k in enumerate(all_keys):
        if k in none_keys or k in special_keys:
            continue
        _keep_flex = bco_instances[k].keep_flex
        _min_ndim = bco_instances[k].min_ndim
        _axis = bco_instances[k].axis
        _expand_axis = bco_instances[k].expand_axis
        _merge_kwargs = bco_instances[k].merge_kwargs
        _context = bco_instances[k].context
        must_reset_index = _merge_kwargs.get("reset_index", None) not in (None, False)
        _reindex_kwargs = resolve_dict(bco_instances[k].reindex_kwargs)
        _fill_value = _reindex_kwargs.get("fill_value", np.nan)

        if k in param_keys:
            # Broadcast parameters
            from vectorbtpro.base.merging import column_stack_merge

            if _axis == 0:
                raise ValueError("Parameters do not support broadcasting with axis=0")
            obj = param_product[k]
            new_obj = []
            any_needs_broadcasting = False
            all_forced_broadcast = True
            for o in obj:
                if isinstance(
                    o, (indexing.index_dict, indexing.IdxSetter, indexing.IdxSetterFactory)
                ):
                    o = wrapper.fill_and_set(
                        o,
                        fill_value=_fill_value,
                        keep_flex=_keep_flex,
                    )
                elif isinstance(o, CustomTemplate):
                    context = merge_dicts(
                        dict(
                            bco_instances=bco_instances,
                            wrapper=wrapper,
                            obj_name=k,
                            bco=bco_instances[k],
                        ),
                        _context,
                    )
                    o = o.substitute(context, eval_id="broadcast")
                o = to_2d_array(o)
                if not _keep_flex or o.shape[0] > 1 or o.shape[1] > 1 and o.shape[1] != to_shape_2d[1]:
                    needs_broadcasting = True
                else:
                    needs_broadcasting = False
                if needs_broadcasting:
                    any_needs_broadcasting = True
                    o = broadcast_array_to(o, to_shape_2d, axis=_axis)
                elif o.size == 1:
                    all_forced_broadcast = False
                    o = np.repeat(o, to_shape_2d[1], axis=1)
                else:
                    all_forced_broadcast = False
                new_obj.append(o)
            if any_needs_broadcasting and not all_forced_broadcast:
                new_obj2 = []
                for o in new_obj:
                    if o.shape[1] != to_shape_2d[1] or (
                        not must_reset_index and o.shape[0] != to_shape_2d[0]
                    ):
                        o = broadcast_array_to(o, to_shape_2d, axis=_axis)
                    new_obj2.append(o)
                new_obj = new_obj2
            obj = column_stack_merge(new_obj, **_merge_kwargs)
            if tile is not None:
                obj = np.tile(obj, (1, tile))
            old_obj = obj
            new_obj = obj
        else:
            # Broadcast regular objects
            old_obj = aligned_objs[k]
            new_obj = ready_objs[k]
            if (
                _axis in (None, 0)
                and new_obj.ndim >= 1
                and new_obj.shape[0] > 1
                and new_obj.shape[0] != to_shape[0]
            ):
                raise ValueError(
                    f"Could not broadcast argument {k} of shape {new_obj.shape} to {to_shape}"
                )
            if (
                _axis in (None, 1)
                and new_obj.ndim == 2
                and new_obj.shape[1] > 1
                and new_obj.shape[1] != to_shape[1]
            ):
                raise ValueError(
                    f"Could not broadcast argument {k} of shape {new_obj.shape} to {to_shape}"
                )
            if _keep_flex:
                if n_params > 0 and _axis in (None, 1):
                    if len(to_shape) == 1:
                        if new_obj.ndim == 1 and new_obj.shape[0] > 1:
                            new_obj = new_obj[:, None]  # product changes is_2d behavior
                    else:
                        if new_obj.ndim == 1 and new_obj.shape[0] > 1:
                            new_obj = np.tile(new_obj, n_params)
                        elif new_obj.ndim == 2 and new_obj.shape[1] > 1:
                            new_obj = np.tile(new_obj, (1, n_params))
            else:
                new_obj = broadcast_array_to(new_obj, to_shape, axis=_axis)
                if n_params > 0 and _axis in (None, 1):
                    if new_obj.ndim == 1:
                        new_obj = new_obj[:, None]  # product changes is_2d behavior
                    new_obj = np.tile(new_obj, (1, n_params))

        new_obj = _adjust_dims(new_obj, _keep_flex, _min_ndim, _expand_axis)
        aligned_objs2[k] = old_obj
        new_objs[k] = new_obj

    # Resolve special objects
    new_objs2 = {}
    for i, k in enumerate(all_keys):
        if k in none_keys:
            continue
        if k in special_keys:
            bco = bco_instances[k]
            if isinstance(
                bco.value, (indexing.index_dict, indexing.IdxSetter, indexing.IdxSetterFactory)
            ):
                _is_pd = bco.to_pd
                if _is_pd is None:
                    _is_pd = is_pd
                _keep_flex = bco.keep_flex
                _min_ndim = bco.min_ndim
                _expand_axis = bco.expand_axis
                _reindex_kwargs = resolve_dict(bco.reindex_kwargs)
                _fill_value = _reindex_kwargs.get("fill_value", np.nan)
                new_obj = wrapper.fill_and_set(
                    bco.value,
                    fill_value=_fill_value,
                    keep_flex=_keep_flex,
                )
                if not _is_pd and not _keep_flex:
                    new_obj = new_obj.values
                new_obj = _adjust_dims(new_obj, _keep_flex, _min_ndim, _expand_axis)
            elif isinstance(bco.value, CustomTemplate):
                context = merge_dicts(
                    dict(
                        bco_instances=bco_instances,
                        new_objs=new_objs,
                        wrapper=wrapper,
                        obj_name=k,
                        bco=bco,
                    ),
                    bco.context,
                )
                new_obj = bco.value.substitute(context, eval_id="broadcast")
            else:
                raise TypeError(f"Special type {type(bco.value)} is not supported")
        else:
            new_obj = new_objs[k]

        # Force to match requirements
        new_obj = np.require(new_obj, **resolve_dict(bco_instances[k].require_kwargs))
        new_objs2[k] = new_obj

    # Perform wrapping and post-processing
    new_objs3 = {}
    for i, k in enumerate(all_keys):
        if k in none_keys:
            continue
        new_obj = new_objs2[k]
        _axis = bco_instances[k].axis
        _keep_flex = bco_instances[k].keep_flex

        if not _keep_flex:
            # Wrap array
            _is_pd = bco_instances[k].to_pd
            if _is_pd is None:
                _is_pd = is_pd
            new_obj = wrap_broadcasted(
                new_obj,
                old_obj=aligned_objs2[k] if k not in special_keys else None,
                axis=_axis,
                is_pd=_is_pd,
                new_index=new_index,
                new_columns=new_columns,
                ignore_ranges=ignore_ranges,
            )

        # Post-process array
        _post_func = bco_instances[k].post_func
        if _post_func is not None:
            new_obj = _post_func(new_obj)
        new_objs3[k] = new_obj

    # Prepare outputs
    return_objs = []
    for k in all_keys:
        if k not in none_keys:
            if k in default_keys and keep_wrap_default:
                return_objs.append(Default(new_objs3[k]))
            else:
                return_objs.append(new_objs3[k])
        else:
            if k in default_keys and keep_wrap_default:
                return_objs.append(Default(None))
            else:
                return_objs.append(None)
    if return_dict:
        return_objs = dict(zip(all_keys, return_objs))
    else:
        return_objs = tuple(return_objs)
    if len(return_objs) > 1 or return_dict:
        if return_wrapper:
            return return_objs, wrapper
        return return_objs
    if return_wrapper:
        return return_objs[0], wrapper
    return return_objs[0]


def broadcast_to(
    arg1: tp.ArrayLike,
    arg2: tp.Union[tp.ArrayLike, tp.ShapeLike, wrapping.ArrayWrapper],
    to_pd: tp.Optional[bool] = None,
    index_from: tp.Optional[tp.IndexFromLike] = None,
    columns_from: tp.Optional[tp.IndexFromLike] = None,
    **kwargs,
) -> tp.Any:
    """Broadcast `arg1` to the shape of `arg2`.

    Args:
        arg1 (ArrayLike): Input array or scalar to broadcast.
        arg2 (Union[ArrayLike, ShapeLike, ArrayWrapper]): Target shape, array-like object,
            or instance of `vectorbtpro.base.wrapping.ArrayWrapper`.
        to_pd (Optional[bool]): Convert result to a Pandas object if True.
        index_from (Optional[IndexFromLike]): Index to use for the output;
            if None, uses the index of `arg2`.
        columns_from (Optional[IndexFromLike]): Columns to use for the output;
            if None, uses the columns of `arg2`.
        **kwargs: Keyword arguments for `broadcast`.

    Returns:
        Any: Broadcasted structure, either as an array or a Pandas object depending on `to_pd`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.reshaping import broadcast_to

        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([4, 5, 6], index=pd.Index(['x', 'y', 'z']), name='a')

        >>> broadcast_to(a, sr)
        x    1
        y    2
        z    3
        Name: a, dtype: int64

        >>> broadcast_to(sr, a)
        array([4, 5, 6])
        ```
    """
    if checks.is_int(arg2) or isinstance(arg2, tuple):
        arg2 = to_tuple_shape(arg2)
    if isinstance(arg2, tuple):
        to_shape = arg2
    elif isinstance(arg2, wrapping.ArrayWrapper):
        to_pd = True
        if index_from is None:
            index_from = arg2.index
        if columns_from is None:
            columns_from = arg2.columns
        to_shape = arg2.shape
    else:
        arg2 = to_any_array(arg2)
        if to_pd is None:
            to_pd = checks.is_pandas(arg2)
        if to_pd:
            # Take index and columns from arg2
            if index_from is None:
                index_from = indexes.get_index(arg2, 0)
            if columns_from is None:
                columns_from = indexes.get_index(arg2, 1)
        to_shape = arg2.shape
    return broadcast(
        arg1,
        to_shape=to_shape,
        to_pd=to_pd,
        index_from=index_from,
        columns_from=columns_from,
        **kwargs,
    )


def broadcast_to_array_of(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> tp.Array:
    """Broadcast `arg1` to an array with shape `(1, *arg2.shape)`.

    Args:
        arg1 (ArrayLike): Scalar, 1-dimensional array, or an array with one extra dimension compared to `arg2`.
        arg2 (ArrayLike): Array whose shape is used as the base for broadcasting.

    Returns:
        Array: Broadcasted array with shape `(1, *arg2.shape)`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.reshaping import broadcast_to_array_of

        >>> broadcast_to_array_of([0.1, 0.2], np.empty((2, 2)))
        [[[0.1 0.1]
          [0.1 0.1]]

         [[0.2 0.2]
          [0.2 0.2]]]
        ```
    """
    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)
    if arg1.ndim == arg2.ndim + 1:
        if arg1.shape[1:] == arg2.shape:
            return arg1
    # From here on arg1 can be only a 1-dim array
    if arg1.ndim == 0:
        arg1 = to_1d(arg1)
    checks.assert_ndim(arg1, 1)

    if arg2.ndim == 0:
        return arg1
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
    return np.tile(arg1, (1, *arg2.shape))


def broadcast_to_axis_of(
    arg1: tp.ArrayLike,
    arg2: tp.ArrayLike,
    axis: int,
    require_kwargs: tp.KwargsLike = None,
) -> tp.Array:
    """Broadcast `arg1` to match the length of a specified axis in `arg2`.

    Args:
        arg1 (ArrayLike): Input value or array to be broadcast.
        arg2 (ArrayLike): Array whose specified axis determines the target length.
        axis (int): Axis index along which to broadcast `arg1`.
        require_kwargs (KwargsLike): Keyword arguments for `np.require`.

    Returns:
        Array: Broadcasted array.

    !!! note
        If `arg2` has fewer dimensions than `axis + 1`, `arg1` is broadcast to a single number.

        For additional keyword arguments, refer to the behavior of `broadcast`.
    """
    if require_kwargs is None:
        require_kwargs = {}
    arg2 = to_any_array(arg2)
    if arg2.ndim < axis + 1:
        return broadcast_array_to(arg1, (1,))[0]  # to a single number
    arg1 = broadcast_array_to(arg1, (arg2.shape[axis],))
    arg1 = np.require(arg1, **require_kwargs)
    return arg1


def broadcast_combs(
    *objs: tp.ArrayLike,
    axis: int = 1,
    comb_func: tp.Callable = itertools.product,
    **broadcast_kwargs,
) -> tp.Any:
    """Align the given arrays along a specified axis using a combinatorial function and broadcast their indexes.

    Args:
        *objs (ArrayLike): Array-like objects to combine.
        axis (int): Axis along which to align and broadcast the arrays.
        comb_func (Callable): Combinatorial function applied to indices (e.g., `itertools.product`).
        **broadcast_kwargs: Keyword arguments for `broadcast`.

    Returns:
        Any: Tuple of broadcasted arrays aligned based on the combinatorial indices.

    Examples:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.reshaping import broadcast_combs

        >>> df = pd.DataFrame([[1, 2, 3], [3, 4, 5]], columns=pd.Index(['a', 'b', 'c'], name='df_param'))
        >>> df2 = pd.DataFrame([[6, 7], [8, 9]], columns=pd.Index(['d', 'e'], name='df2_param'))
        >>> sr = pd.Series([10, 11], name='f')

        >>> new_df, new_df2, new_sr = broadcast_combs((df, df2, sr))

        >>> new_df
        df_param   a     b     c
        df2_param  d  e  d  e  d  e
        0          1  1  2  2  3  3
        1          3  3  4  4  5  5

        >>> new_df2
        df_param   a     b     c
        df2_param  d  e  d  e  d  e
        0          6  7  6  7  6  7
        1          8  9  8  9  8  9

        >>> new_sr
        df_param    a       b       c
        df2_param   d   e   d   e   d   e
        0          10  10  10  10  10  10
        1          11  11  11  11  11  11
        ```
    """
    if broadcast_kwargs is None:
        broadcast_kwargs = {}

    objs = list(objs)
    if len(objs) < 2:
        raise ValueError("At least two arguments are required")
    for i in range(len(objs)):
        obj = to_any_array(objs[i])
        if axis == 1:
            obj = to_2d(obj)
        objs[i] = obj
    indices = []
    for obj in objs:
        indices.append(np.arange(len(indexes.get_index(to_pd_array(obj), axis))))
    new_indices = list(map(list, zip(*list(comb_func(*indices)))))
    results = []
    for i, obj in enumerate(objs):
        if axis == 1:
            if checks.is_pandas(obj):
                results.append(obj.iloc[:, new_indices[i]])
            else:
                results.append(obj[:, new_indices[i]])
        else:
            if checks.is_pandas(obj):
                results.append(obj.iloc[new_indices[i]])
            else:
                results.append(obj[new_indices[i]])
    if axis == 1:
        broadcast_kwargs = merge_dicts(dict(columns_from="stack"), broadcast_kwargs)
    else:
        broadcast_kwargs = merge_dicts(dict(index_from="stack"), broadcast_kwargs)
    return broadcast(*results, **broadcast_kwargs)


def get_multiindex_series(obj: tp.SeriesFrame) -> tp.Series:
    """Return a Series with a MultiIndex.

    Args:
        obj (SeriesFrame): Pandas Series or DataFrame.

            If a DataFrame is provided, it must have at most one row or one column.

    Returns:
        Series: Resulting Series with a MultiIndex.

    !!! note
        If a DataFrame with more than one row and more than one column is provided, a `ValueError` is raised.
    """
    checks.assert_instance_of(obj, (pd.Series, pd.DataFrame))
    if checks.is_frame(obj):
        if obj.shape[0] == 1:
            obj = obj.iloc[0, :]
        elif obj.shape[1] == 1:
            obj = obj.iloc[:, 0]
        else:
            raise ValueError("Supported are either Series or DataFrame with one column/row")
    checks.assert_instance_of(obj.index, pd.MultiIndex)
    return obj


def unstack_to_array(
    obj: tp.SeriesFrame,
    levels: tp.Optional[tp.MaybeLevelSequence] = None,
    sort: bool = True,
    return_indexes: bool = False,
) -> tp.Union[tp.Array, tp.Tuple[tp.Array, tp.List[tp.Index]]]:
    """Reshape a multi-indexed Series or DataFrame into a multi-dimensional NumPy array.

    Args:
        obj (SeriesFrame): Series or DataFrame with a MultiIndex.
        levels (Optional[MaybeLevelSequence]): Index level(s) to unstack and their desired order.
        sort (bool): Whether to sort the new index values.
        return_indexes (bool): If True, also return the list of new index values corresponding to each level.

    Returns:
        Union[Array, Tuple[Array, List[Index]]]: Multi-dimensional array of unstacked data, or a tuple containing
            the array and the list of new index values if `return_indexes` is True.

    Examples:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.reshaping import unstack_to_array

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> unstack_to_array(sr).shape
        (2, 2, 4)

        >>> unstack_to_array(sr)
        [[[ 1. nan nan nan]
         [nan  2. nan nan]]

         [[nan nan  3. nan]
        [nan nan nan  4.]]]

        >>> unstack_to_array(sr, levels=(2, 0))
        [[ 1. nan]
         [ 2. nan]
         [nan  3.]
         [nan  4.]]
        ```
    """
    sr = get_multiindex_series(obj)
    if sr.index.duplicated().any():
        raise ValueError("Index contains duplicate entries, cannot reshape")

    new_index_list = []
    value_indices_list = []
    if levels is None:
        levels = range(sr.index.nlevels)
    if isinstance(levels, (int, str)):
        levels = (levels,)
    for level in levels:
        level_values = indexes.select_levels(sr.index, level)
        new_index = level_values.unique()
        if sort:
            new_index = new_index.sort_values()
        new_index_list.append(new_index)
        index_map = pd.Series(range(len(new_index)), index=new_index)
        value_indices = index_map.loc[level_values]
        value_indices_list.append(value_indices)

    a = np.full(list(map(len, new_index_list)), np.nan)
    a[tuple(zip(value_indices_list))] = sr.values
    if return_indexes:
        return a, new_index_list
    return a


def make_symmetric(obj: tp.SeriesFrame, sort: bool = True) -> tp.Frame:
    """Make the DataFrame symmetric so that its index and columns are identical.

    This operation requires that the index and columns have the same number of levels.

    Args:
        obj (SeriesFrame): Series or DataFrame.
        sort (bool): Whether to sort the combined index and columns.

            If False, the indexes are concatenated in their original order with duplicates removed.

    Returns:
        Frame: Symmetric DataFrame with matching index and columns.

    Examples:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.reshaping import make_symmetric

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['c', 'd'])

        >>> make_symmetric(df)
             a    b    c    d
        a  NaN  NaN  1.0  2.0
        b  NaN  NaN  3.0  4.0
        c  1.0  3.0  NaN  NaN
        d  2.0  4.0  NaN  NaN
        ```
    """
    from vectorbtpro.base.merging import concat_arrays

    checks.assert_instance_of(obj, (pd.Series, pd.DataFrame))
    df = to_2d(obj)
    if isinstance(df.index, pd.MultiIndex) or isinstance(df.columns, pd.MultiIndex):
        checks.assert_instance_of(df.index, pd.MultiIndex)
        checks.assert_instance_of(df.columns, pd.MultiIndex)
        checks.assert_array_equal(df.index.nlevels, df.columns.nlevels)
        names1, names2 = tuple(df.index.names), tuple(df.columns.names)
    else:
        names1, names2 = df.index.name, df.columns.name

    if names1 == names2:
        new_name = names1
    else:
        if isinstance(df.index, pd.MultiIndex):
            new_name = tuple(zip(*[names1, names2]))
        else:
            new_name = (names1, names2)
    if sort:
        idx_vals = np.unique(concat_arrays((df.index, df.columns))).tolist()
    else:
        idx_vals = list(dict.fromkeys(concat_arrays((df.index, df.columns))))
    df_index = df.index.copy()
    df_columns = df.columns.copy()
    if isinstance(df.index, pd.MultiIndex):
        unique_index = pd.MultiIndex.from_tuples(idx_vals, names=new_name)
        df_index.names = new_name
        df_columns.names = new_name
    else:
        unique_index = pd.Index(idx_vals, name=new_name)
        df_index.name = new_name
        df_columns.name = new_name
    df = df.copy(deep=False)
    df.index = df_index
    df.columns = df_columns
    df_out_dtype = np.promote_types(df.values.dtype, np.min_scalar_type(np.nan))
    df_out = pd.DataFrame(index=unique_index, columns=unique_index, dtype=df_out_dtype)
    df_out.loc[:, :] = df
    df_out[df_out.isnull()] = df.transpose()
    return df_out


def unstack_to_df(
    obj: tp.SeriesFrame,
    index_levels: tp.Optional[tp.MaybeLevelSequence] = None,
    column_levels: tp.Optional[tp.MaybeLevelSequence] = None,
    symmetric: bool = False,
    sort: bool = True,
) -> tp.Frame:
    """Reshape a multi-indexed Series or DataFrame into a DataFrame by unstacking specified levels.

    Args:
        obj (SeriesFrame): Series or DataFrame with a MultiIndex.
        index_levels (Optional[MaybeLevelSequence]): Index level(s) to unstack for the new DataFrame's index.
        column_levels (Optional[MaybeLevelSequence]): Index level(s) to unstack for the new DataFrame's columns.
        symmetric (bool): If True, return a symmetric DataFrame with identical index and columns.
        sort (bool): Whether to sort the level values before reshaping.

    Returns:
        Frame: DataFrame resulting from unstacking the specified index levels.

    Examples:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.reshaping import unstack_to_df

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']],
        ...     names=['x', 'y', 'z'])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> unstack_to_df(sr, index_levels=(0, 1), column_levels=2)
        z      a    b    c    d
        x y
        1 3  1.0  NaN  NaN  NaN
        1 4  NaN  2.0  NaN  NaN
        2 3  NaN  NaN  3.0  NaN
        2 4  NaN  NaN  NaN  4.0
        ```
    """
    sr = get_multiindex_series(obj)
    if sr.index.nlevels > 2:
        if index_levels is None:
            raise ValueError("index_levels must be specified")
        if column_levels is None:
            raise ValueError("column_levels must be specified")
    else:
        if index_levels is None:
            index_levels = 0
        if column_levels is None:
            column_levels = 1

    unstacked, (new_index, new_columns) = unstack_to_array(
        sr,
        levels=(index_levels, column_levels),
        sort=sort,
        return_indexes=True,
    )
    df = pd.DataFrame(unstacked, index=new_index, columns=new_columns)
    if symmetric:
        return make_symmetric(df, sort=sort)
    return df
