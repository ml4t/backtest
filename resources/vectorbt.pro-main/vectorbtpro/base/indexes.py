# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functionality for index and column manipulations.

Provides functions for stacking, combining, and cleansing Pandas MultiIndex levels.

!!! note
    In the Pandas context, "Index" refers to both row indexes and columns."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base

__all__ = [
    "ExceptLevel",
    "repeat_index",
    "tile_index",
    "stack_indexes",
    "combine_indexes",
]


@define
class ExceptLevel(DefineMixin):
    """Class for grouping one or more levels to be excluded."""

    value: tp.MaybeLevelSequence = define.field()
    """One or more level positions or names."""


def to_any_index(index_like: tp.IndexLike) -> tp.Index:
    """Convert any index-like object to a Pandas Index.

    Args:
        index_like (IndexLike): Object convertible to a Pandas Index.

    Returns:
        Index: Resulting Pandas Index instance. Index objects are returned unchanged.
    """
    if checks.is_np_array(index_like) and index_like.ndim == 0:
        index_like = index_like[None]
    if not checks.is_index(index_like):
        return pd.Index(index_like)
    return index_like


def get_index(obj: tp.SeriesFrame, axis: int) -> tp.Index:
    """Return the index or columns of a Series or DataFrame based on the specified axis.

    Args:
        obj (Union[pd.Series, pd.DataFrame]): Pandas Series or DataFrame.
        axis (int): Axis number (0 for row index, 1 for columns).

    Returns:
        Index: Row index if axis is 0, or the columns if axis is 1.

            For a Series on axis 1, returns an index containing the series name if available,
            otherwise an index with a single element 0.
    """
    checks.assert_instance_of(obj, (pd.Series, pd.DataFrame))
    checks.assert_in(axis, (0, 1))

    if axis == 0:
        return obj.index
    else:
        if checks.is_series(obj):
            if obj.name is not None:
                return pd.Index([obj.name])
            return pd.Index([0])  # same as how Pandas does it
        else:
            return obj.columns


def index_from_values(
    values: tp.Sequence,
    single_value: bool = False,
    name: tp.Optional[tp.Hashable] = None,
) -> tp.Index:
    """Create a new Pandas Index from a sequence of values.

    Processes each element in the sequence to generate corresponding index labels.
    When the `single_value` flag is True, only the first value is used and repeated for all entries.

    Args:
        values (Sequence): Iterable of values to generate index entries.
        single_value (bool): If True, uses only the first value from `values` for index creation,
            repeating it for all entries.
        name (Optional[Hashable]): Name to assign to the index.

    Returns:
        Index: Pandas Index with labels generated from the provided values.
    """
    scalar_types = (int, float, complex, str, bool, datetime, timedelta, np.generic)
    type_id_number = {}
    value_names = []
    if len(values) == 1:
        single_value = True
    for i in range(len(values)):
        if i > 0 and single_value:
            break
        v = values[i]
        if v is None or isinstance(v, scalar_types):
            value_names.append(v)
        elif isinstance(v, np.ndarray):
            all_same = False
            if np.issubdtype(v.dtype, np.floating):
                if np.isclose(v, v.item(0), equal_nan=True).all():
                    all_same = True
            elif v.dtype.names is not None:
                all_same = False
            else:
                if np.equal(v, v.item(0)).all():
                    all_same = True
            if all_same:
                value_names.append(v.item(0))
            else:
                if single_value:
                    value_names.append("array")
                else:
                    if "array" not in type_id_number:
                        type_id_number["array"] = {}
                    if id(v) not in type_id_number["array"]:
                        type_id_number["array"][id(v)] = len(type_id_number["array"])
                    value_names.append("array_%d" % (type_id_number["array"][id(v)]))
        else:
            type_name = str(type(v).__name__)
            if single_value:
                value_names.append("%s" % type_name)
            else:
                if type_name not in type_id_number:
                    type_id_number[type_name] = {}
                if id(v) not in type_id_number[type_name]:
                    type_id_number[type_name][id(v)] = len(type_id_number[type_name])
                value_names.append("%s_%d" % (type_name, type_id_number[type_name][id(v)]))
    if single_value and len(values) > 1:
        value_names *= len(values)
    return pd.Index(value_names, name=name)


def repeat_index(index: tp.IndexLike, n: int, ignore_ranges: tp.Optional[bool] = None) -> tp.Index:
    """Repeat each element in the provided index n times.

    Args:
        index (IndexLike): Input index to be repeated.
        n (int): Number of repetitions for each element.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.

    Returns:
        Index: New index with each element repeated n times.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_ranges is None:
        ignore_ranges = broadcasting_cfg["ignore_ranges"]

    index = to_any_index(index)
    if n == 1:
        return index
    if checks.is_default_index(index) and ignore_ranges:  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    return index.repeat(n)


def tile_index(index: tp.IndexLike, n: int, ignore_ranges: tp.Optional[bool] = None) -> tp.Index:
    """Tile the entire index by repeating its sequence n times.

    Args:
        index (IndexLike): Input index to be tiled.
        n (int): Number of times to tile the index.
        ignore_ranges (Optional[bool]): Whether to disregard indexes of type `pd.RangeIndex`.

    Returns:
        Index: Tiled index.

            If the input is a MultiIndex, the resulting index is a MultiIndex with tiled levels.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_ranges is None:
        ignore_ranges = broadcasting_cfg["ignore_ranges"]

    index = to_any_index(index)
    if n == 1:
        return index
    if checks.is_default_index(index) and ignore_ranges:  # ignore simple ranges without name
        return pd.RangeIndex(start=0, stop=len(index) * n, step=1)
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)
    return pd.Index(np.tile(index, n), name=index.name)


def clean_index(
    index: tp.IndexLike,
    drop_duplicates: tp.Optional[bool] = None,
    keep: tp.Optional[str] = None,
    drop_redundant: tp.Optional[bool] = None,
) -> tp.Index:
    """Clean the provided index by removing duplicate or redundant levels based on configuration.

    Args:
        index (IndexLike): Index to be cleaned.
        drop_duplicates (Optional[bool]): If True, remove duplicate levels.
        keep (Optional[str]): Indicates which duplicate to retain; valid options are "first" or "last".
        drop_redundant (Optional[bool]): If True, remove redundant levels.

    Returns:
        Index: Cleaned index.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if drop_duplicates is None:
        drop_duplicates = broadcasting_cfg["drop_duplicates"]
    if keep is None:
        keep = broadcasting_cfg["keep"]
    if drop_redundant is None:
        drop_redundant = broadcasting_cfg["drop_redundant"]

    index = to_any_index(index)
    if drop_duplicates:
        index = drop_duplicate_levels(index, keep=keep)
    if drop_redundant:
        index = drop_redundant_levels(index)
    return index


def stack_indexes(*indexes: tp.MaybeTuple[tp.IndexLike], **clean_index_kwargs) -> tp.Index:
    """Stack multiple indexes into a single MultiIndex by combining their levels sequentially.

    Args:
        *indexes (MaybeTuple[IndexLike]): One or more index-like objects to stack.
        **clean_index_kwargs: Keyword arguments for `clean_index`.

    Returns:
        Index: MultiIndex constructed by stacking the levels of the provided indexes.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    levels = []
    for i in range(len(indexes)):
        index = indexes[i]
        if not isinstance(index, pd.MultiIndex):
            levels.append(to_any_index(index))
        else:
            for j in range(index.nlevels):
                levels.append(index.get_level_values(j))

    max_len = max(map(len, levels))
    for i in range(len(levels)):
        if len(levels[i]) < max_len:
            if len(levels[i]) != 1:
                raise ValueError(
                    f"Index at level {i} could not be broadcast to shape ({max_len},) "
                )
            levels[i] = repeat_index(levels[i], max_len, ignore_ranges=False)
    new_index = pd.MultiIndex.from_arrays(levels)
    return clean_index(new_index, **clean_index_kwargs)


def combine_indexes(*indexes: tp.MaybeTuple[tp.IndexLike], **kwargs) -> tp.Index:
    """Combine indexes using a Cartesian product.

    Args:
        *indexes (MaybeTuple[IndexLike]): One or more index-like objects to combine.
        **kwargs: Keyword arguments for `stack_indexes`.

    Returns:
        Index: Combined index.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    new_index = to_any_index(indexes[0])
    for i in range(1, len(indexes)):
        index1, index2 = new_index, to_any_index(indexes[i])
        new_index1 = repeat_index(index1, len(index2), ignore_ranges=False)
        new_index2 = tile_index(index2, len(index1), ignore_ranges=False)
        new_index = stack_indexes([new_index1, new_index2], **kwargs)
    return new_index


def combine_index_with_keys(
    index: tp.IndexLike, keys: tp.IndexLike, lens: tp.Sequence[int], **kwargs
) -> tp.Index:
    """Build a composite index by combining index segments with repeated keys.

    Args:
        index (IndexLike): Original index to segment.
        keys (IndexLike): Keys to repeat for each index segment.
        lens (Sequence[int]): Sequence of lengths for each corresponding segment.
        **kwargs: Keyword arguments for `stack_indexes`.

    Returns:
        Index: Composite index combining the repeated keys and segments.
    """
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if not isinstance(keys, pd.Index):
        keys = pd.Index(keys)
    new_index = None
    new_keys = None
    start_idx = 0
    for i in range(len(keys)):
        _index = index[start_idx : start_idx + lens[i]]
        if new_index is None:
            new_index = _index
        else:
            new_index = new_index.append(_index)
        start_idx += lens[i]
        new_key = keys[[i]].repeat(lens[i])
        if new_keys is None:
            new_keys = new_key
        else:
            new_keys = new_keys.append(new_key)
    return stack_indexes([new_keys, new_index], **kwargs)


def concat_indexes(
    *indexes: tp.MaybeSequence[tp.IndexLike],
    index_concat_method: tp.MaybeTuple[tp.Union[str, tp.Callable]] = "append",
    keys: tp.Optional[tp.IndexLike] = None,
    clean_index_kwargs: tp.KwargsLike = None,
    verify_integrity: bool = True,
    axis: int = 1,
) -> tp.Index:
    """Concatenate multiple indexes.

    Supported concatenation methods include:

    * 'append': Append one index to another.
    * 'union': Build the union of indexes.
    * 'pd_concat': Convert indexes to Pandas Series or DataFrames and concatenate them using `pd.concat`.
    * 'factorize': Factorize the concatenated index.
    * 'factorize_each': Factorize each index and combine them with unique numbering.
    * 'reset': Reset the concatenated index without applying `keys`.
    * Callable: Custom callable that takes a list of indexes and returns the concatenated index.

    If `index_concat_method` is provided as a tuple, the second option is applied if the first fails.

    Args:
        *indexes (MaybeSequence[IndexLike]): Indexes to concatenate.
        index_concat_method (MaybeTuple[Union[str, Callable]]): Method used for concatenating indexes.
        keys (Optional[IndexLike]): Index to add an additional level on top of the concatenated indexes.
        clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        verify_integrity (bool): Flag to verify the integrity of the concatenated index.
        axis (int): Axis corresponding to the indexes (0 for rows, 1 for columns, or other for groups).

    Returns:
        Index: Concatenated index.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)
    if clean_index_kwargs is None:
        clean_index_kwargs = {}
    if axis == 0:
        factorized_name = "row_idx"
    elif axis == 1:
        factorized_name = "col_idx"
    else:
        factorized_name = "group_idx"

    if keys is None:
        all_ranges = True
        for index in indexes:
            if not checks.is_default_index(index):
                all_ranges = False
                break
        if all_ranges:
            return pd.RangeIndex(stop=sum(map(len, indexes)))
    if isinstance(index_concat_method, tuple):
        try:
            return concat_indexes(
                *indexes,
                index_concat_method=index_concat_method[0],
                keys=keys,
                clean_index_kwargs=clean_index_kwargs,
                verify_integrity=verify_integrity,
                axis=axis,
            )
        except Exception:
            return concat_indexes(
                *indexes,
                index_concat_method=index_concat_method[1],
                keys=keys,
                clean_index_kwargs=clean_index_kwargs,
                verify_integrity=verify_integrity,
                axis=axis,
            )
    if not isinstance(index_concat_method, str):
        new_index = index_concat_method(indexes)
    elif index_concat_method.lower() == "append":
        new_index = None
        for index in indexes:
            if new_index is None:
                new_index = index
            else:
                new_index = new_index.append(index)
    elif index_concat_method.lower() == "union":
        if keys is not None:
            raise ValueError("Cannot apply keys after concatenating indexes through union")
        new_index = None
        for index in indexes:
            if new_index is None:
                new_index = index
            else:
                new_index = new_index.union(index)
    elif index_concat_method.lower() == "pd_concat":
        new_index = None
        for index in indexes:
            if isinstance(index, pd.MultiIndex):
                index = index.to_frame().reset_index(drop=True)
            else:
                index = index.to_series().reset_index(drop=True)
            if new_index is None:
                new_index = index
            else:
                if isinstance(new_index, pd.DataFrame):
                    if isinstance(index, pd.Series):
                        index = index.to_frame()
                elif isinstance(index, pd.Series):
                    if isinstance(new_index, pd.DataFrame):
                        new_index = new_index.to_frame()
                new_index = pd.concat((new_index, index), ignore_index=True)
        if isinstance(new_index, pd.Series):
            new_index = pd.Index(new_index)
        else:
            new_index = pd.MultiIndex.from_frame(new_index)
    elif index_concat_method.lower() == "factorize":
        new_index = concat_indexes(
            *indexes,
            index_concat_method="append",
            clean_index_kwargs=clean_index_kwargs,
            verify_integrity=False,
            axis=axis,
        )
        new_index = pd.Index(pd.factorize(new_index)[0], name=factorized_name)
    elif index_concat_method.lower() == "factorize_each":
        new_index = None
        for index in indexes:
            index = pd.Index(pd.factorize(index)[0], name=factorized_name)
            if new_index is None:
                new_index = index
                next_min = index.max() + 1
            else:
                new_index = new_index.append(index + next_min)
                next_min = index.max() + 1 + next_min
    elif index_concat_method.lower() == "reset":
        return pd.RangeIndex(stop=sum(map(len, indexes)))
    else:
        if axis == 0:
            raise ValueError(f"Invalid index concatenation method: '{index_concat_method}'")
        elif axis == 1:
            raise ValueError(f"Invalid column concatenation method: '{index_concat_method}'")
        else:
            raise ValueError(f"Invalid group concatenation method: '{index_concat_method}'")
    if keys is not None:
        if isinstance(keys[0], pd.Index):
            keys = concat_indexes(
                *keys,
                index_concat_method="append",
                clean_index_kwargs=clean_index_kwargs,
                verify_integrity=False,
                axis=axis,
            )
            new_index = stack_indexes((keys, new_index), **clean_index_kwargs)
            keys = None
        elif not isinstance(keys, pd.Index):
            keys = pd.Index(keys)
    if keys is not None:
        top_index = None
        for i, index in enumerate(indexes):
            repeated_index = repeat_index(keys[[i]], len(index))
            if top_index is None:
                top_index = repeated_index
            else:
                top_index = top_index.append(repeated_index)
        new_index = stack_indexes((top_index, new_index), **clean_index_kwargs)
    if verify_integrity:
        if keys is None:
            if axis == 0:
                if not new_index.is_monotonic_increasing:
                    raise ValueError("Concatenated index is not monotonically increasing")
                if "mixed" in new_index.inferred_type:
                    raise ValueError("Concatenated index is mixed")
                if new_index.has_duplicates:
                    raise ValueError("Concatenated index contains duplicates")
            if axis == 1:
                if new_index.has_duplicates:
                    raise ValueError("Concatenated columns contain duplicates")
            if axis == 2:
                if new_index.has_duplicates:
                    len_sum = 0
                    for index in indexes:
                        if len_sum > 0:
                            prev_index = new_index[:len_sum]
                            this_index = new_index[len_sum : len_sum + len(index)]
                            if len(prev_index.intersection(this_index)) > 0:
                                raise ValueError("Concatenated groups contain duplicates")
                        len_sum += len(index)
    return new_index


def drop_levels(
    index: tp.Index,
    levels: tp.Union[ExceptLevel, tp.MaybeLevelSequence],
    strict: bool = True,
) -> tp.Index:
    """Drop specified levels from the given index by name or position.

    If `levels` is provided as an instance of `ExceptLevel`, drop all levels except those specified.

    Args:
        index (Index): MultiIndex from which levels will be dropped.
        levels (Union[ExceptLevel, MaybeLevelSequence]): Level names or positions to select,
            or an `ExceptLevel` indicating the levels to exclude.
        strict (bool): Whether to raise exceptions for invalid level specifications.

    Returns:
        Index: New index with the specified levels removed.
    """
    if not isinstance(index, pd.MultiIndex):
        if strict:
            raise TypeError("Index must be a MultiIndex")
        return index
    if isinstance(levels, ExceptLevel):
        levels = levels.value
        except_mode = True
    else:
        except_mode = False
    levels_to_drop = set()
    if isinstance(levels, str) or not checks.is_sequence(levels):
        levels = [levels]

    for level in levels:
        if level in index.names:
            for level_pos in [i for i, x in enumerate(index.names) if x == level]:
                levels_to_drop.add(level_pos)
        elif checks.is_int(level):
            if level < 0:
                new_level = index.nlevels + level
                if new_level < 0:
                    raise KeyError(f"Level at position {level} not found")
                level = new_level
            if 0 <= level < index.nlevels:
                levels_to_drop.add(level)
            else:
                raise KeyError(f"Level at position {level} not found")
        elif strict:
            raise KeyError(f"Level '{level}' not found")
    if except_mode:
        levels_to_drop = set(range(index.nlevels)).difference(levels_to_drop)
    if len(levels_to_drop) == 0:
        if strict:
            raise ValueError("No levels to drop")
        return index
    if len(levels_to_drop) >= index.nlevels:
        if strict:
            raise ValueError(
                f"Cannot remove {len(levels_to_drop)} levels from an index with {index.nlevels} levels: "
                "at least one level must be left"
            )
        return index
    return index.droplevel(list(levels_to_drop))


def rename_levels(
    index: tp.Index, mapper: tp.MaybeMappingSequence[tp.Level], strict: bool = True
) -> tp.Index:
    """Rename levels in the index using a mapping.

    The mapper can be a single value, a sequence of values, or a dictionary mapping old level names to new ones.

    Args:
        index (Index): Index whose levels will be renamed.
        mapper (MaybeMappingSequence[Level]): New name, sequence of names, or mapping for the levels.
        strict (bool): Whether to raise exceptions for invalid level specifications.

    Returns:
        Index: Index with renamed levels.
    """
    if isinstance(index, pd.MultiIndex):
        nlevels = index.nlevels
        if isinstance(mapper, (int, str)):
            mapper = dict(zip(index.names, [mapper]))
        elif checks.is_complex_sequence(mapper):
            mapper = dict(zip(index.names, mapper))
    else:
        nlevels = 1
        if isinstance(mapper, (int, str)):
            mapper = dict(zip([index.name], [mapper]))
        elif checks.is_complex_sequence(mapper):
            mapper = dict(zip([index.name], mapper))

    for k, v in mapper.items():
        if k in index.names:
            if isinstance(index, pd.MultiIndex):
                index = index.rename(v, level=k)
            else:
                index = index.rename(v)
        elif checks.is_int(k):
            if k < 0:
                new_k = nlevels + k
                if new_k < 0:
                    raise KeyError(f"Level at position {k} not found")
                k = new_k
            if 0 <= k < nlevels:
                if isinstance(index, pd.MultiIndex):
                    index = index.rename(v, level=k)
                else:
                    index = index.rename(v)
            else:
                raise KeyError(f"Level at position {k} not found")
        elif strict:
            raise KeyError(f"Level '{k}' not found")
    return index


def select_levels(
    index: tp.Index,
    levels: tp.Union[ExceptLevel, tp.MaybeLevelSequence],
    strict: bool = True,
) -> tp.Index:
    """Select specified levels from the given index.

    When `levels` is provided as an instance of `ExceptLevel`, select all levels except those specified.
    If the input index is not a MultiIndex, it is converted into one for consistent processing.

    Args:
        index (Index): Index from which levels are selected.
        levels (Union[ExceptLevel, MaybeLevelSequence]): Level names or positions to select,
            or an `ExceptLevel` indicating the levels to exclude.
        strict (bool): Whether to raise exceptions for invalid level specifications.

    Returns:
        Index: New index containing the selected level(s).
    """
    was_multiindex = True
    if not isinstance(index, pd.MultiIndex):
        was_multiindex = False
        index = pd.MultiIndex.from_arrays([index])
    if isinstance(levels, ExceptLevel):
        levels = levels.value
        except_mode = True
    else:
        except_mode = False
    levels_to_select = list()
    if isinstance(levels, str) or not checks.is_sequence(levels):
        levels = [levels]
        single_mode = True
    else:
        single_mode = False

    for level in levels:
        if level in index.names:
            for level_pos in [i for i, x in enumerate(index.names) if x == level]:
                if level_pos not in levels_to_select:
                    levels_to_select.append(level_pos)
        elif checks.is_int(level):
            if level < 0:
                new_level = index.nlevels + level
                if new_level < 0:
                    raise KeyError(f"Level at position {level} not found")
                level = new_level
            if 0 <= level < index.nlevels:
                if level not in levels_to_select:
                    levels_to_select.append(level)
            else:
                raise KeyError(f"Level at position {level} not found")
        elif strict:
            raise KeyError(f"Level '{level}' not found")
    if except_mode:
        levels_to_select = list(set(range(index.nlevels)).difference(levels_to_select))
    if len(levels_to_select) == 0:
        if strict:
            raise ValueError("No levels to select")
        if not was_multiindex:
            return index.get_level_values(0)
        return index
    if len(levels_to_select) == 1 and single_mode:
        return index.get_level_values(levels_to_select[0])
    levels = [index.get_level_values(level) for level in levels_to_select]
    return pd.MultiIndex.from_arrays(levels)


def drop_redundant_levels(index: tp.Index) -> tp.Index:
    """Drop redundant levels from a MultiIndex.

    Removes levels that contain a single unnamed value or that represent a default integer range.

    Args:
        index (Index): MultiIndex from which redundant levels will be removed.

    Returns:
        Index: New index with redundant levels dropped.
    """
    if not isinstance(index, pd.MultiIndex):
        return index

    levels_to_drop = []
    for i in range(index.nlevels):
        if len(index.levels[i]) == 1 and index.levels[i].name is None or checks.is_default_index(index.get_level_values(i)):
            levels_to_drop.append(i)
    if len(levels_to_drop) < index.nlevels:
        return index.droplevel(levels_to_drop)
    return index


def drop_duplicate_levels(index: tp.Index, keep: tp.Optional[str] = None) -> tp.Index:
    """Drop duplicate levels from the index that share the same name and values.

    If level names are identical, the duplicate level is removed based on the specified retention strategy.

    Args:
        index (Index): Index from which duplicate levels are removed.
        keep (Optional[str]): Indicates which duplicate to retain; valid options are "first" or "last".

    Returns:
        Index: Index with duplicate levels dropped.

    !!! info
        For default settings, see `vectorbtpro._settings.broadcasting`.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if keep is None:
        keep = broadcasting_cfg["keep"]
    if not isinstance(index, pd.MultiIndex):
        return index
    checks.assert_in(keep.lower(), ["first", "last"])

    levels_to_drop = set()
    level_values = [index.get_level_values(i) for i in range(index.nlevels)]
    for i in range(index.nlevels):
        level1 = level_values[i]
        for j in range(i + 1, index.nlevels):
            level2 = level_values[j]
            if level1.name is None or level2.name is None or level1.name == level2.name:
                if checks.is_index_equal(level1, level2, check_names=False):
                    if level1.name is None and level2.name is not None:
                        levels_to_drop.add(i)
                    elif level1.name is not None and level2.name is None:
                        levels_to_drop.add(j)
                    else:
                        if keep.lower() == "first":
                            levels_to_drop.add(j)
                        else:
                            levels_to_drop.add(i)
    return index.droplevel(list(levels_to_drop))


@register_jitted(cache=True)
def align_arr_indices_nb(a: tp.Array1d, b: tp.Array1d) -> tp.Array1d:
    """Return indices to align array `a` with array `b`.

    Iterates over `b` to find the first matching index in `a` for each element,
    constructing an array of alignment indices.

    Args:
        a (Array1d): Source array for alignment.
        b (Array1d): Target array used to determine the alignment order.

    Returns:
        Array1d: Array of indices representing the alignment of `a` to `b`.
    """
    idxs = np.empty(b.shape[0], dtype=int_)
    g = 0
    for i in range(b.shape[0]):
        for j in range(a.shape[0]):
            if b[i] == a[j]:
                idxs[g] = j
                g += 1
                break
    return idxs


def align_index_to(
    index1: tp.Index, index2: tp.Index, jitted: tp.JittedOption = None
) -> tp.IndexSlice:
    """Align `index1` to the shape of `index2` based on common index levels.

    Args:
        index1 (Index): Index to be aligned.
        index2 (Index): Reference index providing the desired shape.
        jitted (JittedOption): Option to control JIT compilation.

            See `vectorbtpro.utils.jitting.resolve_jitted_option`.

    Returns:
        IndexSlice: Index slice that aligns `index1` with `index2`.
    """
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    if checks.is_index_equal(index1, index2):
        return pd.IndexSlice[:]
    if len(index1) > len(index2):
        raise ValueError("Longer index cannot be aligned to shorter index")

    mapper = {}
    for i in range(index1.nlevels):
        name1 = index1.names[i]
        for j in range(index2.nlevels):
            name2 = index2.names[j]
            if name1 is None or name2 is None or name1 == name2:
                if set(index2.levels[j]).issubset(set(index1.levels[i])):
                    if i in mapper:
                        raise ValueError(
                            f"There are multiple candidate levels with name {name1} in second index"
                        )
                    mapper[i] = j
                    continue
                if name1 == name2 and name1 is not None:
                    raise ValueError(
                        f"Level {name1} in second index contains values not in first index"
                    )
    if len(mapper) == 0:
        if len(index1) == len(index2):
            return pd.IndexSlice[:]
        raise ValueError("Cannot find common levels to align indexes")

    factorized = []
    for k, v in mapper.items():
        factorized.append(
            pd.factorize(
                pd.concat(
                    (
                        index1.get_level_values(k).to_series(),
                        index2.get_level_values(v).to_series(),
                    )
                )
            )[0],
        )
    stacked = np.transpose(np.stack(factorized))
    indices1 = stacked[: len(index1)]
    indices2 = stacked[len(index1) :]

    if len(indices1) < len(indices2):
        if len(np.unique(indices1, axis=0)) != len(indices1):
            raise ValueError("Cannot align indexes")
        if len(index2) % len(index1) == 0:
            tile_times = len(index2) // len(index1)
            index1_tiled = np.tile(indices1, (tile_times, 1))
            if np.array_equal(index1_tiled, indices2):
                return pd.IndexSlice[np.tile(np.arange(len(index1)), tile_times)]

    unique_indices = np.unique(stacked, axis=0, return_inverse=True)[1]
    unique1 = unique_indices[: len(index1)]
    unique2 = unique_indices[len(index1) :]
    if len(indices1) == len(indices2):
        if np.array_equal(unique1, unique2):
            return pd.IndexSlice[:]
    func = jit_reg.resolve_option(align_arr_indices_nb, jitted)
    return pd.IndexSlice[func(unique1, unique2)]


def align_indexes(
    *indexes: tp.MaybeSequence[tp.Index],
    return_new_index: bool = False,
    **kwargs,
) -> tp.Union[tp.Tuple[tp.IndexSlice, ...], tp.Tuple[tp.Tuple[tp.IndexSlice, ...], tp.Index]]:
    """Align multiple indexes based on common levels using `align_index_to`.

    Args:
        indexes (MaybeSequence[Index]): Indexes to align.
        return_new_index (bool): If True, return a new stacked index along with the index slices.
        **kwargs: Keyword arguments for `align_index_to`.

    Returns:
        Union[Tuple[IndexSlice, ...], Tuple[Tuple[IndexSlice, ...], Index]]:
            * If `return_new_index` is False, a tuple of index slices is returned.
            * If `return_new_index` is True, a tuple is returned where the first element
                is a tuple of index slices and the second element is the new stacked index.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)

    index_items = sorted([(i, indexes[i]) for i in range(len(indexes))], key=lambda x: len(x[1]))
    index_slices = []
    for i in range(len(index_items)):
        index_slice = align_index_to(index_items[i][1], index_items[-1][1], **kwargs)
        index_slices.append((index_items[i][0], index_slice))
    index_slices = list(map(lambda x: x[1], sorted(index_slices, key=lambda x: x[0])))
    if return_new_index:
        new_index = stack_indexes(
            *[indexes[i][index_slices[i]] for i in range(len(indexes))],
            drop_duplicates=True,
        )
        return tuple(index_slices), new_index
    return tuple(index_slices)


@register_jitted(cache=True)
def block_index_product_nb(
    block_group_map1: tp.GroupMap,
    block_group_map2: tp.GroupMap,
    factorized1: tp.Array1d,
    factorized2: tp.Array1d,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Compute block-wise Cartesian product indices from two factorized indexes.

    Args:
        block_group_map1 (GroupMap): Tuple with group indices and group lengths for the first factorized index.
        block_group_map2 (GroupMap): Tuple with group indices and group lengths for the second factorized index.
        factorized1 (Array1d): Factorized values of the first index.
        factorized2 (Array1d): Factorized values of the second index.

    Returns:
        Tuple[Array1d, Array1d]: Two arrays containing indices for matching blocks of the two factorized indexes.
    """
    group_idxs1, group_lens1 = block_group_map1
    group_idxs2, group_lens2 = block_group_map2
    group_start_idxs1 = np.cumsum(group_lens1) - group_lens1
    group_start_idxs2 = np.cumsum(group_lens2) - group_lens2

    matched1 = np.empty(len(factorized1), dtype=np.bool_)
    matched2 = np.empty(len(factorized2), dtype=np.bool_)
    indices1 = np.empty(len(factorized1) * len(factorized2), dtype=int_)
    indices2 = np.empty(len(factorized1) * len(factorized2), dtype=int_)
    k1 = 0
    k2 = 0

    for g1 in range(len(group_lens1)):
        group_len1 = group_lens1[g1]
        group_start1 = group_start_idxs1[g1]

        for g2 in range(len(group_lens2)):
            group_len2 = group_lens2[g2]
            group_start2 = group_start_idxs2[g2]

            for c1 in range(group_len1):
                i = group_idxs1[group_start1 + c1]

                for c2 in range(group_len2):
                    j = group_idxs2[group_start2 + c2]

                    if factorized1[i] == factorized2[j]:
                        matched1[i] = True
                        matched2[j] = True
                        indices1[k1] = i
                        indices2[k2] = j
                        k1 += 1
                        k2 += 1

    if not np.all(matched1) or not np.all(matched2):
        raise ValueError("Cannot match some block level values")
    return indices1[:k1], indices2[:k2]


def cross_index_with(
    index1: tp.Index,
    index2: tp.Index,
    return_new_index: bool = False,
) -> tp.Union[
    tp.Tuple[tp.IndexSlice, tp.IndexSlice],
    tp.Tuple[tp.Tuple[tp.IndexSlice, tp.IndexSlice], tp.Index],
]:
    """Build a Cartesian product of two indexes, accounting for shared levels.

    Args:
        index1 (Index): First index to cross.
        index2 (Index): Second index to cross.
        return_new_index (bool): If True, also return a combined index formed by stacking the two indexes.

    Returns:
        Union[Tuple[IndexSlice, IndexSlice], Tuple[Tuple[IndexSlice, IndexSlice], Index]]:
            * If `return_new_index` is False, a tuple of index slices for alignment is returned.
            * If `return_new_index` is True, a tuple containing the index slices and the
                new combined index is returned.
    """
    from vectorbtpro.base.grouping.nb import get_group_map_nb

    index1_default = checks.is_default_index(index1, check_names=True)
    index2_default = checks.is_default_index(index2, check_names=True)
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    if not index1_default and not index2_default and checks.is_index_equal(index1, index2):
        if return_new_index:
            new_index = stack_indexes(index1, index2, drop_duplicates=True)
            return (pd.IndexSlice[:], pd.IndexSlice[:]), new_index
        return pd.IndexSlice[:], pd.IndexSlice[:]

    levels1 = []
    levels2 = []
    for i in range(index1.nlevels):
        if checks.is_default_index(index1.get_level_values(i), check_names=True):
            continue
        for j in range(index2.nlevels):
            if checks.is_default_index(index2.get_level_values(j), check_names=True):
                continue
            name1 = index1.names[i]
            name2 = index2.names[j]
            if name1 == name2:
                if set(index2.levels[j]) == set(index1.levels[i]):
                    if i in levels1 or j in levels2:
                        raise ValueError(
                            f"There are multiple candidate block levels with name {name1}"
                        )
                    levels1.append(i)
                    levels2.append(j)
                    continue
                if name1 is not None:
                    raise ValueError(
                        f"Candidate block level {name1} in both indexes has different values"
                    )

    if len(levels1) == 0:
        # Regular index product
        indices1 = np.repeat(np.arange(len(index1)), len(index2))
        indices2 = np.tile(np.arange(len(index2)), len(index1))
    else:
        # Block index product
        index_levels1 = select_levels(index1, levels1)
        index_levels2 = select_levels(index2, levels2)

        block_levels1 = list(set(range(index1.nlevels)).difference(levels1))
        block_levels2 = list(set(range(index2.nlevels)).difference(levels2))
        if len(block_levels1) > 0:
            index_block_levels1 = select_levels(index1, block_levels1)
        else:
            index_block_levels1 = pd.Index(np.full(len(index1), 0))
        if len(block_levels2) > 0:
            index_block_levels2 = select_levels(index2, block_levels2)
        else:
            index_block_levels2 = pd.Index(np.full(len(index2), 0))

        factorized = pd.factorize(
            pd.concat((index_levels1.to_series(), index_levels2.to_series()))
        )[0]
        factorized1 = factorized[: len(index_levels1)]
        factorized2 = factorized[len(index_levels1) :]

        block_factorized1, block_unique1 = pd.factorize(index_block_levels1)
        block_factorized2, block_unique2 = pd.factorize(index_block_levels2)
        block_group_map1 = get_group_map_nb(block_factorized1, len(block_unique1))
        block_group_map2 = get_group_map_nb(block_factorized2, len(block_unique2))

        indices1, indices2 = block_index_product_nb(
            block_group_map1,
            block_group_map2,
            factorized1,
            factorized2,
        )
    if return_new_index:
        new_index = stack_indexes(index1[indices1], index2[indices2], drop_duplicates=True)
        return (pd.IndexSlice[indices1], pd.IndexSlice[indices2]), new_index
    return pd.IndexSlice[indices1], pd.IndexSlice[indices2]


def cross_indexes(
    *indexes: tp.MaybeSequence[tp.Index],
    return_new_index: bool = False,
) -> tp.Union[tp.Tuple[tp.IndexSlice, ...], tp.Tuple[tp.Tuple[tp.IndexSlice, ...], tp.Index]]:
    """Compute the Cartesian product over multiple indexes by iteratively applying `cross_index_with`.

    Args:
        *indexes (MaybeSequence[Index]): Indexes to cross.
        return_new_index (bool): If True, also return a combined index from the cross product.

    Returns:
        Union[Tuple[IndexSlice, ...], Tuple[Tuple[IndexSlice, ...], Index]]:
            * If `return_new_index` is False, a tuple of index slices representing
                the aligned indexes is returned.
            * If `return_new_index` is True, a tuple containing the index slices and
                the new combined index is returned.
    """
    if len(indexes) == 1:
        indexes = indexes[0]
    indexes = list(indexes)
    if len(indexes) == 2:
        return cross_index_with(indexes[0], indexes[1], return_new_index=return_new_index)

    index = None
    index_slices = []
    for i in range(len(indexes) - 2, -1, -1):
        index1 = indexes[i]
        if i == len(indexes) - 2:
            index2 = indexes[i + 1]
        else:
            index2 = index
        (index_slice1, index_slice2), index = cross_index_with(
            index1, index2, return_new_index=True
        )
        if i == len(indexes) - 2:
            index_slices.append(index_slice2)
        else:
            for j in range(len(index_slices)):
                if isinstance(index_slices[j], slice):
                    index_slices[j] = np.arange(len(index2))[index_slices[j]]
                index_slices[j] = index_slices[j][index_slice2]
        index_slices.append(index_slice1)

    if return_new_index:
        return tuple(index_slices[::-1]), index
    return tuple(index_slices[::-1])


OptionalLevelSequence = tp.Optional[tp.Sequence[tp.Union[None, tp.Level]]]


def pick_levels(
    index: tp.Index,
    required_levels: OptionalLevelSequence = None,
    optional_levels: OptionalLevelSequence = None,
) -> tp.Tuple[tp.List[int], tp.List[int]]:
    """Select the indices for required and optional levels from a MultiIndex.

    An exception is raised if the index's number of levels does not match the expected configuration.

    Args:
        index (Index): MultiIndex from which to select levels.
        required_levels (Optional[Sequence[Union[None, Level]]]): Sequence specifying
            required levels by name or position.
        optional_levels (Optional[Sequence[Union[None, Level]]]): Sequence specifying
            optional levels by name or position.

    Returns:
        Tuple[List[int], List[int]]: Tuple containing:

            * List of required level indices.
            * List of optional level indices.
    """
    if required_levels is None:
        required_levels = []
    if optional_levels is None:
        optional_levels = []
    checks.assert_instance_of(index, pd.MultiIndex)

    n_opt_set = len(list(filter(lambda x: x is not None, optional_levels)))
    n_req_set = len(list(filter(lambda x: x is not None, required_levels)))
    n_levels_left = index.nlevels - n_opt_set
    if n_req_set < len(required_levels):
        if n_levels_left != len(required_levels):
            n_expected = len(required_levels) + n_opt_set
            raise ValueError(f"Expected {n_expected} levels, found {index.nlevels}")

    levels_left = list(range(index.nlevels))

    _optional_levels = []
    for level in optional_levels:
        level_pos = None
        if level is not None:
            checks.assert_instance_of(level, (int, str))
            if isinstance(level, str):
                level_pos = index.names.index(level)
            else:
                level_pos = level
            if level_pos < 0:
                level_pos = index.nlevels + level_pos
            levels_left.remove(level_pos)
        _optional_levels.append(level_pos)

    _required_levels = []
    for level in required_levels:
        level_pos = None
        if level is not None:
            checks.assert_instance_of(level, (int, str))
            if isinstance(level, str):
                level_pos = index.names.index(level)
            else:
                level_pos = level
            if level_pos < 0:
                level_pos = index.nlevels + level_pos
            levels_left.remove(level_pos)
        _required_levels.append(level_pos)
    for i, level in enumerate(_required_levels):
        if level is None:
            _required_levels[i] = levels_left.pop(0)

    return _required_levels, _optional_levels


def find_first_occurrence(index_value: tp.Any, index: tp.Index) -> int:
    """Return the index position of the first occurrence of a value in an index.

    Args:
        index_value (Any): Value to locate in the index.
        index (Index): Index to search.

    Returns:
        int: Position of the first occurrence of the specified value.
    """
    loc = index.get_loc(index_value)
    if isinstance(loc, slice):
        return loc.start
    elif isinstance(loc, list):
        return loc[0]
    elif isinstance(loc, np.ndarray):
        return np.flatnonzero(loc)[0]
    return loc


IndexApplierT = tp.TypeVar("IndexApplierT", bound="IndexApplier")


class IndexApplier(Base):
    """Abstract class for applying transformations to an instance's index."""

    def apply_to_index(
        self: IndexApplierT, apply_func: tp.Callable, *args, **kwargs
    ) -> IndexApplierT:
        """Apply the specified function to the instance's index and return a new instance.

        Args:
            apply_func (Callable): Callable to apply to the instance's index.
            *args: Positional arguments for `apply_func`.
            **kwargs: Keyword arguments for `apply_func`.

        Returns:
            IndexApplier: New instance with the updated index.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def add_levels(
        self: IndexApplierT,
        *indexes: tp.Index,
        on_top: bool = True,
        drop_duplicates: tp.Optional[bool] = None,
        keep: tp.Optional[str] = None,
        drop_redundant: tp.Optional[bool] = None,
        **kwargs,
    ) -> IndexApplierT:
        """Append or prepend additional levels to the index using `stack_indexes`.

        Args:
            *indexes (Index): One or more indexes to add as new levels.
            on_top (bool): If True, add the new levels before the existing index; if False, add them after.
            drop_duplicates (Optional[bool]): If True, remove duplicate levels.
            keep (Optional[str]): Indicates which duplicate to retain; valid options are "first" or "last".
            drop_redundant (Optional[bool]): If True, remove redundant levels.
            **kwargs: Keyword arguments for `IndexApplier.apply_to_index`.

        Returns:
            IndexApplier: New instance with the modified index.
        """

        def _apply_func(index):
            if on_top:
                return stack_indexes(
                    [*indexes, index],
                    drop_duplicates=drop_duplicates,
                    keep=keep,
                    drop_redundant=drop_redundant,
                )
            return stack_indexes(
                [index, *indexes],
                drop_duplicates=drop_duplicates,
                keep=keep,
                drop_redundant=drop_redundant,
            )

        return self.apply_to_index(_apply_func, **kwargs)

    def drop_levels(
        self: IndexApplierT,
        levels: tp.Union[ExceptLevel, tp.MaybeLevelSequence],
        strict: bool = True,
        **kwargs,
    ) -> IndexApplierT:
        """Drop specified levels from the index using `drop_levels`.

        Args:
            levels (Union[ExceptLevel, MaybeLevelSequence]): Level names or positions to select,
                or an `ExceptLevel` indicating the levels to exclude.
            strict (bool): Whether to raise exceptions for invalid level specifications.
            **kwargs: Keyword arguments for `IndexApplier.apply_to_index`.

        Returns:
            IndexApplier: New instance with the specified levels removed from the index.
        """

        def _apply_func(index):
            return drop_levels(index, levels, strict=strict)

        return self.apply_to_index(_apply_func, **kwargs)

    def rename_levels(
        self: IndexApplierT,
        mapper: tp.MaybeMappingSequence[tp.Level],
        strict: bool = True,
        **kwargs,
    ) -> IndexApplierT:
        """Rename levels in the index using `rename_levels`.

        Args:
            mapper (MaybeMappingSequence[Level]): New name, sequence of names, or mapping for the levels.
            strict (bool): Whether to raise exceptions for invalid level specifications.
            **kwargs: Keyword arguments for `IndexApplier.apply_to_index`.

        Returns:
            IndexApplier: New instance with the index levels renamed.
        """

        def _apply_func(index):
            return rename_levels(index, mapper, strict=strict)

        return self.apply_to_index(_apply_func, **kwargs)

    def select_levels(
        self: IndexApplierT,
        level_names: tp.Union[ExceptLevel, tp.MaybeLevelSequence],
        strict: bool = True,
        **kwargs,
    ) -> IndexApplierT:
        """Select specific levels from the index using `select_levels`.

        Args:
            level_names (Union[ExceptLevel, MaybeLevelSequence]): Level or levels to select from the index.
            strict (bool): Whether to raise exceptions for invalid level specifications.
            **kwargs: Keyword arguments for `IndexApplier.apply_to_index`.

        Returns:
            IndexApplier: New instance with a subset of the index levels.
        """

        def _apply_func(index):
            return select_levels(index, level_names, strict=strict)

        return self.apply_to_index(_apply_func, **kwargs)

    def drop_redundant_levels(self: IndexApplierT, **kwargs) -> IndexApplierT:
        """Remove redundant levels from the index using `drop_redundant_levels`.

        Args:
            **kwargs: Keyword arguments for `IndexApplier.apply_to_index`.

        Returns:
            IndexApplier: New instance with redundant levels removed.
        """

        def _apply_func(index):
            return drop_redundant_levels(index)

        return self.apply_to_index(_apply_func, **kwargs)

    def drop_duplicate_levels(
        self: IndexApplierT, keep: tp.Optional[str] = None, **kwargs
    ) -> IndexApplierT:
        """Remove duplicate levels from the index using `drop_duplicate_levels`.

        Args:
            keep (Optional[str]): Indicates which duplicate to retain; valid options are "first" or "last".
            **kwargs: Keyword arguments for `IndexApplier.apply_to_index`.

        Returns:
            IndexApplier: New instance with duplicate levels removed.
        """

        def _apply_func(index):
            return drop_duplicate_levels(index, keep=keep)

        return self.apply_to_index(_apply_func, **kwargs)
