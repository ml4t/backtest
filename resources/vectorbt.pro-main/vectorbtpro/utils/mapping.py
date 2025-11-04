# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing mapping utilities."""

import re

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks

__all__ = []


def reverse_mapping(mapping: tp.Mapping) -> dict:
    """Return a mapping with keys and values swapped.

    Args:
        mapping (Mapping): Mapping to reverse.

    Returns:
        dict: Dictionary with reversed key-value pairs.
    """
    return {v: k for k, v in mapping.items()}


def to_field_mapping(mapping_like: tp.MappingLike) -> dict:
    """Return a field mapping dictionary from a mapping-like object.

    Args:
        mapping_like (MappingLike): Object convertible to a mapping.

    Returns:
        dict: Dictionary representing the field mapping.
    """
    if isinstance(mapping_like, tp.EnumMeta):
        mapping = {i.name: i.value for i in mapping_like}
    elif checks.is_namedtuple(mapping_like):
        mapping = dict(mapping_like._asdict())
    elif checks.is_record(mapping_like):
        mapping = dict(zip(mapping_like.dtype.names, mapping_like))
    elif checks.is_series(mapping_like):
        mapping = mapping_like.to_dict()
    else:
        mapping = dict(mapping_like)
    return mapping


def to_value_mapping(
    mapping_like: tp.MappingLike, reverse: bool = False, enum_unkval: tp.Any = -1
) -> dict:
    """Return a value mapping dictionary from a mapping-like object.

    If `reverse` is True, applies `reverse_mapping` to swap keys and values.

    Args:
        mapping_like (MappingLike): Object convertible to a mapping.
        reverse (bool): Whether to reverse the mapping's keys and values.
        enum_unkval (Any): Value for unknown enumeration items.

    Returns:
        dict: Dictionary representing the value mapping.
    """
    if isinstance(mapping_like, tp.EnumMeta):
        mapping = {i.value: i.name for i in mapping_like}
    elif checks.is_namedtuple(mapping_like):
        mapping = {v: k for k, v in mapping_like._asdict().items()}
        if enum_unkval not in mapping_like:
            mapping[enum_unkval] = None
    elif checks.is_record(mapping_like):
        mapping = dict(zip(mapping_like, mapping_like.dtype.names))
    elif not checks.is_mapping(mapping_like):
        if checks.is_index(mapping_like):
            mapping_like = mapping_like.to_series().reset_index(drop=True)
        if checks.is_series(mapping_like):
            mapping = mapping_like.to_dict()
        else:
            mapping = dict(enumerate(mapping_like))
    else:
        mapping = dict(mapping_like)
    if reverse:
        mapping = reverse_mapping(mapping)
    return mapping


def apply_mapping(
    obj: tp.Any,
    mapping_like: tp.Optional[tp.MappingLike] = None,
    enum_unkval: tp.Any = -1,
    reverse: bool = False,
    ignore_case: bool = True,
    ignore_underscores: bool = True,
    ignore_invalid: bool = True,
    ignore_type: tp.Optional[tp.MaybeTuple[tp.DTypeLike]] = None,
    ignore_missing: bool = False,
    na_sentinel: tp.Any = None,
) -> tp.Any:
    """Return an object with values transformed based on a mapping-like converter.

    This function applies a mapping to the provided object, which can be a scalar,
    tuple, list, set, frozenset, NumPy array, Pandas Index, Series, or DataFrame. It
    converts the mapping-like object using `to_value_mapping` and then transforms the
    input object based on defined rules for key matching and conversion.

    Args:
        obj (Any): Input object to transform.

            Can be a scalar, tuple, list, set, frozenset, NumPy array, Index, Series, or DataFrame.
        mapping_like (Optional[MappingLike]): Object convertible to a mapping.

            See `to_value_mapping`.
        enum_unkval (Any): Value for unknown enumeration items.
        reverse (bool): Whether to reverse the mapping's keys and values.

            See `to_value_mapping`.
        ignore_case (bool): Whether to ignore case when matching.
        ignore_underscores (bool): Whether to ignore underscores in string keys.
        ignore_invalid (bool): Whether to remove characters not allowed in a Python variable.
        ignore_type (Optional[MaybeTuple[DTypeLike]]): One or multiple data types to ignore.
        ignore_missing (bool): Whether to return the original value if a key is not found.
        na_sentinel (Any): Value used to represent a "not found" state.

    Returns:
        Any: Transformed object with the mapping applied.
    """
    if mapping_like is None:
        return obj

    def _key_func(x):
        if ignore_case:
            x = x.lower()
        if ignore_underscores:
            x = x.replace("_", "")
        if ignore_invalid:
            x = re.sub(r"\W+", "", x)
        return x

    if not isinstance(ignore_type, tuple):
        ignore_type = (ignore_type,)

    mapping = to_value_mapping(mapping_like, enum_unkval=enum_unkval, reverse=reverse)

    new_mapping = dict()
    for k, v in mapping.items():
        if pd.isnull(k):
            na_sentinel = v
        else:
            if isinstance(k, str):
                k = _key_func(k)
            new_mapping[k] = v

    def _compatible_types(x_type: type, item: tp.Any = None) -> bool:
        if item is not None:
            if np.dtype(x_type) == "O":
                x_type = type(item)
        for y_type in ignore_type:
            if y_type is None:
                return False
            if x_type is y_type:
                return True
            x_dtype = np.dtype(x_type)
            y_dtype = np.dtype(y_type)
            if x_dtype is y_dtype:
                return True
            if np.issubdtype(x_dtype, np.integer) and np.issubdtype(y_dtype, np.integer):
                return True
            if np.issubdtype(x_dtype, np.floating) and np.issubdtype(y_dtype, np.floating):
                return True
            if np.issubdtype(x_dtype, np.bool_) and np.issubdtype(y_dtype, np.bool_):
                return True
            if np.issubdtype(x_dtype, np.flexible) and np.issubdtype(y_dtype, np.flexible):
                return True
        return False

    def _converter(x: tp.Any) -> tp.Any:
        if pd.isnull(x):
            return na_sentinel
        if isinstance(x, str):
            x = _key_func(x)
        if ignore_missing:
            try:
                return new_mapping[x]
            except KeyError:
                return x
        return new_mapping[x]

    if isinstance(obj, (tuple, list, set, frozenset)):
        result = [
            apply_mapping(
                v,
                mapping_like=mapping_like,
                reverse=reverse,
                ignore_case=ignore_case,
                ignore_underscores=ignore_underscores,
                ignore_type=ignore_type,
                ignore_missing=ignore_missing,
                na_sentinel=na_sentinel,
            )
            for v in obj
        ]
        return type(obj)(result)
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return obj
        if ignore_type is None or not _compatible_types(obj.dtype, obj.item(0)):
            if obj.ndim == 1:
                return pd.Series(obj).map(_converter).values
            return np.vectorize(_converter)(obj)
        return obj
    if isinstance(obj, pd.Series):
        if obj.size == 0:
            return obj
        if ignore_type is None or not _compatible_types(obj.dtype, obj.iloc[0]):
            return obj.map(_converter)
        return obj
    if isinstance(obj, pd.Index):
        if obj.size == 0:
            return obj
        if ignore_type is None or not _compatible_types(obj.dtype, obj[0]):
            return obj.map(_converter)
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.size == 0:
            return obj
        series = []
        for sr_name, sr in obj.items():
            if ignore_type is None or not _compatible_types(sr.dtype, sr.iloc[0]):
                series.append(sr.map(_converter))
            else:
                series.append(sr)
        return pd.concat(series, axis=1, keys=obj.columns)
    if ignore_type is None or not _compatible_types(type(obj)):
        return _converter(obj)
    return obj
