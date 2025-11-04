# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functions for merging arrays via concatenation and stacking operations."""

from functools import partial

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexes import clean_index, concat_indexes, stack_indexes
from vectorbtpro.base.reshaping import to_1d_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.execution import NoResult, NoResultsException, filter_out_no_results
from vectorbtpro.utils.merging import MergeFunc

__all__ = [
    "concat_arrays",
    "row_stack_arrays",
    "column_stack_arrays",
    "concat_merge",
    "row_stack_merge",
    "column_stack_merge",
    "imageio_merge",
    "mixed_merge",
]

__pdoc__ = {}


def concat_arrays(*arrs: tp.MaybeSequence[tp.AnyArray]) -> tp.Array1d:
    """Concatenate provided array-like objects into a single one-dimensional array.

    Args:
        *arrs (MaybeSequence[AnyArray]): Array-like objects to concatenate.

    Returns:
        Array1d: Concatenated one-dimensional array.
    """
    if len(arrs) == 1:
        arrs = arrs[0]
    arrs = list(arrs)

    arrs = list(map(to_1d_array, arrs))
    return np.concatenate(arrs)


def row_stack_arrays(*arrs: tp.MaybeSequence[tp.AnyArray], expand_axis: int = 1) -> tp.Array2d:
    """Stack provided array-like objects (converted to 2D arrays) vertically into a single 2D array.

    Args:
        *arrs (MaybeSequence[AnyArray]): Array-like objects to stack.
        expand_axis (int): Axis along which to expand the array if necessary.

    Returns:
        Array2d: Two-dimensional array with the input arrays stacked row-wise.
    """
    if len(arrs) == 1:
        arrs = arrs[0]
    arrs = list(arrs)

    arrs = list(map(partial(to_2d_array, expand_axis=expand_axis), arrs))
    return np.concatenate(arrs, axis=0)


def column_stack_arrays(*arrs: tp.MaybeSequence[tp.AnyArray], expand_axis: int = 1) -> tp.Array2d:
    """Stack provided array-like objects (converted to 2D arrays) horizontally into a single 2D array.

    If all arrays have a single dimension or a single column, they are concatenated vertically
    and then transposed; otherwise, arrays are concatenated along columns.

    Args:
        *arrs (MaybeSequence[AnyArray]): Array-like objects to stack.
        expand_axis (int): Axis along which to expand the array if necessary.

    Returns:
        Array2d: Two-dimensional array with the input arrays stacked column-wise.
    """
    if len(arrs) == 1:
        arrs = arrs[0]
    arrs = list(arrs)

    arrs = list(map(partial(to_2d_array, expand_axis=expand_axis), arrs))
    common_shape = None
    can_concatenate = True
    for arr in arrs:
        if common_shape is None:
            common_shape = arr.shape
        if arr.shape != common_shape:
            can_concatenate = False
            continue
        if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)):
            can_concatenate = False
            continue

    if can_concatenate:
        return np.concatenate(arrs, axis=0).reshape((len(arrs), common_shape[0])).T
    return np.concatenate(arrs, axis=1)


def concat_merge(
    *objs: tp.MaybeSequence[tp.MaybeTuple[tp.Any]],
    keys: tp.Optional[tp.Index] = None,
    filter_results: bool = True,
    raise_no_results: bool = True,
    wrap: tp.Optional[bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    clean_index_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like objects by concatenation.

    Args:
        *objs (MaybeSequence[MaybeTuple[Any]]): Array-like objects to merge.

            This function supports passing a sequence of tuples, where each element
            in the tuple is merged separately.
        keys (Optional[Index]): Index or sequence of index objects to assign to the merged result.
        filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
        raise_no_results (bool): Flag indicating whether to raise a
            `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
        wrap (Optional[bool]): If True, wrap each array with a Pandas Series using `pd.concat`.

            If None, the value is inferred from the presence of `wrapper`, `keys`, or `wrap_kwargs`.
        wrapper (Optional[ArrayWrapper]): Array wrapper instance.
        wrap_kwargs (KwargsLikeSequence): Keyword arguments for wrapping the result;
            can be a dictionary or a list of dictionaries.

            See `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.
        clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        **kwargs: Keyword arguments for `pd.concat`.

    Returns:
        MaybeTuple[AnyArray]: Merged array-like object, which may be a Pandas Series,
            a NumPy array, or a tuple/namedtuple of such objects.

    !!! note
        All arrays are assumed to have the same type and dimensionality.
    """
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if len(objs) == 0:
        raise ValueError("No objects to be merged")

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            out_tuple = (
                concat_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        else:
            out_tuple = tuple(
                map(
                    lambda x: concat_merge(
                        x,
                        keys=keys,
                        wrap=wrap,
                        wrapper=wrapper,
                        wrap_kwargs=wrap_kwargs,
                        **kwargs,
                    ),
                    zip(*objs),
                )
            )
        if checks.is_namedtuple(objs[0]):
            return type(objs[0])(*out_tuple)
        return type(objs[0])(out_tuple)

    if filter_results:
        try:
            objs, keys = filter_out_no_results(objs, keys=keys)
        except NoResultsException as e:
            if raise_no_results:
                raise e
            return NoResult

    if isinstance(objs[0], Wrapping):
        raise TypeError("Concatenating Wrapping instances is not supported")

    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = (
            isinstance(objs[0], pd.Series)
            or wrapper is not None
            or keys is not None
            or len(wrap_kwargs) > 0
        )
    if not checks.is_complex_iterable(objs[0]):
        if wrap:
            if keys is not None and isinstance(keys[0], pd.Index):
                if len(keys) == 1:
                    keys = keys[0]
                else:
                    keys = concat_indexes(
                        *keys,
                        index_concat_method="append",
                        clean_index_kwargs=clean_index_kwargs,
                        verify_integrity=False,
                        axis=0,
                    )
            wrap_kwargs = merge_dicts(dict(index=keys), wrap_kwargs)
            return pd.Series(objs, **wrap_kwargs)
        return np.asarray(objs)
    if isinstance(objs[0], pd.Index):
        objs = list(map(lambda x: x.to_series(), objs))

    default_index = True
    if not isinstance(objs[0], pd.Series):
        if isinstance(objs[0], pd.DataFrame):
            raise ValueError("Use row stacking for concatenating DataFrames")
        if wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    if "force_1d" not in _wrap_kwargs:
                        _wrap_kwargs["force_1d"] = True
                    new_objs.append(wrapper.wrap_reduced(obj, **_wrap_kwargs))
                else:
                    new_objs.append(pd.Series(obj, **_wrap_kwargs))
                if default_index and not checks.is_default_index(
                    new_objs[-1].index, check_names=True
                ):
                    default_index = False
            objs = new_objs

    if not wrap:
        return concat_arrays(objs)

    if keys is not None and isinstance(keys[0], pd.Index):
        new_obj = pd.concat(objs, axis=0, **kwargs)
        if len(keys) == 1:
            keys = keys[0]
        else:
            keys = concat_indexes(
                *keys,
                index_concat_method="append",
                verify_integrity=False,
                axis=0,
            )
        if default_index:
            new_obj.index = keys
        else:
            new_obj.index = stack_indexes((keys, new_obj.index))
    else:
        new_obj = pd.concat(objs, axis=0, keys=keys, **kwargs)
    if clean_index_kwargs is None:
        clean_index_kwargs = {}
    new_obj.index = clean_index(new_obj.index, **clean_index_kwargs)
    return new_obj


def row_stack_merge(
    *objs: tp.MaybeSequence[tp.MaybeTuple[tp.Any]],
    keys: tp.Optional[tp.Index] = None,
    filter_results: bool = True,
    raise_no_results: bool = True,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    clean_index_kwargs: tp.KwargsLikeSequence = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like or `vectorbtpro.base.wrapping.Wrapping` objects via row stacking.

    Args:
        *objs (MaybeSequence[MaybeTuple[Any]]): Array-like or wrapping objects to merge.

            This function supports passing a sequence of tuples, where each element
            in the tuple is merged separately.
        keys (Optional[Index]): Keys used for concatenating arrays along the row axis.
        filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
        raise_no_results (bool): Flag indicating whether to raise a
            `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
        wrap (Union[None, str, bool]): Determines wrapping behavior for each object.

            * None: Treated as True if `wrapper`, `keys`, or `wrap_kwargs` is provided.
            * True: Wraps each array as a Pandas Series or DataFrame based on its dimensions.
            * "sr" or "series": Wraps each array as a Pandas Series.
            * "df", "frame", or "dataframe": Wraps each array as a Pandas DataFrame.

            Without wrapping, arrays will be kept as-is and merged using `row_stack_arrays`.
        wrapper (Optional[ArrayWrapper]): Array wrapper instance.
        wrap_kwargs (KwargsLikeSequence): Keyword arguments for wrapping the result;
            can be a dictionary or a list of dictionaries.

            See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
        clean_index_kwargs (KwargsLikeSequence): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        **kwargs: Keyword arguments for `pd.concat` and
            `vectorbtpro.base.wrapping.Wrapping.row_stack`.

    Returns:
        MaybeTuple[AnyArray]: Merged result after row stacking.

    !!! note
        All arrays are assumed to have the same type and dimensionality.
    """
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if len(objs) == 0:
        raise ValueError("No objects to be merged")

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            out_tuple = (
                row_stack_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        else:
            out_tuple = tuple(
                map(
                    lambda x: row_stack_merge(
                        x,
                        keys=keys,
                        wrap=wrap,
                        wrapper=wrapper,
                        wrap_kwargs=wrap_kwargs,
                        **kwargs,
                    ),
                    zip(*objs),
                )
            )
        if checks.is_namedtuple(objs[0]):
            return type(objs[0])(*out_tuple)
        return type(objs[0])(out_tuple)

    if filter_results:
        try:
            objs, keys = filter_out_no_results(objs, keys=keys)
        except NoResultsException as e:
            if raise_no_results:
                raise e
            return NoResult

    if isinstance(objs[0], Wrapping):
        kwargs = merge_dicts(dict(wrapper_kwargs=dict(keys=keys)), kwargs)
        return type(objs[0]).row_stack(objs, **kwargs)
    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = (
            isinstance(objs[0], (pd.Series, pd.DataFrame))
            or wrapper is not None
            or keys is not None
            or len(wrap_kwargs) > 0
        )
    if isinstance(objs[0], pd.Index):
        objs = list(map(lambda x: x.to_series(), objs))

    default_index = True
    if not isinstance(objs[0], (pd.Series, pd.DataFrame)):
        if isinstance(wrap, str) or wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap(obj, **_wrap_kwargs))
                else:
                    if not isinstance(wrap, str):
                        if isinstance(obj, np.ndarray):
                            ndim = obj.ndim
                        else:
                            ndim = np.asarray(obj).ndim
                        if ndim == 1:
                            wrap = "series"
                        else:
                            wrap = "frame"
                    if isinstance(wrap, str):
                        if wrap.lower() in ("sr", "series"):
                            new_objs.append(pd.Series(obj, **_wrap_kwargs))
                        elif wrap.lower() in ("df", "frame", "dataframe"):
                            new_objs.append(pd.DataFrame(obj, **_wrap_kwargs))
                        else:
                            raise ValueError(f"Invalid wrapping option: '{wrap}'")
                if default_index and not checks.is_default_index(
                    new_objs[-1].index, check_names=True
                ):
                    default_index = False
            objs = new_objs

    if not wrap:
        return row_stack_arrays(objs)

    if keys is not None and isinstance(keys[0], pd.Index):
        new_obj = pd.concat(objs, axis=0, **kwargs)
        if len(keys) == 1:
            keys = keys[0]
        else:
            keys = concat_indexes(
                *keys,
                index_concat_method="append",
                verify_integrity=False,
                axis=0,
            )
        if default_index:
            new_obj.index = keys
        else:
            new_obj.index = stack_indexes((keys, new_obj.index))
    else:
        new_obj = pd.concat(objs, axis=0, keys=keys, **kwargs)
    if clean_index_kwargs is None:
        clean_index_kwargs = {}
    new_obj.index = clean_index(new_obj.index, **clean_index_kwargs)
    return new_obj


def column_stack_merge(
    *objs: tp.MaybeSequence[tp.MaybeTuple[tp.Any]],
    reset_index: tp.Union[None, bool, str] = None,
    fill_value: tp.Scalar = np.nan,
    keys: tp.Optional[tp.Index] = None,
    filter_results: bool = True,
    raise_no_results: bool = True,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    clean_index_kwargs: tp.KwargsLikeSequence = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like or `vectorbtpro.base.wrapping.Wrapping` objects via column stacking.

    Args:
        *objs (MaybeSequence[MaybeTuple[Any]]): Array-like or wrapping objects to merge.

            This function supports passing a sequence of tuples, where each element
            in the tuple is merged separately.
        reset_index (Union[None, bool, str]): Option to reset indexes in each object.

            * False or None: Retain the original index.
            * True or "from_start": Reset indexes to start from zero.
            * "from_end": Reset indexes to align at the end.

            !!! note
                Applicable to Pandas, NumPy, and `vectorbtpro.base.wrapping.Wrapping` instances.
        fill_value (Scalar): Value to use for filling missing entries when arrays have different row counts.
        keys (Optional[Index]): Keys used to label columns in the merged result.
        filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
        raise_no_results (bool): Flag indicating whether to raise a
            `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
        wrap (Union[None, str, bool]): Determines wrapping behavior for each object.

            * None: Treated as True if `wrapper`, `keys`, or `wrap_kwargs` is provided.
            * True: Wraps each array as a Pandas Series or DataFrame based on its dimensions.
            * "sr" or "series": Wraps each array as a Pandas Series.
            * "df", "frame", or "dataframe": Wraps each array as a Pandas DataFrame.

            Without wrapping, arrays will be kept as-is and merged using `column_stack_arrays`.
        wrapper (Optional[ArrayWrapper]): Array wrapper instance.
        wrap_kwargs (KwargsLikeSequence): Keyword arguments for wrapping the result;
            can be a dictionary or a list of dictionaries.

            See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
        clean_index_kwargs (KwargsLikeSequence): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        **kwargs: Keyword arguments for `pd.concat` and
            `vectorbtpro.base.wrapping.Wrapping.column_stack`.

    Returns:
        MaybeTuple[AnyArray]: Merged column-stacked array-like object,
            or a tuple of such objects when merging a sequence of tuples.

    !!! note
        All arrays are assumed to have the same type and dimensionality.
    """
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if len(objs) == 0:
        raise ValueError("No objects to be merged")
    if isinstance(reset_index, bool):
        if reset_index:
            reset_index = "from_start"
        else:
            reset_index = None

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            out_tuple = (
                column_stack_merge(
                    list(map(lambda x: x[0], objs)),
                    reset_index=reset_index,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        else:
            out_tuple = tuple(
                map(
                    lambda x: column_stack_merge(
                        x,
                        reset_index=reset_index,
                        keys=keys,
                        wrap=wrap,
                        wrapper=wrapper,
                        wrap_kwargs=wrap_kwargs,
                        **kwargs,
                    ),
                    zip(*objs),
                )
            )
        if checks.is_namedtuple(objs[0]):
            return type(objs[0])(*out_tuple)
        return type(objs[0])(out_tuple)

    if filter_results:
        try:
            objs, keys = filter_out_no_results(objs, keys=keys)
        except NoResultsException as e:
            if raise_no_results:
                raise e
            return NoResult

    if isinstance(objs[0], Wrapping):
        if reset_index is not None:
            max_length = max(map(lambda x: x.wrapper.shape[0], objs))
            new_objs = []
            for obj in objs:
                if isinstance(reset_index, str) and reset_index.lower() == "from_start":
                    new_index = pd.RangeIndex(stop=obj.wrapper.shape[0])
                    new_obj = obj.replace(wrapper=obj.wrapper.replace(index=new_index))
                elif isinstance(reset_index, str) and reset_index.lower() == "from_end":
                    new_index = pd.RangeIndex(
                        start=max_length - obj.wrapper.shape[0], stop=max_length
                    )
                    new_obj = obj.replace(wrapper=obj.wrapper.replace(index=new_index))
                else:
                    raise ValueError(f"Invalid index resetting option: '{reset_index}'")
                new_objs.append(new_obj)
            objs = new_objs
        kwargs = merge_dicts(dict(wrapper_kwargs=dict(keys=keys)), kwargs)
        return type(objs[0]).column_stack(objs, **kwargs)
    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = (
            isinstance(objs[0], (pd.Series, pd.DataFrame))
            or wrapper is not None
            or keys is not None
            or len(wrap_kwargs) > 0
        )
    if isinstance(objs[0], pd.Index):
        objs = list(map(lambda x: x.to_series(), objs))

    default_columns = True
    if not isinstance(objs[0], (pd.Series, pd.DataFrame)):
        if isinstance(wrap, str) or wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap(obj, **_wrap_kwargs))
                else:
                    if not isinstance(wrap, str):
                        if isinstance(obj, np.ndarray):
                            ndim = obj.ndim
                        else:
                            ndim = np.asarray(obj).ndim
                        if ndim == 1:
                            wrap = "series"
                        else:
                            wrap = "frame"
                    if isinstance(wrap, str):
                        if wrap.lower() in ("sr", "series"):
                            new_objs.append(pd.Series(obj, **_wrap_kwargs))
                        elif wrap.lower() in ("df", "frame", "dataframe"):
                            new_objs.append(pd.DataFrame(obj, **_wrap_kwargs))
                        else:
                            raise ValueError(f"Invalid wrapping option: '{wrap}'")
                if (
                    default_columns
                    and isinstance(new_objs[-1], pd.DataFrame)
                    and not checks.is_default_index(new_objs[-1].columns, check_names=True)
                ):
                    default_columns = False
            objs = new_objs

    if not wrap:
        if reset_index is not None:
            min_n_rows = None
            max_n_rows = None
            n_cols = 0
            new_objs = []
            for obj in objs:
                new_obj = to_2d_array(obj)
                new_objs.append(new_obj)
                if min_n_rows is None or new_obj.shape[0] < min_n_rows:
                    min_n_rows = new_obj.shape[0]
                if max_n_rows is None or new_obj.shape[0] > min_n_rows:
                    max_n_rows = new_obj.shape[0]
                n_cols += new_obj.shape[1]
            if min_n_rows == max_n_rows:
                return column_stack_arrays(new_objs)
            new_obj = np.full((max_n_rows, n_cols), fill_value)
            start_col = 0
            for obj in new_objs:
                end_col = start_col + obj.shape[1]
                if isinstance(reset_index, str) and reset_index.lower() == "from_start":
                    new_obj[: len(obj), start_col:end_col] = obj
                elif isinstance(reset_index, str) and reset_index.lower() == "from_end":
                    new_obj[-len(obj) :, start_col:end_col] = obj
                else:
                    raise ValueError(f"Invalid index resetting option: '{reset_index}'")
                start_col = end_col
            return new_obj
        return column_stack_arrays(objs)

    if reset_index is not None:
        max_length = max(map(len, objs))
        new_objs = []
        for obj in objs:
            new_obj = obj.copy(deep=False)
            if isinstance(reset_index, str) and reset_index.lower() == "from_start":
                new_obj.index = pd.RangeIndex(stop=len(new_obj))
            elif isinstance(reset_index, str) and reset_index.lower() == "from_end":
                new_obj.index = pd.RangeIndex(start=max_length - len(new_obj), stop=max_length)
            else:
                raise ValueError(f"Invalid index resetting option: '{reset_index}'")
            new_objs.append(new_obj)
        objs = new_objs
        kwargs = merge_dicts(dict(sort=True), kwargs)

    if keys is not None and isinstance(keys[0], pd.Index):
        new_obj = pd.concat(objs, axis=1, **kwargs)
        if len(keys) == 1:
            keys = keys[0]
        else:
            keys = concat_indexes(
                *keys,
                index_concat_method="append",
                verify_integrity=False,
                axis=1,
            )
        if default_columns:
            new_obj.columns = keys
        else:
            new_obj.columns = stack_indexes((keys, new_obj.columns))
    else:
        new_obj = pd.concat(objs, axis=1, keys=keys, **kwargs)
    if clean_index_kwargs is None:
        clean_index_kwargs = {}
    new_obj.columns = clean_index(new_obj.columns, **clean_index_kwargs)
    return new_obj


def imageio_merge(
    *objs: tp.MaybeSequence[tp.MaybeTuple[tp.Any]],
    keys: tp.Optional[tp.Index] = None,
    filter_results: bool = True,
    raise_no_results: bool = True,
    to_image_kwargs: tp.KwargsLike = None,
    imread_kwargs: tp.KwargsLike = None,
    **imwrite_kwargs,
) -> tp.MaybeTuple[tp.Union[None, bytes]]:
    """Merge multiple figure-like objects into a single image using `imageio`.

    Args:
        *objs (MaybeSequence[MaybeTuple[Any]]): Figure-like objects to merge.

            This function supports passing a sequence of tuples, where each element
            in the tuple is merged separately.
        keys (Optional[Index]): Not used.
        filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
        raise_no_results (bool): Flag indicating whether to raise a
            `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
        to_image_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Figure.to_image`.
        imread_kwargs (KwargsLike): Keyword arguments for `imageio.imread`.
        **imwrite_kwargs: Keyword arguments for `imageio.imwrite`.

    Returns:
        MaybeTuple[Union[None, bytes]]: Merged image data.

            An individual `bytes` object is returned when a single merged image is produced,
            or a tuple of `bytes` objects when multiple are merged.
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("plotly")
    import imageio.v3 as iio
    import plotly.graph_objects as go

    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if len(objs) == 0:
        raise ValueError("No objects to be merged")

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            out_tuple = (
                imageio_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    imread_kwargs=imread_kwargs,
                    to_image_kwargs=to_image_kwargs,
                    **imwrite_kwargs,
                ),
            )
        else:
            out_tuple = tuple(
                map(
                    lambda x: imageio_merge(
                        x,
                        keys=keys,
                        imread_kwargs=imread_kwargs,
                        to_image_kwargs=to_image_kwargs,
                        **imwrite_kwargs,
                    ),
                    zip(*objs),
                )
            )
        if checks.is_namedtuple(objs[0]):
            return type(objs[0])(*out_tuple)
        return type(objs[0])(out_tuple)

    if filter_results:
        try:
            objs, keys = filter_out_no_results(objs, keys=keys)
        except NoResultsException as e:
            if raise_no_results:
                raise e
            return NoResult

    if imread_kwargs is None:
        imread_kwargs = {}
    if to_image_kwargs is None:
        to_image_kwargs = {}

    frames = []
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, (go.Figure, go.FigureWidget)):
            obj = obj.to_image(**to_image_kwargs)
        if not isinstance(obj, np.ndarray):
            obj = iio.imread(obj, **imread_kwargs)
        frames.append(obj)
    return iio.imwrite(image=frames, **imwrite_kwargs)


def mixed_merge(
    *objs: tp.MaybeSequence[tp.Tuple[tp.Any]],
    merge_funcs: tp.Optional[tp.MergeFuncLike] = None,
    mixed_kwargs: tp.Optional[tp.Sequence[tp.KwargsLike]] = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge objects of mixed types element-wise using specified merging functions.

    Args:
        *objs (MaybeSequence[Tuple[Any]]): Tuples of objects to merge.

            Each tuple should have the same length, and merging is performed
            element-wise across these tuples.
        merge_funcs (Optional[MergeFuncLike]): Merging function or a sequence of merging functions
            (or their names) to apply to each group of objects.
        mixed_kwargs (Optional[Sequence[KwargsLike]]): Sequence of keyword argument dictionaries
            for each merging function.
        **kwargs: Keyword arguments for the merging functions if not overridden by `mixed_kwargs`.

    Returns:
        MaybeTuple[AnyArray]: Tuple containing the merged result for each group of objects.
    """
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if len(objs) == 0:
        raise ValueError("No objects to be merged")
    if merge_funcs is None:
        raise ValueError("Merging functions or their names are required")
    if not isinstance(objs[0], tuple):
        raise ValueError("Mixed merging must be applied on tuples")

    outputs = []
    for i, output_objs in enumerate(zip(*objs)):
        output_objs = list(output_objs)
        merge_func = resolve_merge_func(merge_funcs[i])
        if merge_func is None:
            outputs.append(output_objs)
        else:
            if mixed_kwargs is None:
                _kwargs = kwargs
            else:
                _kwargs = merge_dicts(kwargs, mixed_kwargs[i])
            output = merge_func(output_objs, **_kwargs)
            outputs.append(output)
    return tuple(outputs)


merge_func_config = HybridConfig(
    dict(
        concat=concat_merge,
        row_stack=row_stack_merge,
        column_stack=column_stack_merge,
        reset_column_stack=partial(column_stack_merge, reset_index=True),
        from_start_column_stack=partial(column_stack_merge, reset_index="from_start"),
        from_end_column_stack=partial(column_stack_merge, reset_index="from_end"),
        imageio=imageio_merge,
    )
)
"""_"""


__pdoc__["merge_func_config"] = f"""Configuration for merging functions.

This configuration maps merging function names to their respective implementations:

```python
{merge_func_config.prettify_doc()}
```
"""


def resolve_merge_func(merge_func: tp.MergeFuncLike) -> tp.Optional[tp.Callable]:
    """Resolve a merging function into a callable.

    Args:
        merge_func (MergeFuncLike): Merging function to resolve.

            * If provided as a string, it is looked up in `merge_func_config`.
            * If provided as a sequence, a partial application of `mixed_merge` with
                `merge_funcs=merge_func` is returned.
            * If provided as an instance of `vectorbtpro.utils.merging.MergeFunc`,
                its `resolve_merge_func` method is called to obtain the actual callable.

    Returns:
        Optional[Callable]: Resolved merging function as a callable, or None if `merge_func` is None.
    """
    if merge_func is None:
        return None
    if isinstance(merge_func, str):
        if merge_func.lower() not in merge_func_config:
            raise ValueError(f"Invalid merging function name: '{merge_func}'")
        return merge_func_config[merge_func.lower()]
    if checks.is_sequence(merge_func):
        return partial(mixed_merge, merge_funcs=merge_func)
    if isinstance(merge_func, MergeFunc):
        return merge_func.resolve_merge_func()
    return merge_func


def is_merge_func_from_config(merge_func: tp.MergeFuncLike) -> bool:
    """Determine if the provided merging function is defined in `merge_func_config`.

    Args:
        merge_func (MergeFuncLike): Merging function (or its representation) to check.

    Returns:
        bool: True if `merge_func` is found in `merge_func_config`, False otherwise.
    """
    if merge_func is None:
        return False
    if isinstance(merge_func, str):
        return merge_func.lower() in merge_func_config
    if checks.is_sequence(merge_func):
        return all(map(is_merge_func_from_config, merge_func))
    if isinstance(merge_func, MergeFunc):
        return is_merge_func_from_config(merge_func.merge_func)
    return False
