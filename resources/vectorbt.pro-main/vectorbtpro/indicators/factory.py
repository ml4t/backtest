# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing functionality for constructing and managing technical indicators.

Run for the examples below:

```pycon
>>> from vectorbtpro import *

>>> price = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... }, index=pd.date_range("2020", periods=5)).astype(float)
>>> price
            a    b
2020-01-01  1.0  5.0
2020-01-02  2.0  4.0
2020-01-03  3.0  3.0
2020-01-04  4.0  2.0
2020-01-05  5.0  1.0
```
"""

import fnmatch
import functools
import inspect
import itertools
import re
from collections import Counter, OrderedDict
from types import FunctionType, ModuleType

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base import combining, indexes, reshaping
from vectorbtpro.base.indexing import build_param_indexer
from vectorbtpro.base.merging import column_stack_arrays, row_stack_arrays
from vectorbtpro.base.reshaping import Default, broadcast_array_to, resolve_ref
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import BaseAccessor
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.indicators.expr import expr_func_config, expr_res_func_config, wqa101_expr_config
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import build_nan_mask, squeeze_nan, unsqueeze_nan
from vectorbtpro.utils.config import Config, Configured, HybridConfig, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import cacheable_property, class_property, hybrid_method
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.eval_ import evaluate
from vectorbtpro.utils.execution import Task
from vectorbtpro.utils.formatting import camel_to_snake_case, prettify
from vectorbtpro.utils.magic_decorators import (
    attach_binary_magic_methods,
    attach_unary_magic_methods,
)
from vectorbtpro.utils.mapping import apply_mapping, to_value_mapping
from vectorbtpro.utils.module_ import search_package
from vectorbtpro.utils.params import (
    Param,
    broadcast_params,
    combine_params,
    create_param_product,
    is_single_param_value,
    params_to_list,
    to_typed_list,
)
from vectorbtpro.utils.parsing import (
    get_expr_var_names,
    get_func_arg_names,
    get_func_kwargs,
    suppress_stdout,
)
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import Rep, has_templates, substitute_templates
from vectorbtpro.utils.warnings_ import WarningsFiltered, warn

__all__ = [
    "IndicatorBase",
    "IndicatorFactory",
    "IF",
    "indicator",
    "talib",
    "pandas_ta",
    "ta",
    "wqa101",
    "technical",
    "techcon",
    "smc",
]

__pdoc__ = {}

if tp.TYPE_CHECKING:
    from ta.utils import IndicatorMixin as IndicatorMixinT
else:
    IndicatorMixinT = "ta.utils.IndicatorMixin"
if tp.TYPE_CHECKING:
    from technical.consensus import Consensus as ConsensusT
else:
    ConsensusT = "technical.consensus.Consensus"


def prepare_params(
    params: tp.MaybeParams,
    param_names: tp.Sequence[str],
    param_settings: tp.Sequence[tp.KwargsLike],
    input_shape: tp.Optional[tp.Shape] = None,
    to_2d: bool = False,
    context: tp.KwargsLike = None,
) -> tp.Tuple[tp.Params, bool]:
    """Prepare parameters.

    Resolve references in the input parameters and perform broadcasting to match the input shape.

    Args:
        params (MaybeParams): Input parameters, which may include references that need resolution.

            They are mapped based on `param_names`.
        param_names (Sequence[str]): Names of the parameters.
        param_settings (Sequence[KwargsLike]): Sequence of dictionaries providing settings for each parameter.
        input_shape (Optional[Shape]): Shape of the input arrays.
        to_2d (bool): If True, reshapes inputs to two-dimensional arrays.
        context (KwargsLike): Additional context for template substitution.

    Returns:
        Tuple[Params, bool]: Tuple where the first element is the list of processed parameters
            and the second element indicates whether a single parameter combination was provided.
    """
    if context is None:
        context = {}
    pool = dict(zip(param_names, params))
    for k in pool:
        pool[k] = resolve_ref(pool, k)
    params = [pool[k] for k in param_names]

    new_params = []
    single_comb = True
    for i, param_values in enumerate(params):
        _param_settings = resolve_dict(param_settings[i])
        is_tuple = _param_settings.get("is_tuple", False)
        dtype = _param_settings.get("dtype", None)
        if checks.is_mapping_like(dtype):
            dtype_kwargs = _param_settings.get("dtype_kwargs", None)
            if dtype_kwargs is None:
                dtype_kwargs = {}
            if checks.is_namedtuple(dtype):
                param_values = map_enum_fields(param_values, dtype, **dtype_kwargs)
            else:
                param_values = apply_mapping(param_values, dtype, **dtype_kwargs)
        is_array_like = _param_settings.get("is_array_like", False)
        min_one_dim = _param_settings.get("min_one_dim", False)
        bc_to_input = _param_settings.get("bc_to_input", False)
        broadcast_kwargs = merge_dicts(
            dict(require_kwargs=dict(requirements="W")),
            _param_settings.get("broadcast_kwargs", None),
        )
        template = _param_settings.get("template", None)

        if not is_single_param_value(param_values, is_tuple, is_array_like):
            single_comb = False
        new_param_values = params_to_list(param_values, is_tuple, is_array_like)
        if template is not None:
            new_param_values = [
                template.substitute(context={param_names[i]: new_param_values[j], **context})
                for j in range(len(new_param_values))
            ]
        if not bc_to_input:
            if is_array_like:
                if min_one_dim:
                    new_param_values = list(map(reshaping.to_1d_array, new_param_values))
                else:
                    new_param_values = list(map(np.asarray, new_param_values))
        else:
            if is_tuple:
                raise ValueError("Cannot broadcast to input if tuple")
            if input_shape is None:
                raise ValueError(
                    "Cannot broadcast to input if input shape is unknown. Pass input_shape."
                )
            if bc_to_input is True:
                to_shape = input_shape
            else:
                checks.assert_in(bc_to_input, (0, 1))
                if bc_to_input == 0:
                    to_shape = (input_shape[0],)
                else:
                    to_shape = (input_shape[1],) if len(input_shape) > 1 else (1,)
            _new_param_values = reshaping.broadcast(
                *new_param_values, to_shape=to_shape, **broadcast_kwargs
            )
            if len(new_param_values) == 1:
                _new_param_values = [_new_param_values]
            else:
                _new_param_values = list(_new_param_values)
            if to_2d and bc_to_input is True:
                __new_param_values = _new_param_values.copy()
                for j, param in enumerate(__new_param_values):
                    keep_flex = broadcast_kwargs.get("keep_flex", False)
                    if keep_flex is False or (
                        isinstance(keep_flex, (tuple, list)) and not keep_flex[j]
                    ):
                        __new_param_values[j] = reshaping.to_2d(param)
                new_param_values = __new_param_values
            else:
                new_param_values = _new_param_values
        new_params.append(new_param_values)
    return new_params, single_comb


def build_columns(
    params: tp.Params,
    input_columns: tp.IndexLike,
    level_names: tp.Optional[tp.Sequence[str]] = None,
    hide_levels: tp.Optional[tp.Sequence[tp.Union[str, int]]] = None,
    single_value: tp.Optional[tp.Sequence[bool]] = None,
    param_settings: tp.KwargsLikeSequence = None,
    per_column: bool = False,
    ignore_ranges: bool = False,
    **kwargs,
) -> dict:
    """For each parameter in `params`, create a new column level with parameter values
    and stack it on top of `input_columns`.

    Args:
        params (Params): Collection of parameter values.
        input_columns (IndexLike): Initial column index to which parameter levels are added.
        level_names (Optional[Sequence[str]]): List of level names corresponding to each parameter.
        hide_levels (Optional[Sequence[Union[str, int]]): Levels to exclude from visibility.
        single_value (Optional[Sequence[bool]]): Flags indicating if each parameter is a single value.
        param_settings (KwargsLikeSequence): Settings for parameters such as data type mapping and processing options.
        per_column (bool): If True, processes parameters separately for each column.
        ignore_ranges (bool): If True, ignores range checks during column stacking.
        **kwargs: Keyword arguments for `vectorbtpro.base.indexes.stack_indexes`.

    Returns:
        dict: Dictionary containing:

            * `param_indexes`: List of initial parameter indexes.
            * `rep_param_indexes`: List of repeated parameter indexes corresponding to `input_columns`.
            * `vis_param_indexes`: List of visible parameter indexes not hidden.
            * `vis_rep_param_indexes`: List of visible repeated parameter indexes.
            * `param_index`: Combined parameter index, or None if `per_column` is True.
            * `final_index`: Final stacked index combining visible parameter indexes and `input_columns`.
    """
    if level_names is not None:
        checks.assert_len_equal(params, level_names)
    if hide_levels is None:
        hide_levels = []
    input_columns = indexes.to_any_index(input_columns)

    param_indexes = []
    rep_param_indexes = []
    vis_param_indexes = []
    vis_rep_param_indexes = []
    has_per_column = False
    for i in range(len(params)):
        param_values = params[i]
        level_name = None
        if level_names is not None:
            level_name = level_names[i]
        _single_value = False
        if single_value is not None:
            _single_value = single_value[i]
        _param_settings = resolve_dict(param_settings, i=i)
        dtype = _param_settings.get("dtype", None)
        if checks.is_mapping_like(dtype):
            if checks.is_namedtuple(dtype):
                dtype = to_value_mapping(dtype, reverse=False)
            else:
                dtype = to_value_mapping(dtype, reverse=True)
            param_values = apply_mapping(param_values, dtype)
        _per_column = _param_settings.get("per_column", False)
        _post_index_func = _param_settings.get("post_index_func", None)

        if per_column:
            param_index = indexes.index_from_values(
                param_values, single_value=_single_value, name=level_name
            )
            repeat_index = False
            has_per_column = True
        elif _per_column:
            param_index = None
            for p in param_values:
                bc_param = broadcast_array_to(p, len(input_columns))
                _param_index = indexes.index_from_values(
                    bc_param, single_value=False, name=level_name
                )
                if param_index is None:
                    param_index = _param_index
                else:
                    param_index = param_index.append(_param_index)
            if len(param_index) == 1 and len(input_columns) > 1:
                param_index = indexes.repeat_index(
                    param_index, len(input_columns), ignore_ranges=ignore_ranges
                )
            repeat_index = False
            has_per_column = True
        else:
            param_index = indexes.index_from_values(
                param_values, single_value=_single_value, name=level_name
            )
            repeat_index = True

        if _post_index_func is not None:
            param_index = _post_index_func(param_index)
        if repeat_index:
            rep_param_index = indexes.repeat_index(
                param_index, len(input_columns), ignore_ranges=ignore_ranges
            )
        else:
            rep_param_index = param_index
        param_indexes.append(param_index)
        rep_param_indexes.append(rep_param_index)
        if i not in hide_levels and (level_names is None or level_names[i] not in hide_levels):
            vis_param_indexes.append(param_index)
            vis_rep_param_indexes.append(rep_param_index)

    if not per_column:
        n_param_values = len(params[0]) if len(params) > 0 else 1
        input_columns = indexes.tile_index(
            input_columns, n_param_values, ignore_ranges=ignore_ranges
        )
    if len(vis_param_indexes) > 0:
        if has_per_column:
            param_index = None
        else:
            param_index = indexes.stack_indexes(vis_param_indexes, **kwargs)
        final_index = indexes.stack_indexes([*vis_rep_param_indexes, input_columns], **kwargs)
    else:
        param_index = None
        final_index = input_columns
    return dict(
        param_indexes=rep_param_indexes,
        rep_param_indexes=rep_param_indexes,
        vis_param_indexes=vis_param_indexes,
        vis_rep_param_indexes=vis_rep_param_indexes,
        param_index=param_index,
        final_index=final_index,
    )


def combine_objs(
    obj: tp.SeriesFrame,
    other: tp.MaybeTupleList[tp.Union[tp.ArrayLike, BaseAccessor]],
    combine_func: tp.Callable,
    *args,
    level_name: tp.Optional[str] = None,
    keys: tp.Optional[tp.IndexLike] = None,
    allow_multiple: bool = True,
    **kwargs,
) -> tp.SeriesFrame:
    """Combine or compare `obj` with `other` to generate signals by applying a custom combine function.

    Args:
        obj (SeriesFrame): Main Series or DataFrame to operate on.
        other (MaybeTupleList[Union[ArrayLike, BaseAccessor]]): Object or objects to be combined with `obj`.
        combine_func (Callable): Function used to combine or compare elements of `obj` and `other`.
        *args: Positional arguments for `vectorbtpro.base.accessors.BaseAccessor.combine`.
        level_name (Optional[str]): Name for the new column level when multiple values of `other` are provided.
        keys (Optional[IndexLike]): Keys to use when broadcasting multiple objects.
        allow_multiple (bool): If True, permits `other` to be provided as a tuple or list.
        **kwargs: Keyword arguments for `vectorbtpro.base.accessors.BaseAccessor.combine`.

    Returns:
        SeriesFrame: Resulting Series or DataFrame after combining `obj` with `other`.
    """
    if allow_multiple and isinstance(other, (tuple, list)):
        if keys is None:
            keys = indexes.index_from_values(other, name=level_name)
    return obj.vbt.combine(
        other, combine_func, *args, keys=keys, allow_multiple=allow_multiple, **kwargs
    )


IndicatorBaseT = tp.TypeVar("IndicatorBaseT", bound="IndicatorBase")


def combine_indicator_with_other(
    self: IndicatorBaseT,
    other: tp.Union["IndicatorBase", tp.ArrayLike],
    np_func: tp.Callable[[tp.ArrayLike, tp.ArrayLike], tp.Array1d],
) -> tp.SeriesFrame:
    """Combine `IndicatorBase` with another compatible object by applying a specified NumPy function.

    Args:
        other (Union[IndicatorBase, ArrayLike]): Other indicator or array.

            If an instance of `IndicatorBase` is provided, its `main_output` is used.
        np_func (Callable[[ArrayLike, ArrayLike], Array1d]): Function that combines
            the arrays from `IndicatorBase.main_output` and `other`.

    Returns:
        SeriesFrame: Resulting Series or DataFrame after combining `IndicatorBase.main_output`
            with the other object's data.
    """
    if isinstance(other, IndicatorBase):
        other = other.main_output
    return np_func(self.main_output, other)


@attach_binary_magic_methods(combine_indicator_with_other)
@attach_unary_magic_methods(lambda self, np_func: np_func(self.main_output))
class IndicatorBase(Analyzable):
    """Base class for indicators.

    Set properties before instantiation.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        input_list (IFArrayList): List of two-dimensional input arrays.
        input_mapper (IFInputMapper): One-dimensional input mapper array.
        in_output_list (IFArrayList): List of two-dimensional input-output arrays.
        output_list (IFArrayList): List of two-dimensional output arrays.
        param_list (IFParamList): List of parameter value lists.
        mapper_list (IFMapperList): List of mapper indexes.
        short_name (str): Short name of the indicator.
        **kwargs: Keyword arguments for `vectorbtpro.generic.analyzable.Analyzable`.
    """

    _short_name: tp.ClassVar[str]
    _input_names: tp.ClassVar[tp.Tuple[str, ...]]
    _param_names: tp.ClassVar[tp.Tuple[str, ...]]
    _in_output_names: tp.ClassVar[tp.Tuple[str, ...]]
    _output_names: tp.ClassVar[tp.Tuple[str, ...]]
    _lazy_output_names: tp.ClassVar[tp.Tuple[str, ...]]
    _output_flags: tp.ClassVar[tp.Kwargs]

    def __init__(
        self,
        wrapper: ArrayWrapper,
        input_list: tp.IFArrayList,
        input_mapper: tp.IFInputMapper,
        in_output_list: tp.IFArrayList,
        output_list: tp.IFArrayList,
        param_list: tp.IFParamList,
        mapper_list: tp.IFMapperList,
        short_name: str,
        **kwargs,
    ) -> None:
        if input_mapper is not None:
            checks.assert_equal(input_mapper.shape[0], wrapper.shape_2d[1])
        for ts in input_list:
            checks.assert_equal(ts.shape[0], wrapper.shape_2d[0])
        for ts in in_output_list + output_list:
            checks.assert_equal(ts.shape, wrapper.shape_2d)
        for params in param_list:
            checks.assert_len_equal(param_list[0], params)
        for mapper in mapper_list:
            checks.assert_equal(len(mapper), wrapper.shape_2d[1])
        checks.assert_instance_of(short_name, str)
        if "level_names" in kwargs:
            del kwargs["level_names"]

        Analyzable.__init__(
            self,
            wrapper,
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list,
            short_name=short_name,
            **kwargs,
        )

        self._short_name = short_name
        for i, ts_name in enumerate(self.input_names):
            setattr(self, f"_{ts_name}", input_list[i])
        self._input_mapper = input_mapper
        for i, in_output_name in enumerate(self.in_output_names):
            setattr(self, f"_{in_output_name}", in_output_list[i])
        for i, output_name in enumerate(self.output_names):
            setattr(self, f"_{output_name}", output_list[i])
        for i, param_name in enumerate(self.param_names):
            setattr(self, f"_{param_name}_list", param_list[i])
            setattr(self, f"_{param_name}_mapper", mapper_list[i])

        mapper_sr_list = []
        for i, m in enumerate(mapper_list):
            mapper_sr_list.append(pd.Series(m, index=wrapper.columns))
        tuple_mapper = self._tuple_mapper
        if tuple_mapper is not None:
            level_names = tuple(tuple_mapper.names)
            mapper_sr_list.append(pd.Series(tuple_mapper.tolist(), index=wrapper.columns))
        else:
            level_names = ()
        self._level_names = level_names
        for base_cls in type(self).__bases__:
            if base_cls.__name__ == "ParamIndexer":
                base_cls.__init__(self, mapper_sr_list, level_names=[*level_names, level_names])

    def __getattr__(self, k: str) -> tp.Any:
        """Redirect attribute lookup queries for generic output names.

        If the attribute is not found via the standard lookup, attempt to resolve
        it using the indicator's short name and output naming conventions.

        Args:
            k (str): Attribute name to resolve.

        Returns:
            Any: Value of the resolved attribute.
        """
        try:
            return object.__getattribute__(self, k)
        except AttributeError:
            pass
        if k == "vbt":
            return self.main_output.vbt

        short_name = object.__getattribute__(self, "short_name")
        output_names = object.__getattribute__(self, "output_names")
        if len(output_names) == 1:
            if k.startswith("output") and "output" not in output_names:
                new_k = k[len("output") :]
                if len(new_k) == 0 or not new_k[0].isalnum():
                    try:
                        return object.__getattribute__(self, output_names[0] + new_k)
                    except AttributeError:
                        pass
            if k.startswith(short_name) and short_name not in output_names:
                new_k = k[len(short_name) :]
                if len(new_k) == 0 or not new_k[0].isalnum():
                    try:
                        return object.__getattribute__(self, output_names[0] + new_k)
                    except AttributeError:
                        pass
            if k.lower().startswith(short_name.lower()) and short_name.lower() not in output_names:
                new_k = k[len(short_name) :].lower()
                if len(new_k) == 0 or not new_k[0].isalnum():
                    try:
                        return object.__getattribute__(self, output_names[0] + new_k)
                    except AttributeError:
                        pass
            try:
                return object.__getattribute__(self, output_names[0] + "_" + k)
            except AttributeError:
                pass
        elif short_name in output_names:
            try:
                return object.__getattribute__(self, short_name + "_" + k)
            except AttributeError:
                pass
        elif short_name.lower() in output_names:
            try:
                return object.__getattribute__(self, short_name.lower() + "_" + k)
            except AttributeError:
                pass
        return object.__getattribute__(self, k)

    @property
    def main_output(self) -> tp.SeriesFrame:
        """Main output.

        If the indicator has only one output, return that output.
        Otherwise, return the output matching the indicator's short name (case sensitive or lower case).

        Returns:
            SeriesFrame: Main output of the indicator.

        Raises:
            ValueError: If the indicator has no main output.
        """
        if len(self.output_names) == 1:
            return getattr(self, self.output_names[0])
        if self.short_name in self.output_names:
            return getattr(self, self.short_name)
        if self.short_name.lower() in self.output_names:
            return getattr(self, self.short_name.lower())
        raise ValueError(f"Indicator {self} has no main output")

    def __array__(self, dtype: tp.Optional[tp.DTypeLike] = None) -> tp.Array:
        """Convert the main output to a NumPy array.

        Args:
            dtype (Optional[DTypeLike]): Data type to use for the conversion.

        Returns:
            Array: NumPy array representation of the main output.
        """
        return np.asarray(self.main_output, dtype=dtype)

    @classmethod
    def run_pipeline(
        cls,
        num_ret_outputs: int,
        custom_func: tp.Callable,
        *args,
        require_input_shape: bool = False,
        input_shape: tp.Optional[tp.ShapeLike] = None,
        input_index: tp.Optional[tp.IndexLike] = None,
        input_columns: tp.Optional[tp.IndexLike] = None,
        inputs: tp.Optional[tp.MappingSequence[tp.ArrayLike]] = None,
        in_outputs: tp.Optional[tp.MappingSequence[tp.ArrayLike]] = None,
        in_output_settings: tp.Optional[tp.MappingSequence[tp.KwargsLike]] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        params: tp.Optional[tp.MaybeParams] = None,
        param_product: bool = False,
        combine_kwargs: tp.KwargsLike = None,
        random_subset: tp.Optional[int] = None,
        param_settings: tp.Optional[tp.MappingSequence[tp.KwargsLike]] = None,
        run_unique: bool = False,
        silence_warnings: bool = False,
        per_column: tp.Optional[bool] = None,
        keep_pd: bool = False,
        to_2d: bool = True,
        pass_packed: bool = False,
        pass_input_shape: tp.Optional[bool] = None,
        pass_wrapper: bool = False,
        pass_param_index: bool = False,
        pass_final_index: bool = False,
        pass_single_comb: bool = False,
        level_names: tp.Optional[tp.Sequence[str]] = None,
        hide_levels: tp.Optional[tp.Sequence[tp.Union[str, int]]] = None,
        build_col_kwargs: tp.KwargsLike = None,
        return_raw: tp.Union[bool, str] = False,
        use_raw: tp.Optional[tp.IFRawOutput] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        seed: tp.Optional[int] = None,
        **kwargs,
    ) -> tp.Union[tp.IFCacheOutput, tp.IFRawOutput, tp.IFPipelineOutput]:
        """Run a pipeline to compute an indicator using a custom function.

        This method prepares input arrays, parameters, and necessary broadcasting, and then applies the
        custom function to perform indicator calculations. It supports parameter combination, per-column
        processing, and various configurations to adjust input shapes and outputs. This method is used internally
        by `IndicatorFactory`.

        Args:
            num_ret_outputs (int): Number of output arrays returned by `custom_func`.
            custom_func (Callable): Custom function for indicator computation.

                See `IndicatorFactory.with_custom_func`.
            *args: Positional arguments for `custom_func`.
            require_input_shape (bool): Flag indicating whether the input shape is required.

                If True, sets `pass_input_shape` to True and raises an error if `input_shape` is None.
            input_shape (Optional[ShapeLike]): Shape to which each input is broadcast.

                May be passed to `custom_func` if `pass_input_shape` is enabled.
            input_index (Optional[IndexLike]): Index to assign to each input array.

                Used to label inputs when no explicit index is provided.
            input_columns (Optional[IndexLike]): Column labels for each input array.

                Used to label inputs when no explicit columns are provided.
            inputs (Optional[MappingSequence[ArrayLike]]): Input arrays provided as a mapping or sequence.

                If a sequence is given, it is converted to a mapping with keys formatted as `input_{i}`.
            in_outputs (Optional[MappingSequence[ArrayLike]]): In-place output arrays provided
                as a mapping or sequence.

                If a sequence is given, it is converted to a mapping with keys formatted as `in_output_{i}`.
            in_output_settings (Optional[MappingSequence[KwargsLike]]): Settings for each in-place output.

                If provided as a mapping, keys should correspond to those in `in_outputs`. Accepted keys:

                * `dtype`: Data type to use when creating the array with `np.empty`.
                * `doc`: Documentation string for the in-place output.
            broadcast_named_args (KwargsLike): Additional named arguments for broadcasting.

                Use templates such as `vectorbtpro.utils.template.Rep` to substitute
                callback function arguments with their broadcasted values.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            params (Optional[MaybeParams]): Parameters provided as a mapping or sequence.

                If given as a sequence, it is converted to a mapping with keys formatted as `param_{i}`.
                Each parameter can be an array-like object or a single value.
            param_product (bool): Flag to build a Cartesian product from all parameters.
            combine_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.utils.params.combine_params`.
            random_subset (Optional[int]): Select a random subset of parameter combinations.

                Set the seed for reproducibility.
            param_settings (Optional[MappingSequence[KwargsLike]]): Settings for each parameter.

                If provided as a mapping, keys should correspond to those in `params`. Accepted keys:

                * `dtype`: Used for converting a string parameter value based on an enumerated type.
                * `dtype_kwargs`: Keyword arguments for processing the data type,
                    used by `vectorbtpro.utils.enum_.map_enum_fields` for enumerated types.
                * `is_tuple`: Treat a tuple as a single value; use a list to treat it as multiple values.
                * `is_array_like`: Treat an array-like object as a single value; use a list for multiple values.
                * `template`: Template to substitute each parameter value before broadcasting.
                * `min_one_dim`: Convert a scalar into a one-dimensional array.
                * `bc_to_input`: Broadcast the parameter to the input shape or along a specific axis.
                * `broadcast_kwargs`: Keyword arguments for input broadcasting.
                * `per_column`: Allow splitting parameter values per column for multi-indexing.
                * `post_index_func`: Function to transform the final parameter index level.
                * `doc`: Documentation string for the parameter.
            run_unique (bool): Flag to run only on unique parameter combinations.

                !!! note
                    Cache, raw output, and extra outputs beyond `num_ret_outputs` are returned
                    only for unique parameter combinations.
            silence_warnings (bool): Flag to suppress warning messages.
            per_column (Optional[bool]): Flag indicating whether parameter values should be applied per column.

                When True, each list of parameter values is broadcast to the number of input columns and applied
                per column rather than globally. Requires a known input shape.
            keep_pd (bool): If True, retain inputs as Pandas objects; otherwise, convert them to NumPy arrays.
            to_2d (bool): If True, reshapes inputs to two-dimensional arrays.
            pass_packed (bool): Whether to pass inputs, in-place outputs, and parameters as packed tuples.

                For Numba-compiled functions, tuples are passed instead.
            pass_input_shape (Optional[bool]): If True, passes `input_shape` as a keyword argument to `custom_func`.

                Defaults to True if `require_input_shape` is True, otherwise False.
            pass_wrapper (bool): If True, passes the input wrapper to `custom_func` as a keyword argument.
            pass_param_index (bool): If True, passes the parameter index to `custom_func`.
            pass_final_index (bool): If True, passes the final index to `custom_func`.
            pass_single_comb (bool): If True, indicates that there is only one parameter combination,
                and passes this information to `custom_func`.
            level_names (Optional[Sequence[str]]): List of level names corresponding to each parameter.
            hide_levels (Optional[Sequence[Union[str, int]]]): List of level names or indices
                to hide from the output.
            build_col_kwargs (KwargsLike): Keyword arguments for `build_columns`.
            return_raw (Union[bool, str]): If set, returns raw outputs and hashed parameter tuples
                without further post-processing.

                Passing "outputs" returns only the raw outputs.
            use_raw (Optional[IFRawOutput]): If True, uses the raw results obtained previously
                instead of executing `custom_func`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            seed (Optional[int]): Random seed for deterministic output.
            **kwargs: Keyword arguments for `custom_func`.

                Common arguments include `return_cache` to return cache and `use_cache` to control
                caching. If `use_cache` is False, caching is disabled. These apply only to functions
                that support caching, such as those created via `IndicatorFactory.with_apply_func`.

        Returns:
            Union[tp.IFCacheOutput, tp.IFRawOutput, tp.IFPipelineOutput]:
                Tuple containing the following elements:

                * Array wrapper.
                * List of input arrays (`np.ndarray`).
                * Input mapper (`np.ndarray`).
                * List of output arrays for in-place outputs (`np.ndarray`).
                * List of additional output arrays beyond `num_ret_outputs` (`np.ndarray`).
                * List of parameter arrays (`np.ndarray`).
                * List of parameter mapper arrays (`np.ndarray`).
                * List of extra output objects.
        """
        pass_per_column = per_column is not None
        if per_column is None:
            per_column = False
        if len(params) == 0 and per_column:
            raise ValueError("per_column cannot be enabled without parameters")
        if require_input_shape:
            checks.assert_not_none(input_shape, arg_name="input_shape")
            if pass_input_shape is None:
                pass_input_shape = True
        if pass_input_shape is None:
            pass_input_shape = False
        if input_index is not None:
            input_index = indexes.to_any_index(input_index)
        if input_columns is not None:
            input_columns = indexes.to_any_index(input_columns)
        if inputs is None:
            inputs = {}
        if not checks.is_mapping(inputs):
            inputs = {"input_" + str(i): input for i, input in enumerate(inputs)}
        input_names = list(inputs.keys())
        input_list = list(inputs.values())
        if in_outputs is None:
            in_outputs = {}
        if not checks.is_mapping(in_outputs):
            in_outputs = {
                "in_output_" + str(i): in_output for i, in_output in enumerate(in_outputs)
            }
        in_output_names = list(in_outputs.keys())
        in_output_list = list(in_outputs.values())
        if in_output_settings is None:
            in_output_settings = {}
        if checks.is_mapping(in_output_settings):
            checks.assert_dict_valid(in_output_settings, [in_output_names, ["dtype", "doc"]])
            in_output_settings = [in_output_settings.get(k) for k in in_output_names]
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if combine_kwargs is None:
            combine_kwargs = {}
        if template_context is None:
            template_context = {}
        if params is None:
            params = {}
        if not checks.is_mapping(params):
            params = {"param_" + str(i): param for i, param in enumerate(params)}
        if any([isinstance(v, Param) for k, v in params.items()]):
            params = combine_params(
                params,
                build_index=False,
                keep_single_value=True,
                build_product=param_product,
                **combine_kwargs,
            )
            param_product = False
        param_names = list(params.keys())
        param_list = list(params.values())
        if param_settings is None:
            param_settings = {}
        if checks.is_mapping(param_settings):
            checks.assert_dict_valid(
                param_settings,
                [
                    param_names,
                    [
                        "dtype",
                        "dtype_kwargs",
                        "is_tuple",
                        "is_array_like",
                        "template",
                        "min_one_dim",
                        "bc_to_input",
                        "broadcast_kwargs",
                        "per_column",
                        "post_index_func",
                        "doc",
                    ],
                ],
            )
            param_settings = [param_settings.get(k) for k in param_names]
        if hide_levels is None:
            hide_levels = []
        if build_col_kwargs is None:
            build_col_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if keep_pd and checks.is_numba_func(custom_func):
            raise ValueError(
                "Cannot pass Pandas objects to a Numba-compiled custom_func. Set keep_pd to False."
            )

        if seed is not None:
            set_seed(seed)

        if input_shape is not None:
            input_shape = reshaping.to_tuple_shape(input_shape)
        if len(inputs) > 0 or len(in_outputs) > 0 or len(broadcast_named_args) > 0:
            broadcast_args = merge_dicts(inputs, in_outputs, broadcast_named_args)
            broadcast_kwargs = merge_dicts(
                dict(
                    to_shape=input_shape,
                    index_from=input_index,
                    columns_from=input_columns,
                    require_kwargs=dict(requirements="W"),
                    post_func=None if keep_pd else np.asarray,
                    to_pd=True,
                ),
                broadcast_kwargs,
            )
            broadcast_args, wrapper = reshaping.broadcast(
                broadcast_args, return_wrapper=True, **broadcast_kwargs
            )
            input_shape, input_index, input_columns = wrapper.shape, wrapper.index, wrapper.columns
            if input_index is None:
                input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
            if input_columns is None:
                input_columns = pd.RangeIndex(
                    start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1
                )
            input_list = [broadcast_args[input_name] for input_name in input_names]
            in_output_list = [broadcast_args[in_output_name] for in_output_name in in_output_names]
            broadcast_named_args = {
                arg_name: broadcast_args[arg_name] for arg_name in broadcast_named_args
            }
        else:
            wrapper = None

        input_shape_ready = input_shape
        input_shape_2d = input_shape
        if input_shape is not None:
            input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
        if to_2d:
            if input_shape is not None:
                input_shape_ready = input_shape_2d
        if wrapper is not None:
            wrapper_ready = wrapper
        elif (
            input_index is not None and input_columns is not None and input_shape_ready is not None
        ):
            wrapper_ready = ArrayWrapper(input_index, input_columns, len(input_shape_ready))
        else:
            wrapper_ready = None

        input_list_ready = []
        for input in input_list:
            new_input = input
            if to_2d:
                new_input = reshaping.to_2d(input)
            if keep_pd and isinstance(new_input, np.ndarray):
                new_input = ArrayWrapper(input_index, input_columns, new_input.ndim).wrap(new_input)
            input_list_ready.append(new_input)

        param_context = merge_dicts(
            broadcast_named_args,
            dict(
                input_shape=input_shape_ready,
                wrapper=wrapper_ready,
                **dict(zip(input_names, input_list_ready)),
                pre_sub_args=args,
                pre_sub_kwargs=kwargs,
            ),
            template_context,
        )
        param_list, single_comb = prepare_params(
            param_list,
            param_names,
            param_settings,
            input_shape=input_shape,
            to_2d=to_2d,
            context=param_context,
        )
        single_value = list(map(lambda x: len(x) == 1, param_list))
        if len(param_list) > 1:
            if level_names is not None:
                checks.assert_len_equal(param_list, level_names)
                if input_columns is not None:
                    for level_name in level_names:
                        if level_name is not None:
                            checks.assert_level_not_exists(input_columns, level_name)
            if param_product:
                param_list = create_param_product(param_list)
        if len(param_list) > 0:
            if per_column:
                param_list = broadcast_params(param_list, to_n=input_shape_2d[1])
            else:
                param_list = broadcast_params(param_list)
        if random_subset is not None:
            if per_column:
                raise ValueError("Cannot select random subset when per_column=True")
            random_indices = np.sort(
                np.random.permutation(np.arange(len(param_list[0])))[:random_subset]
            )
            param_list = [[params[i] for i in random_indices] for params in param_list]
        n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
        use_run_unique = False
        param_list_unique = param_list
        if not per_column and run_unique:
            try:
                param_tuples = list(zip(*param_list))
                unique_param_tuples = list(OrderedDict.fromkeys(param_tuples).keys())
                if len(unique_param_tuples) < len(param_tuples):
                    param_list_unique = list(map(list, zip(*unique_param_tuples)))
                    use_run_unique = True
            except Exception:
                pass
        if checks.is_numba_func(custom_func):
            param_list_ready = [to_typed_list(params) for params in param_list_unique]
        else:
            param_list_ready = param_list_unique
        n_unique_param_values = len(param_list_unique[0]) if len(param_list_unique) > 0 else 1

        if (
            len(param_list) > 0
            and input_columns is not None
            and (pass_param_index or pass_final_index)
        ):
            build_columns_meta = build_columns(
                param_list,
                input_columns,
                level_names=level_names,
                hide_levels=hide_levels,
                single_value=single_value,
                param_settings=param_settings,
                per_column=per_column,
                **build_col_kwargs,
            )
            rep_param_indexes = build_columns_meta["rep_param_indexes"]
            param_index = build_columns_meta["param_index"]
            final_index = build_columns_meta["final_index"]
        else:
            rep_param_indexes = None
            param_index = None
            final_index = None

        in_output_list_ready = []
        for i in range(len(in_output_list)):
            if input_shape_2d is None:
                raise ValueError("input_shape is required when using in-place outputs")
            if in_output_list[i] is not None:
                in_output_wide = in_output_list[i]
                if isinstance(in_output_list[i], np.ndarray):
                    in_output_wide = np.require(in_output_wide, requirements="W")
                if not per_column:
                    in_output_wide = reshaping.tile(in_output_wide, n_unique_param_values, axis=1)
            else:
                _in_output_settings = resolve_dict(in_output_settings[i])
                dtype = _in_output_settings.get("dtype", None)
                if per_column:
                    in_output_shape = input_shape_ready
                else:
                    in_output_shape = (input_shape_2d[0], input_shape_2d[1] * n_unique_param_values)
                in_output_wide = np.empty(in_output_shape, dtype=dtype)
            in_output_list[i] = in_output_wide
            in_outputs = []
            if per_column:
                in_outputs.append(in_output_wide)
            else:
                for p in range(n_unique_param_values):
                    if isinstance(in_output_wide, pd.DataFrame):
                        in_output = in_output_wide.iloc[
                            :, p * input_shape_2d[1] : (p + 1) * input_shape_2d[1]
                        ]
                        if len(input_shape_ready) == 1:
                            in_output = in_output.iloc[:, 0]
                    else:
                        in_output = in_output_wide[
                            :, p * input_shape_2d[1] : (p + 1) * input_shape_2d[1]
                        ]
                        if len(input_shape_ready) == 1:
                            in_output = in_output[:, 0]
                    if keep_pd and isinstance(in_output, np.ndarray):
                        in_output = ArrayWrapper(input_index, input_columns, in_output.ndim).wrap(
                            in_output
                        )
                    in_outputs.append(in_output)
            in_output_list_ready.append(in_outputs)
        if checks.is_numba_func(custom_func):
            in_output_list_ready = [
                to_typed_list(in_outputs) for in_outputs in in_output_list_ready
            ]

        def _use_raw(_raw):
            _output_list, _param_map, _n_input_cols, _other_list = _raw
            idxs = np.array([_param_map.index(param_tuple) for param_tuple in zip(*param_list)])
            _output_list = [
                np.hstack([o[:, idx * _n_input_cols : (idx + 1) * _n_input_cols] for idx in idxs])
                for o in _output_list
            ]
            return _output_list, _param_map, _n_input_cols, _other_list

        if use_raw is not None:
            output_list, param_map, n_input_cols, other_list = _use_raw(use_raw)
        else:
            func_args = args
            func_kwargs = dict(kwargs)
            if pass_input_shape:
                func_kwargs["input_shape"] = input_shape_ready
            if pass_wrapper:
                func_kwargs["wrapper"] = wrapper_ready
            if pass_param_index:
                func_kwargs["param_index"] = param_index
            if pass_final_index:
                func_kwargs["final_index"] = final_index
            if pass_single_comb:
                func_kwargs["single_comb"] = single_comb
            if pass_per_column:
                func_kwargs["per_column"] = per_column

            if has_templates(func_args) or has_templates(func_kwargs):
                template_context = merge_dicts(
                    broadcast_named_args,
                    dict(
                        input_shape=input_shape_ready,
                        wrapper=wrapper_ready,
                        **dict(zip(input_names, input_list_ready)),
                        **dict(zip(in_output_names, in_output_list_ready)),
                        **dict(zip(param_names, param_list_ready)),
                        pre_sub_args=func_args,
                        pre_sub_kwargs=func_kwargs,
                    ),
                    template_context,
                )
                func_args = substitute_templates(
                    func_args, template_context, eval_id="custom_func_args"
                )
                func_kwargs = substitute_templates(
                    func_kwargs, template_context, eval_id="custom_func_kwargs"
                )

            if checks.is_numba_func(custom_func):
                func_args += tuple(func_kwargs.values())
                func_kwargs = {}
            if pass_packed:
                outputs = custom_func(
                    tuple(input_list_ready),
                    tuple(in_output_list_ready),
                    tuple(param_list_ready),
                    *func_args,
                    **func_kwargs,
                )
            else:
                outputs = custom_func(
                    *input_list_ready,
                    *in_output_list_ready,
                    *param_list_ready,
                    *func_args,
                    **func_kwargs,
                )

            if isinstance(return_raw, str):
                if return_raw.lower() == "outputs":
                    if use_run_unique and not silence_warnings:
                        warn(
                            "Raw outputs are produced by unique parameter combinations when run_unique=True"
                        )
                    return outputs
                else:
                    raise ValueError(f"Invalid return_raw: '{return_raw}'")

            if kwargs.get("return_cache", False):
                if use_run_unique and not silence_warnings:
                    warn("Cache is produced by unique parameter combinations when run_unique=True")
                return outputs

            if outputs is None:
                output_list = []
                other_list = []
            else:
                if isinstance(outputs, (tuple, list, List)):
                    output_list = list(outputs)
                else:
                    output_list = [outputs]
                if len(output_list) > num_ret_outputs:
                    other_list = output_list[num_ret_outputs:]
                    if use_run_unique and not silence_warnings:
                        warn(
                            "Additional output objects are produced by unique parameter combinations "
                            "when run_unique=True"
                        )
                else:
                    other_list = []
                output_list = output_list[:num_ret_outputs]
            if len(output_list) != num_ret_outputs:
                raise ValueError("Number of returned outputs other than expected")
            output_list = list(map(lambda x: reshaping.to_2d_array(x), output_list))
            output_list = in_output_list + output_list
            param_map = list(zip(*param_list_unique))
            output_shape = output_list[0].shape
            for output in output_list:
                if output.shape != output_shape:
                    raise ValueError("All outputs must have the same shape")
            if per_column:
                n_input_cols = output_shape[1]
            else:
                n_input_cols = output_shape[1] // n_unique_param_values
            if input_shape_2d is not None:
                if n_input_cols != input_shape_2d[1]:
                    if per_column:
                        raise ValueError(
                            "All outputs must have the same number of columns as inputs when per_column=True"
                        )
                    else:
                        raise ValueError(
                            "All outputs must have the same number of columns as there "
                            "are input columns times parameter combinations"
                        )
            raw = output_list, param_map, n_input_cols, other_list
            if return_raw:
                if use_run_unique and not silence_warnings:
                    warn(
                        "Raw outputs are produced by unique parameter combinations when run_unique=True"
                    )
                return raw
            if use_run_unique:
                output_list, param_map, n_input_cols, other_list = _use_raw(raw)

        if input_shape is None:
            if n_input_cols == 1:
                input_shape = (output_list[0].shape[0],)
            else:
                input_shape = (output_list[0].shape[0], n_input_cols)
        if input_index is None:
            input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
        if input_columns is None:
            input_columns = pd.RangeIndex(
                start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1
            )

        if len(param_list) > 0:
            if final_index is None:
                build_columns_meta = build_columns(
                    param_list,
                    input_columns,
                    level_names=level_names,
                    hide_levels=hide_levels,
                    single_value=single_value,
                    param_settings=param_settings,
                    per_column=per_column,
                    **build_col_kwargs,
                )
                rep_param_indexes = build_columns_meta["rep_param_indexes"]
                final_index = build_columns_meta["final_index"]

            input_mapper = None
            if len(input_list) > 0:
                if per_column:
                    input_mapper = np.arange(len(input_columns))
                else:
                    input_mapper = np.tile(np.arange(len(input_columns)), n_param_values)
            mapper_list = [rep_param_indexes[i] for i in range(len(param_list))]
        else:
            final_index = input_columns
            input_mapper = None
            mapper_list = []

        new_ndim = len(input_shape) if output_list[0].shape[1] == 1 else output_list[0].ndim
        if new_ndim == 1 and not single_comb:
            new_ndim = 2
        wrapper = ArrayWrapper(input_index, final_index, new_ndim, **wrapper_kwargs)

        return (
            wrapper,
            input_list,
            input_mapper,
            output_list[: len(in_output_list)],
            output_list[len(in_output_list) :],
            param_list,
            mapper_list,
            other_list,
        )

    @classmethod
    def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> tp.IFRunOutput:
        raise NotImplementedError

    @classmethod
    def run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> tp.IFRunOutput:
        """Execute the indicator run operation.

        This method delegates to the internal `_run` method.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            IFRunOutput: Result of running the indicator.
        """
        return cls._run(*args, **kwargs)

    @classmethod
    def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> tp.IFRunCombsOutput:
        raise NotImplementedError

    @classmethod
    def run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> tp.IFRunCombsOutput:
        """Execute the indicator run combinations operation.

        This method delegates to the internal `_run_combs` method.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            IFRunCombsOutput: Result of running the indicator combinations.
        """
        return cls._run_combs(*args, **kwargs)

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[IndicatorBaseT],
        *objs: tp.MaybeSequence[IndicatorBaseT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> IndicatorBaseT:
        """Stack multiple `IndicatorBase` instances along rows.

        This method uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to combine
        the wrappers and stack input, in_output, and output arrays from each indicator.

        All objects to be merged must have the same columns for parameters.

        Args:
            *objs (MaybeSequence[IndicatorBase]): (Additional) indicator instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `IndicatorBase` through
                `IndicatorBase.resolve_row_stack_kwargs` and `IndicatorBase.resolve_stack_kwargs`.

        Returns:
            IndicatorBase: New instance with combined data from the provided indicators.
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
            if not checks.is_instance_of(obj, IndicatorBase):
                raise TypeError("Each object to be merged must be an instance of IndicatorBase")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs], stack_columns=False, **wrapper_kwargs
            )

        if "input_list" not in kwargs:
            new_input_list = []
            for input_name in cls.input_names:
                new_input_list.append(
                    row_stack_arrays([getattr(obj, f"_{input_name}") for obj in objs])
                )
            kwargs["input_list"] = new_input_list
        if "in_output_list" not in kwargs:
            new_in_output_list = []
            for in_output_name in cls.in_output_names:
                new_in_output_list.append(
                    row_stack_arrays([getattr(obj, f"_{in_output_name}") for obj in objs])
                )
            kwargs["in_output_list"] = new_in_output_list
        if "output_list" not in kwargs:
            new_output_list = []
            for output_name in cls.output_names:
                new_output_list.append(
                    row_stack_arrays([getattr(obj, f"_{output_name}") for obj in objs])
                )
            kwargs["output_list"] = new_output_list

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[IndicatorBaseT],
        *objs: tp.MaybeSequence[IndicatorBaseT],
        wrapper_kwargs: tp.KwargsLike = None,
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> IndicatorBaseT:
        """Stack multiple `IndicatorBase` instances along columns for parameters.

        This method uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to combine
        the wrappers and stack input, in_output, output arrays, parameter lists, and mapper lists.

        All objects to be merged must share the same index.

        Args:
            *objs (MaybeSequence[IndicatorBase]): (Additional) indicator instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            reindex_kwargs (KwargsLike): Keyword arguments for `pd.DataFrame.reindex`.
            **kwargs: Keyword arguments for `IndicatorBase` through
                `IndicatorBase.resolve_column_stack_kwargs` and `IndicatorBase.resolve_stack_kwargs`.

        Returns:
            IndicatorBase: New instance with combined data from the provided indicators.
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
            if not checks.is_instance_of(obj, IndicatorBase):
                raise TypeError("Each object to be merged must be an instance of IndicatorBase")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        if "input_mapper" not in kwargs:
            stack_input_mapper_objs = True
            for obj in objs:
                if getattr(obj, "_input_mapper", None) is None:
                    stack_input_mapper_objs = False
                    break
            if stack_input_mapper_objs:
                kwargs["input_mapper"] = np.concatenate(
                    [obj._input_mapper for obj in objs]
                )
        if "in_output_list" not in kwargs:
            new_in_output_list = []
            for in_output_name in cls.in_output_names:
                new_in_output_list.append(
                    column_stack_arrays([getattr(obj, f"_{in_output_name}") for obj in objs])
                )
            kwargs["in_output_list"] = new_in_output_list
        if "output_list" not in kwargs:
            new_output_list = []
            for output_name in cls.output_names:
                new_output_list.append(
                    column_stack_arrays([getattr(obj, f"_{output_name}") for obj in objs])
                )
            kwargs["output_list"] = new_output_list
        if "param_list" not in kwargs:
            new_param_list = []
            for param_name in cls.param_names:
                param_objs = []
                for obj in objs:
                    param_objs.extend(getattr(obj, f"_{param_name}_list"))
                new_param_list.append(param_objs)
            kwargs["param_list"] = new_param_list
        if "mapper_list" not in kwargs:
            new_mapper_list = []
            for param_name in cls.param_names:
                new_mapper = None
                for obj in objs:
                    obj_mapper = getattr(obj, f"_{param_name}_mapper")
                    if new_mapper is None:
                        new_mapper = obj_mapper
                    else:
                        new_mapper = new_mapper.append(obj_mapper)
                new_mapper_list.append(new_mapper)
            kwargs["mapper_list"] = new_mapper_list

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @property
    def _tuple_mapper(self) -> tp.Optional[tp.MultiIndex]:
        if len(self.param_names) <= 1:
            return None
        return pd.MultiIndex.from_arrays(
            [getattr(self, f"_{name}_mapper") for name in self.param_names]
        )

    @property
    def _param_mapper(self) -> tp.Optional[tp.Index]:
        if len(self.param_names) == 0:
            return None
        if len(self.param_names) == 1:
            return getattr(self, f"_{self.param_names[0]}_mapper")
        return self._tuple_mapper

    @property
    def _visible_param_mapper(self) -> tp.Optional[tp.Index]:
        if len(self.param_names) == 0:
            return None
        if len(self.param_names) == 1:
            mapper = getattr(self, f"_{self.param_names[0]}_mapper")
            if mapper.name is None:
                return None
            if mapper.name not in self.wrapper.columns.names:
                return None
            return mapper
        mapper = self._tuple_mapper
        visible_indexes = []
        for i, name in enumerate(mapper.names):
            if name is not None and name in self.wrapper.columns.names:
                visible_indexes.append(mapper.get_level_values(i))
        if len(visible_indexes) == 0:
            return None
        if len(visible_indexes) == 1:
            return visible_indexes[0]
        return pd.MultiIndex.from_arrays(visible_indexes)

    def indexing_func(
        self: IndicatorBaseT, *args, wrapper_meta: tp.DictLike = None, **kwargs
    ) -> IndicatorBaseT:
        """Perform indexing on an `IndicatorBase` instance.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func`.
            wrapper_meta (DictLike): Metadata from the indexing operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func`.

        Returns:
            IndicatorBase: New indicator instance with updated indexing.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        row_idxs = wrapper_meta["row_idxs"]
        col_idxs = wrapper_meta["col_idxs"]
        rows_changed = wrapper_meta["rows_changed"]
        columns_changed = wrapper_meta["columns_changed"]
        if not isinstance(row_idxs, slice):
            row_idxs = reshaping.to_1d_array(row_idxs)
        if not isinstance(col_idxs, slice):
            col_idxs = reshaping.to_1d_array(col_idxs)

        input_mapper = getattr(self, "_input_mapper", None)
        if input_mapper is not None:
            if columns_changed:
                input_mapper = input_mapper[col_idxs]
        input_list = []
        for input_name in self.input_names:
            new_input = ArrayWrapper.select_from_flex_array(
                getattr(self, f"_{input_name}"),
                row_idxs=row_idxs,
                col_idxs=col_idxs if input_mapper is None else None,
                rows_changed=rows_changed,
                columns_changed=columns_changed if input_mapper is None else False,
            )
            input_list.append(new_input)
        in_output_list = []
        for in_output_name in self.in_output_names:
            new_in_output = ArrayWrapper.select_from_flex_array(
                getattr(self, f"_{in_output_name}"),
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
            in_output_list.append(new_in_output)
        output_list = []
        for output_name in self.output_names:
            new_output = ArrayWrapper.select_from_flex_array(
                getattr(self, f"_{output_name}"),
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
            output_list.append(new_output)
        param_list = []
        for param_name in self.param_names:
            param_list.append(getattr(self, f"_{param_name}_list"))
        mapper_list = []
        for param_name in self.param_names:
            mapper_list.append(getattr(self, f"_{param_name}_mapper")[col_idxs])

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list,
        )

    @class_property
    def short_name(cls_or_self) -> str:
        """Short name of the indicator.

        Returns:
            str: Short name of the indicator.
        """
        return cls_or_self._short_name

    @class_property
    def input_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the input arrays.

        Returns:
            Tuple[str, ...]: Tuple of input names.
        """
        return cls_or_self._input_names

    @class_property
    def param_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the parameters.

        Returns:
            Tuple[str, ...]: Tuple of parameter names.
        """
        return cls_or_self._param_names

    @class_property
    def in_output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the in-place output arrays.

        Returns:
            Tuple[str, ...]: Tuple of in-place output names.
        """
        return cls_or_self._in_output_names

    @class_property
    def output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the regular output arrays.

        Returns:
            Tuple[str, ...]: Tuple of output names.
        """
        return cls_or_self._output_names

    @class_property
    def lazy_output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the lazy output arrays.

        Returns:
            Tuple[str, ...]: Tuple of lazy output names.
        """
        return cls_or_self._lazy_output_names

    @class_property
    def output_flags(cls_or_self) -> tp.Kwargs:
        """Dictionary of output flags.

        Returns:
            Kwargs: Dictionary of output flags.
        """
        return cls_or_self._output_flags

    @class_property
    def param_defaults(cls_or_self) -> tp.Dict[str, tp.Any]:
        """Parameter defaults extracted from the signature of `IndicatorBase.run`.

        Returns:
            Dict[str, Any]: Dictionary of parameter defaults.
        """
        func_kwargs = get_func_kwargs(cls_or_self.run)
        out = {}
        for k, v in func_kwargs.items():
            if k in cls_or_self.param_names:
                if isinstance(v, Default):
                    out[k] = v.value
                else:
                    out[k] = v
        return out

    @property
    def level_names(self) -> tp.Tuple[str, ...]:
        """List of level names corresponding to each parameter.

        Returns:
            Tuple[str, ...]: Tuple of level names corresponding to each parameter.
        """
        return self._level_names

    def unpack(self) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Return indicator outputs.

        If there is only one output, return it directly; otherwise, return a tuple of outputs.

        Returns:
            MaybeTuple[SeriesFrame]: Output or outputs of the indicator.
        """
        out = tuple([getattr(self, name) for name in self.output_names])
        if len(out) == 1:
            out = out[0]
        return out

    def to_dict(self, include_all: bool = True) -> tp.Dict[str, tp.SeriesFrame]:
        """Return the indicator outputs as a dictionary.

        Args:
            include_all (bool): Flag to determine whether to include all outputs (regular, in-place, and lazy).

        Returns:
            Dict[str, SeriesFrame]: Mapping of output names to their corresponding data.
        """
        if include_all:
            output_names = self.output_names + self.in_output_names + self.lazy_output_names
        else:
            output_names = self.output_names
        return {name: getattr(self, name) for name in output_names}

    def to_frame(self, include_all: bool = True) -> tp.Frame:
        """Return the indicator outputs as a DataFrame.

        Args:
            include_all (bool): Flag to determine whether to include all outputs (regular, in-place, and lazy).

        Returns:
            Frame: DataFrame combining the outputs with output names as column keys.
        """
        out = self.to_dict(include_all=include_all)
        return pd.concat(list(out.values()), axis=1, keys=pd.Index(list(out.keys()), name="output"))

    def get(self, key: tp.Optional[tp.Hashable] = None) -> tp.Optional[tp.SeriesFrame]:
        """Return a time series output based on a key.

        Args:
            key (Optional[Hashable]): Key corresponding to a specific output.

                If None, the main output is returned.

        Returns:
            Optional[SeriesFrame]: Requested time series or the main output if no key is provided.
        """
        if key is None:
            return self.main_output
        return getattr(self, key)

    def dropna(self: IndicatorBaseT, include_all: bool = True, **kwargs) -> IndicatorBaseT:
        """Drop missing values from the indicator outputs.

        Args:
            include_all (bool): Flag to determine whether to include all outputs (regular, in-place, and lazy).
            **kwargs: Keyword arguments for `pd.Series.dropna` or `pd.DataFrame.dropna`.

        Returns:
            IndicatorBase: New indicator instance with missing values dropped.
        """
        df = self.to_frame(include_all=include_all)
        new_df = df.dropna(**kwargs)
        if new_df.index.equals(df.index):
            return self
        return self.loc[new_df.index]

    def rename(self: IndicatorBaseT, short_name: str) -> IndicatorBaseT:
        """Replace the short name of the indicator.

        Args:
            short_name (str): New short name for the indicator.

        Returns:
            IndicatorBase: New indicator instance with the updated short name.
        """
        new_level_names = ()
        for level_name in self.level_names:
            if level_name.startswith(self.short_name + "_"):
                level_name = level_name.replace(self.short_name, short_name, 1)
            new_level_names += (level_name,)
        new_mapper_list = []
        for i, param_name in enumerate(self.param_names):
            mapper = getattr(self, f"_{param_name}_mapper")
            new_mapper_list.append(mapper.rename(new_level_names[i]))
        new_columns = self.wrapper.columns
        for i, name in enumerate(self.wrapper.columns.names):
            if name in self.level_names:
                new_columns = new_columns.rename(
                    {name: new_level_names[self.level_names.index(name)]}
                )
        new_wrapper = self.wrapper.replace(columns=new_columns)
        return self.replace(wrapper=new_wrapper, mapper_list=new_mapper_list, short_name=short_name)

    def rename_levels(
        self: IndicatorBaseT,
        mapper: tp.MaybeMappingSequence[tp.Level],
        **kwargs,
    ) -> IndicatorBaseT:
        new_self = Analyzable.rename_levels(self, mapper, **kwargs)
        old_column_names = self.wrapper.columns.names
        new_column_names = new_self.wrapper.columns.names
        new_level_names = ()
        for level_name in new_self.level_names:
            if level_name in old_column_names:
                level_name = new_column_names[old_column_names.index(level_name)]
            new_level_names += (level_name,)
        new_mapper_list = []
        for i, param_name in enumerate(new_self.param_names):
            mapper = getattr(new_self, f"_{param_name}_mapper")
            new_mapper_list.append(mapper.rename(new_level_names[i]))
        return new_self.replace(mapper_list=new_mapper_list, level_names=new_level_names)

    # ############# Iteration ############# #

    def items(
        self,
        group_by: tp.GroupByLike = "params",
        apply_group_by: bool = False,
        keep_2d: bool = False,
        key_as_index: bool = False,
    ) -> tp.Items:
        """Iterate over columns or groups.

        Iterates over columns or groups based on the specified grouping criteria. When grouping is enabled via
        `vectorbtpro.base.wrapping.Wrapping.group_select`, groups are returned instead of individual columns.
        The `group_by` parameter can be provided as a column name present in the wrapper, the string "all_params"
        for full parameter mapping, "params" for only visible parameters, or as a specific parameter name.

        Args:
            group_by (GroupByLike): Grouping specification.

                If a string, valid options include:

                * a column name in the wrapper,
                * "all_params" for the full parameter mapper,
                * "params" for only visible parameters, or
                * a parameter name.

                See `vectorbtpro.base.grouping.base.Grouper`.
            apply_group_by (bool): If True, applies the grouping to both iteration and the final output.

                If False, `group_by` is used solely as an iteration instruction.
            keep_2d (bool): Whether to maintain the output data in a two-dimensional format.
            key_as_index (bool): Whether to return the yielded key as an index.

        Returns:
            Items: Iterator over key-value pairs representing each column or group.
        """
        if isinstance(group_by, str):
            if group_by not in self.wrapper.columns.names:
                if group_by.lower() == "all_params":
                    group_by = self._param_mapper
                elif group_by.lower() == "params":
                    group_by = self._visible_param_mapper
                elif group_by in self.param_names:
                    group_by = getattr(self, f"_{group_by}_mapper")
        elif isinstance(group_by, (tuple, list)):
            new_group_by = []
            for g in group_by:
                if isinstance(g, str):
                    if g not in self.wrapper.columns.names:
                        if g in self.param_names:
                            g = getattr(self, f"_{g}_mapper")
                new_group_by.append(g)
            group_by = type(group_by)(new_group_by)
        for k, v in Analyzable.items(
            self,
            group_by=group_by,
            apply_group_by=apply_group_by,
            keep_2d=keep_2d,
            key_as_index=key_as_index,
        ):
            yield k, v

    # ############# Documentation ############# #

    @classmethod
    def clone_docstring(cls, another_cls: tp.Type) -> None:
        """Clone the docstring from another class.

        Args:
            another_cls (Type): Class from which to clone the docstring.

        Returns:
            None
        """
        cls.__doc__ = another_cls.__doc__

    @classmethod
    def clone_method(cls, method: tp.Callable, target_name: tp.Optional[str] = None) -> None:
        """Clone a method to the class.

        Args:
            method (Callable): Method to clone.
            target_name (Optional[str]): Target name for the cloned method.

        Returns:
            None
        """
        from functools import update_wrapper
        from types import FunctionType

        if target_name is None:
            target_name = method.__name__
        new_method = FunctionType(
            method.__code__,
            method.__globals__,
            method.__name__,
            method.__defaults__,
            method.__closure__,
        )
        update_wrapper(new_method, method)
        setattr(cls, target_name, new_method)


class IndicatorFactory(Configured):
    """Factory for creating new indicators.

    Initialize `IndicatorFactory` to create a skeleton. Then, use a class method such as
    `IndicatorFactory.with_custom_func` to bind a calculation function to the skeleton.

    Args:
        class_name (Optional[str]): Name for the created indicator class.
        class_docstring (Optional[str]): Docstring for the created indicator class.
        module_name (Optional[str]): Module name to bind the generated class.
        short_name (Optional[str]): Concise name for the indicator.

            Defaults to lower-case `class_name`.
        prepend_name (bool): Whether to prepend `short_name` to each parameter level.
        input_names (Optional[Sequence[str]]): List of input names.
        param_names (Optional[Sequence[str]]): List of parameter names.
        in_output_names (Optional[Sequence[str]]): List of in-place output names.

            An in-place output is modified in place rather than being returned. Advantages include:

            * It does not need to be returned.
            * It can be passed between functions as easily as inputs.
            * It can use pre-allocated memory to save memory.
            * If data or a default value is not provided, it is created empty to avoid occupying memory.
        output_names (Optional[Sequence[str]]): List of output names.
        output_flags (KwargsLike): Dictionary of flags for in-place and regular outputs.
        lazy_outputs (KwargsLike): Dictionary of user-defined functions bound to the indicator class and
            wrapped with `property` if not already wrapped.
        attr_settings (KwargsLike): Settings for attributes, where each key maps to a dictionary of options.

            Attributes include `input_names`, `in_output_names`, `output_names`, and `lazy_outputs`.

            The following keys are accepted:

            * `dtype`: Data type used to determine which methods to generate around this attribute.
                Set to None to disable. Default is `float_`. Can be a `namedtuple` type acting as an
                enumerated type, or any other mapping. It will then create a property with the suffix `readable`
                that contains data in string format.
            * `enum_unkval`: Value to be considered as unknown (applies only to enumerated data types).
            * `make_cacheable`: Whether to make the property cacheable (applies only to inputs).
            * `doc`: Documentation string for the attribute.
        metrics (KwargsLike): Metrics supported by `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

            If a dictionary is provided, it will be converted to a `vectorbtpro.utils.config.Config`.
        stats_defaults (Union[None, Callable, Kwargs]): Defaults for
            `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

            If a dictionary is provided, it will be converted into a property.
        subplots (KwargsLike): Subplots supported by `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

            If a dictionary is provided, it will be converted to a `vectorbtpro.utils.config.Config`.
        plots_defaults (Union[None, Callable, Kwargs]): Defaults for
            `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

            If a dictionary is provided, it will be converted into a property.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! note
        The `__init__` method is not used for running the indicator; use `run` instead.
        Indexing requires a clean `__init__` method to create a new indicator object with re-indexed attributes.
    """

    def __init__(
        self,
        class_name: tp.Optional[str] = None,
        class_docstring: tp.Optional[str] = None,
        module_name: tp.Optional[str] = __name__,
        short_name: tp.Optional[str] = None,
        prepend_name: bool = True,
        input_names: tp.Optional[tp.Sequence[str]] = None,
        param_names: tp.Optional[tp.Sequence[str]] = None,
        in_output_names: tp.Optional[tp.Sequence[str]] = None,
        output_names: tp.Optional[tp.Sequence[str]] = None,
        output_flags: tp.KwargsLike = None,
        lazy_outputs: tp.KwargsLike = None,
        attr_settings: tp.KwargsLike = None,
        metrics: tp.KwargsLike = None,
        stats_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None,
        subplots: tp.KwargsLike = None,
        plots_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            class_name=class_name,
            class_docstring=class_docstring,
            module_name=module_name,
            short_name=short_name,
            prepend_name=prepend_name,
            input_names=input_names,
            param_names=param_names,
            in_output_names=in_output_names,
            output_names=output_names,
            output_flags=output_flags,
            lazy_outputs=lazy_outputs,
            attr_settings=attr_settings,
            metrics=metrics,
            stats_defaults=stats_defaults,
            subplots=subplots,
            plots_defaults=plots_defaults,
            **kwargs,
        )

        if class_name is None:
            class_name = "Indicator"
        checks.assert_instance_of(class_name, str)
        if class_docstring is None:
            class_docstring = ""
        checks.assert_instance_of(class_docstring, str)
        if module_name is not None:
            checks.assert_instance_of(module_name, str)
        if short_name is None:
            if class_name == "Indicator":
                short_name = "custom"
            else:
                short_name = class_name.lower()
        checks.assert_instance_of(short_name, str)
        checks.assert_instance_of(prepend_name, bool)
        if input_names is None:
            input_names = []
        else:
            checks.assert_sequence(input_names)
            input_names = list(input_names)
        if param_names is None:
            param_names = []
        else:
            checks.assert_sequence(param_names)
            param_names = list(param_names)
        if in_output_names is None:
            in_output_names = []
        else:
            checks.assert_sequence(in_output_names)
            in_output_names = list(in_output_names)
        if output_names is None:
            output_names = []
        else:
            checks.assert_sequence(output_names)
            output_names = list(output_names)
        all_output_names = in_output_names + output_names
        if len(all_output_names) == 0:
            raise ValueError("Must have at least one in-place or regular output")
        if len(set.intersection(set(input_names), set(in_output_names), set(output_names))) > 0:
            raise ValueError("Inputs, in-place outputs, and parameters must all have unique names")
        if output_flags is None:
            output_flags = {}
        checks.assert_instance_of(output_flags, dict)
        if len(output_flags) > 0:
            checks.assert_dict_valid(output_flags, all_output_names)
        if lazy_outputs is None:
            lazy_outputs = {}
        checks.assert_instance_of(lazy_outputs, dict)
        if attr_settings is None:
            attr_settings = {}
        checks.assert_instance_of(attr_settings, dict)
        all_attr_names = input_names + all_output_names + list(lazy_outputs.keys())
        if len(attr_settings) > 0:
            checks.assert_dict_valid(
                attr_settings,
                [
                    all_attr_names,
                    ["dtype", "enum_unkval", "make_cacheable", "doc"],
                ],
            )

        ParamIndexer = build_param_indexer(
            param_names + (["tuple"] if len(param_names) > 1 else []),
            module_name=module_name,
        )
        Indicator = type(class_name, (IndicatorBase, ParamIndexer), {})
        Indicator.__doc__ = class_docstring
        if module_name is not None:
            Indicator.__module__ = module_name

        Indicator._short_name = short_name
        Indicator._input_names = tuple(input_names)
        Indicator._param_names = tuple(param_names)
        Indicator._in_output_names = tuple(in_output_names)
        Indicator._output_names = tuple(output_names)
        Indicator._lazy_output_names = tuple(lazy_outputs.keys())
        Indicator._output_flags = output_flags

        for param_name in param_names:

            def param_list_prop(self, _param_name=param_name) -> tp.List[tp.ParamValue]:
                return getattr(self, f"_{_param_name}_list")

            param_list_prop.__doc__ = f"List of values for the `{param_name}` parameter."
            param_list_prop.__name__ = f"{param_name}_list"
            param_list_prop.__module__ = Indicator.__module__
            param_list_prop.__qualname__ = f"{Indicator.__name__}.{param_list_prop.__name__}"
            setattr(Indicator, f"{param_name}_list", property(param_list_prop))

        for input_name in input_names:
            _attr_settings = attr_settings.get(input_name, {})
            make_cacheable = _attr_settings.get("make_cacheable", False)

            def input_prop(self, _input_name: str = input_name) -> tp.SeriesFrame:
                old_input = reshaping.to_2d_array(getattr(self, "_" + _input_name))
                input_mapper = self._input_mapper
                if input_mapper is None:
                    return self.wrapper.wrap(old_input)
                return self.wrapper.wrap(old_input[:, input_mapper])

            input_prop.__doc__ = f"Input array for `{input_name}`."
            input_prop.__name__ = input_name
            input_prop.__module__ = Indicator.__module__
            input_prop.__qualname__ = f"{Indicator.__name__}.{input_prop.__name__}"
            if make_cacheable:
                setattr(Indicator, input_name, cacheable_property(input_prop))
            else:
                setattr(Indicator, input_name, property(input_prop))

        for output_name in all_output_names:

            def output_prop(self, _output_name: str = output_name) -> tp.SeriesFrame:
                return self.wrapper.wrap(getattr(self, "_" + _output_name))

            if output_name in in_output_names:
                output_prop.__doc__ = f"In-place output array for `{output_name}`."
            else:
                output_prop.__doc__ = f"Output array for `{output_name}`."
            output_prop.__name__ = output_name
            output_prop.__module__ = Indicator.__module__
            output_prop.__qualname__ = f"{Indicator.__name__}.{output_prop.__name__}"
            if output_name in output_flags:
                _output_flags = output_flags[output_name]
                if isinstance(_output_flags, (tuple, list)):
                    _output_flags = ", ".join(_output_flags)
                output_prop.__doc__ += "\n\n" + _output_flags
            setattr(Indicator, output_name, property(output_prop))

        for prop_name, prop in lazy_outputs.items():
            prop.__name__ = prop_name
            prop.__module__ = Indicator.__module__
            prop.__qualname__ = f"{Indicator.__name__}.{prop.__name__}"
            if prop.__doc__ is None:
                prop.__doc__ = "Custom property."
            if not isinstance(prop, property):
                prop = property(prop)
            setattr(Indicator, prop_name, prop)

        def assign_combine_method(
            func_name: str,
            combine_func: tp.Callable,
            def_kwargs: tp.Kwargs,
            attr_name: str,
            docstring: str,
        ) -> None:
            def combine_method(
                self: IndicatorBaseT,
                other: tp.MaybeTupleList[tp.Union[IndicatorBaseT, tp.ArrayLike, BaseAccessor]],
                level_name: tp.Optional[str] = None,
                allow_multiple: bool = True,
                _prepend_name: bool = prepend_name,
                **kwargs,
            ) -> tp.SeriesFrame:
                if allow_multiple and isinstance(other, (tuple, list)):
                    other = list(other)
                    for i in range(len(other)):
                        if isinstance(other[i], IndicatorBase):
                            other[i] = getattr(other[i], attr_name)
                        elif isinstance(other[i], str):
                            other[i] = getattr(self, other[i])
                else:
                    if isinstance(other, IndicatorBase):
                        other = getattr(other, attr_name)
                    elif isinstance(other, str):
                        other = getattr(self, other)
                if level_name is None:
                    if _prepend_name:
                        if attr_name == self.short_name:
                            level_name = f"{self.short_name}_{func_name}"
                        else:
                            level_name = f"{self.short_name}_{attr_name}_{func_name}"
                    else:
                        level_name = f"{attr_name}_{func_name}"
                out = combine_objs(
                    getattr(self, attr_name),
                    other,
                    combine_func,
                    level_name=level_name,
                    allow_multiple=allow_multiple,
                    **merge_dicts(def_kwargs, kwargs),
                )
                return out

            combine_method.__name__ = f"{attr_name}_{func_name}"
            combine_method.__module__ = Indicator.__module__
            combine_method.__qualname__ = f"{Indicator.__name__}.{combine_method.__name__}"
            combine_method.__doc__ = docstring
            setattr(Indicator, f"{attr_name}_{func_name}", combine_method)

        for attr_name in all_attr_names:
            _attr_settings = attr_settings.get(attr_name, {})
            dtype = _attr_settings.get("dtype", float_)
            enum_unkval = _attr_settings.get("enum_unkval", -1)

            if checks.is_mapping_like(dtype):

                def attr_readable(
                    self,
                    _attr_name: str = attr_name,
                    _mapping: tp.MappingLike = dtype,
                    _enum_unkval: tp.Any = enum_unkval,
                ) -> tp.SeriesFrame:
                    return (
                        getattr(self, _attr_name)
                        .vbt(mapping=_mapping)
                        .apply_mapping(enum_unkval=_enum_unkval)
                    )

                attr_readable.__name__ = f"{attr_name}_readable"
                attr_readable.__module__ = Indicator.__module__
                attr_readable.__qualname__ = f"{Indicator.__name__}.{attr_readable.__name__}"
                attr_readable.__doc__ = inspect.cleandoc(
                    f"""
                    `{attr_name}` in a human-readable format based on the mapping below:

                    ```python
                    {prettify(to_value_mapping(dtype, enum_unkval=enum_unkval), indent=5, indent_head=False)}
                    ```
                    """
                )

                setattr(Indicator, f"{attr_name}_readable", property(attr_readable))

                def attr_stats(
                    self,
                    *args,
                    _attr_name: str = attr_name,
                    _mapping: tp.MappingLike = dtype,
                    **kwargs,
                ) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).stats(*args, **kwargs)

                attr_stats.__name__ = f"{attr_name}_stats"
                attr_stats.__module__ = Indicator.__module__
                attr_stats.__qualname__ = f"{Indicator.__name__}.{attr_stats.__name__}"
                attr_stats.__doc__ = inspect.cleandoc(
                    f"""
                    Compute statistics for `{attr_name}` based on the mapping below:

                    ```python
                    {prettify(to_value_mapping(dtype), indent=5, indent_head=False)}
                    ```

                    Args:
                        *args: Positional arguments for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.
                        **kwargs: Keyword arguments for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                    Returns:
                        SeriesFrame: Computed statistics for `{attr_name}`.
                    """
                )
                setattr(Indicator, f"{attr_name}_stats", attr_stats)

            elif np.issubdtype(dtype, np.number):
                func_info = [
                    ("above", np.greater, dict()),
                    ("below", np.less, dict()),
                    ("equal", np.equal, dict()),
                    (
                        "crossed_above",
                        lambda x, y, wait=0, dropna=False: jit_reg.resolve(
                            generic_nb.crossed_above_nb
                        )(
                            x,
                            y,
                            wait=wait,
                            dropna=dropna,
                        ),
                        dict(to_2d=True),
                    ),
                    (
                        "crossed_below",
                        lambda x, y, wait=0, dropna=False: jit_reg.resolve(
                            generic_nb.crossed_above_nb
                        )(
                            y,
                            x,
                            wait=wait,
                            dropna=dropna,
                        ),
                        dict(to_2d=True),
                    ),
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = inspect.cleandoc(
                        f"""
                        Return a boolean array indicating where `{attr_name}` is {func_name} compared to `other`.

                        Args:
                            other (MaybeTupleList[Union[IndicatorBase, ArrayLike, BaseAccessor]]):
                                Indicator, array, or accessor to compare.
                            level_name (Optional[str]): Output level name.

                                If not provided, a name is auto-generated.
                            allow_multiple (bool): Flag indicating whether multiple comparisons are permitted.
                            **kwargs: Keyword arguments for `vectorbtpro.indicators.factory.combine_objs`.

                        Returns:
                            SeriesFrame: Resulting boolean array.
                        """
                    )
                    assign_combine_method(
                        func_name, np_func, def_kwargs, attr_name, method_docstring
                    )

                def attr_stats(
                    self, *args, _attr_name: str = attr_name, **kwargs
                ) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.stats(*args, **kwargs)

                attr_stats.__name__ = f"{attr_name}_stats"
                attr_stats.__module__ = Indicator.__module__
                attr_stats.__qualname__ = f"{Indicator.__name__}.{attr_stats.__name__}"
                attr_stats.__doc__ = inspect.cleandoc(
                    f"""
                    Compute generic statistics for `{attr_name}`.

                    Args:
                        *args: Positional arguments for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.
                        **kwargs: Keyword arguments for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                    Returns:
                        SeriesFrame: Computed generic statistics for `{attr_name}`.
                    """
                )
                setattr(Indicator, f"{attr_name}_stats", attr_stats)

            elif np.issubdtype(dtype, np.bool_):
                func_info = [
                    ("and", np.logical_and, dict()),
                    ("or", np.logical_or, dict()),
                    ("xor", np.logical_xor, dict()),
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = inspect.cleandoc(
                        f"""
                        Return a boolean array representing the element-wise `{func_name.upper()}`
                        operation between `{attr_name}` and `other`.

                        Args:
                            other (MaybeTupleList[Union[IndicatorBase, ArrayLike, BaseAccessor]]):
                                Indicator, array, or accessor to compare.
                            level_name (Optional[str]): Output level name.

                                If not provided, a name is auto-generated.
                            allow_multiple (bool): Flag indicating whether multiple comparisons are permitted.
                            **kwargs: Keyword arguments for `vectorbtpro.indicators.factory.combine_objs`.

                        Returns:
                            SeriesFrame: Resulting boolean array.
                        """
                    )
                    assign_combine_method(
                        func_name, np_func, def_kwargs, attr_name, method_docstring
                    )

                def attr_stats(
                    self, *args, _attr_name: str = attr_name, **kwargs
                ) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.signals.stats(*args, **kwargs)

                attr_stats.__name__ = f"{attr_name}_stats"
                attr_stats.__module__ = Indicator.__module__
                attr_stats.__qualname__ = f"{Indicator.__name__}.{attr_stats.__name__}"
                attr_stats.__doc__ = inspect.cleandoc(
                    f"""
                    Compute signal statistics for `{attr_name}`.

                    Args:
                        *args: Positional arguments for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.
                        **kwargs: Keyword arguments for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                    Returns:
                        SeriesFrame: Computed signal statistics for `{attr_name}`.
                    """
                )
                setattr(Indicator, f"{attr_name}_stats", attr_stats)

        if metrics is not None:
            if not isinstance(metrics, Config):
                metrics = Config(metrics, options_=dict(copy_kwargs=dict(copy_mode="deep")))
            Indicator._metrics = metrics.copy()

        if stats_defaults is not None:
            if isinstance(stats_defaults, dict):

                def stats_defaults_prop(
                    self, _stats_defaults: tp.Kwargs = stats_defaults
                ) -> tp.Kwargs:
                    return _stats_defaults

            else:

                def stats_defaults_prop(
                    self, _stats_defaults: tp.Kwargs = stats_defaults
                ) -> tp.Kwargs:
                    return stats_defaults(self)

            stats_defaults_prop.__name__ = "stats_defaults"
            stats_defaults_prop.__module__ = Indicator.__module__
            stats_defaults_prop.__qualname__ = (
                f"{Indicator.__name__}.{stats_defaults_prop.__name__}"
            )
            Indicator.stats_defaults = property(stats_defaults_prop)

        if subplots is not None:
            if not isinstance(subplots, Config):
                subplots = Config(subplots, options_=dict(copy_kwargs=dict(copy_mode="deep")))
            Indicator._subplots = subplots.copy()

        if plots_defaults is not None:
            if isinstance(plots_defaults, dict):

                def plots_defaults_prop(
                    self, _plots_defaults: tp.Kwargs = plots_defaults
                ) -> tp.Kwargs:
                    return _plots_defaults

            else:

                def plots_defaults_prop(
                    self, _plots_defaults: tp.Kwargs = plots_defaults
                ) -> tp.Kwargs:
                    return plots_defaults(self)

            plots_defaults_prop.__name__ = "plots_defaults"
            plots_defaults_prop.__module__ = Indicator.__module__
            plots_defaults_prop.__qualname__ = (
                f"{Indicator.__name__}.{plots_defaults_prop.__name__}"
            )
            Indicator.plots_defaults = property(plots_defaults_prop)

        self._class_name = class_name
        self._class_docstring = class_docstring
        self._module_name = module_name
        self._short_name = short_name
        self._prepend_name = prepend_name
        self._input_names = input_names
        self._param_names = param_names
        self._in_output_names = in_output_names
        self._output_names = output_names
        self._output_flags = output_flags
        self._lazy_outputs = lazy_outputs
        self._attr_settings = attr_settings
        self._metrics = metrics
        self._stats_defaults = stats_defaults
        self._subplots = subplots
        self._plots_defaults = plots_defaults

        self._Indicator = Indicator

    @property
    def class_name(self) -> str:
        """Name of the created indicator class.

        Returns:
            str: Name of the created indicator class.
        """
        return self._class_name

    @property
    def class_docstring(self) -> str:
        """Docstring for the created indicator class.

        Returns:
            str: Docstring for the created indicator class.
        """
        return self._class_docstring

    @property
    def module_name(self) -> str:
        """Module name from which the class originates.

        Returns:
            str: Module name from which the class originates.
        """
        return self._module_name

    @property
    def short_name(self) -> str:
        """Concise name for the indicator.

        Returns:
            str: Concise name for the indicator.
        """
        return self._short_name

    @property
    def prepend_name(self) -> bool:
        """Whether `IndicatorFactory.short_name` should be prepended to each parameter level.

        Returns:
            bool: True if `short_name` should be prepended to each parameter level, False otherwise.
        """
        return self._prepend_name

    @property
    def input_names(self) -> tp.List[str]:
        """List of input names.

        Returns:
            List[str]: List of input names.
        """
        return self._input_names

    @property
    def param_names(self) -> tp.List[str]:
        """List of parameter names.

        Returns:
            List[str]: List of parameter names.
        """
        return self._param_names

    @property
    def in_output_names(self) -> tp.List[str]:
        """List of in-place output names.

        Returns:
            List[str]: List of in-place output names.
        """
        return self._in_output_names

    @property
    def output_names(self) -> tp.List[str]:
        """List of output names.

        Returns:
            List[str]: List of output names.
        """
        return self._output_names

    @property
    def output_flags(self) -> tp.Kwargs:
        """Dictionary of flags for in-place and regular outputs.

        Returns:
            Kwargs: Dictionary of flags for in-place and regular outputs.
        """
        return self._output_flags

    @property
    def lazy_outputs(self) -> tp.Kwargs:
        """Dictionary of user-defined functions converted into properties.

        Returns:
            Kwargs: Dictionary of user-defined functions converted into properties.
        """
        return self._lazy_outputs

    @property
    def attr_settings(self) -> tp.Kwargs:
        """Dictionary specifying attribute settings.

        Returns:
            Kwargs: Dictionary specifying attribute settings.
        """
        return self._attr_settings

    @property
    def metrics(self) -> Config:
        """Metrics supported by `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

        Returns:
            Config: Metrics supported by the stats builder.
        """
        return self._metrics

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        return self._stats_defaults

    @property
    def subplots(self) -> Config:
        """Subplots configuration supported by `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

        Returns:
            Config: Subplots configuration supported by the plots builder.
        """
        return self._subplots

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        return self._plots_defaults

    @property
    def Indicator(self) -> tp.Type[IndicatorBase]:
        """Built indicator class.

        Returns:
            Type[IndicatorBase]: Built indicator class.
        """
        return self._Indicator

    # ############# Construction ############# #

    def with_custom_func(
        self,
        custom_func: tp.Callable,
        require_input_shape: bool = False,
        param_settings: tp.KwargsLike = None,
        in_output_settings: tp.KwargsLike = None,
        hide_params: tp.Union[None, bool, tp.Sequence[str]] = None,
        hide_default: bool = True,
        var_args: bool = False,
        keyword_only_args: bool = False,
        **pipeline_kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class based on a custom calculation function.

        This method offers full flexibility compared to `IndicatorFactory.with_apply_func`.
        The caller is responsible for handling caching and concatenation of columns for each
        parameter (e.g., via `vectorbtpro.base.combining.apply_and_concat`). Additionally,
        ensure that each output array has the correct number of columns, which should equal
        the number of input array columns multiplied by the number of parameter combinations.

        Args:
            custom_func (Callable): Function that processes broadcast arrays corresponding to
                `input_names`, in-place output arrays corresponding to `in_output_names`, and
                broadcast parameter arrays corresponding to `param_names`, along with additional
                positional and keyword arguments.

                It returns outputs corresponding to `output_names` and any extra objects to be
                included with the indicator instance.

                It can be Numba-compiled.

                !!! note
                    Each output's shape must match the input shape repeated n times (where n is
                    the number of parameter values) along the column axis.
            require_input_shape (bool): Flag indicating whether the input shape is required.
            param_settings (KwargsLike): Dictionary of parameter settings keyed by name.

                See `IndicatorBase.run_pipeline`.
            in_output_settings (KwargsLike): Dictionary of in-place output settings keyed by name.

                See `IndicatorBase.run_pipeline`.
            hide_params (Union[None, bool, Sequence[str]]): Either a boolean to hide all parameter column
                levels or a list of parameter names for which the column levels should be hidden.
            hide_default (bool): If True, hides column levels for parameters that have default values.
            var_args (bool): Specifies whether run methods should accept variable positional arguments (`*args`).

                Set to True if `custom_func` requires additional positional arguments
                not defined in the configuration.
            keyword_only_args (bool): Specifies whether run methods should enforce keyword-only arguments.

                Set to True to require keyword arguments and avoid misplacement.
            **pipeline_kwargs: Keyword arguments for `IndicatorBase.run_pipeline`.

                These can include default values and references using `vectorbtpro.base.reshaping.Ref`.

        Returns:
            Indicator: Instance of the indicator.

                If `custom_func` returns more objects than defined in `output_names`,
                the additional objects are returned as a tuple alongside the indicator instance.

        Examples:
            Following example produces the same indicator as the `IndicatorFactory.with_apply_func` example.

            ```pycon
            >>> @njit
            >>> def apply_func_nb(i, ts1, ts2, p1, p2, arg1, arg2):
            ...     return ts1 * p1[i] + arg1, ts2 * p2[i] + arg2

            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            ...     return vbt.base.combining.apply_and_concat_multiple_nb(
            ...         len(p1), apply_func_nb, ts1, ts2, p1, p2, arg1, arg2)

            >>> MyInd = vbt.IF(
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).with_custom_func(custom_func, var_args=True, arg2=200)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.o1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.o2
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  206.0  230.0  208.0  240.0
            2020-01-02  212.0  224.0  216.0  232.0
            2020-01-03  218.0  218.0  224.0  224.0
            2020-01-04  224.0  212.0  232.0  216.0
            2020-01-05  230.0  206.0  240.0  208.0
            ```

            The primary difference between `apply_func_nb` in this example and in
            `IndicatorFactory.with_apply_func` is that here the function receives an index for
            the current parameter combination, which can be used for parameter selection.

            Alternatively, you can omit the separate `apply_func_nb` function and implement your
            logic directly in `custom_func` (which need not be Numba-compiled):

            ```pycon
            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            ...     input_shape = ts1.shape
            ...     n_params = len(p1)
            ...     out1 = np.empty((input_shape[0], input_shape[1] * n_params), dtype=float_)
            ...     out2 = np.empty((input_shape[0], input_shape[1] * n_params), dtype=float_)
            ...     for k in range(n_params):
            ...         for col in range(input_shape[1]):
            ...             for i in range(input_shape[0]):
            ...                 out1[i, input_shape[1] * k + col] = ts1[i, col] * p1[k] + arg1
            ...                 out2[i, input_shape[1] * k + col] = ts2[i, col] * p2[k] + arg2
            ...     return out1, out2
            ```
        """
        Indicator = self.Indicator

        short_name = self.short_name
        prepend_name = self.prepend_name
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names
        output_names = self.output_names

        all_input_names = input_names + param_names + in_output_names

        def custom_func_prop(self, _custom_func=custom_func) -> tp.Callable:
            return _custom_func

        custom_func_prop.__doc__ = "Custom function."
        custom_func_prop.__name__ = "custom_func"
        custom_func_prop.__module__ = Indicator.__module__
        custom_func_prop.__qualname__ = f"{Indicator.__name__}.{custom_func_prop.__name__}"
        setattr(Indicator, custom_func_prop.__name__, class_property(custom_func_prop))

        def _split_args(
            args: tp.Sequence,
        ) -> tp.Tuple[
            tp.Dict[str, tp.ArrayLike],
            tp.Dict[str, tp.ArrayLike],
            tp.Dict[str, tp.ParamValues],
            tp.Args,
        ]:
            inputs = dict(zip(input_names, args[: len(input_names)]))
            checks.assert_len_equal(inputs, input_names)
            args = args[len(input_names) :]

            params = dict(zip(param_names, args[: len(param_names)]))
            checks.assert_len_equal(params, param_names)
            args = args[len(param_names) :]

            in_outputs = dict(zip(in_output_names, args[: len(in_output_names)]))
            checks.assert_len_equal(in_outputs, in_output_names)
            args = args[len(in_output_names) :]
            if not var_args and len(args) > 0:
                raise TypeError(
                    "Variable length arguments are not supported by this function (var_args is set to False)"
                )

            return inputs, in_outputs, params, args

        for k, v in pipeline_kwargs.items():
            if k in param_names and not isinstance(v, Default):
                pipeline_kwargs[k] = Default(v)
        pipeline_kwargs = merge_dicts(dict.fromkeys(in_output_names), pipeline_kwargs)

        default_kwargs = {}
        for k in list(pipeline_kwargs.keys()):
            if k in input_names or k in param_names or k in in_output_names:
                default_kwargs[k] = pipeline_kwargs.pop(k)

        if var_args and keyword_only_args:
            raise ValueError("var_args and keyword_only_args cannot be used together")

        def_run_kwargs = dict(
            short_name=short_name,
            hide_params=hide_params,
            hide_default=hide_default,
            **default_kwargs,
        )

        def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> tp.IFRunOutput:
            _short_name = kwargs.pop("short_name", def_run_kwargs["short_name"])
            _hide_params = kwargs.pop("hide_params", def_run_kwargs["hide_params"])
            _hide_default = kwargs.pop("hide_default", def_run_kwargs["hide_default"])
            _param_settings = merge_dicts(param_settings, kwargs.pop("param_settings", {}))
            _in_output_settings = merge_dicts(
                in_output_settings, kwargs.pop("in_output_settings", {})
            )

            if isinstance(_hide_params, bool):
                if not _hide_params:
                    _hide_params = None
                else:
                    _hide_params = param_names
            if _hide_params is None:
                _hide_params = []
            args = list(args)
            inputs, in_outputs, params, args = _split_args(args)
            level_names = []
            hide_levels = []
            for pname in param_names:
                level_name = _short_name + "_" + pname if prepend_name else pname
                level_names.append(level_name)
                if pname in _hide_params or (_hide_default and isinstance(params[pname], Default)):
                    hide_levels.append(level_name)
            for k, v in params.items():
                if isinstance(v, Default):
                    params[k] = v.value

            results = Indicator.run_pipeline(
                len(output_names),
                custom_func,
                *args,
                require_input_shape=require_input_shape,
                inputs=inputs,
                in_outputs=in_outputs,
                params=params,
                level_names=level_names,
                hide_levels=hide_levels,
                param_settings=_param_settings,
                in_output_settings=_in_output_settings,
                **merge_dicts(pipeline_kwargs, kwargs),
            )
            if kwargs.get("return_raw", False) or kwargs.get("return_cache", False):
                return results
            (
                wrapper,
                new_input_list,
                input_mapper,
                in_output_list,
                output_list,
                new_param_list,
                mapper_list,
                other_list,
            ) = results
            obj = cls(
                wrapper,
                new_input_list,
                input_mapper,
                in_output_list,
                output_list,
                new_param_list,
                mapper_list,
                short_name,
            )
            if len(other_list) > 0:
                return (obj, *tuple(other_list))
            return obj

        Indicator._run = classmethod(_run)

        def compile_run_function(
            func_name: str, docstring: str, _default_kwargs: tp.KwargsLike = None
        ) -> tp.Callable:
            pos_names = []
            main_kw_names = []
            other_kw_names = []
            if _default_kwargs is None:
                _default_kwargs = {}
            for k in input_names + param_names:
                if k in _default_kwargs:
                    main_kw_names.append(k)
                else:
                    pos_names.append(k)
            main_kw_names.extend(in_output_names)
            for k, v in _default_kwargs.items():
                if k not in pos_names and k not in main_kw_names:
                    other_kw_names.append(k)

            _0 = func_name
            _1 = "*, " if keyword_only_args else ""
            _2 = []
            if require_input_shape:
                _2.append("input_shape")
            _2.extend(pos_names)
            _2 = ", ".join(_2) + ", " if len(_2) > 0 else ""
            _3 = "*args, " if var_args else ""
            _4 = [f"{k}={k}" for k in main_kw_names + other_kw_names]
            if require_input_shape:
                _4 += ["input_index=None", "input_columns=None"]
            _4 = ", ".join(_4) + ", " if len(_4) > 0 else ""
            _5 = docstring
            _6 = all_input_names
            _6 = ", ".join(_6) + ", " if len(_6) > 0 else ""
            _7 = []
            if require_input_shape:
                _7.append("input_shape")
            _7.extend(other_kw_names)
            _7 = [f"{k}={k}" for k in _7]
            if require_input_shape:
                _7 += ["input_index=input_index", "input_columns=input_columns"]
            _7 = ", ".join(_7) + ", " if len(_7) > 0 else ""
            func_str = (
                "@classmethod\n"
                f"def {_0}(cls, {_1}{_2}{_3}{_4}**kwargs):\n"
                f'    """{_5}"""\n'
                f"    return cls._{_0}({_6}{_3}{_7}**kwargs)"
            )
            scope = {**dict(Default=Default), **_default_kwargs}
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, "single")
            exec(code, scope)
            return scope[func_name]

        def _prepare_name(name: str) -> str:
            if self.attr_settings.get(name, {}).get("doc", None):
                return f"\n    * `{name}`: {self.attr_settings[name]['doc']}"
            if param_settings is not None and param_settings.get(name, {}).get("doc", None):
                return f"\n    * `{name}`: {param_settings[name]['doc']}"
            if in_output_settings is not None and in_output_settings.get(name, {}).get("doc", None):
                return f"\n    * `{name}`: {in_output_settings[name]['doc']}"
            return f"\n    * `{name}`"

        _0 = Indicator.__name__
        _1 = ""
        if len(self.input_names) > 0:
            _1 += "\n\nInputs:" + "".join(map(_prepare_name, self.input_names))
        if len(self.in_output_names) > 0:
            _1 += "\n\nIn-place outputs:" + "".join(map(_prepare_name, self.in_output_names))
        if len(self.param_names) > 0:
            _1 += "\n\nParameters:" + "".join(map(_prepare_name, self.param_names))
        if len(self.output_names) > 0:
            _1 += "\n\nOutputs:" + "".join(map(_prepare_name, self.output_names))
        if len(self.lazy_outputs) > 0:
            _1 += "\n\nLazy outputs:" + "".join(map(_prepare_name, list(self.lazy_outputs.keys())))
        run_docstring = inspect.cleandoc(
            """
            Run `{0}` indicator.{1}

            Pass a list of parameter names as `hide_params` to hide their column levels, or `True` to hide all.
            Set `hide_default` to False to display column levels for parameters with default values.

            Args:
                *args: Positional arguments corresponding to inputs, parameters, and in-place outputs.
                **kwargs: Keyword arguments for `{0}.run_pipeline`.

            Returns:
                Indicator: Instance of the `{0}` indicator, or a tuple of additional objects if applicable.
            """
        ).format(_0, _1)
        run = compile_run_function("run", run_docstring, def_run_kwargs)
        run.__name__ = "run"
        run.__module__ = Indicator.__module__
        run.__qualname__ = f"{Indicator.__name__}.{run.__name__}"
        Indicator.run = run

        if len(param_names) > 0:
            def_run_combs_kwargs = dict(
                r=2,
                param_product=False,
                comb_func=itertools.combinations,
                run_unique=True,
                short_names=None,
                hide_params=hide_params,
                hide_default=hide_default,
                **default_kwargs,
            )

            def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> tp.IFRunCombsOutput:
                _r = kwargs.pop("r", def_run_combs_kwargs["r"])
                _param_product = kwargs.pop("param_product", def_run_combs_kwargs["param_product"])
                _comb_func = kwargs.pop("comb_func", def_run_combs_kwargs["comb_func"])
                _run_unique = kwargs.pop("run_unique", def_run_combs_kwargs["run_unique"])
                _short_names = kwargs.pop("short_names", def_run_combs_kwargs["short_names"])
                _hide_params = kwargs.pop("hide_params", def_run_kwargs["hide_params"])
                _hide_default = kwargs.pop("hide_default", def_run_kwargs["hide_default"])
                _param_settings = merge_dicts(param_settings, kwargs.get("param_settings", {}))

                if isinstance(_hide_params, bool):
                    if not _hide_params:
                        _hide_params = None
                    else:
                        _hide_params = param_names
                if _hide_params is None:
                    _hide_params = []
                if _short_names is None:
                    _short_names = [f"{short_name}_{str(i + 1)}" for i in range(_r)]
                args = list(args)
                inputs, in_outputs, params, args = _split_args(args)
                for pname in param_names:
                    if _hide_default and isinstance(params[pname], Default):
                        params[pname] = params[pname].value
                        if pname not in _hide_params:
                            _hide_params.append(pname)
                checks.assert_len_equal(params, param_names)
                input_list = list(inputs.values())
                in_output_list = list(in_outputs.values())
                param_list = list(params.values())
                for i, pname in enumerate(param_names):
                    is_tuple = _param_settings.get(pname, {}).get("is_tuple", False)
                    is_array_like = _param_settings.get(pname, {}).get("is_array_like", False)
                    param_list[i] = params_to_list(params[pname], is_tuple, is_array_like)
                if _param_product:
                    param_list = create_param_product(param_list)
                else:
                    param_list = broadcast_params(param_list)
                if _run_unique:
                    raw_results = cls._run(
                        *input_list,
                        *param_list,
                        *in_output_list,
                        *args,
                        return_raw=True,
                        run_unique=False,
                        **kwargs,
                    )
                    kwargs["use_raw"] = raw_results
                instances = []
                if _comb_func == itertools.product:
                    param_lists = zip(*_comb_func(zip(*param_list), repeat=_r))
                else:
                    param_lists = zip(*_comb_func(zip(*param_list), _r))
                for i, param_list in enumerate(param_lists):
                    instances.append(
                        cls._run(
                            *input_list,
                            *zip(*param_list),
                            *in_output_list,
                            *args,
                            short_name=_short_names[i],
                            hide_params=_hide_params,
                            hide_default=_hide_default,
                            run_unique=False,
                            **kwargs,
                        )
                    )
                return tuple(instances)

            Indicator._run_combs = classmethod(_run_combs)

            _0 = Indicator.__name__
            _1 = ""
            if len(self.input_names) > 0:
                _1 += "\n\nInputs:" + "".join(map(_prepare_name, self.input_names))
            if len(self.in_output_names) > 0:
                _1 += "\n\nIn-place outputs:" + "".join(map(_prepare_name, self.in_output_names))
            if len(self.param_names) > 0:
                _1 += "\n\nParameters:" + "".join(map(_prepare_name, self.param_names))
            if len(self.output_names) > 0:
                _1 += "\n\nOutputs:" + "".join(map(_prepare_name, self.output_names))
            if len(self.lazy_outputs) > 0:
                _1 += "\n\nLazy outputs:" + "".join(
                    map(_prepare_name, list(self.lazy_outputs.keys()))
                )
            run_combs_docstring = inspect.cleandoc(
                """
                Create multiple `{0}` indicator instances based on parameter combinations.{1}

                `comb_func` must accept an iterable of parameter tuples along with `r`.
                It also supports combinatoric iterators from `itertools`, like `itertools.combinations`.

                Args:
                    *args: Positional arguments corresponding to inputs, parameters, and in-place outputs.
                    **kwargs: Keyword arguments for pipeline configuration and combination settings, including:

                        * `r`: Number of indicators to run.
                        * `param_product`: Flag controlling parameter combination behavior.
                        * `comb_func`: Function to combine parameter tuples (e.g., `itertools.combinations`).
                        * `run_unique`: If True, computes raw outputs before constructing indicators.
                        * `short_names`: Custom short names for each indicator.
                        * `hide_params`: Parameter names to hide column levels.
                        * `hide_default`: Whether to hide parameters with default values.

                        Other keyword arguments are passed to `{0}.run`.

                Returns:
                    Tuple[Indicator, ...]: Tuple of indicator instances generated.

                !!! note
                    Use this method only when multiple indicator instances are required.
                    To test multiple parameters, pass them as lists to `{0}.run`.
                """
            ).format(_0, _1)
            run_combs = compile_run_function("run_combs", run_combs_docstring, def_run_combs_kwargs)
            run_combs.__name__ = "run_combs"
            run_combs.__module__ = Indicator.__module__
            run_combs.__qualname__ = f"{Indicator.__name__}.{run_combs.__name__}"
            Indicator.run_combs = run_combs

        return Indicator

    def with_apply_func(
        self,
        apply_func: tp.Callable,
        cache_func: tp.Optional[tp.Callable] = None,
        takes_1d: bool = False,
        select_params: bool = True,
        pass_packed: bool = False,
        cache_pass_packed: tp.Optional[bool] = None,
        pass_per_column: bool = False,
        cache_pass_per_column: tp.Optional[bool] = None,
        forward_skipna: bool = False,
        kwargs_as_args: tp.Optional[tp.Iterable[str]] = None,
        jit_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[str, tp.Type[IndicatorBase]]:
        """Build indicator class around a custom apply function.

        Construct and return an indicator class that wraps a custom apply function for calculations.
        This method simplifies indicator creation by handling caching, parameter selection, and
        concatenation of outputs automatically. In contrast to `IndicatorFactory.with_custom_func`,
        it works with one parameter selection at a time, limiting the ability to view all combinations.

        The computation and concatenation are executed using `vectorbtpro.base.combining.apply_and_concat_each`.

        Args:
            apply_func (Callable): Function that receives inputs, a selection of parameters,
                and additional arguments, and performs calculations.

                Arguments are passed in the following order:

                * `i`, the index of the parameter combination, if `select_params` is False.
                * `input_shape` if `pass_input_shape` is True and `input_shape` is not in `kwargs_as_args`.
                * Input arrays corresponding to `input_names`, passed as a tuple if `pass_packed` is True,
                    or unpacked otherwise. When `select_params` is True, each argument is a list of arrays
                    (one per parameter combination). If `per_column` is True, each array corresponds to a column;
                    otherwise, they all refer to the same array. If `takes_1d` is True, each array is further split
                    into multiple column arrays. Still passed as a single array to the caching function.
                * In-place output arrays corresponding to `in_output_names`, with similar behavior as input arrays.
                * Parameter values corresponding to `param_names`, passed as a tuple if `pass_packed` is True,
                    or unpacked otherwise. If `select_params` is True, each argument is a list with one value per
                    parameter combination. When `per_column` is True, each value corresponds to a column; if
                    `takes_1d` is True, each value is repeated by the number of columns.
                * Variable arguments if `var_args` is set to True.
                * `per_column` argument if `pass_per_column` is True and not present in `kwargs_as_args`
                    and if `jitted_loop` is True.
                * Positional arguments listed in `kwargs_as_args`.
                * Other keyword arguments if `jitted_loop` is False, also including `takes_1d` and `per_column`
                    if needed.

                !!! note
                    The shape of each output must match the corresponding input.
            cache_func (Optional[Callable]): Function to preprocess inputs via caching
                before invoking `apply_func`.

                It accepts the same arguments as `apply_func` and must return a single object or
                a tuple of objects. The returned objects are appended as additional arguments to `apply_func`.
            takes_1d (bool): Whether to split 2D arrays into multiple 1D arrays along the column axis.
            select_params (bool): Whether to automatically select in-place outputs and parameters.

                If False, the current iteration index is prepended to the arguments.
            pass_packed (bool): Whether to pass inputs, in-place outputs, and parameters as packed tuples.

                For Numba-compiled functions, tuples are passed instead.
            cache_pass_packed (Optional[bool]): Overrides `pass_packed` for the caching function.
            pass_per_column (bool): Whether to pass the `per_column` flag to the apply function.
            cache_pass_per_column (Optional[bool]): Overrides `pass_per_column` for the caching function.
            forward_skipna (bool): Whether to forward the `skipna` argument to the apply function.
            kwargs_as_args (Optional[Iterable[str]]): Names of keyword arguments from `**kwargs`
                to pass as positional arguments to the apply function.

                Should be used with `jitted_loop=True` as Numba does not support variable keyword arguments.
            jit_kwargs (KwargsLike): Keyword arguments for the `@njit` decorator of the parameter
                selection function.

                Has `nogil` set to True by default.
            **kwargs: Keyword arguments for `IndicatorFactory.with_custom_func`
                and ultimately to `vectorbtpro.base.combining.apply_and_concat_each`.

        Returns:
            Indicator: Indicator class constructed around the provided apply function.

        !!! note
            If `apply_func` is a Numba-compiled function:

            * All inputs are automatically converted to NumPy arrays.
            * Each positional argument must be of a Numba-compatible type.
            * Keyword arguments cannot be passed.
            * Output arrays must have identical shapes, data types, and data orders.

        !!! note
            Reserved arguments such as `per_column` are passed as positional arguments when
            `jitted_loop` is True, and as keyword arguments otherwise.

        Examples:
            Following example produces the same indicator as the `IndicatorFactory.with_custom_func` example.

            ```pycon
            >>> @njit
            ... def apply_func_nb(ts1, ts2, p1, p2, arg1, arg2):
            ...     return ts1 * p1 + arg1, ts2 * p2 + arg2

            >>> MyInd = vbt.IF(
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['out1', 'out2']
            ... ).with_apply_func(
            ...     apply_func_nb, var_args=True,
            ...     kwargs_as_args=['arg2'], arg2=200)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.out1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.out2
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  206.0  230.0  208.0  240.0
            2020-01-02  212.0  224.0  216.0  232.0
            2020-01-03  218.0  218.0  224.0  224.0
            2020-01-04  224.0  212.0  232.0  216.0
            2020-01-05  230.0  206.0  240.0  208.0
            ```

            To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def apply_func(ts, p):
            ...     time.sleep(1)
            ...     return ts * p

            >>> MyInd = vbt.IF(
            ...     input_names=['ts'],
            ...     param_names=['p'],
            ...     output_names=['out']
            ... ).with_apply_func(apply_func)

            >>> %timeit MyInd.run(price, [1, 2, 3])
            3.02 s ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit MyInd.run(price, [1, 2, 3], execute_kwargs=dict(engine='dask'))
            1.02 s ± 2.67 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        Indicator = self.Indicator

        def apply_func_prop(self, _apply_func=apply_func) -> tp.Callable:
            return _apply_func

        apply_func_prop.__doc__ = "Apply function."
        apply_func_prop.__name__ = "apply_func"
        apply_func_prop.__module__ = Indicator.__module__
        apply_func_prop.__qualname__ = f"{Indicator.__name__}.{apply_func_prop.__name__}"
        setattr(Indicator, apply_func_prop.__name__, class_property(apply_func_prop))

        if cache_func is not None:

            def cache_func_prop(self, _cache_func=cache_func) -> tp.Callable:
                return _cache_func

            cache_func_prop.__doc__ = "Cache function."
            cache_func_prop.__name__ = "cache_func"
            cache_func_prop.__module__ = Indicator.__module__
            cache_func_prop.__qualname__ = f"{Indicator.__name__}.{cache_func_prop.__name__}"
            setattr(Indicator, cache_func_prop.__name__, class_property(cache_func_prop))

        module_name = self.module_name
        input_names = self.input_names
        output_names = self.output_names
        in_output_names = self.in_output_names
        param_names = self.param_names

        num_ret_outputs = len(output_names)

        if kwargs_as_args is None:
            kwargs_as_args = []

        if checks.is_numba_func(apply_func):
            _0 = "i"
            _0 += ", args_before"
            if len(input_names) > 0:
                _0 += ", " + ", ".join(input_names)
            if len(in_output_names) > 0:
                _0 += ", " + ", ".join(in_output_names)
            if len(param_names) > 0:
                _0 += ", " + ", ".join(param_names)
            _0 += ", *args"
            if select_params:
                _1 = "*args_before"
            else:
                _1 = "i, *args_before"
            if pass_packed:
                if len(input_names) > 0:
                    _1 += (
                        ", ("
                        + ", ".join(
                            map(lambda x: x + ("[i]" if select_params else ""), input_names)
                        )
                        + ",)"
                    )
                else:
                    _1 += ", ()"
                if len(in_output_names) > 0:
                    _1 += (
                        ", ("
                        + ", ".join(
                            map(lambda x: x + ("[i]" if select_params else ""), in_output_names)
                        )
                        + ",)"
                    )
                else:
                    _1 += ", ()"
                if len(param_names) > 0:
                    _1 += (
                        ", ("
                        + ", ".join(
                            map(lambda x: x + ("[i]" if select_params else ""), param_names)
                        )
                        + ",)"
                    )
                else:
                    _1 += ", ()"
            else:
                if len(input_names) > 0:
                    _1 += ", " + ", ".join(
                        map(lambda x: x + ("[i]" if select_params else ""), input_names)
                    )
                if len(in_output_names) > 0:
                    _1 += ", " + ", ".join(
                        map(lambda x: x + ("[i]" if select_params else ""), in_output_names)
                    )
                if len(param_names) > 0:
                    _1 += ", " + ", ".join(
                        map(lambda x: x + ("[i]" if select_params else ""), param_names)
                    )
            _1 += ", *args"
            func_str = f"def param_select_func_nb({_0}):\n   return apply_func({_1})"
            scope = {"apply_func": apply_func}
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, "single")
            exec(code, scope)
            param_select_func_nb = scope["param_select_func_nb"]
            param_select_func_nb.__doc__ = "Parameter selection function."
            if module_name is not None:
                param_select_func_nb.__module__ = module_name
            jit_kwargs = merge_dicts(dict(nogil=True), jit_kwargs)
            param_select_func_nb = njit(param_select_func_nb, **jit_kwargs)

            Indicator.param_select_func_nb = param_select_func_nb
        else:
            param_select_func_nb = None

        def custom_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.List[tp.AnyArray], ...],
            param_tuple: tp.Tuple[tp.List[tp.ParamValue], ...],
            *args_,
            input_shape: tp.Optional[tp.Shape] = None,
            per_column: bool = False,
            split_columns: bool = False,
            skipna: bool = False,
            return_cache: bool = False,
            use_cache: tp.Union[bool, tp.IFCacheOutput] = True,
            jitted_loop: bool = False,
            jitted_warmup: bool = False,
            param_index: tp.Optional[tp.Index] = None,
            final_index: tp.Optional[tp.Index] = None,
            single_comb: bool = False,
            execute_kwargs: tp.KwargsLike = None,
            **kwargs_,
        ) -> tp.Union[None, tp.IFCacheOutput, tp.Array2d, tp.List[tp.Array2d]]:
            """Forward inputs and parameters to `{0}`, performing caching and pre-processing.

            Caching is performed using `{1}` if provided, and the results are passed to
            `{0}`. If `{0}` is a Numba-compiled function, the actual apply-function becomes
            `{2}`, which handles the selection of parameters and the passing of arguments.

            Args:
                input_tuple (Tuple[AnyArray, ...]): Tuple of input arrays.
                in_output_tuple (Tuple[List[AnyArray], ...]): Tuple of lists of in-place output arrays.
                param_tuple (Tuple[List[ParamValue], ...]): Tuple of lists of parameter values.
                *args_: Additional positional arguments.
                input_shape (Optional[Shape]): Shape of the input arrays.
                per_column (bool): If True, processes parameters separately for each column.
                split_columns (bool): Whether to split arrays into separate columns.
                skipna (bool): Whether to skip NaN values.
                return_cache (bool): If True, return the cache result instead of processing further.
                use_cache (Union[bool, IFCacheOutput]): Flag or configuration indicating whether to use caching.
                jitted_loop (bool): Flag indicating whether to use a JIT-compiled loop.
                jitted_warmup (bool): If True, perform a warm-up call for the JIT-compiled function.
                param_index (Optional[Index]): Index for parameter combinations.
                final_index (Optional[Index]): Final index used for the output.
                single_comb (bool): Flag indicating whether there is only one parameter combination.
                execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                    See `vectorbtpro.utils.execution.execute`.
                **kwargs_: Additional keyword arguments.

            Returns:
                Union[None, IFCacheOutput, Array2d, List[Array2d]]:
                    Result of applying `{0}`, which may be:

                    * Cache output if `return_cache` is True.
                    * 2D array.
                    * List of 2D arrays.
                    * None.
            """
            if jitted_loop and not checks.is_numba_func(apply_func):
                raise ValueError("Apply function must be Numba-compiled for jitted_loop=True")
            if skipna and len(in_output_tuple) > 1:
                raise ValueError("NaNs cannot be skipped for in-place outputs")
            if skipna and jitted_loop:
                raise ValueError("NaNs cannot be skipped when jitted_loop=True")
            if forward_skipna:
                kwargs_["skipna"] = skipna
                skipna = False
            if execute_kwargs is None:
                execute_kwargs = {}
            else:
                execute_kwargs = dict(execute_kwargs)

            _cache_pass_packed = cache_pass_packed
            _cache_pass_per_column = cache_pass_per_column

            args_before = ()
            if input_shape is not None and "input_shape" not in kwargs_as_args:
                if per_column or takes_1d:
                    args_before += ((input_shape[0],),)
                elif split_columns and len(input_shape) == 2:
                    args_before += ((input_shape[0], 1),)
                else:
                    args_before += (input_shape,)

            more_args = ()
            for k in kwargs_as_args:
                if k == "per_column" or k == "takes_1d" or k == "split_columns":
                    value = per_column
                else:
                    value = kwargs_.pop(k)
                more_args += (value,)

            if len(input_tuple) > 0:
                if input_tuple[0].ndim == 1:
                    n_cols = 1
                else:
                    n_cols = input_tuple[0].shape[1]
            elif input_shape is not None:
                if len(input_shape) == 1:
                    n_cols = 1
                else:
                    n_cols = input_shape[1]
            else:
                n_cols = None
            if per_column:
                n_params = n_cols
            else:
                n_params = len(param_tuple[0]) if len(param_tuple) > 0 else 1

            cache = use_cache
            if isinstance(cache, bool):
                if cache and cache_func is not None:
                    _input_tuple = input_tuple
                    _in_output_tuple = ()
                    for in_outputs in in_output_tuple:
                        if checks.is_numba_func(cache_func):
                            _in_outputs = to_typed_list(in_outputs)
                        else:
                            _in_outputs = in_outputs
                        _in_output_tuple += (_in_outputs,)
                    _param_tuple = ()
                    for params in param_tuple:
                        if checks.is_numba_func(cache_func):
                            _params = to_typed_list(params)
                        else:
                            _params = params
                        _param_tuple += (_params,)

                    if _cache_pass_packed is None:
                        _cache_pass_packed = pass_packed
                    if _cache_pass_per_column is None and per_column:
                        _cache_pass_per_column = True
                    if _cache_pass_per_column is None:
                        _cache_pass_per_column = pass_per_column
                    cache_more_args = tuple(more_args)
                    cache_kwargs = dict(kwargs_)
                    if _cache_pass_per_column:
                        if "per_column" not in kwargs_as_args:
                            if jitted_loop:
                                cache_more_args += (per_column,)
                            else:
                                cache_kwargs["per_column"] = per_column

                    if _cache_pass_packed:
                        cache = cache_func(
                            *args_before,
                            _input_tuple,
                            _in_output_tuple,
                            _param_tuple,
                            *args_,
                            *cache_more_args,
                            **cache_kwargs,
                        )
                    else:
                        cache = cache_func(
                            *args_before,
                            *_input_tuple,
                            *_in_output_tuple,
                            *_param_tuple,
                            *args_,
                            *cache_more_args,
                            **cache_kwargs,
                        )
                else:
                    cache = None
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, tuple):
                cache = (cache,)

            def _expand_input(
                input: tp.MaybeList[tp.AnyArray], multiple: bool = False
            ) -> tp.List[tp.AnyArray]:
                if jitted_loop:
                    _inputs = List()
                else:
                    _inputs = []
                if per_column:
                    if multiple:
                        _input = input[0]
                    else:
                        _input = input
                    if _input.ndim == 2:
                        for i in range(_input.shape[1]):
                            if takes_1d:
                                if isinstance(_input, pd.DataFrame):
                                    _inputs.append(_input.iloc[:, i])
                                else:
                                    _inputs.append(_input[:, i])
                            else:
                                if isinstance(_input, pd.DataFrame):
                                    _inputs.append(_input.iloc[:, i : i + 1])
                                else:
                                    _inputs.append(_input[:, i : i + 1])
                    else:
                        _inputs.append(_input)
                else:
                    for p in range(n_params):
                        if multiple:
                            _input = input[p]
                        else:
                            _input = input
                        if takes_1d or split_columns:
                            if isinstance(_input, pd.DataFrame):
                                for i in range(_input.shape[1]):
                                    if takes_1d:
                                        _inputs.append(_input.iloc[:, i])
                                    else:
                                        _inputs.append(_input.iloc[:, i : i + 1])
                            elif _input.ndim == 2:
                                for i in range(_input.shape[1]):
                                    if takes_1d:
                                        _inputs.append(_input[:, i])
                                    else:
                                        _inputs.append(_input[:, i : i + 1])
                            else:
                                _inputs.append(_input)
                        else:
                            _inputs.append(_input)
                return _inputs

            _input_tuple = ()
            for input in input_tuple:
                _inputs = _expand_input(input)
                _input_tuple += (_inputs,)
            if skipna and len(_input_tuple) > 0:
                new_input_tuple = tuple([[] for _ in range(len(_input_tuple))])
                nan_masks = []
                any_nan = False
                for i in range(len(_input_tuple[0])):
                    inputs = []
                    for k in range(len(_input_tuple)):
                        input = _input_tuple[k][i]
                        if input.ndim == 2 and input.shape[1] > 1:
                            raise ValueError(
                                "NaNs cannot be skipped for multi-columnar inputs. Use split_columns=True."
                            )
                        inputs.append(input)
                    nan_mask = build_nan_mask(*inputs)
                    nan_masks.append(nan_mask)
                    if not any_nan and nan_mask is not None and np.any(nan_mask):
                        any_nan = True
                    if any_nan:
                        inputs = squeeze_nan(*inputs, nan_mask=nan_mask)
                    for k in range(len(_input_tuple)):
                        new_input_tuple[k].append(inputs[k])
                _input_tuple = new_input_tuple
                if any_nan:

                    def _post_execute_func(outputs, nan_masks):
                        new_outputs = []
                        for i, output in enumerate(outputs):
                            if isinstance(output, (tuple, list, List)):
                                output = unsqueeze_nan(*output, nan_mask=nan_masks[i])
                            else:
                                output = unsqueeze_nan(output, nan_mask=nan_masks[i])
                            new_outputs.append(output)
                        return new_outputs

                    if "post_execute_func" in execute_kwargs:
                        raise ValueError("Cannot use custom post_execute_func when skipna=True")
                    execute_kwargs["post_execute_func"] = _post_execute_func
                    if "post_execute_kwargs" in execute_kwargs:
                        raise ValueError("Cannot use custom post_execute_kwargs when skipna=True")
                    execute_kwargs["post_execute_kwargs"] = dict(
                        outputs=Rep("results"), nan_masks=nan_masks
                    )
                    if "post_execute_on_sorted" in execute_kwargs:
                        raise ValueError(
                            "Cannot use custom post_execute_on_sorted when skipna=True"
                        )
                    execute_kwargs["post_execute_on_sorted"] = True

            _in_output_tuple = ()
            for in_outputs in in_output_tuple:
                _in_outputs = _expand_input(in_outputs, multiple=True)
                _in_output_tuple += (_in_outputs,)

            _param_tuple = ()
            for params in param_tuple:
                if not per_column and (takes_1d or split_columns):
                    _params = [params[p] for p in range(len(params)) for _ in range(n_cols)]
                else:
                    _params = params
                if jitted_loop:
                    if len(_params) > 0 and np.isscalar(_params[0]):
                        _params = np.asarray(_params)
                    else:
                        _params = to_typed_list(_params)
                _param_tuple += (_params,)
            if not per_column and (takes_1d or split_columns):
                _n_params = n_params * n_cols
                keys = final_index
            else:
                _n_params = n_params
                keys = param_index
            execute_kwargs = merge_dicts(
                dict(show_progress=False if single_comb else None), execute_kwargs
            )
            execute_kwargs["keys"] = keys

            if pass_per_column:
                if "per_column" not in kwargs_as_args:
                    if jitted_loop:
                        more_args += (per_column,)
                    else:
                        kwargs_["per_column"] = per_column

            if jitted_loop:
                return combining.apply_and_concat(
                    _n_params,
                    param_select_func_nb,
                    args_before,
                    *_input_tuple,
                    *_in_output_tuple,
                    *_param_tuple,
                    *args_,
                    *more_args,
                    *cache,
                    **kwargs_,
                    n_outputs=num_ret_outputs,
                    jitted_loop=True,
                    jitted_warmup=jitted_warmup,
                    execute_kwargs=execute_kwargs,
                )

            tasks = []
            for i in range(_n_params):
                if select_params:
                    _inputs = tuple(_inputs[i] for _inputs in _input_tuple)
                    _in_outputs = tuple(_in_outputs[i] for _in_outputs in _in_output_tuple)
                    _params = tuple(_params[i] for _params in _param_tuple)
                else:
                    _inputs = _input_tuple
                    _in_outputs = _in_output_tuple
                    _params = _param_tuple
                tasks.append(
                    Task(
                        apply_func,
                        *((i,) if not select_params else ()),
                        *args_before,
                        *((_inputs,) if pass_packed else _inputs),
                        *((_in_outputs,) if pass_packed else _in_outputs),
                        *((_params,) if pass_packed else _params),
                        *args_,
                        *more_args,
                        *cache,
                        **kwargs_,
                    )
                )
            return combining.apply_and_concat_each(
                tasks,
                n_outputs=num_ret_outputs,
                execute_kwargs=execute_kwargs,
            )

        custom_func.__name__ = "custom_func"
        custom_func.__module__ = Indicator.__module__
        custom_func.__qualname__ = f"{Indicator.__name__}.{custom_func.__name__}"
        custom_func.__doc__ = custom_func.__doc__.format(
            Indicator.__name__ + ".apply_func",
            Indicator.__name__ + ".cache_func" if cache_func is not None else "cache_func",
            (
                Indicator.__name__ + ".param_select_func_nb"
                if param_select_func_nb is not None
                else "param_select_func_nb"
            ),
        )
        return self.with_custom_func(
            custom_func,
            pass_packed=True,
            pass_param_index=True,
            pass_final_index=True,
            pass_single_comb=True,
            **kwargs,
        )

    # ############# Exploration ############# #

    _custom_indicators: tp.ClassVar[Config] = HybridConfig()

    @class_property
    def custom_indicators(cls) -> Config:
        """Custom indicators keyed by their custom locations.

        Returns:
            Config: Dictionary-like object containing custom indicators.
        """
        return cls._custom_indicators

    @classmethod
    def list_custom_locations(cls) -> tp.List[str]:
        """List of custom indicator locations in the order they were registered.

        Returns:
            List[str]: List of custom indicator locations.
        """
        return list(cls.custom_indicators.keys())

    @classmethod
    def list_builtin_locations(cls) -> tp.List[str]:
        """List of built-in indicator locations in the order defined by the author.

        Returns:
            List[str]: List of built-in indicator locations.
        """
        return [
            "vbt",
            "talib_func",
            "talib",
            "pandas_ta",
            "ta",
            "technical",
            "techcon",
            "smc",
            "wqa101",
        ]

    @classmethod
    def list_locations(cls) -> tp.List[str]:
        """List of all supported indicator locations, with custom locations listed before built-in locations.

        Returns:
            List[str]: List of all indicator locations.
        """
        return [*cls.list_custom_locations(), *cls.list_builtin_locations()]

    @classmethod
    def match_location(cls, location: str) -> tp.Optional[str]:
        """Return the matching location name for the provided input.

        Args:
            location (str): Location name to match (case-insensitive).

        Returns:
            Optional[str]: Matching location if found; otherwise, None.
        """
        for k in cls.list_locations():
            if k.lower() == location.lower():
                return k
        return None

    @classmethod
    def split_indicator_name(cls, name: str) -> tp.Tuple[tp.Optional[str], tp.Optional[str]]:
        """Split an indicator name into its constituent location and indicator name.

        Args:
            name (str): Indicator name, which may include location information separated
                by a colon or underscore.

        Returns:
            Tuple[Optional[str], Optional[str]]: Tuple where the first element is the location
                (if detected) and the second element is the indicator name.

                If the name itself matches a known location, returns (location, None).
        """
        locations = cls.list_locations()

        matched_location = cls.match_location(name)
        if matched_location is not None:
            return matched_location, None
        if ":" in name:
            location = name.split(":")[0].strip()
            name = name.split(":")[1].strip()
        else:
            location = None
            found_location = False
            if "_" in name:
                for location in locations:
                    if name.lower().startswith(location.lower() + "_"):
                        found_location = True
                        break
            if found_location:
                name = name[len(location) + 1 :]
            else:
                location = None
        return location, name

    @classmethod
    def register_custom_indicator(
        cls,
        indicator: tp.Union[str, tp.Type[IndicatorBase]],
        name: tp.Optional[str] = None,
        location: tp.Optional[str] = None,
        if_exists: str = "raise",
    ) -> None:
        """Register a custom indicator under a custom location.

        Args:
            indicator (Union[str, IndicatorBase]): Custom indicator to register,
                specified as a string reference or a type.
            name (Optional[str]): Name under which to register the indicator.
            location (Optional[str]): Custom location where the indicator should be registered.
            if_exists (str): Behavior if an indicator with the same name already exists;
                must be "raise", "skip", or "override".

        Returns:
            None
        """
        if isinstance(indicator, str):
            indicator = cls.get_indicator(indicator)
        if name is None:
            name = indicator.__name__
        elif location is None:
            location, name = cls.split_indicator_name(name)
        if location is None:
            location = "custom"
        else:
            matched_location = cls.match_location(location)
            if matched_location is not None:
                location = matched_location
        if not name.isidentifier():
            raise ValueError(f"Custom name '{name}' must be a valid variable name")
        if not location.isidentifier():
            raise ValueError(f"Custom location '{location}' must be a valid variable name")
        if location in cls.list_builtin_locations():
            raise ValueError(
                f"Custom location '{location}' shadows a built-in location with the same name"
            )
        if location not in cls.custom_indicators:
            cls.custom_indicators[location] = dict()
        for k in cls.custom_indicators[location]:
            if name.upper() == k.upper():
                if if_exists.lower() == "raise":
                    raise ValueError(
                        f"Indicator with name '{name}' already exists under location '{location}'"
                    )
                if if_exists.lower() == "skip":
                    return None
                if if_exists.lower() == "override":
                    break
                raise ValueError(f"Invalid if_exists: '{if_exists}'")
        cls.custom_indicators[location][name] = indicator

    @classmethod
    def deregister_custom_indicator(
        cls,
        name: tp.Optional[str] = None,
        location: tp.Optional[str] = None,
        remove_location: bool = True,
    ) -> None:
        """Deregister a custom indicator based on its name and location.

        Args:
            name (Optional[str]): Name of the indicator to remove.
            location (Optional[str]): Location from which to remove the indicator.

                If None, all custom locations will be searched for the indicator.
            remove_location (bool): Whether to remove a location if it becomes empty
                after the indicator is removed.

        Returns:
            None
        """
        if location is not None:
            matched_location = cls.match_location(location)
            if matched_location is not None:
                location = matched_location
        if name is None:
            if location is None:
                for k in list(cls.custom_indicators.keys()):
                    del cls.custom_indicators[k]
            else:
                del cls.custom_indicators[location]
        else:
            if location is None:
                location, name = cls.split_indicator_name(name)
            if location is None:
                for k, v in list(cls.custom_indicators.items()):
                    for k2 in list(cls.custom_indicators[k].keys()):
                        if name.upper() == k2.upper():
                            del cls.custom_indicators[k][k2]
                            if remove_location and len(cls.custom_indicators[k]) == 0:
                                del cls.custom_indicators[k]
            else:
                for k in list(cls.custom_indicators[location].keys()):
                    if name.upper() == k.upper():
                        del cls.custom_indicators[location][k]
                        if remove_location and len(cls.custom_indicators[location]) == 0:
                            del cls.custom_indicators[location]

    @classmethod
    def get_custom_indicator(
        cls,
        name: str,
        location: tp.Optional[str] = None,
        return_first: bool = False,
    ) -> tp.Type[IndicatorBase]:
        """Return a custom indicator based on its name and optional location.

        Args:
            name (str): Name of the custom indicator.
            location (Optional[str]): Location in which to search for the indicator.

                If None, the location will be inferred from the indicator name.
            return_first (bool): If multiple indicators match, return the first one when True.

        Returns:
            IndicatorBase: Custom indicator class.
        """
        if location is None:
            location, name = cls.split_indicator_name(name)
        else:
            matched_location = cls.match_location(location)
            if matched_location is not None:
                location = matched_location
        name = name.upper()
        if location is None:
            found_indicators = []
            for k, v in cls.custom_indicators.items():
                for k2, v2 in v.items():
                    k2 = k2.upper()
                    if k2 == name:
                        found_indicators.append(v2)
            if len(found_indicators) == 1:
                return found_indicators[0]
            if len(found_indicators) > 1:
                if return_first:
                    return found_indicators[0]
                raise KeyError(f"Found multiple custom indicators with name '{name}'")
            raise KeyError(f"Found no custom indicator with name '{name}'")
        else:
            for k, v in cls.custom_indicators[location].items():
                k = k.upper()
                if k == name:
                    return v
            raise KeyError(
                f"Found no custom indicator with name '{name}' under location '{location}'"
            )

    @classmethod
    def list_custom_indicators(
        cls,
        uppercase: bool = False,
        location: tp.Optional[str] = None,
        prepend_location: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """Return a list of custom indicator names.

        Args:
            uppercase (bool): Whether to convert indicator names to uppercase.

                Otherwise, names are returned in their original case.
            location (Optional[str]): Filter indicators by a specific location.
            prepend_location (Optional[bool]): When True, indicator names are prefixed with their location.

        Returns:
            List[str]: List of custom indicator names.
        """
        if location is not None:
            matched_location = cls.match_location(location)
            if matched_location is not None:
                location = matched_location
        locations_names = []
        non_custom_location = False
        for k, v in cls.custom_indicators.items():
            if location is not None:
                if k != location:
                    continue
            for k2, v2 in v.items():
                if uppercase:
                    k2 = k2.upper()
                if not non_custom_location and k != "custom":
                    non_custom_location = True
                locations_names.append((k, k2))
        locations_names = sorted(locations_names, key=lambda x: (x[0].upper(), x[1]))
        if prepend_location is None:
            prepend_location = location is None and non_custom_location
        if prepend_location:
            return list(map(lambda x: x[0] + ":" + x[1], locations_names))
        return list(map(lambda x: x[1], locations_names))

    @classmethod
    def list_vbt_indicators(cls) -> tp.List[str]:
        """Return a sorted list of all vectorbtpro indicators.

        Returns:
            List[str]: Sorted list of all vectorbtpro indicator names.
        """
        import vectorbtpro as vbt

        return sorted(
            [
                attr
                for attr in dir(vbt)
                if not attr.startswith("_")
                and isinstance(getattr(vbt, attr), type)
                and getattr(vbt, attr) is not IndicatorBase
                and issubclass(getattr(vbt, attr), IndicatorBase)
            ]
        )

    @classmethod
    def list_indicators(
        cls,
        pattern: tp.Optional[str] = None,
        case_sensitive: bool = False,
        use_regex: bool = False,
        location: tp.Optional[str] = None,
        prepend_location: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """Return a list of indicator names optionally filtered by a pattern or location.

        A pattern may also represent a location, in which case all indicators from that location are returned.
        For supported locations, see `IndicatorFactory.list_locations`.

        Args:
            pattern (Optional[str]): Pattern to filter indicator names.

                If the pattern matches a location, all indicators from that location are returned.
            case_sensitive (bool): Whether to treat the pattern as case-sensitive.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.
            location (Optional[str]): Specific location from which to list indicators.
            prepend_location (Optional[bool]): When True, indicator names are prefixed with their location.

        Returns:
            List[str]: List of matching indicator names.
        """
        if pattern is not None:
            if not case_sensitive:
                pattern = pattern.lower()
            if location is None and cls.match_location(pattern) is not None:
                location = pattern
                pattern = None
        if prepend_location is None:
            if location is not None:
                prepend_location = False
            else:
                prepend_location = True
        with WarningsFiltered():
            if location is not None:
                matched_location = cls.match_location(location)
                if matched_location is not None:
                    location = matched_location
                if location in cls.list_custom_locations():
                    all_indicators = cls.list_custom_indicators(
                        location=location, prepend_location=prepend_location
                    )
                else:
                    all_indicators = map(
                        lambda x: location + ":" + x if prepend_location else x,
                        getattr(cls, f"list_{location}_indicators")(),
                    )
            else:
                from vectorbtpro.utils.module_ import check_installed

                all_indicators = [
                    *cls.list_custom_indicators(prepend_location=prepend_location),
                    *map(
                        lambda x: "vbt:" + x if prepend_location else x, cls.list_vbt_indicators()
                    ),
                    *map(
                        lambda x: "talib:" + x if prepend_location else x,
                        cls.list_talib_indicators() if check_installed("talib") else [],
                    ),
                    *map(
                        lambda x: "pandas_ta:" + x if prepend_location else x,
                        cls.list_pandas_ta_indicators() if check_installed("pandas_ta") else [],
                    ),
                    *map(
                        lambda x: "ta:" + x if prepend_location else x,
                        cls.list_ta_indicators() if check_installed("ta") else [],
                    ),
                    *map(
                        lambda x: "technical:" + x if prepend_location else x,
                        cls.list_technical_indicators() if check_installed("technical") else [],
                    ),
                    *map(
                        lambda x: "techcon:" + x if prepend_location else x,
                        cls.list_techcon_indicators() if check_installed("technical") else [],
                    ),
                    *map(
                        lambda x: "smc:" + x if prepend_location else x,
                        cls.list_smc_indicators() if check_installed("smartmoneyconcepts") else [],
                    ),
                    *map(
                        lambda x: "wqa101:" + str(x) if prepend_location else str(x), range(1, 102)
                    ),
                ]
        found_indicators = []
        for indicator in all_indicators:
            if prepend_location and location is not None:
                indicator = location + ":" + indicator
            if case_sensitive:
                indicator_name = indicator
            else:
                indicator_name = indicator.lower()
            if pattern is not None:
                if use_regex:
                    if location is not None:
                        if not re.match(pattern, indicator_name):
                            continue
                    else:
                        if not re.match(pattern, indicator_name.split(":")[1]):
                            continue
                else:
                    if location is not None:
                        if not re.match(fnmatch.translate(pattern), indicator_name):
                            continue
                    else:
                        if not re.match(fnmatch.translate(pattern), indicator_name.split(":")[1]):
                            continue
            found_indicators.append(indicator)
        return found_indicators

    @classmethod
    def get_indicator(cls, name: str, location: tp.Optional[str] = None) -> tp.Type[IndicatorBase]:
        """Return the indicator class corresponding to the given name and location.

        The indicator name can include a location prefix separated by a colon. For example,
        `"talib:sma"` or `"talib_sma"` returns the TA-Lib SMA indicator. If no location is specified,
        the indicator is searched across all available sources, including vectorbtpro indicators.

        Args:
            name (str): Name of the indicator, optionally including a location prefix.
            location (Optional[str]): Location to filter the search for the indicator.

        Returns:
            Type[IndicatorBase]: Indicator class matching the provided name.
        """
        if location is None:
            location, name = cls.split_indicator_name(name)
        else:
            matched_location = cls.match_location(location)
            if matched_location is not None:
                location = matched_location
        if name is not None:
            name = name.upper()
            if location is not None:
                if location in cls.list_custom_locations():
                    return cls.get_custom_indicator(name, location=location)
                if location == "vbt":
                    import vectorbtpro as vbt

                    return getattr(vbt, name.upper())
                if location == "talib":
                    return cls.from_talib(name)
                if location == "pandas_ta":
                    return cls.from_pandas_ta(name)
                if location == "ta":
                    return cls.from_ta(name)
                if location == "technical":
                    return cls.from_technical(name)
                if location == "techcon":
                    return cls.from_techcon(name)
                if location == "smc":
                    return cls.from_smc(name)
                if location == "wqa101":
                    return cls.from_wqa101(int(name))
                raise ValueError(f"Location '{location}' not found")
            else:
                import vectorbtpro as vbt
                from vectorbtpro.utils.module_ import check_installed

                if name in cls.list_custom_indicators(uppercase=True, prepend_location=False):
                    return cls.get_custom_indicator(name, return_first=True)
                if hasattr(vbt, name):
                    return getattr(vbt, name)
                if str(name).isnumeric():
                    return cls.from_wqa101(int(name))
                if check_installed("smc") and name in cls.list_smc_indicators():
                    return cls.from_smc(name)
                if check_installed("technical") and name in cls.list_techcon_indicators():
                    return cls.from_techcon(name)
                if check_installed("talib") and name in cls.list_talib_indicators():
                    return cls.from_talib(name)
                if check_installed("ta") and name in cls.list_ta_indicators(uppercase=True):
                    return cls.from_ta(name)
                if check_installed("pandas_ta") and name in cls.list_pandas_ta_indicators():
                    return cls.from_pandas_ta(name)
                if check_installed("technical") and name in cls.list_technical_indicators():
                    return cls.from_technical(name)
        raise ValueError(f"Indicator '{name}' not found")

    # ############# Third party ############# #

    @classmethod
    def list_talib_indicators(cls) -> tp.List[str]:
        """Return a sorted list of all parseable indicator names from TA-Lib.

        Returns:
            List[str]: Sorted list of indicator names from the TA-Lib module.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("talib")
        import talib

        return sorted(talib.get_functions())

    @classmethod
    def from_talib(
        cls, func_name: str, factory_kwargs: tp.KwargsLike = None, **kwargs
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class using a TA-Lib function.

        Requires [TA-Lib](https://github.com/mrjbq7/ta-lib) to be installed.

        For input, parameter, and output names, see the
        [TA-Lib documentation](https://github.com/mrjbq7/ta-lib/blob/master/docs/index.md).

        Args:
            func_name (str): Name of the TA-Lib function to wrap.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            IndicatorBase: New indicator class based on the TA-Lib function.

        Examples:
            ```pycon
            >>> SMA = vbt.IF.from_talib('SMA')

            >>> sma = SMA.run(price, timeperiod=[2, 3])
            >>> sma.real
            sma_timeperiod         2         3
                              a    b    a    b
            2020-01-01      NaN  NaN  NaN  NaN
            2020-01-02      1.5  4.5  NaN  NaN
            2020-01-03      2.5  3.5  2.0  4.0
            2020-01-04      3.5  2.5  3.0  3.0
            2020-01-05      4.5  1.5  4.0  2.0
            ```

            To plot an indicator:

            ```pycon
            >>> sma.plot(column=(2, 'a')).show()
            ```

            ![](/assets/images/api/talib_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/talib_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("talib")
        from talib import abstract

        from vectorbtpro.indicators.talib_ import talib_func, talib_plot_func

        func_name = func_name.upper()
        info = abstract.Function(func_name).info
        input_names = []
        for in_names in info["input_names"].values():
            if isinstance(in_names, (list, tuple)):
                input_names.extend(list(in_names))
            else:
                input_names.append(in_names)
        class_name = info["name"]
        class_docstring = "{}, {}".format(info["display_name"], info["group"])
        param_names = list(info["parameters"].keys()) + ["timeframe"]
        output_names = info["output_names"]
        output_flags = info["output_flags"]

        _talib_func = talib_func(func_name)
        _talib_plot_func = talib_plot_func(func_name)

        def apply_func(
            input_tuple: tp.Tuple[tp.Array2d, ...],
            in_output_tuple: tp.Tuple[tp.Array2d, ...],
            param_tuple: tp.Tuple[tp.ParamValue, ...],
            timeframe: tp.Optional[tp.FrequencyLike] = None,
            **kwargs_,
        ) -> tp.MaybeTuple[tp.Array2d]:
            if len(param_tuple) == len(param_names):
                if timeframe is not None:
                    raise ValueError(
                        "Time frame is set both as a parameter and as a keyword argument"
                    )
                timeframe = param_tuple[-1]
                param_tuple = param_tuple[:-1]
            elif len(param_tuple) > len(param_names):
                raise ValueError("Provided more parameters than registered")

            return _talib_func(
                *input_tuple,
                *param_tuple,
                timeframe=timeframe,
                **kwargs_,
            )

        apply_func.__doc__ = inspect.cleandoc(
            f"""
            Apply the TA-Lib function with the provided inputs and parameters.

            Based on `vbt.talib_func("{func_name}")`.

            Args:
                input_tuple (Tuple[Array2d, ...]): Input arrays for the TA-Lib function.

                    These arrays provide the indicator data.
                in_output_tuple (Tuple[Array2d, ...]): Intermediate output arrays.

                    Not used by this function.
                param_tuple (Tuple[ParamValue, ...]): Tuple of parameter values.

                    May include the timeframe as the last element.
                timeframe (Optional[FrequencyLike]): Timeframe specification (e.g., "daily", "15 minutes").

                    See `vectorbtpro.utils.datetime_.to_freq`.
                **kwargs_: Additional keyword arguments.

            Returns:
                MaybeTuple[Array2d]: Output computed by the TA-Lib function.
            """
        )
        kwargs = merge_dicts(
            {k: Default(v) for k, v in info["parameters"].items()}, dict(timeframe=None), kwargs
        )
        Indicator = cls(
            **merge_dicts(
                dict(
                    class_name=class_name,
                    class_docstring=class_docstring,
                    module_name=__name__ + ".talib",
                    input_names=input_names,
                    param_names=param_names,
                    output_names=output_names,
                    output_flags=output_flags,
                ),
                factory_kwargs,
            )
        ).with_apply_func(
            apply_func,
            pass_packed=True,
            pass_wrapper=True,
            forward_skipna=True,
            **kwargs,
        )

        def plot(
            self,
            column: tp.Optional[tp.Column] = None,
            add_shape_kwargs: tp.KwargsLike = None,
            add_trace_kwargs: tp.KwargsLike = None,
            fig: tp.Optional[tp.BaseFigure] = None,
            **kwargs,
        ) -> tp.BaseFigure:
            self_col = self.select_col(column=column, group_by=False)

            return _talib_plot_func(
                *[getattr(self_col, output_name) for output_name in output_names],
                add_shape_kwargs=add_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
                **kwargs,
            )

        plot.__doc__ = inspect.cleandoc(
            f"""
            Plot the indicator output using the TA-Lib plotting function.

            Based on `vbt.talib_plot_func("{func_name}")`.

            Args:
                column (Optional[Column]): Identifier of the column to plot.

                    If provided, selects the corresponding column from the indicator output.
                add_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for each shape.
                add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                    for example, `dict(row=1, col=1)`.
                fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
                **kwargs: Keyword arguments for the plotting function.

            Returns:
                BaseFigure: Figure containing the plotted indicator.
            """
        )
        Indicator.plot = plot

        return Indicator

    @classmethod
    def parse_pandas_ta_config(
        cls,
        func: tp.Callable,
        test_input_names: tp.Optional[tp.Sequence[str]] = None,
        test_index_len: int = 100,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Kwargs:
        """Parse the configuration of a Pandas TA indicator.

        This class method inspects the signature of the provided indicator function to determine
        its inputs, parameters, and outputs. It creates a test DataFrame using random data, passes
        it to the function, and parses the output to extract column names. The resulting configuration
        dictionary encapsulates details required for further processing.

        Args:
            func (Callable): Pandas TA indicator function to parse.

                The function is inspected to detect input and parameter names.
            test_input_names (Optional[Sequence[str]]): Collection of potential input parameter names.

                Used to identify input columns if not explicitly annotated.
            test_index_len (int): Number of rows in the generated test DataFrame.
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for the indicator function.

        Returns:
            Kwargs: Dictionary containing the following keys:

                * `class_name`: Uppercase name of the indicator function.
                * `class_docstring`: Original docstring of the indicator function.
                * `input_names`: List of detected input parameter names.
                * `param_names`: List of parameter names that have default values.
                * `output_names`: List of output column names extracted from the function's result.
                * `defaults`: Mapping of parameter names to their default values.
        """
        if test_input_names is None:
            test_input_names = {
                "open_",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "dividends",
                "split",
            }

        input_names = []
        param_names = []
        output_names = []
        defaults = {}

        sig = inspect.signature(func)
        for k, v in sig.parameters.items():
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                if v.annotation != inspect.Parameter.empty and v.annotation == pd.Series or k in test_input_names or v.default == inspect.Parameter.empty:
                    input_names.append(k)
                else:
                    param_names.append(k)
                    defaults[k] = v.default

        test_df = pd.DataFrame(
            {c: np.random.uniform(1, 10, size=(test_index_len,)) for c in input_names},
            index=pd.date_range("2020", periods=test_index_len),
        )
        new_args = merge_dicts({c: test_df[c] for c in input_names}, kwargs)
        result = suppress_stdout(func)(**new_args)

        if isinstance(result, tuple):
            results = []
            for i, r in enumerate(result):
                if len(r.index) != len(test_df.index):
                    if not silence_warnings:
                        warn(f"Couldn't parse the output at index {i}: mismatching index")
                else:
                    results.append(r)
            if len(results) > 1:
                result = pd.concat(results, axis=1)
            elif len(results) == 1:
                result = results[0]
            else:
                raise ValueError("Couldn't parse the output")
        if len(result.index) != len(test_df.index):
            raise ValueError("Couldn't parse the output: mismatching index")
        output_cols = result.columns.tolist() if isinstance(result, pd.DataFrame) else [result.name]
        new_output_cols = []
        for i in range(len(output_cols)):
            name_parts = []
            for name_part in output_cols[i].split("_"):
                try:
                    float(name_part)
                    continue
                except Exception:
                    name_parts.append(name_part.replace("-", "_").lower())
            output_col = "_".join(name_parts)
            new_output_cols.append(output_col)

        for k, v in Counter(new_output_cols).items():
            if v == 1:
                output_names.append(k)
            else:
                for i in range(v):
                    output_names.append(k + str(i))

        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults,
        )

    @classmethod
    def list_pandas_ta_indicators(cls, silence_warnings: bool = True, **kwargs) -> tp.List[str]:
        """List all parseable indicators in Pandas TA.

        This class method iterates over the indicator functions available in Pandas TA and
        attempts to parse each one's configuration using `IndicatorFactory.parse_pandas_ta_config`.
        Only indicator functions that are successfully parsed are included in the final list.

        Args:
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `IndicatorFactory.parse_pandas_ta_config`.

        Returns:
            List[str]: Sorted list of indicator names in uppercase that were successfully parsed.

        !!! note
            Returns only the indicators that have been successfully parsed.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pandas_ta")
        try:
            import pandas_ta
        except ImportError as e:
            if "cannot import name 'NaN' from 'numpy'" in str(e):
                warn(
                    "Cannot import name 'NaN' from 'numpy'. "
                    "Current version of Pandas TA is not compatible with the latest NumPy. "
                    "Please install an older version of NumPy, e.g., 1.26.4."
                )
                return []
            raise e

        indicators = set()
        for func_name in [_k for k, v in pandas_ta.Category.items() for _k in v]:
            try:
                cls.parse_pandas_ta_config(
                    getattr(pandas_ta, func_name), silence_warnings=silence_warnings, **kwargs
                )
                indicators.add(func_name.upper())
            except Exception as e:
                if not silence_warnings:
                    warn(f"Function {func_name}: " + str(e))
        return sorted(indicators)

    @classmethod
    def from_pandas_ta(
        cls,
        func_name: str,
        parse_kwargs: tp.KwargsLike = None,
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a Pandas TA function.

        Requires `pandas-ta` installed. See https://github.com/twopirllc/pandas-ta for details.

        Args:
            func_name (str): Name of the `pandas_ta` function to wrap.

                The function name is case-insensitive.
            parse_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory.parse_pandas_ta_config`.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            Type[IndicatorBase]: New indicator class wrapping the specified Pandas TA function.

        Examples:
            ```pycon
            >>> SMA = vbt.IF.from_pandas_ta('SMA')

            >>> sma = SMA.run(price, length=[2, 3])
            >>> sma.sma
            sma_length         2         3
                          a    b    a    b
            2020-01-01  NaN  NaN  NaN  NaN
            2020-01-02  1.5  4.5  NaN  NaN
            2020-01-03  2.5  3.5  2.0  4.0
            2020-01-04  3.5  2.5  3.0  3.0
            2020-01-05  4.5  1.5  4.0  2.0
            ```
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pandas_ta")
        try:
            import pandas_ta
        except ImportError as e:
            if "cannot import name 'NaN' from 'numpy'" in str(e):
                raise ImportError(
                    "Current version of Pandas TA is not compatible with the latest NumPy. "
                    "Please install an older version of NumPy, e.g., 1.26.4."
                ) from e
            raise e

        func_name = func_name.lower()
        func = getattr(pandas_ta, func_name)

        if parse_kwargs is None:
            parse_kwargs = {}
        config = cls.parse_pandas_ta_config(func, **parse_kwargs)

        def apply_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
            param_tuple: tp.Tuple[tp.ParamValue, ...],
            **kwargs_,
        ) -> tp.MaybeTuple[tp.Array2d]:
            is_series = isinstance(input_tuple[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_tuple[0].columns)
            outputs = []
            for col in range(n_input_cols):
                output = suppress_stdout(func)(
                    **{
                        name: input_tuple[i] if is_series else input_tuple[i].iloc[:, col]
                        for i, name in enumerate(config["input_names"])
                    },
                    **{name: param_tuple[i] for i, name in enumerate(config["param_names"])},
                    **kwargs_,
                )
                if isinstance(output, tuple):
                    _outputs = []
                    for o in output:
                        if len(input_tuple[0].index) == len(o.index):
                            _outputs.append(o)
                    if len(_outputs) > 1:
                        output = pd.concat(_outputs, axis=1)
                    elif len(_outputs) == 1:
                        output = _outputs[0]
                    else:
                        raise ValueError("No valid outputs were returned")
                if isinstance(output, pd.DataFrame):
                    output = tuple([output.iloc[:, i] for i in range(len(output.columns))])
                outputs.append(output)
            if isinstance(outputs[0], tuple):
                outputs = list(zip(*outputs))
                return tuple(map(column_stack_arrays, outputs))
            return column_stack_arrays(outputs)

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(
            **merge_dicts(dict(module_name=__name__ + ".pandas_ta"), config, factory_kwargs),
        ).with_apply_func(apply_func, pass_packed=True, keep_pd=True, to_2d=False, **kwargs)
        return Indicator

    @classmethod
    def list_ta_indicators(cls, uppercase: bool = False) -> tp.List[str]:
        """Return a sorted list of parseable indicator class names from the TA module.

        Args:
            uppercase (bool): Whether to convert indicator names to uppercase.

                Otherwise, names are returned in their original case.

        Returns:
            List[str]: Sorted list of indicator class names.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ta")
        import ta

        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        indicators = set()
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and obj != ta.utils.IndicatorMixin
                    and issubclass(obj, ta.utils.IndicatorMixin)
                ):
                    if uppercase:
                        indicators.add(obj.__name__.upper())
                    else:
                        indicators.add(obj.__name__)
        return sorted(indicators)

    @classmethod
    def find_ta_indicator(cls, cls_name: str) -> IndicatorMixinT:
        """Return a TA indicator class by its name.

        Searches through modules in the TA package for an indicator class
        whose name matches the provided value (case-insensitive).

        Args:
            cls_name (str): Name of the indicator class to find.

        Returns:
            IndicatorMixin: Corresponding TA indicator class.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ta")
        import ta

        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            for attr in dir(module):
                if cls_name.upper() == attr.upper():
                    return getattr(module, attr)
        raise AttributeError(f"Indicator '{cls_name}' not found")

    @classmethod
    def parse_ta_config(cls, ind_cls: IndicatorMixinT) -> tp.Kwargs:
        """Parse the configuration of a TA indicator class.

        Inspects the signature and docstring of the given indicator class to extract
        input names, parameter names, default values, and output names.

        Args:
            ind_cls (IndicatorMixin): TA indicator class to parse.

        Returns:
            dict: Dictionary containing:

                * `class_name`: Name of the indicator class.
                * `class_docstring`: Docstring of the indicator class.
                * `input_names`: List of input parameter names.
                * `param_names`: List of parameter names.
                * `output_names`: List of output attribute names.
                * `defaults`: Mapping of parameter names to their default values.
        """
        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        sig = inspect.signature(ind_cls)
        for k, v in sig.parameters.items():
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                if v.annotation == inspect.Parameter.empty:
                    raise ValueError(f'Argument "{k}" has no annotation')
                if v.annotation == pd.Series:
                    input_names.append(k)
                else:
                    param_names.append(k)
                    if v.default != inspect.Parameter.empty:
                        defaults[k] = v.default

        for attr in dir(ind_cls):
            if not attr.startswith("_"):
                if inspect.signature(getattr(ind_cls, attr)).return_annotation == pd.Series or "Returns:\n            pandas.Series" in getattr(ind_cls, attr).__doc__:
                    output_names.append(attr)

        return dict(
            class_name=ind_cls.__name__,
            class_docstring=ind_cls.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults,
        )

    @classmethod
    def from_ta(
        cls, cls_name: str, factory_kwargs: tp.KwargsLike = None, **kwargs
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a TA technical analysis indicator.

        Requires [ta](https://github.com/bukosabino/ta) to be installed.

        Args:
            cls_name (str): Name of the target TA class.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            Type[IndicatorBase]: Built indicator class.

        Examples:
            ```pycon
            >>> SMAIndicator = vbt.IF.from_ta('SMAIndicator')

            >>> sma = SMAIndicator.run(price, window=[2, 3])
            >>> sma.sma_indicator
            smaindicator_window    2         3
                                   a    b    a    b
            2020-01-01           NaN  NaN  NaN  NaN
            2020-01-02           1.5  4.5  NaN  NaN
            2020-01-03           2.5  3.5  2.0  4.0
            2020-01-04           3.5  2.5  3.0  3.0
            2020-01-05           4.5  1.5  4.0  2.0
            ```
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ta")

        ind_cls = cls.find_ta_indicator(cls_name)
        config = cls.parse_ta_config(ind_cls)

        def apply_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
            param_tuple: tp.Tuple[tp.ParamValue, ...],
            **kwargs_,
        ) -> tp.MaybeTuple[tp.Array2d]:
            is_series = isinstance(input_tuple[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_tuple[0].columns)
            outputs = []
            for col in range(n_input_cols):
                ind = ind_cls(
                    **{
                        name: input_tuple[i] if is_series else input_tuple[i].iloc[:, col]
                        for i, name in enumerate(config["input_names"])
                    },
                    **{name: param_tuple[i] for i, name in enumerate(config["param_names"])},
                    **kwargs_,
                )
                output = []
                for output_name in config["output_names"]:
                    output.append(getattr(ind, output_name)())
                if len(output) == 1:
                    output = output[0]
                else:
                    output = tuple(output)
                outputs.append(output)
            if isinstance(outputs[0], tuple):
                outputs = list(zip(*outputs))
                return tuple(map(column_stack_arrays, outputs))
            return column_stack_arrays(outputs)

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(
            **merge_dicts(dict(module_name=__name__ + ".ta"), config, factory_kwargs)
        ).with_apply_func(
            apply_func,
            pass_packed=True,
            keep_pd=True,
            to_2d=False,
            **kwargs,
        )
        return Indicator

    @classmethod
    def parse_technical_config(cls, func: tp.Callable, test_index_len: int = 100) -> tp.Kwargs:
        """Parse the configuration for a technical indicator function.

        Generates a test DataFrame and inspects the provided function's signature and output
        to extract configuration details, including input names, parameter names, output names,
        and default parameter values.

        Args:
            func (Callable): Technical indicator function to parse.
            test_index_len (int): Number of rows in the generated test DataFrame.

        Returns:
            dict: Configuration dictionary containing:

                * `class_name`: Uppercase name of the function.
                * `class_docstring`: Original function docstring.
                * `input_names`: List of input names derived from the function arguments.
                * `param_names`: List of parameter names for the function.
                * `output_names`: List of output names inferred from the function's return.
                * `defaults`: Default parameter values extracted from the function signature.
        """
        df = pd.DataFrame(
            np.random.randint(1, 10, size=(test_index_len, 5)),
            index=pd.date_range("2020", periods=test_index_len),
            columns=["open", "high", "low", "close", "volume"],
        )

        func_arg_names = get_func_arg_names(func)
        func_kwargs = get_func_kwargs(func)
        args = ()
        input_names = []
        param_names = []
        output_names = []
        defaults = {}

        for arg_name in func_arg_names:
            if arg_name == "field":
                continue
            if arg_name in ("dataframe", "df", "bars"):
                args += (df,)
                if "field" in func_kwargs:
                    input_names.append(func_kwargs["field"])
                else:
                    input_names.extend(["open", "high", "low", "close", "volume"])
            elif arg_name in ("series", "sr"):
                args += (df["close"],)
                input_names.append("close")
            elif arg_name in ("open", "high", "low", "close", "volume"):
                args += (df["close"],)
                input_names.append(arg_name)
            else:
                if arg_name not in func_kwargs:
                    args += (5,)
                else:
                    defaults[arg_name] = func_kwargs[arg_name]
                param_names.append(arg_name)
        if len(input_names) == 0:
            raise ValueError("Couldn't parse the output: unknown input arguments")

        def _validate_series(sr, name: tp.Optional[str] = None):
            if not isinstance(sr, pd.Series):
                raise TypeError("Couldn't parse the output: wrong output type")
            if len(sr.index) != len(df.index):
                raise ValueError("Couldn't parse the output: mismatching index")
            if np.issubdtype(sr.dtype, object):
                raise ValueError("Couldn't parse the output: wrong output data type")
            if name is None and sr.name is None:
                raise ValueError("Couldn't parse the output: missing output name")

        out = suppress_stdout(func)(*args)
        if isinstance(out, list):
            out = np.asarray(out)
        if isinstance(out, np.ndarray):
            out = pd.Series(out)
        if isinstance(out, dict):
            out = pd.DataFrame(out)
        if isinstance(out, tuple):
            out = pd.concat(out, axis=1)
        if isinstance(out, (pd.Series, pd.DataFrame)):
            if isinstance(out, pd.DataFrame):
                for c in out.columns:
                    _validate_series(out[c], name=c)
                    output_names.append(c)
            else:
                if out.name is not None:
                    out_name = out.name
                else:
                    out_name = func.__name__.lower()
                _validate_series(out, name=out_name)
                output_names.append(out_name)
        else:
            raise TypeError("Couldn't parse the output: wrong output type")

        new_output_names = []
        for name in output_names:
            name = name.replace(" ", "").lower()
            if len(output_names) == 1 and name == "close":
                new_output_names.append(func.__name__.lower())
                continue
            if name in ("open", "high", "low", "close", "volume", "data"):
                continue
            new_output_names.append(name)
        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=new_output_names,
            defaults=defaults,
        )

    @classmethod
    def list_technical_indicators(cls, silence_warnings: bool = True, **kwargs) -> tp.List[str]:
        """List all parseable technical indicator functions from the technical module.

        Scans the technical package for functions and attempts to parse each using
        `IndicatorFactory.parse_technical_config`. Returns a sorted list of indicator names in uppercase.

        Args:
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `IndicatorFactory.parse_technical_config`.

        Returns:
            list[str]: Sorted list of technical indicator names in uppercase.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")
        import technical

        match_func = lambda k, v: isinstance(v, FunctionType)
        funcs = search_package(technical, match_func, blacklist=["technical.util"])
        indicators = set()
        for func_name, func in funcs.items():
            try:
                cls.parse_technical_config(func, **kwargs)
                indicators.add(func_name.upper())
            except Exception as e:
                if not silence_warnings:
                    warn(f"Function {func_name}: " + str(e))
        return sorted(indicators)

    @classmethod
    def find_technical_indicator(cls, func_name: str) -> IndicatorMixinT:
        """Get the technical indicator function corresponding to the given name.

        Args:
            func_name (str): Name of the technical indicator function.

        Returns:
            IndicatorMixin: Technical indicator function matching the given name.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")
        import technical

        match_func = lambda k, v: isinstance(v, FunctionType)
        funcs = search_package(technical, match_func, blacklist=["technical.util"])
        for k, v in funcs.items():
            if func_name.upper() == k.upper():
                return v
        raise AttributeError(f"Indicator '{func_name}' not found")

    @classmethod
    def from_technical(
        cls,
        func_name: str,
        parse_kwargs: tp.KwargsLike = None,
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class using the specified technical function.

        This method requires the [technical](https://github.com/freqtrade/technical) package to be installed.

        Args:
            func_name (str): Name of the technical indicator function to wrap.
            parse_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory.parse_technical_config`.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            Indicator: Indicator class constructed around the given technical function.

        Examples:
            ```pycon
            >>> ROLLING_MEAN = vbt.IF.from_technical("ROLLING_MEAN")

            >>> rolling_mean = ROLLING_MEAN.run(price, window=[3, 4])
            >>> rolling_mean.rolling_mean
            rolling_mean_window         3         4
                                   a    b    a    b
            2020-01-01           NaN  NaN  NaN  NaN
            2020-01-02           NaN  NaN  NaN  NaN
            2020-01-03           2.0  4.0  NaN  NaN
            2020-01-04           3.0  3.0  2.5  3.5
            2020-01-05           4.0  2.0  3.5  2.5
            ```
        """
        func = cls.find_technical_indicator(func_name)
        func_arg_names = get_func_arg_names(func)

        if parse_kwargs is None:
            parse_kwargs = {}
        config = cls.parse_technical_config(func, **parse_kwargs)

        def apply_func(
            input_tuple: tp.Tuple[tp.Series, ...],
            in_output_tuple: tp.Tuple[tp.Series, ...],
            param_tuple: tp.Tuple[tp.ParamValue, ...],
            *args_,
            **kwargs_,
        ) -> tp.MaybeTuple[tp.Array1d]:
            input_series = {name: input_tuple[i] for i, name in enumerate(config["input_names"])}
            kwargs_ = {
                **{name: param_tuple[i] for i, name in enumerate(config["param_names"])},
                **kwargs_,
            }
            __args = ()
            for arg_name in func_arg_names:
                if arg_name in ("dataframe", "df", "bars"):
                    __args += (pd.DataFrame(input_series),)
                elif arg_name in ("series", "sr") or arg_name in ("open", "high", "low", "close", "volume"):
                    __args += (input_series["close"],)
                else:
                    break

            out = suppress_stdout(func)(*__args, *args_, **kwargs_)
            if isinstance(out, list):
                out = np.asarray(out)
            if isinstance(out, np.ndarray):
                out = pd.Series(out)
            if isinstance(out, dict):
                out = pd.DataFrame(out)
            if isinstance(out, tuple):
                out = pd.concat(out, axis=1)
            if isinstance(out, pd.DataFrame):
                outputs = []
                for c in out.columns:
                    if len(out.columns) == len(config["output_names"]) or c.replace(" ", "").lower() not in (
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "data",
                    ):
                        outputs.append(out[c].values)
                return tuple(outputs)
            return out.values

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(
            **merge_dicts(dict(module_name=__name__ + ".technical"), config, factory_kwargs),
        ).with_apply_func(apply_func, pass_packed=True, keep_pd=True, takes_1d=True, **kwargs)
        return Indicator

    @classmethod
    def from_custom_techcon(
        cls,
        consensus_cls: tp.Type[ConsensusT],
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Create an indicator based on a technical consensus class that subclasses
        `technical.consensus.consensus.Consensus`.

        Requires the Technical library: https://github.com/freqtrade/technical

        Args:
            consensus_cls (Type): Consensus class that subclasses `technical.consensus.consensus.Consensus`.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            Type: Dynamically created indicator class.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")
        from technical.consensus.consensus import Consensus

        checks.assert_subclass_of(consensus_cls, Consensus)

        def apply_func(
            open: tp.Series,
            high: tp.Series,
            low: tp.Series,
            close: tp.Series,
            volume: tp.Series,
            smooth: tp.Optional[int] = None,
            _consensus_cls: tp.Type[ConsensusT] = consensus_cls,
        ) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d, tp.Array1d]:
            """Apply the consensus function to compute indicator scores.

            Args:
                open (Series): Series of open prices.
                high (Series): Series of high prices.
                low (Series): Series of low prices.
                close (Series): Series of close prices.
                volume (Series): Series of volume data.
                smooth (Optional[int]): Smoothing parameter.

            Returns:
                Tuple[Array1d, Array1d, Array1d, Array1d, Array1d, Array1d]: Tuple containing arrays for:

                    * buy scores,
                    * sell scores,
                    * buy agreement,
                    * sell agreement,
                    * buy disagreement, and
                    * sell disagreement.
            """
            dataframe = pd.DataFrame(
                {
                    "open": open,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
            consensus = _consensus_cls(dataframe)
            score = consensus.score(smooth=smooth)
            return (
                score["buy"].values,
                score["sell"].values,
                score["buy_agreement"].values,
                score["sell_agreement"].values,
                score["buy_disagreement"].values,
                score["sell_disagreement"].values,
            )

        if factory_kwargs is None:
            factory_kwargs = {}
        factory_kwargs = merge_dicts(
            dict(
                class_name="CON",
                module_name=__name__ + ".custom_techcon",
                short_name=None,
                input_names=["open", "high", "low", "close", "volume"],
                param_names=["smooth"],
                output_names=[
                    "buy",
                    "sell",
                    "buy_agreement",
                    "sell_agreement",
                    "buy_disagreement",
                    "sell_disagreement",
                ],
            ),
            factory_kwargs,
        )
        Indicator = cls(**factory_kwargs).with_apply_func(
            apply_func,
            takes_1d=True,
            keep_pd=True,
            smooth=None,
            **kwargs,
        )

        def plot(
            self,
            column: tp.Optional[tp.Column] = None,
            buy_trace_kwargs: tp.KwargsLike = None,
            sell_trace_kwargs: tp.KwargsLike = None,
            add_trace_kwargs: tp.KwargsLike = None,
            fig: tp.Optional[tp.BaseFigure] = None,
            **layout_kwargs,
        ) -> tp.BaseFigure:
            """Plot the buy and sell traces of the indicator.

            Args:
                column (Optional[Column]): Identifier of the column to plot.
                buy_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the buy line.
                sell_trace_kwargs (KwargsLike): Keyword arguments for `plotly.graph_objects.Scatter` for the sell line.
                add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace;
                    for example, `dict(row=1, col=1)`.
                fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
                **layout_kwargs: Keyword arguments for `fig.update_layout`.

            Returns:
                BaseFigure: Updated figure with plotted buy and sell traces.

            !!! info
                For default settings, see `vectorbtpro._settings.plotting`.
            """
            from vectorbtpro._settings import settings
            from vectorbtpro.utils.figure import make_figure

            plotting_cfg = settings["plotting"]

            self_col = self.select_col(column=column, group_by=False)

            if fig is None:
                fig = make_figure()
            fig.update_layout(**layout_kwargs)

            if buy_trace_kwargs is None:
                buy_trace_kwargs = {}
            if sell_trace_kwargs is None:
                sell_trace_kwargs = {}
            buy_trace_kwargs = merge_dicts(
                dict(name="Buy", line=dict(color=plotting_cfg["color_schema"]["green"])),
                buy_trace_kwargs,
            )
            sell_trace_kwargs = merge_dicts(
                dict(name="Sell", line=dict(color=plotting_cfg["color_schema"]["red"])),
                sell_trace_kwargs,
            )

            fig = self_col.buy.vbt.lineplot(
                trace_kwargs=buy_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            fig = self_col.sell.vbt.lineplot(
                trace_kwargs=sell_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

            return fig

        Indicator.plot = plot
        return Indicator

    @classmethod
    def from_techcon(cls, cls_name: str, **kwargs) -> tp.Type[IndicatorBase]:
        """Create an indicator from a preset technical consensus.

        Args:
            cls_name (str): Name of the technical consensus indicator.

                Supported values (case-insensitive) include:

                * `MACON` (or `MovingAverageConsensus`),
                * `OSCCON` (or `OscillatorConsensus`), and
                * `SUMCON` (or `SummaryConsensus`).
            **kwargs: Keyword arguments for `IndicatorFactory.from_custom_techcon`.

        Returns:
            Type[IndicatorBase]: Created indicator class.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("technical")

        if cls_name.lower() in ("MACON".lower(), "MovingAverageConsensus".lower()):
            from technical.consensus.movingaverage import MovingAverageConsensus

            return cls.from_custom_techcon(
                MovingAverageConsensus,
                factory_kwargs=dict(module_name=__name__ + ".techcon", class_name="MACON"),
                **kwargs,
            )
        if cls_name.lower() in ("OSCCON".lower(), "OscillatorConsensus".lower()):
            from technical.consensus.oscillator import OscillatorConsensus

            return cls.from_custom_techcon(
                OscillatorConsensus,
                factory_kwargs=dict(module_name=__name__ + ".techcon", class_name="OSCCON"),
                **kwargs,
            )
        if cls_name.lower() in ("SUMCON".lower(), "SummaryConsensus".lower()):
            from technical.consensus.summary import SummaryConsensus

            return cls.from_custom_techcon(
                SummaryConsensus,
                factory_kwargs=dict(module_name=__name__ + ".techcon", class_name="SUMCON"),
                **kwargs,
            )
        raise ValueError(f"Unknown technical consensus class '{cls_name}'")

    @classmethod
    def list_techcon_indicators(cls) -> tp.List[str]:
        """List all consensus indicators available in technical.

        Returns:
            List[str]: Sorted list of consensus indicator names.
        """
        return sorted({"MACON", "OSCCON", "SUMCON"})

    @classmethod
    def find_smc_indicator(
        cls, func_name: str, raise_error: bool = True
    ) -> tp.Optional[tp.Callable]:
        """Get the Smart Money Concepts indicator class by its name.

        Args:
            func_name (str): Name of the smart money concepts indicator.
            raise_error (bool): Flag indicating whether to raise an error if the indicator is not found.

        Returns:
            Optional[Callable]: Indicator class if found; otherwise, None.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("smartmoneyconcepts")
        from smartmoneyconcepts import smc

        for k in dir(smc):
            if not k.startswith("_"):
                if camel_to_snake_case(func_name) == camel_to_snake_case(k):
                    return getattr(smc, k)
        if raise_error:
            raise AttributeError(f"Indicator '{func_name}' not found")
        return None

    @classmethod
    def parse_smc_config(
        cls, func: tp.Callable, collapse: bool = True, snake_case: bool = True
    ) -> tp.Kwargs:
        """Parse the configuration of a Smart Money Concepts indicator function.

        Inspects the signature and source code of the given function to extract input names,
        parameter names, output names, default values, and nested indicator configurations.

        Args:
            func (Callable): Smartmoneyconcepts indicator function to parse.
            collapse (bool): Flag to collapse nested indicators' configurations into a single set.
            snake_case (bool): Flag to convert names to snake case.

        Returns:
            dict: Dictionary containing the parsed configuration with the following keys:

                * `class_name`: Uppercase name of the function.
                * `class_docstring`: Original docstring of the function.
                * `input_names`: List of input column names.
                * `param_names`: List of parameter names.
                * `output_names`: List of output names extracted from the function source.
                * `defaults`: Default values for the parameters.
                * `dep_input_names`: Mapping of nested indicator names to their dependent input names.
        """
        func_arg_names = get_func_arg_names(func)
        input_names = []
        param_names = []
        defaults = {}
        dep_input_names = {}
        sig = inspect.signature(func)
        for k in func_arg_names:
            if k == "ohlc":
                input_names.extend(["open", "high", "low", "close", "volume"])
            else:
                found_smc_indicator = cls.find_smc_indicator(k, raise_error=False)
                if found_smc_indicator is not None:
                    dep_input_names[k] = []
                    k_func_config = cls.parse_smc_config(found_smc_indicator)
                    if collapse:
                        for input_name in k_func_config["input_names"]:
                            if input_name not in input_names:
                                input_names.append(input_name)
                        for param_name in k_func_config["param_names"]:
                            if param_name not in param_names:
                                param_names.append(param_name)
                        for k2, v2 in k_func_config["defaults"].items():
                            defaults[k2] = v2
                    else:
                        for output_name in k_func_config["output_names"]:
                            if output_name not in input_names:
                                input_names.append(output_name)
                            dep_input_names[k].append(output_name)
                else:
                    v = sig.parameters[k]
                    if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                        if v.default == inspect.Parameter.empty and v.annotation == pd.DataFrame:
                            if k not in input_names:
                                input_names.append(k)
                        else:
                            if k not in param_names:
                                param_names.append(k)
                            if v.default != inspect.Parameter.empty:
                                defaults[k] = v.default

        func_doc = inspect.getsource(func)
        output_names = re.findall(r'name="([^"]+)"', func_doc)
        output_names = [k.replace("%", "") for k in output_names]
        if snake_case:
            input_names = list(map(camel_to_snake_case, input_names))
            param_names = list(map(camel_to_snake_case, param_names))
            output_names = list(map(camel_to_snake_case, output_names))
        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults,
            dep_input_names=dep_input_names,
        )

    @classmethod
    def list_smc_indicators(cls, silence_warnings: bool = True, **kwargs) -> tp.List[str]:
        """List all parseable indicators from the Smart Money Concepts package.

        Inspects each public function in the `smartmoneyconcepts.smc` module and returns
        those that can be successfully parsed into a configuration.

        Args:
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `IndicatorFactory.parse_smc_config`.

        Returns:
            List[str]: Sorted list of indicator names in uppercase.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("smartmoneyconcepts")
        from smartmoneyconcepts import smc

        indicators = set()
        for func_name in dir(smc):
            if not func_name.startswith("_"):
                try:
                    cls.parse_smc_config(getattr(smc, func_name), **kwargs)
                    indicators.add(func_name.upper())
                except Exception as e:
                    if not silence_warnings:
                        warn(f"Function {func_name}: " + str(e))
        return sorted(indicators)

    @classmethod
    def from_smc(
        cls,
        func_name: str,
        collapse: bool = True,
        parse_kwargs: tp.KwargsLike = None,
        factory_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Type[IndicatorBase]:
        """Build an indicator class using a Smart Money Concepts function.

        Requires [smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts)
        to be installed.

        Args:
            func_name (str): Name of the smartmoneyconcepts function to wrap.
            collapse (bool): Flag to collapse nested indicators' configurations into a single set.
            parse_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory.parse_smc_config`.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            Type[IndicatorBase]: Indicator class built around the specified smartmoneyconcepts function.
        """
        func = cls.find_smc_indicator(func_name)
        func_arg_names = get_func_arg_names(func)

        if parse_kwargs is None:
            parse_kwargs = {}
        collapsed_config = cls.parse_smc_config(
            func, collapse=True, snake_case=True, **parse_kwargs
        )
        _ = collapsed_config.pop("dep_input_names")
        expanded_config = cls.parse_smc_config(
            func, collapse=False, snake_case=True, **parse_kwargs
        )
        dep_input_names = expanded_config.pop("dep_input_names")
        if collapse:
            config = collapsed_config
        else:
            config = expanded_config

        def apply_func(
            input_tuple: tp.Tuple[tp.Series, ...],
            in_output_tuple: tp.Tuple[tp.Series, ...],
            param_tuple: tp.Tuple[tp.ParamValue, ...],
            **kwargs_,
        ) -> tp.MaybeTuple[tp.Array1d]:
            named_args = dict(kwargs_)
            for i, input_name in enumerate(config["input_names"]):
                named_args[input_name] = input_tuple[i]
            for i, param_name in enumerate(config["param_names"]):
                named_args[param_name] = param_tuple[i]
            named_args["ohlc"] = pd.concat(
                [
                    named_args["open"].rename("open"),
                    named_args["high"].rename("high"),
                    named_args["low"].rename("low"),
                    named_args["close"].rename("close"),
                    named_args["volume"].rename("volume"),
                ],
                axis=1,
            )
            if collapse and len(dep_input_names) > 0:
                for dep_func_name in dep_input_names:
                    dep_func = cls.find_smc_indicator(dep_func_name)
                    dep_func_arg_names = get_func_arg_names(dep_func)
                    dep_output = dep_func(
                        *[named_args[camel_to_snake_case(k)] for k in dep_func_arg_names]
                    )
                    dep_output.index = input_tuple[0].index
                    named_args[dep_func_name] = dep_output
            elif not collapse:
                for dep_func_name in dep_input_names:
                    dep_func = cls.find_smc_indicator(dep_func_name)
                    dep_config = cls.parse_smc_config(
                        dep_func, collapse=False, snake_case=False, **parse_kwargs
                    )
                    named_args[dep_func_name] = pd.concat(
                        [named_args[input_name] for input_name in dep_input_names[dep_func_name]],
                        axis=1,
                        keys=dep_config["output_names"],
                    )
            output = func(*[named_args[camel_to_snake_case(k)] for k in func_arg_names])
            return tuple([output[c] for c in output.columns])

        kwargs = merge_dicts({k: Default(v) for k, v in config.pop("defaults").items()}, kwargs)
        Indicator = cls(
            **merge_dicts(dict(module_name=__name__ + ".smc"), config, factory_kwargs),
        ).with_apply_func(apply_func, pass_packed=True, keep_pd=True, takes_1d=True, **kwargs)
        return Indicator

    @hybrid_method
    def from_expr(
        cls_or_self,
        expr: str,
        parse_annotations: bool = True,
        factory_kwargs: tp.KwargsLike = None,
        magnet_inputs: tp.Iterable[str] = None,
        magnet_in_outputs: tp.Iterable[str] = None,
        magnet_params: tp.Iterable[str] = None,
        func_mapping: tp.KwargsLike = None,
        res_func_mapping: tp.KwargsLike = None,
        use_pd_eval: tp.Optional[bool] = None,
        pd_eval_kwargs: tp.KwargsLike = None,
        return_clean_expr: bool = False,
        **kwargs,
    ) -> tp.Union[str, tp.Type[IndicatorBase]]:
        """Build an indicator class from an indicator expression.

        Builds a new indicator class based on a Python expression string.

        Args:
            expr (str): Expression string.

                Must contain valid Python code and can be single-line or multi-line.
            parse_annotations (bool): Flag to parse annotations starting with `@`.
            factory_kwargs (KwargsLike): Keyword arguments for `IndicatorFactory`.

                Applied only when invoking the class method.
            magnet_inputs (Iterable[str]): Names to be recognized as input variables.

                Defaults to `open`, `high`, `low`, `close`, and `volume`.
            magnet_in_outputs (Iterable[str]): Names to be recognized as in-place output variables.

                Defaults to an empty list.
            magnet_params (Iterable[str]): Names to be recognized as parameter variables.

                Defaults to an empty list.
            func_mapping (KwargsLike): Mapping to merge with `vectorbtpro.indicators.expr.expr_func_config`.

                Each key is a function name with a dictionary value containing a `func` entry and optionally
                `magnet_inputs`, `magnet_in_outputs`, and `magnet_params`.
            res_func_mapping (KwargsLike): Mapping to merge with `vectorbtpro.indicators.expr.expr_res_func_config`.

                Each key is a function name with a dictionary value containing a `func` entry and optionally
                `magnet_inputs`, `magnet_in_outputs`, and `magnet_params`.
            use_pd_eval (Optional[bool]): Whether to use `pd.eval` for evaluation.

                Defaults to False. Otherwise, uses `vectorbtpro.utils.eval_.evaluate`.

                !!! hint
                    By default, operates on NumPy objects using NumExpr.
                    If you want to operate on Pandas objects, set `keep_pd` to True.
            pd_eval_kwargs (KwargsLike): Keyword arguments for `pd.eval`.
            return_clean_expr (bool): Flag indicating whether to return the cleaned expression.
            **kwargs: Keyword arguments for `IndicatorFactory.with_apply_func`.

        Returns:
            Union[str, Type[IndicatorBase]]: If `return_clean_expr` is True, returns the cleaned
                expression string; otherwise, returns the generated indicator class.

        Searches each variable name parsed from `expr` in:

        * `vectorbtpro.indicators.expr.expr_res_func_config` (calls are executed immediately)
        * `vectorbtpro.indicators.expr.expr_func_config`
        * Input, in-place output, and parameter names
        * Keyword arguments
        * Attributes of `np`
        * Attributes of `vectorbtpro.generic.nb` (both with and without `_nb` suffix)
        * Attributes of `vbt`

        The configurations in `vectorbtpro.indicators.expr.expr_func_config` and
        `vectorbtpro.indicators.expr.expr_res_func_config` can be overridden via `func_mapping`
        and `res_func_mapping`, respectively.

        !!! note
            Each variable name is case-sensitive.

        When invoked as a class method, variable names are parsed directly from the expression.
        If any of `open`, `high`, `low`, `close`, or `volume` appear in the expression or are listed in
        `magnet_inputs` (within either `expr_func_config` or `expr_res_func_config`), they are automatically
        added to `input_names`. Set `magnet_inputs` to an empty list to disable this behavior.

        If the expression starts with a valid variable name followed by a colon (`:`),
        that name is used as the generated class name. Provide an additional variable name enclosed in square
        brackets immediately before the colon to specify the indicator's short name.

        If `parse_annotations` is True, annotations beginning with `@` define variable roles:

        * `@in_*`: Input variable.
        * `@inout_*`: In-place output variable.
        * `@p_*`: Parameter variable.
        * `@out_*`: Output variable.
        * `@out_*:`: Indicates that the subsequent part until a comma represents an output.
        * `@talib_*`: Specifies a TA-Lib function to be used via the indicator's `apply_func`.
        * `@res_*`: Specifies an indicator to resolve automatically. Input names may overlap with those
            of other indicators; all other details are prefixed with the indicator's short name.
        * `@settings(*)`: Settings to merge with the current `IndicatorFactory.from_expr` configuration.
            The contents within the parentheses are evaluated with Python's `eval` and must yield a dictionary.
            These settings override defaults but are themselves overridden by any arguments passed to this method.
            The parameters `expr` and `parse_annotations` cannot be overridden.

        !!! note
            Variable names are parsed in the order they appear in the expression, except for magnet
            input names which follow the order in `magnet_inputs`.

        The number of outputs is determined by counting commas outside any bracket pair.
        If there is only one output, it is named `out`; for multiple outputs, they are named `out1`, `out2`, etc.

        Any of these settings can be overridden using `factory_kwargs`.

        The code context includes all variables from `vectorbtpro.imported_star`.

        Examples:
            ```pycon
            >>> WMA = vbt.IF(
            ...     class_name='WMA',
            ...     input_names=['close'],
            ...     param_names=['window'],
            ...     output_names=['wma']
            ... ).from_expr("wm_mean_nb(close, window)")

            >>> wma = WMA.run(price, window=[2, 3])
            >>> wma.wma
            wma_window                   2                   3
                               a         b         a         b
            2020-01-01       NaN       NaN       NaN       NaN
            2020-01-02  1.666667  4.333333       NaN       NaN
            2020-01-03  2.666667  3.333333  2.333333  3.666667
            2020-01-04  3.666667  2.333333  3.333333  2.666667
            2020-01-05  4.666667  1.333333  4.333333  1.666667
            ```

            The same can be achieved by calling the class method and providing prefixes
            to the variable names to indicate their type:

            ```pycon
            >>> expr = "WMA: @out_wma:wm_mean_nb((@in_high + @in_low) / 2, @p_window)"
            >>> WMA = vbt.IF.from_expr(expr)
            >>> wma = WMA.run(price + 1, price, window=[2, 3])
            >>> wma.wma
            wma_window                   2                   3
                               a         b         a         b
            2020-01-01       NaN       NaN       NaN       NaN
            2020-01-02  2.166667  4.833333       NaN       NaN
            2020-01-03  3.166667  3.833333  2.833333  4.166667
            2020-01-04  4.166667  2.833333  3.833333  3.166667
            2020-01-05  5.166667  1.833333  4.833333  2.166667
            ```

            Magnet names are recognized automatically:

            ```pycon
            >>> expr = "WMA: @out_wma:wm_mean_nb((high + low) / 2, @p_window)"
            ```

            Most settings of this method can be overridden within the expression:

            ```pycon
            >>> expr = \"\"\"
            ... @settings({factory_kwargs={'class_name': 'WMA', 'param_names': ['window']}})
            ... @out_wma:wm_mean_nb((high + low) / 2, window)
            ... \"\"\"
            ```
        """

        def _clean_expr(expr: str) -> str:
            expr = inspect.cleandoc(expr).strip()
            if expr.endswith(","):
                expr = expr[:-1]
            if expr.startswith("(") and expr.endswith(")"):
                n_open_brackets = 0
                remove_brackets = True
                for i, s in enumerate(expr):
                    if s == "(":
                        n_open_brackets += 1
                    elif s == ")":
                        n_open_brackets -= 1
                        if n_open_brackets == 0 and i < len(expr) - 1:
                            remove_brackets = False
                            break
                if remove_brackets:
                    expr = expr[1:-1]
            if expr.endswith(","):
                expr = expr[:-1]
            return expr

        if isinstance(cls_or_self, type):
            settings = dict(
                factory_kwargs=dict(
                    class_name=None,
                    input_names=[],
                    in_output_names=[],
                    param_names=[],
                    output_names=[],
                )
            )

            match = re.match(
                r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[([a-zA-Z_][a-zA-Z0-9_]*)\])?\s*:\s*", expr
            )
            if match:
                settings["factory_kwargs"]["class_name"] = match.group(1)
                if match.group(2):
                    settings["factory_kwargs"]["short_name"] = match.group(2)
                expr = expr[len(match.group(0)) :]

            if "@settings" in expr:
                remove_chars = set()
                for m in re.finditer("@settings", expr):
                    n_open_brackets = 0
                    from_i = None
                    to_i = None
                    for i in range(m.start(), m.end()):
                        remove_chars.add(i)
                    for i in range(m.end(), len(expr)):
                        remove_chars.add(i)
                        s = expr[i]
                        if s in "(":
                            if n_open_brackets == 0:
                                from_i = i + 1
                            n_open_brackets += 1
                        elif s in ")":
                            n_open_brackets -= 1
                            if n_open_brackets == 0:
                                to_i = i
                                break
                    if n_open_brackets != 0:
                        raise ValueError("Couldn't parse the settings: mismatching brackets")
                    settings = merge_dicts(settings, eval(_clean_expr(expr[from_i:to_i])))
                expr = "".join([expr[i] for i in range(len(expr)) if i not in remove_chars])

            expr = _clean_expr(expr)

            parsed_factory_kwargs = settings.pop("factory_kwargs")
            magnet_inputs = settings.pop("magnet_inputs", magnet_inputs)
            magnet_in_outputs = settings.pop("magnet_in_outputs", magnet_in_outputs)
            magnet_params = settings.pop("magnet_params", magnet_params)
            func_mapping = merge_dicts(
                expr_func_config, settings.pop("func_mapping", None), func_mapping
            )
            res_func_mapping = merge_dicts(
                expr_res_func_config,
                settings.pop("res_func_mapping", None),
                res_func_mapping,
            )
            use_pd_eval = settings.pop("use_pd_eval", use_pd_eval)
            pd_eval_kwargs = merge_dicts(settings.pop("pd_eval_kwargs", None), pd_eval_kwargs)

            if use_pd_eval is None:
                use_pd_eval = False
            if magnet_inputs is None:
                magnet_inputs = ["open", "high", "low", "close", "volume"]
            if magnet_in_outputs is None:
                magnet_in_outputs = []
            if magnet_params is None:
                magnet_params = []
            found_magnet_inputs = []
            found_magnet_in_outputs = []
            found_magnet_params = []
            found_defaults = {}
            remove_defaults = set()

            if parse_annotations:
                for var_name in re.findall(r"@[a-z]+_[a-zA-Z_][a-zA-Z0-9_]*", expr):
                    var_name = var_name.replace("@", "")
                    if var_name.startswith("in_"):
                        var_name = var_name[3:]
                        if var_name in magnet_inputs:
                            if var_name not in found_magnet_inputs:
                                found_magnet_inputs.append(var_name)
                        else:
                            if var_name not in parsed_factory_kwargs["input_names"]:
                                parsed_factory_kwargs["input_names"].append(var_name)
                    elif var_name.startswith("inout_"):
                        var_name = var_name[6:]
                        if var_name in magnet_in_outputs:
                            if var_name not in found_magnet_in_outputs:
                                found_magnet_in_outputs.append(var_name)
                        else:
                            if var_name not in parsed_factory_kwargs["in_output_names"]:
                                parsed_factory_kwargs["in_output_names"].append(var_name)
                    elif var_name.startswith("p_"):
                        var_name = var_name[2:]
                        if var_name in magnet_params:
                            if var_name not in found_magnet_params:
                                found_magnet_params.append(var_name)
                        else:
                            if var_name not in parsed_factory_kwargs["param_names"]:
                                parsed_factory_kwargs["param_names"].append(var_name)
                    elif var_name.startswith("res_"):
                        ind_name = var_name[4:]
                        if ind_name.startswith("talib_"):
                            ind_name = ind_name[6:]
                            I = cls_or_self.from_talib(ind_name)
                        else:
                            I = kwargs[ind_name]
                        if not issubclass(I, IndicatorBase):
                            raise TypeError(
                                f"Indicator class '{ind_name}' must subclass IndicatorBase"
                            )

                        def _ind_func(context: tp.Kwargs, _I: IndicatorBase = I) -> tp.Any:
                            args_ = ()
                            kwargs_ = {}
                            signature = inspect.signature(_I.run)
                            for p in signature.parameters.values():
                                if p.name in _I.input_names:
                                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                                        args_ += (context[p.name],)
                                    else:
                                        kwargs_[p.name] = context[p.name]
                                else:
                                    ind_p_name = _I.short_name + "_" + p.name
                                    if ind_p_name in context:
                                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                                            args_ += (context[ind_p_name],)
                                        elif p.kind == p.VAR_POSITIONAL:
                                            args_ += context[ind_p_name]
                                        elif p.kind == p.VAR_KEYWORD:
                                            for k, v in context[ind_p_name].items():
                                                kwargs_[k] = v
                                        else:
                                            kwargs_[p.name] = context[ind_p_name]
                            return_raw = kwargs_.pop("return_raw", True)
                            ind = _I.run(*args_, return_raw=return_raw, **kwargs_)
                            if return_raw:
                                raw_outputs = ind[0]
                                if len(raw_outputs) == 1:
                                    return raw_outputs[0]
                                return raw_outputs
                            return ind

                        res_func_mapping["__" + var_name] = dict(
                            func=_ind_func,
                            magnet_inputs=I.input_names,
                            magnet_in_outputs=[
                                I.short_name + "_" + name for name in I.in_output_names
                            ],
                            magnet_params=[I.short_name + "_" + name for name in I.param_names],
                        )
                        run_kwargs = get_func_kwargs(I.run)

                        def _add_defaults(names, prefix=None):
                            for k in names:
                                if prefix is None:
                                    k_prefixed = k
                                else:
                                    k_prefixed = prefix + "_" + k
                                if k in run_kwargs:
                                    if k_prefixed in found_defaults:
                                        if not checks.is_deep_equal(
                                            found_defaults[k_prefixed], run_kwargs[k]
                                        ):
                                            remove_defaults.add(k_prefixed)
                                    else:
                                        found_defaults[k_prefixed] = run_kwargs[k]

                        _add_defaults(I.input_names)
                        _add_defaults(I.in_output_names, I.short_name)
                        _add_defaults(I.param_names, I.short_name)

                expr = expr.replace("@in_", "__in_")
                expr = expr.replace("@inout_", "__inout_")
                expr = expr.replace("@p_", "__p_")
                expr = expr.replace("@talib_", "__talib_")
                expr = expr.replace("@res_", "__res_")

                to_replace = []
                for var_name in re.findall(r"@out_[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*", expr):
                    to_replace.append(var_name)
                    var_name = var_name.split(":")[0].strip()[5:]
                    if var_name not in parsed_factory_kwargs["output_names"]:
                        parsed_factory_kwargs["output_names"].append(var_name)
                for s in to_replace:
                    expr = expr.replace(s, "")

                for var_name in re.findall(r"@out_[a-zA-Z_][a-zA-Z0-9_]*", expr):
                    var_name = var_name.replace("@", "")
                    if var_name.startswith("out_"):
                        var_name = var_name[4:]
                        if var_name not in parsed_factory_kwargs["output_names"]:
                            parsed_factory_kwargs["output_names"].append(var_name)

                expr = expr.replace("@out_", "__out_")

                if len(parsed_factory_kwargs["output_names"]) == 0:
                    lines = expr.split("\n")
                    if len(lines) > 1:
                        last_line = _clean_expr(lines[-1])
                        valid_output_names = []
                        found_not_valid = False
                        for i, out in enumerate(last_line.split(",")):
                            out = out.strip()
                            if not out.startswith("__") and out.isidentifier():
                                valid_output_names.append(out)
                            else:
                                found_not_valid = True
                                break
                        if not found_not_valid:
                            parsed_factory_kwargs["output_names"] = valid_output_names

            var_names = get_expr_var_names(expr)

            def _find_magnets(magnet_type, magnet_names, magnet_lst, found_magnet_lst):
                for var_name in var_names:
                    if var_name in magnet_lst:
                        if var_name not in found_magnet_lst:
                            found_magnet_lst.append(var_name)
                    if var_name in func_mapping:
                        for magnet_name in func_mapping[var_name].get(magnet_type, []):
                            if magnet_name not in found_magnet_lst:
                                found_magnet_lst.append(magnet_name)
                    if var_name in res_func_mapping:
                        for magnet_name in res_func_mapping[var_name].get(magnet_type, []):
                            if magnet_name not in found_magnet_lst:
                                found_magnet_lst.append(magnet_name)
                for magnet_name in magnet_lst:
                    if magnet_name in found_magnet_lst and magnet_name not in magnet_names:
                        magnet_names.append(magnet_name)
                for magnet_name in found_magnet_lst:
                    if magnet_name not in magnet_names and magnet_name not in magnet_names:
                        magnet_names.append(magnet_name)

            _find_magnets(
                "magnet_inputs",
                parsed_factory_kwargs["input_names"],
                magnet_inputs,
                found_magnet_inputs,
            )
            _find_magnets(
                "magnet_in_outputs",
                parsed_factory_kwargs["in_output_names"],
                magnet_in_outputs,
                found_magnet_in_outputs,
            )
            _find_magnets(
                "magnet_params",
                parsed_factory_kwargs["param_names"],
                magnet_params,
                found_magnet_params,
            )

            for k in remove_defaults:
                found_defaults.pop(k, None)

            def _sort_names(names_name):
                new_names = []
                for k in parsed_factory_kwargs[names_name]:
                    if k not in found_defaults:
                        new_names.append(k)
                for k in parsed_factory_kwargs[names_name]:
                    if k in found_defaults:
                        new_names.append(k)
                parsed_factory_kwargs[names_name] = new_names

            _sort_names("input_names")
            _sort_names("in_output_names")
            _sort_names("param_names")

            if len(parsed_factory_kwargs["output_names"]) == 0:
                lines = expr.split("\n")
                last_line = _clean_expr(lines[-1])
                n_open_brackets = 0
                n_outputs = 1
                for i, s in enumerate(last_line):
                    if s == "," and n_open_brackets == 0:
                        n_outputs += 1
                    elif s in "([{":
                        n_open_brackets += 1
                    elif s in ")]}":
                        n_open_brackets -= 1
                if n_open_brackets != 0:
                    raise ValueError("Couldn't parse the number of outputs: mismatching brackets")
                elif len(parsed_factory_kwargs["output_names"]) == 0:
                    if n_outputs == 1:
                        parsed_factory_kwargs["output_names"] = ["out"]
                    else:
                        parsed_factory_kwargs["output_names"] = [
                            "out%d" % (i + 1) for i in range(n_outputs)
                        ]

            factory = cls_or_self(**merge_dicts(parsed_factory_kwargs, factory_kwargs))
            kwargs = merge_dicts(settings, found_defaults, kwargs)
        else:
            func_mapping = merge_dicts(expr_func_config, func_mapping)
            res_func_mapping = merge_dicts(expr_res_func_config, res_func_mapping)

            var_names = get_expr_var_names(expr)

            factory = cls_or_self

        if return_clean_expr:
            return expr

        input_names = factory.input_names
        in_output_names = factory.in_output_names
        param_names = factory.param_names

        def apply_func(
            input_tuple: tp.Tuple[tp.AnyArray, ...],
            in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
            param_tuple: tp.Tuple[tp.ParamValue, ...],
            **kwargs_,
        ) -> tp.MaybeTuple[tp.Array2d]:
            import vectorbtpro as vbt

            input_context = dict(vbt.imported_star)
            for i, input in enumerate(input_tuple):
                input_context[input_names[i]] = input
            for i, in_output in enumerate(in_output_tuple):
                input_context[in_output_names[i]] = in_output
            for i, param in enumerate(param_tuple):
                input_context[param_names[i]] = param
            merged_context = merge_dicts(input_context, kwargs_)
            context = {}

            for var_name in var_names:
                if var_name in context:
                    continue
                if var_name.startswith("__in_"):
                    var = merged_context[var_name[5:]]
                elif var_name.startswith("__inout_"):
                    var = merged_context[var_name[8:]]
                elif var_name.startswith("__p_"):
                    var = merged_context[var_name[4:]]
                elif var_name.startswith("__talib_"):
                    from vectorbtpro.indicators.talib_ import talib_func

                    talib_func_name = var_name[8:].upper()
                    _talib_func = talib_func(talib_func_name)
                    var = functools.partial(_talib_func, wrapper=kwargs_["wrapper"])
                elif var_name in res_func_mapping:
                    var = res_func_mapping[var_name]["func"]
                elif var_name in func_mapping:
                    var = func_mapping[var_name]["func"]
                elif var_name in merged_context:
                    var = merged_context[var_name]
                elif hasattr(np, var_name):
                    var = getattr(np, var_name)
                elif hasattr(generic_nb, var_name):
                    var = getattr(generic_nb, var_name)
                elif hasattr(generic_nb, var_name + "_nb"):
                    var = getattr(generic_nb, var_name + "_nb")
                elif hasattr(vbt, var_name):
                    var = getattr(vbt, var_name)
                else:
                    continue
                try:
                    if callable(var) and "context" in get_func_arg_names(var):
                        var = functools.partial(var, context=merged_context)
                except Exception:
                    pass
                if var_name in res_func_mapping:
                    var = var()
                context[var_name] = var

            if use_pd_eval:
                return pd.eval(expr, local_dict=context, **resolve_dict(pd_eval_kwargs))
            return evaluate(expr, context=context)

        return factory.with_apply_func(apply_func, pass_packed=True, pass_wrapper=True, **kwargs)

    @classmethod
    def from_wqa101(cls, alpha_idx: tp.Union[str, int], **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class from one of the WorldQuant's 101 alpha expressions.

        Uses a specified WorldQuant alpha expression index to build an indicator class
        based on the expression configuration in `vectorbtpro.indicators.expr.wqa101_expr_config`.

        Args:
            alpha_idx (Union[str, int]): WorldQuant 101 alpha expression index.

                If a string is provided, the prefix "WQA" is removed and the remaining value
                is converted to an integer.
            **kwargs: Keyword arguments for `IndicatorFactory.from_expr`.

        Returns:
            Type[IndicatorBase]: Constructed indicator class.

        !!! note
            Some expressions that utilize cross-sectional operations require columns to be
            a multi-index with a level `sector`, `subindustry`, or `industry`.

        Examples:
            ```pycon
            >>> data = vbt.YFData.pull(['BTC-USD', 'ETH-USD'])

            >>> WQA1 = vbt.IF.from_wqa101(1)
            >>> wqa1 = WQA1.run(data.get('Close'))
            >>> wqa1.out
            symbol                     BTC-USD  ETH-USD
            Date
            2014-09-17 00:00:00+00:00     0.25     0.25
            2014-09-18 00:00:00+00:00     0.25     0.25
            2014-09-19 00:00:00+00:00     0.25     0.25
            2014-09-20 00:00:00+00:00     0.25     0.25
            2014-09-21 00:00:00+00:00     0.25     0.25
            ...                            ...      ...
            2022-01-21 00:00:00+00:00     0.00     0.50
            2022-01-22 00:00:00+00:00     0.00     0.50
            2022-01-23 00:00:00+00:00     0.25     0.25
            2022-01-24 00:00:00+00:00     0.50     0.00
            2022-01-25 00:00:00+00:00     0.50     0.00

            [2688 rows x 2 columns]
            ```
        """
        if isinstance(alpha_idx, str):
            alpha_idx = int(alpha_idx.upper().replace("WQA", ""))
        return cls.from_expr(
            wqa101_expr_config[alpha_idx],
            factory_kwargs=dict(class_name="WQA%d" % alpha_idx, module_name=__name__ + ".wqa101"),
            **kwargs,
        )

    @classmethod
    def list_wqa101_indicators(cls) -> tp.List[str]:
        """List all WorldQuant's 101 alpha indicators.

        Returns:
            List[str]: List of all WorldQuant alpha expression indices as strings.
        """
        return [str(i) for i in range(1, 102)]


IF = IndicatorFactory
"""Shortcut alias for `IndicatorFactory`."""

__pdoc__["IF"] = False


def indicator(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get an indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.get_indicator`.
        **kwargs: Keyword arguments for `IndicatorFactory.get_indicator`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.get_indicator(*args, **kwargs)


def talib(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a TA-Lib indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_talib`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_talib`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_talib(*args, **kwargs)


def pandas_ta(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a Pandas TA indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_pandas_ta`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_pandas_ta`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_pandas_ta(*args, **kwargs)


def ta(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a TA indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_ta`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_ta`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_ta(*args, **kwargs)


def wqa101(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a WorldQuant's 101 alpha indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_wqa101`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_wqa101`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_wqa101(*args, **kwargs)


def technical(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a Technical indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_technical`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_technical`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_technical(*args, **kwargs)


def techcon(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a Technical Consensus indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_techcon`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_techcon`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_techcon(*args, **kwargs)


def smc(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Get a Smart Money Concepts indicator.

    Args:
        *args: Positional arguments for `IndicatorFactory.from_smc`.
        **kwargs: Keyword arguments for `IndicatorFactory.from_smc`.

    Returns:
        IndicatorBase: Indicator class.
    """
    return IndicatorFactory.from_smc(*args, **kwargs)
