# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with parameters.

!!! info
    For default settings, see `vectorbtpro._settings.params`.
"""

import inspect
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.annotations import Annotatable, has_annotatables
from vectorbtpro.utils.attr_ import MISSING, DefineMixin, define
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import Configured, FrozenConfig, merge_dicts
from vectorbtpro.utils.eval_ import Evaluable
from vectorbtpro.utils.execution import NoResult, NoResultsException, execute, filter_out_no_results
from vectorbtpro.utils.merging import MergeFunc, parse_merge_func
from vectorbtpro.utils.parsing import (
    ann_args_to_args,
    annotate_args,
    flatten_ann_args,
    unflatten_ann_args,
)
from vectorbtpro.utils.search_ import find_in_obj, replace_in_obj
from vectorbtpro.utils.selection import LabelSel, PosSel
from vectorbtpro.utils.template import CustomTemplate, substitute_templates

__all__ = [
    "generate_param_combs",
    "pick_from_param_grid",
    "Param",
    "Itemable",
    "Paramable",
    "ItemParamable",
    "combine_params",
    "Parameterizer",
    "parameterized",
]


def to_typed_list(lst: list) -> List:
    """Cast a Python list to a Numba-typed List.

    Due to a known issue in Numba 0.52.0, directly constructing a List may fail.
    Refer to https://github.com/numba/numba/issues/6651 for further details.

    Args:
        lst (list): Python list to cast.

    Returns:
        List: Numba-typed List containing the elements of the input list.
    """
    nb_lst = List()
    for elem in lst:
        nb_lst.append(elem)
    return nb_lst


def flatten_param_tuples(param_tuples: tp.Sequence) -> tp.Params:
    """Flatten a nested sequence of parameter tuples using unzipping.

    Args:
        param_tuples (Sequence): Nested sequence of parameter tuples to be flattened.

    Returns:
        Params: Flattened sequence of parameter values.
    """
    params = []
    unzipped_tuples = zip(*param_tuples)
    for i, unzipped in enumerate(unzipped_tuples):
        unzipped = list(unzipped)
        if isinstance(unzipped[0], tuple):
            params.extend(flatten_param_tuples(unzipped))
        else:
            params.append(unzipped)
    return params


def generate_param_combs(op_tree: tp.Tuple, depth: int = 0) -> tp.Params:
    """Generate arbitrary parameter combinations from an operation tree.

    The operation tree `op_tree` is a tuple with nested instructions for generating parameters.
    The first element must be either a callable or a string corresponding to a callable in `itertools`.
    Any element that is itself a tuple with a callable as its first item is recursively unpacked in the same way.

    Args:
        op_tree (Tuple): Tuple that defines nested instructions for generating parameters.
        depth (int): Internal recursion depth indicator.

    Returns:
        Params: Sequence of generated parameter combinations.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.generate_param_combs(("product", ("combinations", [0, 1, 2, 3], 2), [4, 5]))
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
         [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
         [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]]

        >>> vbt.generate_param_combs(("product", (zip, [0, 1, 2, 3], [4, 5, 6, 7]), [8, 9]))
        [[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 7, 7], [8, 9, 8, 9, 8, 9, 8, 9]]
        ```
    """
    checks.assert_instance_of(op_tree, tuple)
    checks.assert_instance_of(op_tree[0], (Callable, str))
    new_op_tree = (op_tree[0],)
    for elem in op_tree[1:]:
        if isinstance(elem, tuple) and isinstance(elem[0], (Callable, str)):
            new_op_tree += (generate_param_combs(elem, depth=depth + 1),)
        else:
            new_op_tree += (elem,)
    if isinstance(new_op_tree[0], Callable):
        out = list(new_op_tree[0](*new_op_tree[1:]))
    else:
        import itertools

        out = list(getattr(itertools, new_op_tree[0])(*new_op_tree[1:]))
    if depth == 0:
        return flatten_param_tuples(out)
    return out


def broadcast_params(
    params_or_dict: tp.ParamsOrDict, to_n: tp.Optional[int] = None
) -> tp.ParamsOrDict:
    """Broadcast parameter lists to a uniform length.

    If `params_or_dict` is a dictionary, its values are considered parameter lists.
    Each parameter list is adjusted so that its length matches `to_n` (or the maximum length
    among lists if `to_n` is not provided). A single-value parameter is repeated to match the target length.

    Args:
        params_or_dict (ParamsOrDict): Dictionary or sequence of parameter lists.
        to_n (Optional[int]): Target length for each parameter list.

            If not provided, the maximum length among the lists is used.

    Returns:
        ParamsOrDict: Broadcasted parameters with uniform length.
    """
    if isinstance(params_or_dict, dict):
        params = list(params_or_dict.values())
    else:
        params = params_or_dict
    if to_n is None:
        to_n = max(list(map(len, params)))
    new_params = []
    for i in range(len(params)):
        param_values = params[i]
        if len(param_values) in [1, to_n]:
            if len(param_values) < to_n:
                new_params.append([p for _ in range(to_n) for p in param_values])
            else:
                new_params.append(list(param_values))
        else:
            raise ValueError(
                f"Parameters at index {i} have length {len(param_values)} that cannot be broadcast to {to_n}"
            )
    if isinstance(params_or_dict, dict):
        return dict(zip(params_or_dict.keys(), new_params))
    return new_params


def create_param_product(params_or_dict: tp.ParamsOrDict) -> tp.ParamsOrDict:
    """Compute the Cartesian product of parameter lists.

    Given a dictionary or sequence of parameter lists, generate all possible combinations
    by computing their Cartesian product, then restructure the result to mirror the input format.

    Args:
        params_or_dict (ParamsOrDict): Dictionary or sequence of parameter lists.

    Returns:
        ParamsOrDict: Cartesian product of the input parameters.
    """
    import itertools

    if isinstance(params_or_dict, dict):
        params = list(params_or_dict.values())
    else:
        params = params_or_dict
    new_params = list(map(list, zip(*itertools.product(*params))))
    if isinstance(params_or_dict, dict):
        return dict(zip(params_or_dict.keys(), new_params))
    return new_params


def is_single_param_value(
    param_values: tp.MaybeParamValues,
    is_tuple: bool = False,
    is_array_like: bool = False,
) -> bool:
    """Determine whether a parameter represents a single value.

    A value is considered single if it is not an instance of list, a Numba List, tuple
    (unless `is_tuple` is True), or a NumPy array (unless `is_array_like` is True).
    Ranges are also considered non-single.

    Args:
        param_values (MaybeParamValues): Value or iterable to evaluate.
        is_tuple (bool): If True, treat tuples as single values.
        is_array_like (bool): If True, treat array-like objects as single values.

    Returns:
        bool: True if `param_values` is a single value, False otherwise.
    """
    check_against = [list, List]
    if not is_tuple:
        check_against.append(tuple)
    if not is_array_like:
        check_against.append(np.ndarray)
    if isinstance(param_values, tuple(check_against)):
        return False
    if isinstance(param_values, range):
        return False
    return True


def params_to_list(
    param_values: tp.MaybeParamValues,
    is_tuple: bool = False,
    is_array_like: bool = False,
) -> list:
    """Convert the parameter value to a list.

    If `param_values` is a single value, it is wrapped in a list.
    Otherwise, it is converted directly into a list.

    Args:
        param_values (MaybeParamValues): Value or iterable to evaluate.
        is_tuple (bool): If True, treat tuples as single values.
        is_array_like (bool): If True, treat array-like objects as single values.

    Returns:
        list: List of parameter values.
    """
    if is_single_param_value(param_values, is_tuple, is_array_like):
        return [param_values]
    return list(param_values)


def get_param_grid_len(param_grid: tp.ParamGrid) -> int:
    """Calculate the total number of parameter combinations in a grid.

    Each parameter in `param_grid` can be represented as an iterable of values or as an
    integer indicating the count of values. This function multiplies the lengths (or counts)
    of each parameter to determine the total number of combinations.

    Args:
        param_grid (ParamGrid): Parameter grid from which to select combinations.

            Each parameter value can be a sequence or an integer indicating the count of values.

    Returns:
        int: Total number of parameter combinations.
    """
    if isinstance(param_grid, dict):
        params_or_lens = list(param_grid.values())
    else:
        params_or_lens = param_grid
    grid_len = 1
    for param_values_or_len in params_or_lens:
        if checks.is_int(param_values_or_len):
            grid_len *= param_values_or_len
        else:
            grid_len *= len(param_values_or_len)
    return grid_len


def pick_from_param_grid(
    param_grid: tp.ParamGrid,
    i: tp.Union[None, int, tp.Array1d] = None,
) -> tp.Union[tp.ParamCombOrDict, tp.List[tp.Array1d]]:
    """Generate parameter combinations from a parameter grid.

    This function picks one or more parameter combinations from the provided `param_grid`.

    Parameter values can be specified either as a collection or as an integer representing the number of values.

    Args:
        param_grid (ParamGrid): Parameter grid from which to select combinations.

            Each parameter value can be a sequence or an integer indicating the count of values.
        i (Union[None, int, Array1d]): Index for selecting a specific combination.

            If None, a random index is chosen.

    Returns:
        Union[ParamCombOrDict, List[Array1d]]: Selected parameter combination.

            Returns a dictionary if `param_grid` is a dict, or a list otherwise.
    """
    if isinstance(param_grid, dict):
        params_or_lens = list(param_grid.values())
    else:
        params_or_lens = param_grid
    grid_len = get_param_grid_len(params_or_lens)
    if i is None:
        i = np.random.randint(grid_len, dtype=np.int64)
    param_comb = []
    for param_values_or_len in params_or_lens:
        if checks.is_int(param_values_or_len):
            param_len = param_values_or_len
        else:
            param_len = len(param_values_or_len)
        index = i * param_len // grid_len
        block_len = grid_len // param_len
        i = i - index * block_len
        grid_len = block_len
        if checks.is_int(param_values_or_len):
            param_comb.append(index)
        else:
            param_comb.append(param_values_or_len[index])
    if isinstance(param_grid, dict):
        return dict(zip(param_grid.keys(), param_comb))
    return param_comb


ParamT = tp.TypeVar("ParamT", bound="Param")


@define
class Param(Evaluable, Annotatable, DefineMixin):
    """Class that represents a parameter."""

    value: tp.Union[tp.MaybeParamValues, tp.Dict[tp.Hashable, tp.ParamValue]] = (
        define.required_field()
    )
    """One or more parameter values for the parameter."""

    is_tuple: bool = define.optional_field(default=False)
    """Indicates whether `Param.value` is interpreted as a tuple.

    When True, a tuple is considered a single parameter value.
    """

    is_array_like: bool = define.optional_field(default=False)
    """Indicates whether `Param.value` should be treated as array-like.

    When True, a NumPy array is considered a single parameter value.
    """

    map_template: tp.Optional[CustomTemplate] = define.optional_field(default=None)
    """Mapping template applied to `Param.value` prior to constructing parameter combinations."""

    random_subset: tp.Union[None, int, float] = define.optional_field(default=None)
    """Specifies a random subset of parameter values to select."""

    level: tp.Optional[int] = define.optional_field(default=None)
    """Specifies the level for the parameter in the product combination.

    Parameters sharing the same level are grouped together.
    Lower-level parameters are processed before higher-level ones.
    Levels must be consecutive starting from 0 without gaps.
    """

    condition: tp.Union[None, str, CustomTemplate] = define.optional_field(default=None)
    """Specifies a condition to filter parameter combinations.

    The condition can be provided as a template or an expression where `x` (or the parameter's name)
    represents this parameter. If provided as an expression, it is pre-compiled for efficiency.
    To reference a parameter index value, enclose the level name with double underscores
    (e.g., `__fast_sma_timeperiod__`).
    """

    context: tp.KwargsLike = define.optional_field(default=None)
    """Context for evaluating `Param.condition` and applying `Param.map_template`."""

    keys: tp.Optional[tp.IndexLike] = define.optional_field(default=None)
    """Specifies keys to serve as the index level.

    If not provided, `Param.value` is converted to an index via `vectorbtpro.base.indexes.index_from_values`.
    """

    hide: bool = define.optional_field(default=False)
    """Indicates whether the parameter should be hidden from the parameter index."""

    name: tp.Optional[tp.Hashable] = define.optional_field(default=None)
    """Specifies the name of the parameter.

    If not provided, it defaults to the index name from `Param.keys` or the key from `param_dct`
    in `combine_params`.
    """

    mono_reduce: bool = define.optional_field(default=False)
    """Indicates if a mono-chunk of identical values should be reduced to a single value."""

    mono_merge_func: tp.MergeFuncLike = define.optional_field(default=None)
    """Merging function or a dictionary of such to combine parameter values.

    See `vectorbtpro.utils.merging.MergeFunc`.
    """

    mono_merge_kwargs: tp.KwargsLike = define.optional_field(default=None)
    """Keyword arguments for `Param.mono_merge_func`."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.optional_field(default=None)
    """Identifier(s) indicating where to evaluate this instance."""

    def map_value(self: ParamT, func: tp.Callable, old_as_keys: bool = False) -> ParamT:
        """Apply a function to each element of `Param.value` and return a new `Param` instance.

        Args:
            func (Callable): Function to apply to each value in `Param.value`.
            old_as_keys (bool): Flag indicating whether to use parameter values as keys if not provided.

        Returns:
            Param: New `Param` instance with the updated values.
        """
        self.assert_field_not_missing("value")
        attr_dct = self.asdict()
        is_tuple = self.resolve_field("is_tuple")
        is_array_like = self.resolve_field("is_array_like")
        keys = self.resolve_field("keys")

        if isinstance(attr_dct["value"], dict):
            attr_dct["value"] = {k: v for k, v in attr_dct["value"].items()}
        elif isinstance(attr_dct["value"], pd.Index):
            attr_dct["value"] = pd.Index(map(func, attr_dct["value"]))
        elif isinstance(attr_dct["value"], pd.Series):
            attr_dct["value"] = pd.Series(
                map(func, attr_dct["value"].values), index=attr_dct["value"].index
            )
        elif not is_single_param_value(attr_dct["value"], is_tuple, is_array_like):
            if old_as_keys and keys is None:
                from vectorbtpro.base import indexes

                attr_dct["keys"] = indexes.index_from_values(attr_dct["value"])
            attr_dct["value"] = list(map(func, attr_dct["value"]))
        else:
            if old_as_keys and keys is None and not isinstance(attr_dct["value"], Paramable):
                from vectorbtpro.base import indexes

                attr_dct["keys"] = indexes.index_from_values([attr_dct["value"]])
            attr_dct["value"] = func(attr_dct["value"])
        return type(self)(**attr_dct)


class Itemable(Base):
    """Class representing an object that can be returned as items."""

    def items(self, **kwargs) -> tp.Items:
        """Return this instance as items.

        Args:
            **kwargs: Keyword arguments for the underlying implementation.

        Returns:
            Items: Items representation of the instance.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class Paramable(Base):
    """Class representing an object that can be returned as a parameter."""

    def as_param(self, **kwargs) -> Param:
        """Return this instance as a parameter.

        Args:
            **kwargs: Keyword arguments for the underlying implementation.

        Returns:
            Param: Parameter representation of the instance.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class ItemParamable(Itemable, Paramable):
    """Class representing an object that can be returned as both items and parameters."""

    def items(self, key_as_index: bool = False, **kwargs) -> tp.Items:
        raise NotImplementedError

    def as_param(self, **kwargs) -> Param:
        param_values = []
        index_values = []
        first_index = None
        keys = None
        for k, v in self.items(key_as_index=True, **kwargs):
            param_values.append(v)
            index_values.append(k[0])
            if first_index is None:
                first_index = k
        if isinstance(first_index, pd.MultiIndex):
            keys = pd.MultiIndex.from_tuples(index_values, names=first_index.names)
        elif isinstance(first_index, pd.Index):
            keys = pd.Index(index_values, name=first_index.name)
        return Param(param_values, keys=keys)


def combine_params(
    param_dct: tp.Dict[tp.Hashable, Param],
    build_grid: tp.Optional[bool] = None,
    grid_indices: tp.Union[None, slice, tp.Sequence[int]] = None,
    random_subset: tp.Union[None, int, float] = None,
    random_replace: bool = False,
    random_sort: bool = True,
    max_guesses: tp.Union[None, int, float] = None,
    max_misses: tp.Union[None, int, float] = None,
    seed: tp.Optional[int] = None,
    name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
    clean_index_kwargs: tp.KwargsLike = None,
    build_product: bool = True,
    build_index: bool = True,
    raise_empty_error: bool = False,
    keep_single_value: bool = False,
) -> tp.Union[dict, tp.Tuple[dict, pd.Index]]:
    """Combine a dictionary of parameters into a unified configuration.

    This function combines a dictionary of `Param` objects into a single configuration.

    If `build_grid` is True, the entire grid is built and then filtered by conditions.
    If `build_grid` is False, combinations are selected and merged on the fly.
    Materializing the grid is recommended when the total number of combinations is relatively low
    (typically less than one million) or when parameters include conditions.

    Arguments `max_guesses` and `max_misses` are useful for limiting the number of guesses; without
    them, the search may continue forever. Once any of these two numbers is reached, the search will stop.

    Args:
        param_dct (Dict[Hashable, Param]): Dictionary mapping keys to `Param` objects.
        build_grid (Optional[bool]): Flag indicating whether to materialize the full parameter grid.
        grid_indices (Union[None, slice, Sequence[int]]): Slice (for example, `slice(None, None, 2)`
            for `::2`) or sequence of indices to skip certain combinations.

            It can be used to skip a some combinations before a random subset is drawn.
        random_subset (Union[None, int, float]): Integer count or a float fraction specifying
            how many combinations to select randomly.

            If parameters have conditions, `random_subset` is drawn from the subset of
            combinations whose conditions have been met, not the other way around.
        random_replace (bool): If True, select combinations with replacement, allowing duplicates.

            If False, each drawn combination becomes unique.
        random_sort (bool): If True, sort the positions of the selected combinations.

            Otherwise, they remain in their randomly-selected positions.
        max_guesses (Union[None, int, float]): Maximum number of guesses when searching for a valid
            combination (applies when not building the full grid).
        max_misses (Union[None, int, float]): Maximum number of misses allowed in the search for a valid
            combination (applies when not building the full grid).
        seed (Optional[int]): Random seed for deterministic output.
        name_tuple_to_str (Union[None, bool, Callable]): Flag or function to convert name tuples to
            strings for the parameter index.
        clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        build_product (bool): If True, compute the Cartesian product of parameter values.
        build_index (bool): If True, return an index along with the combined parameters.
        raise_empty_error (bool): If True, raise an error when no valid combination is found.
        keep_single_value (bool): If True, retain parameters that have a single value.

    Returns:
        Union[dict, Tuple[dict, pd.Index]]: Combined parameter dictionary,
            and if `build_index` is True, also an index.

    !!! note
        The parameters `random_subset`, `max_guesses`, and `max_misses` only apply when conditions are defined.

        The `grid_indices` parameter determines a subsequence of combinations prior to random selection.
    """
    from vectorbtpro.base import indexes

    if clean_index_kwargs is None:
        clean_index_kwargs = {}
    rng = np.random.default_rng(seed=seed)

    def _name_tuple_to_str(name_tuple):
        return "_".join(map(lambda x: str(x).strip().lower(), name_tuple))

    if isinstance(name_tuple_to_str, bool):
        if name_tuple_to_str:
            name_tuple_to_str = _name_tuple_to_str
        else:
            name_tuple_to_str = None

    def _str_name(name):
        if isinstance(name, tuple):
            if name_tuple_to_str:
                return name_tuple_to_str(name)
            return _name_tuple_to_str(name)
        return str(name)

    level_map = OrderedDict()
    param_level = {}
    param_keys = {}
    param_visible_keys = {}
    level_seen = False
    explicit_level_seen = False
    implicit_level_seen = False
    curr_idx = 0
    max_idx = 0
    conditions = {}
    contexts = {}
    names = {}
    is_single_value = {}
    for i, (k, p) in enumerate(param_dct.items()):
        if isinstance(p, Paramable):
            p = p.as_param()
        if not isinstance(p, Param):
            p = Param(p)
        if isinstance(p.value, Paramable):
            p2 = p.value.as_param()
            p = p.merge_over(p2, value=p2.value)
        p = p.resolve()

        if p.condition is not None:
            conditions[k] = p.condition
            if p.context is not None:
                contexts[k] = p.context
            else:
                contexts[k] = {}
        if p.level is None:
            if not build_product:
                level = -1
                implicit_level_seen = True
            else:
                if explicit_level_seen:
                    raise ValueError("Please provide level for all product parameters")
                level = curr_idx
                curr_idx += 1
                level_seen = True
        else:
            level = p.level
            if level != -1:
                if curr_idx > 0 and not explicit_level_seen:
                    raise ValueError("Please provide level for all product parameters")
                explicit_level_seen = True
                level_seen = True
            else:
                implicit_level_seen = True
        if level > max_idx:
            max_idx = level

        keys_name = None
        p_name = p.name
        sr_name = None
        index_name = None

        keys = p.keys
        if keys_name is None and keys is not None:
            if not isinstance(keys, pd.Index):
                keys = pd.Index(keys)
            if isinstance(keys, pd.MultiIndex):
                keys_name = keys.names
            else:
                keys_name = keys.name

        p.assert_field_not_missing("value")
        value = p.value
        if isinstance(value, dict):
            if keys is None:
                keys = pd.Index(value.keys())
            value = list(value.values())
        elif isinstance(value, pd.Index):
            if keys is None:
                keys = value
            if isinstance(value, pd.MultiIndex):
                index_name = value.names
            else:
                index_name = value.name
            value = value.tolist()
        elif isinstance(value, pd.Series):
            if not checks.is_default_index(value.index):
                if keys is None:
                    keys = value.index
                if isinstance(value.index, pd.MultiIndex):
                    index_name = value.index.names
                else:
                    index_name = value.index.name
            sr_name = value.name
            value = value.values.tolist()
        is_single_value[k] = is_single_param_value(
            value, is_tuple=p.is_tuple, is_array_like=p.is_array_like
        )
        values = params_to_list(value, is_tuple=p.is_tuple, is_array_like=p.is_array_like)

        if keys_name is None:
            if p_name is not None:
                keys_name = p_name
            elif sr_name is not None:
                keys_name = sr_name
            elif index_name is not None:
                keys_name = index_name
            else:
                keys_name = k
        if keys is None and not p.hide:
            keys = indexes.index_from_values(values, name=keys_name)
        elif keys is not None:
            keys = keys.rename(keys_name)

        if p.random_subset is not None:
            if checks.is_float(p.random_subset):
                _random_subset = int(p.random_subset * len(values))
            else:
                _random_subset = p.random_subset
            random_indices = np.sort(rng.permutation(np.arange(len(values)))[:_random_subset])
        else:
            random_indices = None
        if random_indices is not None:
            values = [values[r] for r in random_indices]
            if keys is not None:
                keys = keys[random_indices]

        if p.map_template is not None:
            param_context = merge_dicts(
                dict(
                    param=p,
                    values=values,
                    keys=keys,
                    random_indices=random_indices,
                ),
                p.context,
            )
            values = p.map_template.substitute(param_context, eval_id="map_template")

        if level not in level_map:
            level_map[level] = OrderedDict()
        level_map[level][k] = values
        param_level[k] = level
        param_keys[k] = keys
        if not p.hide:
            param_visible_keys[k] = keys
        if not isinstance(keys, pd.MultiIndex):
            names[k] = keys_name

    if implicit_level_seen:
        if level_seen:
            max_idx += 1
        new_level_map = OrderedDict()
        for k, v in level_map.items():
            if k == -1:
                k = max_idx
            new_level_map[k] = v
        level_map = new_level_map
        new_param_level = {}
        for k, v in param_level.items():
            if v == -1:
                v = max_idx
            new_param_level[k] = v
        param_level = new_param_level

    level_params = []
    level_lens = []
    param_dct_keys = []
    level_indexes = []
    shown_levels = []
    hidden_levels = []
    n_combs = None
    for level in range(max_idx + 1):
        if level not in level_map:
            raise ValueError("Levels must come in a strict order starting with 0 and without gaps")
        for k in level_map[level].keys():
            param_dct_keys.append(k)

        params = tuple(level_map[level].values())
        if len(params) > 1:
            params = broadcast_params(params)
        level_params.append(params)
        level_lens.append(len(params[0]))
        if n_combs is None:
            n_combs = len(params[0])
        else:
            n_combs *= len(params[0])

        if build_index:
            levels = []
            for k in level_map[level].keys():
                if k in param_visible_keys:
                    levels.append(param_visible_keys[k])
            if len(levels) > 1:
                _param_index = indexes.stack_indexes(levels, **clean_index_kwargs)
                shown_levels.append(level)
            elif len(levels) == 1:
                _param_index = levels[0]
                shown_levels.append(level)
            else:
                _param_index = range(len(params[0]))
                hidden_levels.append(level)
            level_indexes.append(_param_index)

    if len(conditions) > 0:
        condition_funcs = {}
        for k, expr in conditions.items():
            if isinstance(expr, str):
                arg_names = (
                    {"x"}
                    | set(map(lambda x: f"__{_str_name(x)}__", param_dct_keys))
                    | set(map(lambda x: f"__{_str_name(x)}__", names.values()))
                    | set(map(_str_name, param_dct_keys))
                    | set(map(_str_name, names.values()))
                    | set(contexts[k].keys())
                )
                for level_index in level_indexes:
                    if level_index is not None:
                        if isinstance(level_index, pd.MultiIndex):
                            for level_name in level_index.names:
                                arg_names.add(f"__{_str_name(level_name)}__")
                        elif isinstance(level_index, pd.Index):
                            arg_names.add(f"__{_str_name(level_index.name)}__")
                condition_funcs[k] = eval(f"lambda {'=None, '.join(arg_names)}=None: {expr}")
            else:
                condition_funcs[k] = expr
    else:
        condition_funcs = None

    if grid_indices is not None:
        if isinstance(grid_indices, slice):
            slice_start = grid_indices.start
            if slice_start is None:
                slice_start = 0
            slice_stop = grid_indices.stop
            if slice_stop is None:
                slice_stop = n_combs
            grid_indices = np.arange(slice_start, slice_stop, grid_indices.step)
        else:
            grid_indices = np.asarray(grid_indices)

    if build_grid is None:
        if grid_indices is not None:
            build_grid = False
        elif random_subset is None:
            build_grid = True
        else:
            if len(conditions) == 0:
                build_grid = False
            else:
                if checks.is_float(random_subset):
                    build_grid = True
                else:
                    if n_combs >= 1_000_000:
                        build_grid = False
                    else:
                        build_grid = True

    if not build_grid:
        if len(conditions) == 0:
            if grid_indices is not None:
                n_combs = len(grid_indices)
            if random_subset is not None:
                if checks.is_float(random_subset):
                    random_subset = int(random_subset * n_combs)
                if grid_indices is not None:
                    random_grid_indices = rng.choice(
                        grid_indices, size=random_subset, replace=random_replace
                    )
                else:
                    random_grid_indices = rng.choice(
                        n_combs, size=random_subset, replace=random_replace
                    )
                if random_sort:
                    random_grid_indices = np.sort(random_grid_indices)
                picked_level_indices = pick_from_param_grid(level_lens, i=random_grid_indices)
            else:
                if grid_indices is not None:
                    picked_grid_indices = grid_indices
                else:
                    picked_grid_indices = np.arange(n_combs)
                picked_level_indices = pick_from_param_grid(level_lens, i=picked_grid_indices)

            params_ready = True
        elif len(conditions) > 0 and grid_indices is None:
            if random_subset is None:
                raise ValueError("Must build the grid for conditions without a random subset")
            if checks.is_float(random_subset):
                raise ValueError("Must build the grid for conditions with a floating random subset")
            if max_guesses is not None and checks.is_float(max_guesses):
                max_guesses = int(max_guesses * random_subset)
            if max_misses is not None and checks.is_float(max_misses):
                max_misses = int(max_misses * random_subset)
            picked_indices_list = []
            picked_indices_set = set()
            visited_indices_set = set()
            n_misses = 0
            n_guesses = 0
            while len(picked_indices_list) < random_subset and len(visited_indices_set) < n_combs:
                n_guesses += 1
                picked_indices = []
                k = 0
                param_comb_keys = {}
                param_comb = {}
                for level, params in enumerate(level_params):
                    i = rng.choice(len(params[0]), replace=True)
                    for j in range(len(params)):
                        p_keys = param_keys[param_dct_keys[k]]
                        if p_keys is not None:
                            param_comb_keys[f"__{_str_name(param_dct_keys[k])}__"] = p_keys[i]
                            if param_dct_keys[k] in names:
                                param_comb_keys[f"__{_str_name(names[param_dct_keys[k]])}__"] = (
                                    p_keys[i]
                                )
                            if isinstance(p_keys, pd.MultiIndex):
                                for l, level_name in enumerate(p_keys.names):
                                    if level_name is not None:
                                        param_comb_keys[f"__{_str_name(level_name)}__"] = p_keys[i][
                                            l
                                        ]
                            elif isinstance(p_keys, pd.Index):
                                if p_keys.name is not None:
                                    param_comb_keys[f"__{_str_name(p_keys.name)}__"] = p_keys[i]
                        picked_value = params[j][i]
                        param_comb[_str_name(param_dct_keys[k])] = picked_value
                        if param_dct_keys[k] in names:
                            param_comb[_str_name(names[param_dct_keys[k]])] = picked_value
                        k += 1
                    picked_indices.append(i)
                visited_indices_set.add(tuple(picked_indices))
                if not random_replace and tuple(picked_indices) in picked_indices_set:
                    continue
                conditions_met = True
                for k, condition_func in condition_funcs.items():
                    param_context = {
                        "x": param_comb[_str_name(k)],
                        **param_comb_keys,
                        **param_comb,
                        **contexts[k],
                    }
                    if isinstance(condition_func, CustomTemplate):
                        condition_met = condition_func.substitute(param_context)
                    else:
                        condition_met = condition_func(**param_context)
                    if not condition_met:
                        conditions_met = False
                        break
                if conditions_met:
                    picked_indices_list.append(tuple(picked_indices))
                    picked_indices_set.add(tuple(picked_indices))
                else:
                    n_misses += 1
                    if max_misses is not None:
                        if n_misses >= max_misses:
                            break
                if max_guesses is not None:
                    if n_guesses >= max_guesses:
                        break
            if random_sort:
                picked_level_indices = list(map(list, zip(*sorted(picked_indices_list))))
            else:
                picked_level_indices = list(map(list, zip(*picked_indices_list)))

            params_ready = True
        else:
            n_combs = len(grid_indices)
            picked_level_indices = pick_from_param_grid(level_lens, i=grid_indices)

            params_ready = False

        if len(picked_level_indices) == 0 or len(picked_level_indices[0]) == 0:
            param_product = {k: [] for k in param_dct_keys}
        else:
            param_product = dict()
            k = 0
            for level in range(len(picked_level_indices)):
                for j in range(len(level_params[level])):
                    param_key = param_dct_keys[k]
                    if param_key not in param_product:
                        param_product[param_key] = []
                    param_values = level_params[level][j]
                    picked_param_values = [param_values[i] for i in picked_level_indices[level]]
                    param_product[param_key] = picked_param_values
                    k += 1

        if build_index and len(shown_levels) > 0:
            shown_indexes = []
            for level, index in enumerate(level_indexes):
                if level in shown_levels:
                    if len(picked_level_indices) == 0 or len(picked_level_indices[0]) == 0:
                        shown_indexes.append(index[0:0])
                    else:
                        shown_indexes.append(index[picked_level_indices[level]])
            if len(shown_indexes) > 1:
                param_index = indexes.stack_indexes(shown_indexes, **clean_index_kwargs)
            else:
                param_index = shown_indexes[0]
        else:
            param_index = None
    else:
        op_tree_operands = []
        for params in level_params:
            if len(params) > 1:
                op_tree_operands.append((zip, *broadcast_params(params)))
            else:
                op_tree_operands.append(params[0])
        if len(op_tree_operands) > 1:
            param_product = dict(
                zip(param_dct_keys, generate_param_combs(("product", *op_tree_operands)))
            )
        elif isinstance(op_tree_operands[0], tuple):
            param_product = dict(zip(param_dct_keys, generate_param_combs(op_tree_operands[0])))
        else:
            param_product = dict(zip(param_dct_keys, op_tree_operands))

        if build_index and len(shown_levels) > 0:
            if len(level_indexes) > 1:
                param_index = indexes.combine_indexes(level_indexes, **clean_index_kwargs)
                if len(hidden_levels) > 0:
                    if len(shown_levels) > 1:
                        param_index = indexes.select_levels(param_index, shown_levels)
                    else:
                        param_index = indexes.select_levels(param_index, shown_levels[0])
            else:
                param_index = level_indexes[0]
        else:
            param_index = None

        if grid_indices is not None:
            n_combs = len(grid_indices)
            param_product = {k: [v[i] for i in grid_indices] for k, v in param_product.items()}
            if build_index and len(shown_levels) > 0:
                param_index = param_index[grid_indices]

        params_ready = False

    if not params_ready:
        if len(conditions) > 0:
            indices = np.arange(n_combs)
            if (
                random_subset is not None
                and not checks.is_float(random_subset)
                and not random_replace
            ):
                pre_random_subset = True
            else:
                pre_random_subset = False
            if pre_random_subset:
                indices = rng.permutation(indices)
            keep_indices = []
            for i in indices:
                param_comb_keys = {}
                level_indices = pick_from_param_grid(level_lens, i=i)
                for k in param_product:
                    p_keys = param_keys[k]
                    if p_keys is not None:
                        p_keys_value = p_keys[level_indices[param_level[k]]]
                        param_comb_keys[f"__{_str_name(k)}__"] = p_keys_value
                        if k in names:
                            param_comb_keys[f"__{_str_name(names[k])}__"] = p_keys_value
                if param_index is not None:
                    if isinstance(param_index, pd.MultiIndex):
                        for l, level_name in enumerate(param_index.names):
                            if level_name is not None:
                                param_comb_keys[f"__{_str_name(level_name)}__"] = param_index[i][l]
                    elif isinstance(param_index, pd.Index):
                        if param_index.name is not None:
                            param_comb_keys[f"__{_str_name(param_index.name)}__"] = param_index[i]
                param_comb = {}
                for k in param_product:
                    param_comb[_str_name(k)] = param_product[k][i]
                    if k in names:
                        param_comb[_str_name(names[k])] = param_product[k][i]
                conditions_met = True
                for k, condition_func in condition_funcs.items():
                    param_context = {
                        "x": param_comb[_str_name(k)],
                        **param_comb_keys,
                        **param_comb,
                        **contexts[k],
                    }
                    if isinstance(condition_func, CustomTemplate):
                        condition_met = condition_func.substitute(param_context)
                    else:
                        condition_met = condition_func(**param_context)
                    if not condition_met:
                        conditions_met = False
                        break
                if conditions_met:
                    keep_indices.append(i)
                    if pre_random_subset:
                        if len(keep_indices) == random_subset:
                            break
            if len(keep_indices) > 0:
                if pre_random_subset and random_sort:
                    keep_indices = np.sort(keep_indices)
                param_product = {k: [v[i] for i in keep_indices] for k, v in param_product.items()}
                n_combs = len(keep_indices)
                if build_index and len(shown_levels) > 0:
                    param_index = param_index[keep_indices]
            else:
                param_product = {k: [] for k, v in param_product.items()}
                n_combs = 0
                if build_index and len(shown_levels) > 0:
                    param_index = param_index[0:0]
        else:
            pre_random_subset = False

        if random_subset is not None and not pre_random_subset and n_combs > 0:
            if checks.is_float(random_subset):
                random_subset = int(random_subset * n_combs)
            random_indices = rng.choice(n_combs, size=random_subset, replace=random_replace)
            if random_sort:
                random_indices = np.sort(random_indices)
            param_product = {k: [v[i] for i in random_indices] for k, v in param_product.items()}
            if build_index and len(shown_levels) > 0:
                param_index = param_index[random_indices]

    if build_index and len(shown_levels) > 0:
        if name_tuple_to_str is not None:
            found_tuple = False
            new_names = []
            for name in param_index.names:
                if isinstance(name, tuple):
                    name = name_tuple_to_str(name)
                    found_tuple = True
                new_names.append(name)
            if found_tuple:
                if isinstance(param_index, pd.MultiIndex):
                    param_index.rename(new_names, inplace=True)
                else:
                    param_index.rename(new_names[0], inplace=True)
        param_index = indexes.clean_index(param_index, **clean_index_kwargs)

    if raise_empty_error:
        if len(param_product[list(param_product.keys())[0]]) == 0:
            raise ValueError("Set of parameter combinations is empty")
    if keep_single_value and all(is_single_value.values()):
        param_product = {k: v[0] for k, v in param_product.items()}
    if build_index:
        return param_product, param_index
    return param_product


class Parameterizer(Configured):
    """Class responsible for parameterizing and running a function.

    Does the following:

    1. Search for values wrapped with the class `Param` in any nested dicts and tuples
        using `Parameterizer.find_params_in_obj`
    2. Use `combine_params` to build parameter combinations
    3. Map parameter combinations to configs using `Parameterizer.param_product_to_objs`
    4. Generate and resolve parameter configs by combining combinations from the previous step with
        `param_configs` optionally provided by the user. User-defined `param_configs` have priority.
    5. Substitute `selection` as a template if provided, convert it to indices for mapping to `param_index`,
        and select the corresponding objects.
    6. Build mono-chunks if `mono_n_chunks`, `mono_chunk_len`, or `mono_chunk_meta` is provided.
    7. Extract arguments and keyword arguments from each parameter config and substitute any templates lazily.
    8. Pass each function and its arguments to `vectorbtpro.utils.execution.execute` for execution.
    9. Optionally, post-process and merge the results by passing them along with `**merge_kwargs` to `merge_func`.

    Argument `param_configs` accepts either a list of dictionaries with argument names corresponding
    to the function signature or a dictionary of dictionaries, where keys are configuration names.
    If a list is provided, each dictionary may also include the key `_name` to assign a name to the config.
    Variable arguments can be provided either in rolled format (`args=(...), kwargs={...}`)
    or unrolled format (`args_0=..., args_1=..., some_kwarg=...`).

    !!! important
        Defining a parameter and listing the same argument in `param_configs` will prioritize the config
        over the parameter, even though the parameter remains visible in the final columns.
        No error is raised in this case.

    If mono-chunking is enabled, parameter configs are distributed over chunks. Any argument wrapped
    with `Param` or included in `mono_merge_func` is aggregated into a list and merged using either
    `Param.mono_merge_func` or `mono_merge_func` similar to `merge_func`. If an argument is not subject
    to these rules and all its values within a chunk are identical, the value is passed as a scalar.
    Arguments `mono_merge_func` and `mono_merge_kwargs` must be dictionaries mapping argument names
    (from the flattened signature) to functions and keyword arguments, respectively.

    If `vectorbtpro.utils.execution.NoResult` is returned, the current iteration is skipped and removed
    from the final index.

    Args:
        param_search_kwargs (KwargsLike): Keyword arguments for parameter search.

            See `Parameterizer.find_params_in_obj`.
        skip_single_comb (Optional[bool]): Flag indicating direct execution when only one
            parameter combination exists.
        template_context (KwargsLike): Additional context for template substitution.
        build_grid (Optional[bool]): Flag controlling whether to materialize the full parameter grid.

            See `combine_params`.
        grid_indices (Union[None, slice, Sequence[int]]): Indices used to skip certain combinations.

            See `combine_params`.
        random_subset (Union[None, int, float]): Size of the random subset to select.

            See `combine_params`.
        random_replace (Optional[bool]): Flag determining whether to select combinations with replacement.

            See `combine_params`.
        random_sort (Optional[bool]): Flag determining whether to sort the positions of selected combinations.

            See `combine_params`.
        max_guesses (Union[None, int, float]): Maximum number of guesses during parameter search.

            See `combine_params`.
        max_misses (Union[None, int, float]): Maximum allowed misses during parameter search.

            See `combine_params`.
        seed (Optional[int]): Random seed for deterministic output.

            See `combine_params`.
        name_tuple_to_str (Union[None, bool, Callable]): Flag or function to convert name tuples to
            strings for the parameter index.

            See `combine_params`.
        clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

            See `vectorbtpro.base.indexes.clean_index`.
        selection (Optional[Selection]): Template or criteria to select specific parameter indices.
        forward_kwargs_as (KwargsLike): Mapping for renaming keyword arguments when forwarding them.
        mono_min_size (Optional[int]): Minimum number of parameter values to split.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        mono_n_chunks (Optional[Union[str, int]]): Specification for the number of mono-chunks.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        mono_chunk_len (Optional[Union[str, int]]):  Specification for the length of each mono-chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        mono_chunk_meta (Optional[Iterable[ChunkMeta]]): Iterable containing metadata for each mono-chunk.

            See `vectorbtpro.utils.chunking.iter_chunk_meta`.
        mono_reduce (Union[None, bool, Kwargs]): Flag or settings for reducing mono-chunk results.
        mono_merge_func (MaybeDict[MergeFuncLike]): Merging function or a dictionary of such to
            combine parameter values.

            See `vectorbtpro.utils.merging.MergeFunc`.
        mono_merge_kwargs (KwargsLike): Keyword arguments for `mono_merge_func`.
        filter_results (Optional[bool]): Flag indicating whether to filter
            `vectorbtpro.utils.execution.NoResult` results after execution.
        raise_no_results (Optional[bool]): Flag indicating whether to raise a
            `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
        merge_func (MergeFuncLike): Merging function used to aggregate individual results.

            See `vectorbtpro.utils.merging.MergeFunc`.
        merge_kwargs (KwargsLike): Keyword arguments for `merge_func`.
        return_meta (Optional[bool]): Flag indicating if additional metadata should be returned.
        return_param_index (Optional[bool]): Flag indicating if the parameter index should be included in the result.
        execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

            See `vectorbtpro.utils.execution.execute`.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.params`.
    """

    _settings_path: tp.SettingsPath = "params"

    def __init__(
        self,
        param_search_kwargs: tp.KwargsLike = None,
        skip_single_comb: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        build_grid: tp.Optional[bool] = None,
        grid_indices: tp.Union[None, slice, tp.Sequence[int]] = None,
        random_subset: tp.Union[None, int, float] = None,
        random_replace: tp.Optional[bool] = None,
        random_sort: tp.Optional[bool] = None,
        max_guesses: tp.Union[None, int, float] = None,
        max_misses: tp.Union[None, int, float] = None,
        seed: tp.Optional[int] = None,
        name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
        clean_index_kwargs: tp.KwargsLike = None,
        selection: tp.Optional[tp.Selection] = None,
        forward_kwargs_as: tp.KwargsLike = None,
        mono_min_size: tp.Optional[int] = None,
        mono_n_chunks: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_len: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
        mono_reduce: tp.Union[None, bool, tp.Kwargs] = None,
        mono_merge_func: tp.MaybeDict[tp.MergeFuncLike] = None,
        mono_merge_kwargs: tp.KwargsLike = None,
        filter_results: tp.Optional[bool] = None,
        raise_no_results: tp.Optional[bool] = None,
        merge_func: tp.MergeFuncLike = None,
        merge_kwargs: tp.KwargsLike = None,
        return_meta: tp.Optional[bool] = None,
        return_param_index: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            param_search_kwargs=param_search_kwargs,
            skip_single_comb=skip_single_comb,
            template_context=template_context,
            build_grid=build_grid,
            grid_indices=grid_indices,
            random_subset=random_subset,
            random_replace=random_replace,
            random_sort=random_sort,
            max_guesses=max_guesses,
            max_misses=max_misses,
            seed=seed,
            name_tuple_to_str=name_tuple_to_str,
            clean_index_kwargs=clean_index_kwargs,
            selection=selection,
            forward_kwargs_as=forward_kwargs_as,
            mono_min_size=mono_min_size,
            mono_n_chunks=mono_n_chunks,
            mono_chunk_len=mono_chunk_len,
            mono_chunk_meta=mono_chunk_meta,
            mono_reduce=mono_reduce,
            mono_merge_func=mono_merge_func,
            mono_merge_kwargs=mono_merge_kwargs,
            filter_results=filter_results,
            raise_no_results=raise_no_results,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            return_meta=return_meta,
            return_param_index=return_param_index,
            execute_kwargs=execute_kwargs,
            **kwargs,
        )

        self._param_search_kwargs = self.resolve_setting(
            param_search_kwargs, "param_search_kwargs", merge=True
        )
        self._skip_single_comb = self.resolve_setting(skip_single_comb, "skip_single_comb")
        self._template_context = self.resolve_setting(
            template_context, "template_context", merge=True
        )
        self._build_grid = self.resolve_setting(build_grid, "build_grid")
        self._grid_indices = self.resolve_setting(grid_indices, "grid_indices")
        self._random_subset = self.resolve_setting(random_subset, "random_subset")
        self._random_replace = self.resolve_setting(random_replace, "random_replace")
        self._random_sort = self.resolve_setting(random_sort, "random_sort")
        self._max_guesses = self.resolve_setting(max_guesses, "max_guesses")
        self._max_misses = self.resolve_setting(max_misses, "max_misses")
        self._seed = self.resolve_setting(seed, "seed")
        self._name_tuple_to_str = self.resolve_setting(name_tuple_to_str, "name_tuple_to_str")
        self._clean_index_kwargs = self.resolve_setting(
            clean_index_kwargs, "clean_index_kwargs", merge=True
        )
        self._selection = self.resolve_setting(selection, "selection")
        self._forward_kwargs_as = self.resolve_setting(
            forward_kwargs_as, "forward_kwargs_as", merge=True
        )
        self._mono_min_size = self.resolve_setting(mono_min_size, "mono_min_size")
        self._mono_n_chunks = self.resolve_setting(mono_n_chunks, "mono_n_chunks")
        self._mono_chunk_len = self.resolve_setting(mono_chunk_len, "mono_chunk_len")
        self._mono_chunk_meta = self.resolve_setting(mono_chunk_meta, "mono_chunk_meta")
        self._mono_reduce = self.resolve_setting(mono_reduce, "mono_reduce")
        self._mono_merge_func = self.resolve_setting(mono_merge_func, "mono_merge_func")
        self._mono_merge_kwargs = self.resolve_setting(
            mono_merge_kwargs, "mono_merge_kwargs", merge=True
        )
        self._filter_results = self.resolve_setting(filter_results, "filter_results")
        self._raise_no_results = self.resolve_setting(raise_no_results, "raise_no_results")
        self._merge_func = self.resolve_setting(merge_func, "merge_func")
        self._merge_kwargs = self.resolve_setting(merge_kwargs, "merge_kwargs", merge=True)
        self._return_meta = self.resolve_setting(return_meta, "return_meta")
        self._return_param_index = self.resolve_setting(return_param_index, "return_param_index")
        self._execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)

    @property
    def param_search_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for parameter search.

        See `Parameterizer.find_params_in_obj`.

        Returns:
            Kwargs: Dictionary of keyword arguments used for parameter search.
        """
        return self._param_search_kwargs

    @property
    def skip_single_comb(self) -> bool:
        """Flag indicating direct execution when only one parameter combination exists.

        Returns:
            bool: True if direct execution is enabled when there is only one combination; otherwise False.
        """
        return self._skip_single_comb

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def build_grid(self) -> tp.Optional[bool]:
        """Flag controlling whether to materialize the full parameter grid.

        See `combine_params`.

        Returns:
            Optional[bool]: True if the full parameter grid should be built, False if not,
                or None for default behavior.
        """
        return self._build_grid

    @property
    def grid_indices(self) -> tp.Union[None, slice, tp.Sequence[int]]:
        """Indices used to skip certain combinations.

        See `combine_params`.

        Returns:
            Union[None, slice, Sequence[int]]: Indices used to select specific combinations from the grid.
        """
        return self._grid_indices

    @property
    def random_subset(self) -> tp.Union[None, int, float]:
        """Size of the random subset to select.

        See `combine_params`.

        Returns:
            Union[None, int, float]: Size or fraction used to select a random subset of parameter combinations.
        """
        return self._random_subset

    @property
    def random_replace(self) -> bool:
        """Flag determining whether to select combinations with replacement.

        See `combine_params`.

        Returns:
            bool: True if sampling is performed with replacement; otherwise False.
        """
        return self._random_replace

    @property
    def random_sort(self) -> bool:
        """Flag determining whether to sort the positions of selected combinations.

        See `combine_params`.

        Returns:
            bool: True if the selected random indices are sorted; otherwise False.
        """
        return self._random_sort

    @property
    def max_guesses(self) -> tp.Union[None, int, float]:
        """Maximum number of guesses during parameter search.

        See `combine_params`.

        Returns:
            Union[None, int, float]: Maximum number of parameter guesses allowed.
        """
        return self._max_guesses

    @property
    def max_misses(self) -> tp.Union[None, int, float]:
        """Maximum allowed misses during parameter search.

        See `combine_params`.

        Returns:
            Union[None, int, float]: Maximum number of misses allowed during parameter search.
        """
        return self._max_misses

    @property
    def seed(self) -> tp.Optional[int]:
        """Random seed for reproducibility.

        See `combine_params`.

        Returns:
            Optional[int]: Seed value used for random number generation.
        """
        return self._seed

    @property
    def name_tuple_to_str(self) -> tp.Union[bool, tp.Callable]:
        """Flag or function to convert name tuples to strings for the parameter index.

        See `combine_params`.

        Returns:
            Union[bool, Callable]: Flag or function used to convert name tuples to string format.
        """
        return self._name_tuple_to_str

    @property
    def clean_index_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for cleaning MultiIndex levels.

        See `vectorbtpro.base.indexes.clean_index`.

        Returns:
            Kwargs: Dictionary of keyword arguments for index cleaning.
        """
        return self._clean_index_kwargs

    @property
    def selection(self) -> tp.Optional[tp.Selection]:
        """Template or criteria to select specific parameter indices.

        Returns:
            Optional[Selection]: Selection criteria used to filter parameter combinations.
        """
        return self._selection

    @property
    def forward_kwargs_as(self) -> tp.Kwargs:
        """Mapping to forward keyword arguments.

        Returns:
            Kwargs: Dictionary mapping original keyword arguments to new names.
        """
        return self._forward_kwargs_as

    @property
    def mono_min_size(self) -> tp.Optional[int]:
        """Minimum number of parameter values to split, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Optional[int]: Minimum allowed size for mono-chunks.
        """
        return self._mono_min_size

    @property
    def mono_n_chunks(self) -> tp.Optional[tp.Union[str, int]]:
        """Specification for the number of mono-chunks, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Optional[Union[str, int]]: Number or method indicator for determining the number of chunks.
        """
        return self._mono_n_chunks

    @property
    def mono_chunk_len(self) -> tp.Optional[tp.Union[str, int]]:
        """Specification for the length of each mono-chunk, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Optional[Union[str, int]]: Length or method indicator for the size of each mono-chunk.
        """
        return self._mono_chunk_len

    @property
    def mono_chunk_meta(self) -> tp.Optional[tp.Iterable[tp.ChunkMeta]]:
        """Custom metadata for mono-chunks, as defined in `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            Optional[Iterable[ChunkMeta]]: Metadata information for each mono-chunk.
        """
        return self._mono_chunk_meta

    @property
    def mono_reduce(self) -> tp.Union[bool, tp.Kwargs]:
        """Flag or settings for reducing mono-chunk results.

        Returns:
            Union[bool, Kwargs]: Configuration determining if mono-chunks should be reduced,
                or a dictionary of settings.
        """
        return self._mono_reduce

    @property
    def mono_merge_func(self) -> tp.MaybeDict[tp.MergeFuncLike]:
        """Merging function or a dictionary of such to combine parameter values.

        See `vectorbtpro.utils.merging.MergeFunc`.

        Returns:
            MaybeDict[MergeFuncLike]: Merging function or a dictionary of such to combine parameter values.
        """
        return self._mono_merge_func

    @property
    def mono_merge_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Parameterizer.mono_merge_func`.

        Returns:
            Kwargs: Dictionary of keyword arguments passed to the merging function.
        """
        return self._mono_merge_kwargs

    @property
    def filter_results(self) -> bool:
        """Flag indicating whether to filter `vectorbtpro.utils.execution.NoResult` results during execution.

        Returns:
            bool: True if results marked as `vectorbtpro.utils.execution.NoResult`
                should be filtered out; otherwise False.
        """
        return self._filter_results

    @property
    def raise_no_results(self) -> bool:
        """Flag indicating whether to raise a `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.

        Returns:
            bool: True if a `vectorbtpro.utils.execution.NoResultsException` should be raised
                when there are no results; otherwise False.
        """
        return self._raise_no_results

    @property
    def merge_func(self) -> tp.MergeFuncLike:
        """Merging function used to aggregate individual results.

        See `vectorbtpro.utils.merging.MergeFunc`.

        Returns:
            MergeFuncLike: Merging function used to aggregate individual results.
        """
        return self._merge_func

    @property
    def merge_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `Parameterizer.merge_func`.

        Returns:
            Kwargs: Dictionary of keyword arguments used with the merging function.
        """
        return self._merge_kwargs

    @property
    def return_meta(self) -> bool:
        """Flag indicating if additional metadata should be returned.

        Returns:
            bool: True if metadata should be returned along with the main results; otherwise False.
        """
        return self._return_meta

    @property
    def return_param_index(self) -> bool:
        """Flag indicating if the parameter index should be included in the result.

        Returns:
            bool: True if the parameter index is to be returned with the results; otherwise False.
        """
        return self._return_param_index

    @property
    def execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for the execution handler.

        See `vectorbtpro.utils.execution.execute`.

        Returns:
            Kwargs: Dictionary of keyword arguments passed to the execution function.
        """
        return self._execute_kwargs

    @classmethod
    def find_params_in_obj(
        cls, obj: tp.Any, eval_id: tp.Optional[tp.Hashable] = None, **kwargs
    ) -> tp.PathDict:
        """Return a dictionary of `Param`-wrapped values found recursively within the provided object.

        Only parameters meeting the specified evaluation identifier are included.

        Args:
            obj (Any): Object to search for `Param` instances.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **kwargs: Keyword arguments for `vectorbtpro.utils.search_.find_in_obj`.

        Returns:
            PathDict: Dictionary containing the paths to `Param` instances found in the object.
        """
        return find_in_obj(
            obj, lambda k, v: isinstance(v, Param) and v.meets_eval_id(eval_id), **kwargs
        )

    @classmethod
    def param_product_to_objs(cls, obj: tp.Any, param_product: dict) -> tp.List[tp.Any]:
        """Return a list of objects produced by resolving the parameter product in the original object.

        If the parameter product is empty, returns an empty list.

        Uses `vectorbtpro.utils.search_.replace_in_obj` to replace parameter templates in the object.

        Iterates over each combination of parameter values to generate a new object.

        Args:
            obj (Any): Original object containing parameter templates.
            param_product (dict): Dictionary of parameter combinations.

        Returns:
            List[Any]: List of objects with parameter values replaced in the original object.
        """
        if len(param_product) == 0:
            return []
        param_product_items = list(param_product.items())
        n_values = len(param_product_items[0][1])
        new_objs = []
        for i in range(n_values):
            param_dct = {k: v[i] for k, v in param_product.items()}
            new_objs.append(replace_in_obj(obj, param_dct))
        return new_objs

    @classmethod
    def parse_and_inject_params(
        cls,
        flat_ann_args: tp.FlatAnnArgs,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.FlatAnnArgs:
        """Return a dictionary of flattened annotated arguments with parsed `Param` instances injected.

        Iterates over each key in the input dictionary and, if the annotation is a `Param` subclass or
        instance that meets the evaluation identifier, updates or merges the corresponding value.

        Processes function annotations to correctly parse and inject parameter values.

        Args:
            flat_ann_args (FlatAnnArgs): Flattened annotated arguments.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            FlatAnnArgs: Dictionary of flattened annotated arguments with injected `Param` instances.
        """
        new_flat_ann_args = dict()
        for k, v in flat_ann_args.items():
            new_flat_ann_args[k] = v = dict(v)
            if "annotation" in v:
                if isinstance(v["annotation"], type) and issubclass(v["annotation"], Param):
                    v["annotation"] = v["annotation"]()
                if isinstance(v["annotation"], Param) and v["annotation"].meets_eval_id(eval_id):
                    if "value" in v:
                        if not isinstance(v["value"], Param):
                            v["value"] = v["annotation"].replace(value=v["value"])
                        else:
                            v["value"] = v["value"].merge_over(v["annotation"])
        return new_flat_ann_args

    @classmethod
    def get_var_arg_names(cls, ann_args: tp.AnnArgs) -> tp.Tuple[str, str]:
        """Return a tuple with the names of variable positional and keyword arguments found in the
        provided annotations.

        Iterates over the annotations and extracts the names for parameters of kind
        `VAR_POSITIONAL` and `VAR_KEYWORD`.

        Args:
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            Tuple[str, str]: Tuple containing the names of variable positional and keyword arguments.
        """
        var_args_name = None
        var_kwargs_name = None
        for k, v in ann_args.items():
            if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                var_args_name = k
            if v["kind"] == inspect.Parameter.VAR_KEYWORD:
                var_kwargs_name = k
        return var_args_name, var_kwargs_name

    @classmethod
    def unroll_param_config(cls, param_config: dict, ann_args: tp.AnnArgs) -> dict:
        """Unroll a parameter configuration dictionary into discrete argument entries.

        Args:
            param_config (dict): Parameter configuration dictionary.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            dict: Updated configuration with expanded variable arguments.
        """
        var_args_name, var_kwargs_name = cls.get_var_arg_names(ann_args)
        new_param_config = dict(param_config)
        if var_args_name is not None and var_args_name in new_param_config:
            for i, arg in enumerate(new_param_config.pop(var_args_name)):
                new_param_config[f"{var_args_name}_{i}"] = arg
        if var_kwargs_name is not None and var_kwargs_name in new_param_config:
            for k, v in new_param_config.pop(var_kwargs_name).items():
                new_param_config[k] = v
        return new_param_config

    @classmethod
    def roll_param_config(cls, param_config: dict, ann_args: tp.AnnArgs) -> dict:
        """Roll a parameter configuration dictionary by aggregating discrete variable arguments.

        Args:
            param_config (dict): Parameter configuration dictionary.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.

        Returns:
            dict: Updated configuration with collapsed variable arguments.
        """
        var_args_name, var_kwargs_name = cls.get_var_arg_names(ann_args)
        new_param_config = dict(param_config)
        if var_args_name is not None:
            _args = ()
            while True:
                k = f"{var_args_name}_{len(_args)}"
                if k in new_param_config:
                    _args += (new_param_config.pop(k),)
                else:
                    break
            new_param_config[var_args_name] = _args
        if var_kwargs_name is not None:
            new_param_config[var_kwargs_name] = {}
            for k in list(new_param_config.keys()):
                if k not in ann_args:
                    new_param_config[var_kwargs_name][k] = new_param_config.pop(k)
        return new_param_config

    @classmethod
    def select_comb(
        cls,
        param_configs: tp.List[tp.Kwargs],
        param_index: tp.Optional[tp.Index],
        selection: tp.Selection,
        single_comb: bool = False,
        template_context: tp.KwargsLike = None,
        raise_no_results: bool = True,
    ) -> tp.Tuple[tp.List[tp.Kwargs], tp.Optional[tp.Index], bool]:
        """Select a parameter combination from the provided configurations and index.

        Args:
            param_configs (List[Kwargs]): List of parameter configuration dictionaries.
            param_index (Optional[Index]): Index for parameter combinations.
            selection (Selection): Selection criteria for choosing a parameter combination.
            single_comb (bool): Flag indicating whether to enforce a single combination.
            template_context (KwargsLike): Additional context for template substitution.
            raise_no_results (bool): Flag indicating whether to raise a
                `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.

        Returns:
            Tuple[List[Kwargs], Optional[Index], bool]: Tuple containing the selected configurations,
                updated index, and a flag for single combination.
        """
        selection = substitute_templates(selection, template_context, eval_id="selection")
        if selection is NoResult:
            if raise_no_results:
                raise NoResultsException
            return NoResult
        found_param = False
        kind = None
        if isinstance(selection, PosSel):
            selection = selection.value
            kind = "positions"
        elif isinstance(selection, LabelSel):
            selection = selection.value
            kind = "labels"
        if checks.is_hashable(selection):
            if kind == "positions" or (kind is None and checks.is_int(selection)):
                selection = {selection}
                found_param = True
                single_comb = True
            elif kind == "labels" or (
                kind is None and param_index is not None and selection in param_index
            ):
                selection = {param_index.get_loc(selection)}
                found_param = True
                single_comb = True
        if not found_param:
            if checks.is_iterable(selection):
                new_selection = set()
                for s in selection:
                    if isinstance(s, PosSel):
                        s = s.value
                        kind = "positions"
                    elif isinstance(s, LabelSel):
                        s = s.value
                        kind = "labels"
                    if kind == "positions" or (kind is None and checks.is_int(s)):
                        new_selection.add(s)
                    elif kind == "labels" or (
                        kind is None and param_index is not None and s in param_index
                    ):
                        new_selection.add(param_index.get_loc(s))
                    else:
                        raise ValueError(
                            f"Selection {selection} couldn't be matched with parameter index"
                        )
                selection = new_selection
                if len(selection) == 0:
                    raise ValueError(f"Selection {selection} is empty")
            else:
                raise ValueError(f"Selection {selection} couldn't be matched with parameter index")
        if param_index is not None:
            param_index = param_index[list(selection)]
        new_param_configs = []
        selection = selection.copy()
        for i, x in enumerate(param_configs):
            if i in selection:
                new_param_configs.append(x)
                selection.remove(i)
                if len(selection) == 0:
                    break
        if len(selection) > 0:
            raise ValueError(f"Selection {selection} couldn't be matched")
        return new_param_configs, param_index, single_comb

    @classmethod
    def iter_tasks(
        cls,
        func: tp.Callable,
        ann_args: tp.AnnArgs,
        param_configs: tp.List[tp.Kwargs],
        template_context: tp.KwargsLike = None,
    ) -> tp.TasksLike:
        """Yield tasks for execution by pairing a function with its arguments.

        Args:
            func (Callable): Function to be executed.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            param_configs (List[Kwargs]): List of parameter configuration dictionaries.
            template_context (KwargsLike): Additional context for template substitution.

        Yields:
            TasksLike: Tuple consisting of the function, its positional arguments, and its keyword arguments.
        """
        for p, param_config in enumerate(param_configs):
            _template_context = dict(template_context)
            _template_context["config_idx"] = p
            _ann_args = dict()
            for k, v in ann_args.items():
                v = dict(v)
                v["value"] = param_config[k]
                _ann_args[k] = v
            _args, _kwargs = ann_args_to_args(_ann_args)
            _args = substitute_templates(_args, _template_context, eval_id="args")
            _kwargs = substitute_templates(_kwargs, _template_context, eval_id="kwargs")
            yield func, _args, _kwargs

    @classmethod
    def get_mono_chunk_indices(
        cls,
        param_configs: tp.List[tp.Kwargs],
        mono_min_size: tp.Optional[int] = None,
        mono_n_chunks: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_len: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    ) -> tp.List[tp.List[int]]:
        """Get the indices for each mono-chunk based on parameter configurations.

        Args:
            param_configs (List[Kwargs]): List of parameter configuration dictionaries.
            mono_min_size (Optional[int]): Minimum number of parameter values to split.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.
            mono_n_chunks (Optional[Union[str, int]]): Specification for the number of mono-chunks.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.
            mono_chunk_len (Optional[Union[str, int]]):  Specification for the length of each mono-chunk.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.
            mono_chunk_meta (Optional[Iterable[ChunkMeta]]): Iterable containing metadata for each mono-chunk.

                See `vectorbtpro.utils.chunking.iter_chunk_meta`.

        Returns:
            List[List[int]]: List of index lists, each representing a mono-chunk.
        """
        if mono_chunk_meta is None:
            from vectorbtpro.utils.chunking import iter_chunk_meta

            mono_chunk_meta = iter_chunk_meta(
                n_chunks=mono_n_chunks,
                size=len(param_configs),
                min_size=mono_min_size,
                chunk_len=mono_chunk_len,
            )

        last_idx = -1
        indices_sorted = True
        mono_chunk_indices = []
        for _chunk_meta in mono_chunk_meta:
            if _chunk_meta.indices is not None:
                chunk_indices = list(_chunk_meta.indices)
            else:
                if _chunk_meta.start is None or _chunk_meta.end is None:
                    raise ValueError("Each chunk must have a start and an end index")
                chunk_indices = list(range(_chunk_meta.start, _chunk_meta.end))
            if indices_sorted:
                for idx in chunk_indices:
                    if idx != last_idx + 1:
                        indices_sorted = False
                        break
                    last_idx = idx
            mono_chunk_indices.append(chunk_indices)
        return mono_chunk_indices

    def build_mono_chunk_config(
        self,
        chunk_indices: tp.List[int],
        param_configs: tp.List[tp.Kwargs],
        param_config_keys: tp.Set[str],
        ann_args: tp.AnnArgs,
        flat_ann_args: tp.Optional[tp.FlatAnnArgs] = None,
        mono_reduce: tp.Union[bool, tp.Kwargs] = None,
        mono_merge_func: tp.MaybeDict[tp.MergeFuncLike] = None,
        mono_merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        """Build a mono-chunk parameter configuration by aggregating multiple configurations.

        Args:
            chunk_indices (List[int]): List of indices defining the mono-chunk.
            param_configs (List[Kwargs]): List of parameter configuration dictionaries.
            param_config_keys (Set[str]): Set of keys required in each parameter configuration.
            ann_args (AnnArgs): Annotated arguments.

                See `vectorbtpro.utils.parsing.annotate_args`.
            flat_ann_args (Optional[FlatAnnArgs]): Flattened annotation arguments.
            mono_reduce (Union[bool, Kwargs]): Configuration for reducing parameters.
            mono_merge_func (MaybeDict[MergeFuncLike]): Merging function or a dictionary of such to
                combine parameter values.

                See `vectorbtpro.utils.merging.MergeFunc`.
            mono_merge_kwargs (KwargsLike): Keyword arguments for `mono_merge_func`.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            Kwargs: Mono-chunk parameter configuration.
        """
        if flat_ann_args is None:
            flat_ann_args = flatten_ann_args(ann_args)
        new_param_config = dict()
        all_same = dict()
        for j, i in enumerate(chunk_indices):
            param_config = self.unroll_param_config(param_configs[i], ann_args)
            for k, v in param_config.items():
                if j == 0:
                    new_param_config[k] = [v]
                    all_same[k] = True
                else:
                    if k not in new_param_config:
                        raise ValueError(
                            f"Mono-chunks cannot be built because key '{k}' is not present in all parameter configs"
                        )
                    if v is not new_param_config[k][-1]:
                        all_same[k] = False
                    new_param_config[k].append(v)

        for k, v in new_param_config.items():
            k_reduce = None
            k_merge_func = None
            k_merge_kwargs = None
            if mono_reduce is not None:
                if isinstance(mono_reduce, dict):
                    if k in mono_reduce:
                        k_reduce = mono_reduce[k]
                else:
                    k_reduce = mono_reduce
            if mono_merge_func is not None:
                if isinstance(mono_merge_func, dict):
                    if k in mono_merge_func:
                        k_merge_func = mono_merge_func[k]
                else:
                    k_merge_func = mono_merge_func
            if mono_merge_kwargs is not None:
                if set(mono_merge_kwargs.keys()) <= param_config_keys:
                    if k in mono_merge_kwargs:
                        k_merge_kwargs = mono_merge_kwargs[k]
                else:
                    k_merge_kwargs = mono_merge_kwargs
            if k in flat_ann_args:
                ann_arg = flat_ann_args[k]
                if "value" in ann_arg and isinstance(ann_arg["value"], Param):
                    if k_reduce is None:
                        param_k_reduce = ann_arg["value"].resolve_field("mono_reduce")
                        if param_k_reduce is not None:
                            k_reduce = param_k_reduce
                    if k_merge_func is None:
                        param_k_merge_func = ann_arg["value"].resolve_field("mono_merge_func")
                        if param_k_merge_func is not None:
                            k_merge_func = param_k_merge_func
                    if k_merge_kwargs is None:
                        param_k_merge_kwargs = ann_arg["value"].resolve_field("mono_merge_kwargs")
                        if param_k_merge_kwargs is not None:
                            k_merge_kwargs = param_k_merge_kwargs
            if k_reduce is None:
                k_reduce = all_same[k]
            elif k_reduce and not all_same[k]:
                k_reduce = False
            if k_reduce:
                new_param_config[k] = new_param_config[k][0]
            elif k_merge_func is not None:
                if isinstance(k_merge_func, MergeFunc):
                    k_merge_func = k_merge_func.replace(
                        merge_kwargs=k_merge_kwargs,
                        context=template_context,
                        eval_id_prefix="mono_",
                    )
                else:
                    k_merge_func = MergeFunc(
                        k_merge_func,
                        merge_kwargs=k_merge_kwargs,
                        context=template_context,
                        eval_id_prefix="mono_",
                    )
                new_param_config[k] = k_merge_func(v)

        return self.roll_param_config(new_param_config, ann_args)

    def run(
        self,
        func: tp.Callable,
        *args,
        param_configs: tp.Optional[tp.MaybeSequence[tp.Kwargs]] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Union[dict, tp.MergeableResults, tp.Tuple[tp.MergeableResults, tp.Optional[tp.Index]]]:
        """Parameterize the function's arguments and execute the function.

        Args:
            func (Callable): Target function to execute.
            *args: Positional arguments for `func`.
            param_configs (Optional[MaybeSequence[dict]]): Configuration(s) used to parameterize `func`'s arguments.
            eval_id (Optional[Hashable]): Evaluation identifier.
            **kwargs: Keyword arguments for `func`.

        Returns:
            Union[dict, MergeableResults, Tuple[MergeableResults, Optional[Index]]]: Result of
                executing `func` with parameterized arguments.
        """
        param_search_kwargs = self.param_search_kwargs
        skip_single_comb = self.skip_single_comb
        template_context = self.template_context
        build_grid = self.build_grid
        grid_indices = self.grid_indices
        random_subset = self.random_subset
        random_replace = self.random_replace
        random_sort = self.random_sort
        max_guesses = self.max_guesses
        max_misses = self.max_misses
        seed = self.seed
        name_tuple_to_str = self.name_tuple_to_str
        clean_index_kwargs = self.clean_index_kwargs
        selection = self.selection
        forward_kwargs_as = self.forward_kwargs_as
        mono_min_size = self.mono_min_size
        mono_n_chunks = self.mono_n_chunks
        mono_chunk_len = self.mono_chunk_len
        mono_chunk_meta = self.mono_chunk_meta
        mono_reduce = self.mono_reduce
        mono_merge_func = self.mono_merge_func
        mono_merge_kwargs = self.mono_merge_kwargs
        filter_results = self.filter_results
        raise_no_results = self.raise_no_results
        merge_func = self.merge_func
        merge_kwargs = self.merge_kwargs
        return_meta = self.return_meta
        return_param_index = self.return_param_index
        execute_kwargs = self.execute_kwargs

        template_context["eval_id"] = eval_id

        if param_configs is None:
            param_configs = []

        parsed_merge_func = parse_merge_func(func, eval_id=eval_id)
        if parsed_merge_func is not None:
            if merge_func is not None:
                raise ValueError(
                    f"Two conflicting merge functions: {parsed_merge_func} (annotations) and {merge_func} (merge_func)"
                )
            merge_func = parsed_merge_func

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

        template_context["ann_args"] = annotate_args(
            func,
            args,
            kwargs,
            allow_partial=True,
            attach_annotations=True,
        )
        template_context["flat_ann_args"] = flatten_ann_args(template_context["ann_args"])
        if has_annotatables(func):
            template_context["flat_ann_args"] = self.parse_and_inject_params(
                template_context["flat_ann_args"],
                eval_id=eval_id,
            )
            template_context["ann_args"] = unflatten_ann_args(
                template_context["flat_ann_args"],
                partial_ann_args=template_context["ann_args"],
            )

        pc_names = []
        pc_names_none = True
        n_param_configs = 0
        if isinstance(param_configs, dict):
            new_param_configs = []
            for k, v in param_configs.items():
                v = dict(v)
                v["_name"] = k
                new_param_configs.append(v)
            param_configs = new_param_configs
        else:
            param_configs = list(param_configs)
        for i, param_config in enumerate(param_configs):
            param_config = self.unroll_param_config(param_config, template_context["ann_args"])
            if "_name" in param_config and param_config["_name"] is not None:
                pc_names.append(param_config.pop("_name"))
                pc_names_none = False
            else:
                pc_names.append(n_param_configs)
            param_configs[i] = param_config
            n_param_configs += 1

        paramable_kwargs = {}
        for k, v in template_context["flat_ann_args"].items():
            if "value" in v:
                paramable_kwargs[k] = v["value"]
        param_dct = self.find_params_in_obj(
            paramable_kwargs, eval_id=eval_id, **param_search_kwargs
        )
        param_index = None
        if len(param_dct) > 0:
            param_product, param_index = combine_params(
                param_dct,
                build_grid=build_grid,
                grid_indices=grid_indices,
                random_subset=random_subset,
                random_replace=random_replace,
                random_sort=random_sort,
                max_guesses=max_guesses,
                max_misses=max_misses,
                seed=seed,
                name_tuple_to_str=name_tuple_to_str,
                clean_index_kwargs=clean_index_kwargs,
                raise_empty_error=True,
            )
            product_param_configs = self.param_product_to_objs(paramable_kwargs, param_product)
            if len(param_configs) == 0:
                param_configs = product_param_configs
            else:
                new_param_configs = []
                for i in range(len(product_param_configs)):
                    for param_config in param_configs:
                        new_param_config = merge_dicts(product_param_configs[i], param_config)
                        new_param_configs.append(new_param_config)
                param_configs = new_param_configs

        n_config_params = len(pc_names)
        if param_index is not None:
            if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                new_param_index = param_index
            else:
                from vectorbtpro.base.indexes import combine_indexes

                new_param_index = combine_indexes(
                    (
                        param_index,
                        pd.Index(pc_names, name="param_config"),
                    ),
                    **clean_index_kwargs,
                )
        else:
            if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                new_param_index = None
            else:
                new_param_index = pd.Index(pc_names, name="param_config")
        template_context["param_index"] = new_param_index

        if len(param_configs) == 0:
            template_context["single_comb"] = True
            param_configs.append(dict())
        else:
            template_context["single_comb"] = False

        param_config_keys = set()
        new_param_configs = []
        for param_config in param_configs:
            new_param_config = merge_dicts(paramable_kwargs, param_config)
            for k in new_param_config:
                param_config_keys.add(k)
            new_param_config = self.roll_param_config(
                new_param_config, template_context["ann_args"]
            )
            new_param_configs.append(new_param_config)
        template_context["param_configs"] = new_param_configs

        if selection is not None:
            new_param_configs, new_param_index, new_single_comb = self.select_comb(
                template_context["param_configs"],
                template_context["param_index"],
                selection,
                single_comb=template_context["single_comb"],
                template_context=template_context,
                raise_no_results=raise_no_results,
            )
            template_context["param_configs"] = new_param_configs
            template_context["param_index"] = new_param_index
            template_context["single_comb"] = new_single_comb

        if skip_single_comb and template_context["single_comb"]:
            tasks = list(
                self.iter_tasks(
                    func,
                    template_context["ann_args"],
                    template_context["param_configs"],
                    template_context=template_context,
                )
            )
            result = tasks[0][0](*tasks[0][1], **tasks[0][2])
            if result is NoResult:
                if raise_no_results:
                    raise NoResultsException
                return NoResult
            return result

        if mono_n_chunks is not None or mono_chunk_len is not None or mono_chunk_meta is not None:
            template_context["mono_chunk_indices"] = self.get_mono_chunk_indices(
                template_context["param_configs"],
                mono_min_size=mono_min_size,
                mono_n_chunks=mono_n_chunks,
                mono_chunk_len=mono_chunk_len,
                mono_chunk_meta=mono_chunk_meta,
            )
            new_param_configs = []
            new_param_index = []
            for chunk_indices in template_context["mono_chunk_indices"]:
                new_param_configs.append(
                    self.build_mono_chunk_config(
                        chunk_indices,
                        template_context["param_configs"],
                        param_config_keys,
                        template_context["ann_args"],
                        flat_ann_args=template_context["flat_ann_args"],
                        mono_reduce=mono_reduce,
                        mono_merge_func=mono_merge_func,
                        mono_merge_kwargs=mono_merge_kwargs,
                        template_context=template_context,
                    )
                )
                if template_context["param_index"] is not None:
                    new_param_index.append(template_context["param_index"][chunk_indices])
            template_context["param_configs"] = new_param_configs
            if template_context["param_index"] is not None:
                template_context["param_index"] = new_param_index
        else:
            template_context["mono_chunk_indices"] = None

        template_context["tasks"] = self.iter_tasks(
            func,
            template_context["ann_args"],
            template_context["param_configs"],
            template_context=template_context,
        )

        if return_meta:
            return dict(
                ann_args=template_context["ann_args"],
                flat_ann_args=template_context["flat_ann_args"],
                single_comb=template_context["single_comb"],
                param_configs=template_context["param_configs"],
                param_index=template_context["param_index"],
                mono_chunk_indices=template_context["mono_chunk_indices"],
                tasks=template_context["tasks"],
            )

        execute_kwargs = merge_dicts(
            dict(show_progress=False if template_context["single_comb"] else None),
            execute_kwargs,
        )
        keys = template_context["param_index"]
        if keys is not None and eval_id is not None:
            new_keys = []
            for key in keys:
                if isinstance(keys, pd.MultiIndex):
                    new_keys.append((MISSING, *key))
                else:
                    new_keys.append((MISSING, key))
            keys = pd.MultiIndex.from_tuples(new_keys, names=(f"eval_id={eval_id}", *keys.names))
        results = execute(
            template_context["tasks"],
            size=len(template_context["param_configs"]),
            keys=keys,
            **execute_kwargs,
        )
        if filter_results:
            try:
                results, template_context["param_index"] = filter_out_no_results(
                    results,
                    keys=template_context["param_index"],
                )
            except NoResultsException as e:
                if raise_no_results:
                    raise e
                if return_param_index:
                    return NoResult, None
                return NoResult
            no_results_filtered = True
        else:
            no_results_filtered = False

        if merge_func is not None:
            from vectorbtpro.base.merging import is_merge_func_from_config

            if is_merge_func_from_config(merge_func):
                merge_kwargs = merge_dicts(
                    dict(
                        keys=template_context["param_index"],
                        filter_results=not no_results_filtered,
                        raise_no_results=raise_no_results,
                    ),
                    merge_kwargs,
                )
            if isinstance(merge_func, MergeFunc):
                merge_func = merge_func.replace(merge_kwargs=merge_kwargs, context=template_context)
            else:
                merge_func = MergeFunc(
                    merge_func, merge_kwargs=merge_kwargs, context=template_context
                )
            if return_param_index:
                return merge_func(results), template_context["param_index"]
            return merge_func(results)
        if return_param_index:
            return results, template_context["param_index"]
        return results


def parameterized(
    *args,
    parameterizer: tp.Optional[tp.Type[Parameterizer]] = None,
    replace_parameterizer: tp.Optional[bool] = None,
    merge_to_execute_kwargs: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Callable:
    """Decorator that parameterizes a function's inputs using `Parameterizer`.

    Parameter values can be modified via the wrapper function's `options` attribute or
    by passing keyword arguments prefixed with an underscore.

    If `merge_to_execute_kwargs` is True, keyword arguments not found in `Parameterizer` or
    `execute_kwargs` are merged into `execute_kwargs`. Otherwise, they are passed directly to `Parameterizer`.

    If a `parameterizer` instance is provided and `replace_parameterizer` is True, create a new
    `Parameterizer` instance by replacing any arguments that are not None.

    Args:
        func (Callable): Function to be decorated.
        parameterizer (Optional[Type[Parameterizer]]): `Parameterizer` class or instance for
            parameterizing inputs.
        replace_parameterizer (Optional[bool]): Flag to create a new `Parameterizer` instance with
            non-None arguments replaced.
        merge_to_execute_kwargs (Optional[bool]): Flag that determines whether to merge unspecified
            keyword arguments into `execute_kwargs`.
        **kwargs: Keyword arguments for `Parameterizer` or the decorated function.

    Returns:
        Callable: New function with the same signature as the provided function.

    !!! info
        For default settings, see `vectorbtpro._settings.params`.

    Examples:
        No parameters, no parameter configs:

        ```pycon
        >>> from vectorbtpro import *

        >>> @vbt.parameterized(merge_func="column_stack")
        ... def my_ma(sr_or_df, window, wtype="simple", minp=0, adjust=False):
        ...     return sr_or_df.vbt.ma(window, wtype=wtype, minp=minp, adjust=adjust)

        >>> sr = pd.Series([1, 2, 3, 4, 3, 2, 1])
        >>> my_ma(sr, 3)
        0    1.000000
        1    1.500000
        2    2.000000
        3    3.000000
        4    3.333333
        5    3.000000
        6    2.000000
        dtype: float64
        ```

        One parameter, no parameter configs:

        ```pycon
        >>> my_ma(sr, vbt.Param([3, 4, 5]))
        window         3    4    5
        0       1.000000  1.0  1.0
        1       1.500000  1.5  1.5
        2       2.000000  2.0  2.0
        3       3.000000  2.5  2.5
        4       3.333333  3.0  2.6
        5       3.000000  3.0  2.8
        6       2.000000  2.5  2.6
        ```

        Product of two parameters, no parameter configs:

        ```pycon
        >>> my_ma(
        ...     sr,
        ...     vbt.Param([3, 4, 5]),
        ...     wtype=vbt.Param(["simple", "exp"])
        ... )
        window         3                4                5
        wtype     simple       exp simple       exp simple       exp
        0       1.000000  1.000000    1.0  1.000000    1.0  1.000000
        1       1.500000  1.500000    1.5  1.400000    1.5  1.333333
        2       2.000000  2.250000    2.0  2.040000    2.0  1.888889
        3       3.000000  3.125000    2.5  2.824000    2.5  2.592593
        4       3.333333  3.062500    3.0  2.894400    2.6  2.728395
        5       3.000000  2.531250    3.0  2.536640    2.8  2.485597
        6       2.000000  1.765625    2.5  1.921984    2.6  1.990398
        ```

        No parameters, one partial parameter config:

        ```pycon
        >>> my_ma(sr, param_configs=[dict(window=3)])
        param_config         0
        0             1.000000
        1             1.500000
        2             2.000000
        3             3.000000
        4             3.333333
        5             3.000000
        6             2.000000
        ```

        No parameters, one full parameter config:

        ```pycon
        >>> my_ma(param_configs=[dict(sr_or_df=sr, window=3)])
        param_config         0
        0             1.000000
        1             1.500000
        2             2.000000
        3             3.000000
        4             3.333333
        5             3.000000
        6             2.000000
        ```

        No parameters, multiple parameter configs:

        ```pycon
        >>> my_ma(param_configs=[
        ...     dict(sr_or_df=sr + 1, window=2),
        ...     dict(sr_or_df=sr - 1, window=3)
        ... ], minp=None)
        param_config    0         1
        0             NaN       NaN
        1             2.5       NaN
        2             3.5  1.000000
        3             4.5  2.000000
        4             4.5  2.333333
        5             3.5  2.000000
        6             2.5  1.000000
        ```

        Multiple parameters, multiple parameter configs:

        ```pycon
        >>> my_ma(param_configs=[
        ...     dict(sr_or_df=sr + 1, minp=0),
        ...     dict(sr_or_df=sr - 1, minp=None)
        ... ], window=vbt.Param([2, 3]))
        window          2              3
        param_config    0    1         0         1
        0             2.0  NaN  2.000000       NaN
        1             2.5  0.5  2.500000       NaN
        2             3.5  1.5  3.000000  1.000000
        3             4.5  2.5  4.000000  2.000000
        4             4.5  2.5  4.333333  2.333333
        5             3.5  1.5  4.000000  2.000000
        6             2.5  0.5  3.000000  1.000000
        ```

        Using annotations:

        ```pycon
        >>> @vbt.parameterized
        ... def my_ma(
        ...     sr_or_df,
        ...     window: vbt.Param,
        ...     wtype: vbt.Param = "simple",
        ...     minp=0,
        ...     adjust=False
        ... ) -> vbt.MergeFunc("column_stack"):
        ...     return sr_or_df.vbt.ma(window, wtype=wtype, minp=minp, adjust=adjust)

        >>> my_ma(sr, [3, 4, 5], ["simple", "exp"])
        window         3                4                5
        wtype     simple       exp simple       exp simple       exp
        0       1.000000  1.000000    1.0  1.000000    1.0  1.000000
        1       1.500000  1.500000    1.5  1.400000    1.5  1.333333
        2       2.000000  2.250000    2.0  2.040000    2.0  1.888889
        3       3.000000  3.125000    2.5  2.824000    2.5  2.592593
        4       3.333333  3.062500    3.0  2.894400    2.6  2.728395
        5       3.000000  2.531250    3.0  2.536640    2.8  2.485597
        6       2.000000  1.765625    2.5  1.921984    2.6  1.990398
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        from vectorbtpro._settings import settings

        params_cfg = settings["params"]

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            parameterizer = kwargs.get("_parameterizer")
            if parameterizer is None:
                parameterizer = wrapper.options["parameterizer"]
            if parameterizer is None:
                parameterizer = params_cfg["parameterizer"]
            if parameterizer is None:
                parameterizer = Parameterizer

            arg_names_set = set(parameterizer._expected_keys)
            kwargs_options = {}
            for k in list(kwargs.keys()):
                if k.startswith("_"):
                    if k[1:] in wrapper.options or k[1:] in arg_names_set:
                        kwargs_options[k[1:]] = kwargs.pop(k)
            parameterizer_kwargs = merge_dicts(wrapper.options, kwargs_options)
            _ = parameterizer_kwargs.pop("parameterizer")
            replace_parameterizer = parameterizer_kwargs.pop("replace_parameterizer")
            merge_to_execute_kwargs = parameterizer_kwargs.pop("merge_to_execute_kwargs")
            eval_id = parameterizer_kwargs.pop("eval_id", None)

            if merge_to_execute_kwargs is None:
                merge_to_execute_kwargs = params_cfg["merge_to_execute_kwargs"]
            if merge_to_execute_kwargs and len(parameterizer_kwargs) > 0:
                arg_names_set = set(parameterizer._expected_keys)
                execute_kwargs = parameterizer_kwargs.pop("execute_kwargs", None)
                if execute_kwargs is None:
                    _execute_kwargs = {}
                else:
                    _execute_kwargs = dict(execute_kwargs)
                execute_kwargs_changed = False
                for k in list(parameterizer_kwargs.keys()):
                    if k not in arg_names_set and k not in _execute_kwargs:
                        _execute_kwargs[k] = parameterizer_kwargs.pop(k)
                        execute_kwargs_changed = True
                if execute_kwargs_changed:
                    parameterizer_kwargs["execute_kwargs"] = _execute_kwargs
                else:
                    parameterizer_kwargs["execute_kwargs"] = execute_kwargs
            if isinstance(parameterizer, type):
                checks.assert_subclass_of(parameterizer, Parameterizer, arg_name="parameterizer")
                parameterizer = parameterizer(**parameterizer_kwargs)
            else:
                checks.assert_instance_of(parameterizer, Parameterizer, arg_name="parameterizer")
                if replace_parameterizer is None:
                    replace_parameterizer = params_cfg["replace_parameterizer"]
                if replace_parameterizer and len(parameterizer_kwargs) > 0:
                    parameterizer = parameterizer.replace(**parameterizer_kwargs)
            return parameterizer.run(func, *args, eval_id=eval_id, **kwargs)

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_parameterized = True
        wrapper.options = FrozenConfig(
            parameterizer=parameterizer,
            replace_parameterizer=replace_parameterizer,
            merge_to_execute_kwargs=merge_to_execute_kwargs,
            **kwargs,
        )
        signature = inspect.signature(wrapper)
        lists_var_kwargs = False
        for k, v in signature.parameters.items():
            if v.kind == v.VAR_KEYWORD:
                lists_var_kwargs = True
                break
        if not lists_var_kwargs:
            var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
            wrapper.__signature__ = signature.replace(parameters=new_parameters)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
