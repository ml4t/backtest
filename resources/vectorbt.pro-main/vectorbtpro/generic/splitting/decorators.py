# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing decorators for splitting functionality."""

import inspect
from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.generic.splitting.base import Splitter, Takeable
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import FrozenConfig, merge_dicts
from vectorbtpro.utils.execution import NoResult, NoResultsException
from vectorbtpro.utils.params import parameterized
from vectorbtpro.utils.parsing import (
    ann_args_to_args,
    annotate_args,
    flatten_ann_args,
    get_func_arg_names,
    match_ann_arg,
    unflatten_ann_args,
)
from vectorbtpro.utils.template import Rep, RepEval, substitute_templates

__all__ = [
    "split",
    "cv_split",
]


def split(
    *args,
    splitter: tp.Union[None, str, Splitter, tp.Callable] = None,
    splitter_cls: tp.Optional[tp.Type[Splitter]] = None,
    splitter_kwargs: tp.KwargsLike = None,
    index: tp.Optional[tp.IndexLike] = None,
    index_from: tp.Optional[tp.AnnArgQuery] = None,
    takeable_args: tp.Optional[tp.MaybeIterable[tp.AnnArgQuery]] = None,
    template_context: tp.KwargsLike = None,
    forward_kwargs_as: tp.KwargsLike = None,
    return_splitter: bool = False,
    apply_kwargs: tp.KwargsLike = None,
    **var_kwargs,
) -> tp.Callable:
    """Decorator that splits the inputs of a function.

    Resolves a `Splitter` instance and applies splitting to the inputs of the decorated function.

    The decorator performs the following operations:

    1. Resolves a splitter of type `vectorbtpro.generic.splitting.base.Splitter` using the
        `splitter` argument. The splitter is constructed using the provided `index` and `splitter_kwargs`.
    2. Wraps arguments specified in `takeable_args` using `vectorbtpro.generic.splitting.base.Takeable`.
    3. Applies the splitter operation using `vectorbtpro.generic.splitting.base.Splitter.apply`
        with the function's arguments and additional `apply_kwargs`.

    Arguments `splitter_kwargs` are forwarded to the splitter factory method, and `apply_kwargs`
    are passed to `vectorbtpro.generic.splitting.base.Splitter.apply`. If variable keyword arguments
    are provided, they are used to update `splitter_kwargs` or `apply_kwargs` based on the context.
    An error is raised if both `splitter_kwargs` and `apply_kwargs` are explicitly set.

    Args:
        func (Callable): Function to be decorated.
        splitter (Union[None, str, Splitter, Callable]): Splitter instance, the name of a factory method
            (e.g. "from_n_rolling"), or the factory method itself.

            If None, the appropriate splitter is determined using
            `vectorbtpro.generic.splitting.base.Splitter.guess_method`.
        splitter_cls (Optional[Type[Splitter]]): Splitter class to use.

            Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
        splitter_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.generic.splitting.base.Splitter`.
        index (Optional[IndexLike]): Index used for splitting.

            If not provided, it is derived from `index_from` or by parsing the first argument in `takeable_args`.
        index_from (Optional[AnnArgQuery]): Argument name or position used to extract the index
            when `index` is not supplied.
        takeable_args (Optional[MaybeIterable[AnnArgQuery]]): Argument name(s) or position(s)
            to be wrapped with `vectorbtpro.generic.splitting.base.Takeable`.
        template_context (KwargsLike): Additional context for template substitution.
        forward_kwargs_as (KwargsLike): Mapping for renaming keyword arguments when forwarding them.
        return_splitter (bool): If True, returns the constructed splitter instance instead of
            applying it to the function.
        apply_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.generic.splitting.base.Splitter.apply`.
        **var_kwargs: Keyword arguments to be distributed between `splitter_kwargs` and `apply_kwargs`.

    Returns:
        Callable: Wrapper function that executes the original function using the splitter.

    Examples:
        Split a Series and return its sum:

        ```pycon
        >>> from vectorbtpro import *

        >>> @vbt.split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=2),
        ...     takeable_args=["sr"]
        ... )
        ... def f(sr):
        ...     return sr.sum()

        >>> index = pd.date_range("2020-01-01", "2020-01-06")
        >>> sr = pd.Series(np.arange(len(index)), index=index)
        >>> f(sr)
        split
        0     3
        1    12
        dtype: int64
        ```

        Perform a split manually:

        ```pycon
        >>> @vbt.split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=2),
        ...     takeable_args=["index"]
        ... )
        ... def f(index, sr):
        ...     return sr[index].sum()

        >>> f(index, sr)
        split
        0     3
        1    12
        dtype: int64
        ```

        Construct splitter and mark arguments as "takeable" manually:

        ```pycon
        >>> splitter = vbt.Splitter.from_n_rolling(index, n=2)

        >>> @vbt.split(splitter=splitter)
        ... def f(sr):
        ...     return sr.sum()

        >>> f(vbt.Takeable(sr))
        split
        0     3
        1    12
        dtype: int64
        ```

        Split multiple timeframes using a custom index:

        ```pycon
        >>> @vbt.split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=2),
        ...     index=index,
        ...     takeable_args=["h12_sr", "d2_sr"]
        ... )
        ... def f(h12_sr, d2_sr):
        ...     return h12_sr.sum() + d2_sr.sum()

        >>> h12_index = pd.date_range("2020-01-01", "2020-01-06", freq="12H")
        >>> d2_index = pd.date_range("2020-01-01", "2020-01-06", freq="2D")
        >>> h12_sr = pd.Series(np.arange(len(h12_index)), index=h12_index)
        >>> d2_sr = pd.Series(np.arange(len(d2_index)), index=d2_index)
        >>> f(h12_sr, d2_sr)
        split
        0    15
        1    42
        dtype: int64
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            splitter = kwargs.pop("_splitter", wrapper.options["splitter"])
            splitter_cls = kwargs.pop("_splitter_cls", wrapper.options["splitter_cls"])
            splitter_kwargs = merge_dicts(
                wrapper.options["splitter_kwargs"], kwargs.pop("_splitter_kwargs", {})
            )
            index = kwargs.pop("_index", wrapper.options["index"])
            index_from = kwargs.pop("_index_from", wrapper.options["index_from"])
            takeable_args = kwargs.pop("_takeable_args", wrapper.options["takeable_args"])
            if takeable_args is None:
                takeable_args = set()
            elif checks.is_iterable(takeable_args) and not isinstance(takeable_args, str):
                takeable_args = set(takeable_args)
            else:
                takeable_args = {takeable_args}
            template_context = merge_dicts(
                wrapper.options["template_context"], kwargs.pop("_template_context", {})
            )
            apply_kwargs = merge_dicts(
                wrapper.options["apply_kwargs"], kwargs.pop("_apply_kwargs", {})
            )
            return_splitter = kwargs.pop("_return_splitter", wrapper.options["return_splitter"])
            forward_kwargs_as = merge_dicts(
                wrapper.options["forward_kwargs_as"], kwargs.pop("_forward_kwargs_as", {})
            )
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

            if splitter_cls is None:
                splitter_cls = Splitter
            if len(var_kwargs) > 0:
                var_splitter_kwargs = {}
                var_apply_kwargs = {}
                if splitter is None or not isinstance(splitter, splitter_cls):
                    apply_arg_names = get_func_arg_names(splitter_cls.apply)
                    if splitter is not None:
                        if isinstance(splitter, str):
                            splitter_arg_names = get_func_arg_names(getattr(splitter_cls, splitter))
                        else:
                            splitter_arg_names = get_func_arg_names(splitter)
                        for k, v in var_kwargs.items():
                            assigned = False
                            if k in splitter_arg_names:
                                var_splitter_kwargs[k] = v
                                assigned = True
                            if k != "split" and k in apply_arg_names:
                                var_apply_kwargs[k] = v
                                assigned = True
                            if not assigned:
                                raise ValueError(f"Argument '{k}' couldn't be assigned")
                    else:
                        for k, v in var_kwargs.items():
                            if k == "freq":
                                var_splitter_kwargs[k] = v
                                var_apply_kwargs[k] = v
                            elif k == "split" or k not in apply_arg_names:
                                var_splitter_kwargs[k] = v
                            else:
                                var_apply_kwargs[k] = v
                else:
                    var_apply_kwargs = var_kwargs
                splitter_kwargs = merge_dicts(var_splitter_kwargs, splitter_kwargs)
                apply_kwargs = merge_dicts(var_apply_kwargs, apply_kwargs)
            if len(splitter_kwargs) > 0:
                if splitter is None:
                    splitter = splitter_cls.guess_method(**splitter_kwargs)
                if splitter is None:
                    raise ValueError("Splitter method couldn't be guessed")
            else:
                if splitter is None:
                    raise ValueError("Must provide splitter or splitter method")
            if not isinstance(splitter, splitter_cls) and index is not None:
                if isinstance(splitter, str):
                    splitter = getattr(splitter_cls, splitter)
                splitter = splitter(index, template_context=template_context, **splitter_kwargs)
                if return_splitter:
                    return splitter

            ann_args = annotate_args(func, args, kwargs, attach_annotations=True)
            flat_ann_args = flatten_ann_args(ann_args)
            if isinstance(splitter, splitter_cls):
                flat_ann_args = splitter.parse_and_inject_takeables(flat_ann_args)
            else:
                flat_ann_args = splitter_cls.parse_and_inject_takeables(flat_ann_args)
            for k, v in flat_ann_args.items():
                if isinstance(v["value"], Takeable):
                    takeable_args.add(k)
            for takeable_arg in takeable_args:
                arg_name = match_ann_arg(ann_args, takeable_arg, return_name=True)
                if not isinstance(flat_ann_args[arg_name]["value"], Takeable):
                    flat_ann_args[arg_name]["value"] = Takeable(flat_ann_args[arg_name]["value"])
            new_ann_args = unflatten_ann_args(flat_ann_args)
            args, kwargs = ann_args_to_args(new_ann_args)

            if not isinstance(splitter, splitter_cls):
                if index is None and index_from is not None:
                    index = splitter_cls.get_obj_index(match_ann_arg(ann_args, index_from))
                if index is None and len(takeable_args) > 0:
                    first_takeable = match_ann_arg(ann_args, list(takeable_args)[0])
                    if isinstance(first_takeable, Takeable):
                        first_takeable = first_takeable.obj
                    index = splitter_cls.get_obj_index(first_takeable)
                if index is None:
                    raise ValueError("Must provide splitter, index, index_from, or takeable_args")
                if isinstance(splitter, str):
                    splitter = getattr(splitter_cls, splitter)
                splitter = splitter(index, template_context=template_context, **splitter_kwargs)
            if return_splitter:
                return splitter

            return splitter.apply(
                func,
                *args,
                **kwargs,
                **apply_kwargs,
            )

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_split = True
        wrapper.options = FrozenConfig(
            splitter=splitter,
            splitter_cls=splitter_cls,
            splitter_kwargs=splitter_kwargs,
            index=index,
            index_from=index_from,
            takeable_args=takeable_args,
            template_context=template_context,
            forward_kwargs_as=forward_kwargs_as,
            return_splitter=return_splitter,
            apply_kwargs=apply_kwargs,
            var_kwargs=var_kwargs,
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


def cv_split(
    *args,
    parameterized_kwargs: tp.KwargsLike = None,
    selection: tp.Union[str, tp.Selection] = "max",
    return_grid: tp.Union[bool, str] = False,
    skip_errored: bool = False,
    raise_no_results: bool = True,
    template_context: tp.KwargsLike = None,
    **split_kwargs,
) -> tp.Callable:
    """Combine cross-validation splitting and parameterized execution for decorated functions.

    Decorator that integrates `split` and `vectorbtpro.utils.params.parameterized`
    to facilitate cross-validation. For each split/set range, the decorated function is applied as follows:

    * In the training set, the function is parameterized across the entire grid of parameters
        and its results are stored.
    * For testing sets, the stored grid results are used to evaluate a selection
        that determines the best parameter combination, which is then executed.
    * Optionally, grid results can be returned in addition to the selection,
        controlled by `return_grid`.

    Handles errors by either skipping an iteration (if `skip_errored` is True or a
    `NoResultsException` is raised) or propagating the exception based on `raise_no_results`.

    Args:
        func (Callable): Function to be decorated.
        parameterized_kwargs (KwargsLike): Keyword arguments for `vectorbtpro.utils.params.parameterized`.

            Their templates are substituted with a context that includes split-related information
            (including `split_idx`, `set_idx`, etc., see `vectorbtpro.generic.splitting.base.Splitter.apply`).
        selection (Union[str, Selection]): Selection method for evaluating grid results.

            Can be a template evaluating `grid_results`, or "min" for `np.nanargmin`
            and "max" for `np.nanargmax`.
        return_grid (Union[bool, str]): Determines whether to return grid results along with the selection.

            If True or "first", returns both the grid for the training set (it gets duplicated for
            each set for technical reasons) and selection. If "all", executes the grid on each set
            and returns both. Otherwise, returns only the selection.
        skip_errored (bool): If True, skips the current iteration upon encountering
            an error or `NoResultsException`, omitting it from the final results.
        raise_no_results (bool): Flag indicating whether to raise a
            `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
        template_context (KwargsLike): Additional context for template substitution.
        **split_kwargs: Keyword arguments for `split`.

    Returns:
        Callable: Decorated function that applies cross-validation via splitting and parameterized execution.

    Examples:
        Permutate a series and pick the first value. Make the seed parameterizable.
        Cross-validate based on the highest picked value:

        ```pycon
        >>> from vectorbtpro import *

        >>> @vbt.cv_split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=3, split=0.5),
        ...     takeable_args=["sr"],
        ...     merge_func="concat",
        ... )
        ... def f(sr, seed):
        ...     np.random.seed(seed)
        ...     return np.random.permutation(sr)[0]

        >>> index = pd.date_range("2020-01-01", "2020-02-01")
        >>> np.random.seed(0)
        >>> sr = pd.Series(np.random.permutation(np.arange(len(index))), index=index)
        >>> f(sr, vbt.Param([41, 42, 43]))
        split  set    seed
        0      set_0  41      22
             set_1  41      28
        1      set_0  43       8
             set_1  43      31
        2      set_0  43      19
             set_1  43       0
        dtype: int64
        ```

        Extend the example above to also return the grid results of each set:

        ```pycon
        >>> f(sr, vbt.Param([41, 42, 43]), _return_grid="all")
        (split  set    seed
         0      set_0  41      22
                       42      22
                       43       2
                set_1  41      28
                       42      28
                       43      20
         1      set_0  41       5
                       42       5
                       43       8
                set_1  41      23
                       42      23
                       43      31
         2      set_0  41      18
                       42      18
                       43      19
                set_1  41      27
                       42      27
                       43       0
         dtype: int64,
         split  set    seed
         0      set_0  41      22
                set_1  41      28
         1      set_0  43       8
                set_1  43      31
         2      set_0  43      19
                set_1  43       0
         dtype: int64)
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        if getattr(func, "is_split", False) or getattr(func, "is_parameterized", False):
            raise ValueError("Function is already decorated with split or parameterized")

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            parameterized_kwargs = merge_dicts(
                wrapper.options["parameterized_kwargs"],
                kwargs.pop("_parameterized_kwargs", {}),
            )
            selection = kwargs.pop("_selection", wrapper.options["selection"])
            if isinstance(selection, str) and selection.lower() == "min":
                selection = RepEval("[np.nanargmin(grid_results)]")
            if isinstance(selection, str) and selection.lower() == "max":
                selection = RepEval("[np.nanargmax(grid_results)]")
            return_grid = kwargs.pop("_return_grid", wrapper.options["return_grid"])
            if isinstance(return_grid, bool):
                if return_grid:
                    return_grid = "first"
                else:
                    return_grid = None
            skip_errored = kwargs.pop("_skip_errored", wrapper.options["skip_errored"])
            template_context = merge_dicts(
                wrapper.options["template_context"],
                kwargs.pop("_template_context", {}),
            )
            split_kwargs = merge_dicts(
                wrapper.options["split_kwargs"],
                kwargs.pop("_split_kwargs", {}),
            )
            if "merge_func" in split_kwargs and "merge_func" not in parameterized_kwargs:
                parameterized_kwargs["merge_func"] = split_kwargs["merge_func"]
            if "show_progress" not in parameterized_kwargs:
                parameterized_kwargs["show_progress"] = False

            all_grid_results = []

            @wraps(func)
            def apply_wrapper(
                *_args, __template_context: tp.KwargsLike = None, **_kwargs
            ) -> tp.Any:
                try:
                    if __template_context is None:
                        __template_context = {}
                    __template_context = dict(__template_context)
                    __template_context["all_grid_results"] = all_grid_results
                    _parameterized_kwargs = substitute_templates(
                        parameterized_kwargs,
                        __template_context,
                        eval_id="parameterized_kwargs",
                    )
                    parameterized_func = parameterized(
                        func,
                        template_context=__template_context,
                        **_parameterized_kwargs,
                    )
                    if __template_context["set_idx"] == 0:
                        try:
                            grid_results = parameterized_func(*_args, **_kwargs)
                            all_grid_results.append(grid_results)
                        except Exception as e:
                            if skip_errored or isinstance(e, NoResultsException):
                                all_grid_results.append(NoResult)
                            raise e
                    if all_grid_results[-1] is NoResult:
                        if raise_no_results:
                            raise NoResultsException
                        return NoResult
                    result = parameterized_func(
                        *_args,
                        _selection=selection,
                        _template_context=dict(grid_results=all_grid_results[-1]),
                        **_kwargs,
                    )
                    if return_grid is not None:
                        if return_grid.lower() == "first":
                            return all_grid_results[-1], result
                        if return_grid.lower() == "all":
                            grid_results = parameterized_func(
                                *_args,
                                _template_context=dict(grid_results=all_grid_results[-1]),
                                **_kwargs,
                            )
                            return grid_results, result
                        else:
                            raise ValueError(f"Invalid return_grid: '{return_grid}'")
                    return result
                except Exception as e:
                    if skip_errored or isinstance(e, NoResultsException):
                        return NoResult
                    raise e

            signature = inspect.signature(apply_wrapper)
            lists_var_kwargs = False
            for k, v in signature.parameters.items():
                if v.kind == v.VAR_KEYWORD:
                    lists_var_kwargs = True
                    break
            if not lists_var_kwargs:
                var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
                new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
                apply_wrapper.__signature__ = signature.replace(parameters=new_parameters)
            split_func = split(apply_wrapper, template_context=template_context, **split_kwargs)
            return split_func(
                *args, __template_context=Rep("context", eval_id="apply_kwargs"), **kwargs
            )

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_parameterized = True
        wrapper.is_split = True
        wrapper.options = FrozenConfig(
            parameterized_kwargs=parameterized_kwargs,
            selection=selection,
            return_grid=return_grid,
            skip_errored=skip_errored,
            template_context=template_context,
            split_kwargs=split_kwargs,
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
