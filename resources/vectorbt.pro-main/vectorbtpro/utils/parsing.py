# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for parsing."""

import ast
import contextlib
import inspect
import io
import re
import sys
from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.utils.annotations import VarArgs, VarKwargs, get_annotations
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base

__all__ = [
    "Regex",
    "PrintsSuppressed",
]


@define
class Regex(DefineMixin):
    """Class for matching a regular expression pattern."""

    pattern: str = define.field()
    """Pattern."""

    flags: int = define.field(default=0)
    """Flags."""

    def matches(self, string: str) -> bool:
        """Return whether the given string matches the regular expression pattern.

        Args:
            string (str): String to test.

        Returns:
            bool: True if the string matches the pattern, otherwise False.
        """
        return re.match(self.pattern, string, self.flags) is not None


def get_func_kwargs(func: tp.Callable) -> tp.Kwargs:
    """Return a dictionary mapping parameter names to their default values for the given function.

    Args:
        func (Callable): Function to inspect.

    Returns:
        Kwargs: Mapping of parameter names to default values.
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_func_arg_names(
    func: tp.Callable,
    arg_kind: tp.Optional[tp.MaybeTuple[int]] = None,
    req_only: bool = False,
    opt_only: bool = False,
    incl_var: bool = False,
) -> tp.List[str]:
    """Return a list of parameter names from a function signature with optional filtering.

    Args:
        func (Callable): Function to inspect.
        arg_kind (Optional[Tuple[int]]): Tuple of parameter kinds to include.

            If None, variable arguments are excluded.
        req_only (bool): If True, include only parameters without defaults.
        opt_only (bool): If True, include only parameters with defaults.
        incl_var (bool): If True, include variable parameters.

    Returns:
        List[str]: Filtered list of parameter names.
    """
    signature = inspect.signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    arg_names = []
    for p in signature.parameters.values():
        if arg_kind is None:
            if not incl_var and (p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD):
                continue
        else:
            if p.kind not in arg_kind:
                continue
        if req_only and p.default is not inspect.Parameter.empty:
            continue
        if opt_only and p.default is inspect.Parameter.empty:
            continue
        arg_names.append(p.name)
    return arg_names


def get_variable_args_name(func: tp.Callable) -> tp.Optional[str]:
    """Return the name of the variable positional arguments of the given function.

    Args:
        func (Callable): Function whose signature is inspected.

    Returns:
        Optional[str]: Variable positional argument name if present, otherwise None.
    """
    signature = inspect.signature(func)
    for p in signature.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            return p.name
    return None


def has_variable_args(func: tp.Callable) -> bool:
    """Return whether the given function accepts variable positional arguments.

    Args:
        func (Callable): Function to check.

    Returns:
        bool: True if the function accepts variable positional arguments, otherwise False.
    """
    return get_variable_args_name(func) is not None


def get_variable_kwargs_name(func: tp.Callable) -> tp.Optional[str]:
    """Return the name of the variable keyword arguments of the given function.

    Args:
        func (Callable): Function whose signature is inspected.

    Returns:
        Optional[str]: Variable keyword argument name if present, otherwise None.
    """
    signature = inspect.signature(func)
    for p in signature.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            return p.name
    return None


def has_variable_kwargs(func: tp.Callable) -> bool:
    """Return whether the given function accepts variable keyword arguments.

    Args:
        func (Callable): Function to check.

    Returns:
        bool: True if the function accepts variable keyword arguments, otherwise False.
    """
    return get_variable_kwargs_name(func) is not None


def get_forward_args(func: tp.Callable, local_dict: tp.Kwargs, **kwargs) -> tp.ArgsKwargs:
    """Return a tuple containing positional and keyword arguments to forward to the given function.

    Args:
        func (Callable): Target function to inspect.
        local_dict (Kwargs): Dictionary of local variables to consider.
        **kwargs: Keyword arguments to match against the function's parameters.

    Returns:
        ArgsKwargs: Tuple where the first element is a tuple of positional arguments
            and the second is a dictionary of keyword arguments.
    """
    new_args = ()
    new_kwargs = {}
    signature = inspect.signature(func)
    for p in signature.parameters.values():
        k = p.name
        if k in kwargs:
            v = kwargs.pop(k)
        elif k in local_dict:
            v = local_dict[k]
        else:
            continue
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            new_args += (v,)
        elif p.kind == p.VAR_POSITIONAL:
            new_args += v
        elif p.kind == p.KEYWORD_ONLY:
            new_kwargs[k] = v
        else:
            for _k, _v in v.items():
                new_kwargs[_k] = _v
    return new_args, new_kwargs


def extend_args(
    func: tp.Callable, args: tp.Args, kwargs: tp.Kwargs, **with_kwargs
) -> tp.ArgsKwargs:
    """Extend given positional and keyword arguments with additional keyword arguments
    based on the function's signature.

    Args:
        func (Callable): Target function whose signature is used.
        args (Args): Original positional arguments.
        kwargs (Kwargs): Original keyword arguments.
        **with_kwargs: Keyword arguments to include.

    Returns:
        ArgsKwargs: Extended positional and keyword arguments.
    """
    kwargs = dict(kwargs)
    new_args = ()
    new_kwargs = dict()
    signature = inspect.signature(func)
    for p in signature.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            new_args += args
            args = ()
            continue
        if p.kind == p.VAR_KEYWORD:
            for k in list(kwargs.keys()):
                new_kwargs[k] = kwargs.pop(k)
            continue

        arg_name = p.name.lower()
        took_from_args = False
        if arg_name not in kwargs and arg_name in with_kwargs:
            arg_value = with_kwargs[arg_name]
        elif len(args) > 0:
            arg_value = args[0]
            args = args[1:]
            took_from_args = True
        elif arg_name in kwargs:
            arg_value = kwargs.pop(arg_name)
        else:
            continue
        if p.kind == p.POSITIONAL_ONLY or len(args) > 0 or took_from_args:
            new_args += (arg_value,)
        else:
            new_kwargs[arg_name] = arg_value

    return new_args + args, {**new_kwargs, **kwargs}


def annotate_args(
    func: tp.Callable,
    args: tp.Args,
    kwargs: tp.Kwargs,
    only_passed: bool = False,
    allow_partial: bool = False,
    attach_annotations: bool = False,
    flatten: bool = False,
) -> tp.AnnArgs:
    """Annotate a function's arguments based on its signature.

    This function binds positional and keyword arguments according to the signature of the given
    function and annotates them with type hints if available. If `allow_partial` is True,
    missing required arguments do not raise an error, but extra arguments not in the signature
    will still trigger an error.

    Args:
        func (Callable): Function whose signature is used for annotation.
        args (Args): Positional arguments to annotate.
        kwargs (Kwargs): Keyword arguments to annotate.
        only_passed (bool): If True, annotate only the arguments that were explicitly passed.
        allow_partial (bool): Whether to allow partial binding of arguments.
        attach_annotations (bool): If True, attach type annotations from the function's signature.
        flatten (bool): If True, flatten the annotation dictionary before returning.

    Returns:
        AnnArgs: Dictionary of annotated arguments.
    """
    kwargs = dict(kwargs)
    signature = inspect.signature(func)
    if not allow_partial:
        signature.bind(*args, **kwargs)
    ann_args = dict()
    if attach_annotations:
        annotations = get_annotations(func)
    else:
        annotations = dict()

    last_pos = None
    var_positional = False
    var_keyword = False
    for p in signature.parameters.values():
        if p.kind == p.POSITIONAL_ONLY:
            if len(args) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[0])
                args = args[1:]
                last_pos = p.name
            elif not only_passed:
                if allow_partial:
                    ann_args[p.name] = dict(kind=p.kind)
                else:
                    raise TypeError(f"missing a required argument: '{p.name}'")
        elif p.kind == p.VAR_POSITIONAL:
            var_positional = True
            if len(args) > 0 or not only_passed:
                ann_args[p.name] = dict(kind=p.kind, value=args)
                args = ()
                last_pos = p.name
        elif p.kind == p.POSITIONAL_OR_KEYWORD:
            if len(args) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[0])
                args = args[1:]
                last_pos = p.name
            elif p.name in kwargs:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
            elif not only_passed:
                if p.default is not p.empty:
                    ann_args[p.name] = dict(kind=p.kind, value=p.default)
                else:
                    if allow_partial:
                        ann_args[p.name] = dict(kind=p.kind)
                    else:
                        raise TypeError(f"missing a required argument: '{p.name}'")
        elif p.kind == p.KEYWORD_ONLY:
            if p.name in kwargs:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
            elif not only_passed:
                ann_args[p.name] = dict(kind=p.kind, value=p.default)
        else:
            var_keyword = True
            if not only_passed or len(kwargs) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs)
        if p.name in ann_args and p.name in annotations:
            ann_args[p.name]["annotation"] = annotations[p.name]

    if not var_positional:
        if len(args) == 1:
            raise TypeError(
                f"{func.__name__}() got an unexpected positional argument after '{last_pos}'"
            )
        if len(args) > 1:
            raise TypeError(
                f"{func.__name__}() got {len(args)} unexpected positional arguments after '{last_pos}'"
            )
    if not var_keyword:
        if len(kwargs) == 1:
            raise TypeError(
                f"{func.__name__}() got an unexpected keyword argument '{list(kwargs.keys())[0]}'"
            )
        if len(kwargs) > 1:
            raise TypeError(
                f"{func.__name__}() got unexpected keyword arguments {list(kwargs.keys())}"
            )
    if flatten:
        return flatten_ann_args(ann_args)
    return ann_args


def ann_args_to_args(ann_args: tp.AnnArgs) -> tp.ArgsKwargs:
    """Convert annotated arguments to positional and keyword arguments.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        ArgsKwargs: Tuple containing positional arguments and keyword arguments.
    """
    args = ()
    kwargs = {}
    p = inspect.Parameter
    for k, v in ann_args.items():
        if v["kind"] in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            args += (v["value"],)
        elif v["kind"] == p.VAR_POSITIONAL:
            args += v["value"]
        elif v["kind"] == p.KEYWORD_ONLY:
            kwargs[k] = v["value"]
        else:
            for _k, _v in v["value"].items():
                kwargs[_k] = _v
    return args, kwargs


def flat_ann_args_to_args(ann_args: tp.AnnArgs) -> tp.ArgsKwargs:
    """Convert annotated arguments to positional and keyword arguments after flattening them.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        ArgsKwargs: Tuple containing positional arguments and keyword arguments.
    """
    return ann_args_to_args(flatten_ann_args(ann_args))


def flatten_ann_args(ann_args: tp.AnnArgs) -> tp.FlatAnnArgs:
    """Flatten annotated arguments into a dictionary of flattened argument entries.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.

    Returns:
        FlatAnnArgs: Flattened dictionary representation of the annotated arguments.
    """
    flat_ann_args = {}
    for arg_name, ann_arg in ann_args.items():
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            for i, v in enumerate(ann_arg["value"]):
                dct = dict(var_name=arg_name, kind=ann_arg["kind"], value=v)
                if "annotation" in ann_arg:
                    if isinstance(ann_arg["annotation"], VarArgs):
                        dct["annotation"] = ann_arg["annotation"].args[i]
                        dct["var_annotation"] = ann_arg["annotation"]
                    else:
                        if isinstance(ann_arg["annotation"], VarKwargs):
                            raise TypeError("VarKwargs used for variable positional arguments")
                        dct["annotation"] = ann_arg["annotation"]
                new_arg_name = f"{arg_name}_{i}"
                if new_arg_name in flat_ann_args:
                    raise ValueError(
                        f"Unpacked key {new_arg_name} already exists in annotated arguments"
                    )
                flat_ann_args[new_arg_name] = dct
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            for var_arg_name, var_value in ann_arg["value"].items():
                dct = dict(var_name=arg_name, kind=ann_arg["kind"], value=var_value)
                if "annotation" in ann_arg:
                    if isinstance(ann_arg["annotation"], VarKwargs):
                        dct["annotation"] = ann_arg["annotation"].kwargs[var_arg_name]
                        dct["var_annotation"] = ann_arg["annotation"]
                    else:
                        if isinstance(ann_arg["annotation"], VarArgs):
                            raise TypeError("VarArgs used for variable keyword arguments")
                        dct["annotation"] = ann_arg["annotation"]
                if var_arg_name in flat_ann_args:
                    raise ValueError(
                        f"Unpacked key {var_arg_name} already exists in annotated arguments"
                    )
                flat_ann_args[var_arg_name] = dct
        else:
            dct = dict(kind=ann_arg["kind"])
            if "value" in ann_arg:
                dct["value"] = ann_arg["value"]
            if "annotation" in ann_arg:
                dct["annotation"] = ann_arg["annotation"]
            flat_ann_args[arg_name] = dct
    return flat_ann_args


def unflatten_ann_args(
    flat_ann_args: tp.FlatAnnArgs, partial_ann_args: tp.Optional[tp.AnnArgs] = None
) -> tp.AnnArgs:
    """Reconstruct original annotated arguments from flattened entries.

    Args:
        flat_ann_args (FlatAnnArgs): Flattened annotated arguments.
        partial_ann_args (Optional[AnnArgs]): Partial annotated arguments to integrate, if provided.

    Returns:
        AnnArgs: Reconstructed annotated arguments.
    """
    ann_args = dict()
    for arg_name, ann_arg in flat_ann_args.items():
        ann_arg = dict(ann_arg)
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            var_arg_name = ann_arg.pop("var_name")
            if var_arg_name not in ann_args:
                dct = dict(value=(), kind=ann_arg["kind"])
                if "var_annotation" in ann_arg:
                    dct["annotation"] = ann_arg["var_annotation"]
                elif "annotation" in ann_arg:
                    dct["annotation"] = ann_arg["annotation"]
                ann_args[var_arg_name] = dct
            ann_args[var_arg_name]["value"] = ann_args[var_arg_name]["value"] + (ann_arg["value"],)
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            var_arg_name = ann_arg.pop("var_name")
            if var_arg_name not in ann_args:
                dct = dict(value={}, kind=ann_arg["kind"])
                if "var_annotation" in ann_arg:
                    dct["annotation"] = ann_arg["var_annotation"]
                elif "annotation" in ann_arg:
                    dct["annotation"] = ann_arg["annotation"]
                ann_args[var_arg_name] = dct
            ann_args[var_arg_name]["value"][arg_name] = ann_arg["value"]
        else:
            ann_args[arg_name] = ann_arg
    if partial_ann_args is not None:
        if ann_args.keys() > partial_ann_args.keys():
            raise ValueError("Unflattened annotated arguments contain unexpected keys")
        for k, v in partial_ann_args.items():
            if k not in ann_args:
                ann_args[k] = v
        new_ann_args = dict()
        for k in partial_ann_args:
            new_ann_args[k] = ann_args[k]
        return new_ann_args
    return ann_args


def match_flat_ann_arg(
    flat_ann_args: tp.FlatAnnArgs,
    query: tp.AnnArgQuery,
    return_name: bool = False,
    return_index: bool = False,
) -> tp.Any:
    """Match an argument from flattened annotated arguments.

    Args:
        flat_ann_args (FlatAnnArgs): Flattened annotated arguments.
        query (AnnArgQuery): Query to identify the argument by position, name, or regular expression.
        return_name (bool): If True, return the argument's name.
        return_index (bool): If True, return the argument's positional index.

    Returns:
        Any: Matched argument value, or its name/index if specified.

    !!! note
        Only the first matching argument is returned.
    """
    if return_name and return_index:
        raise ValueError("Either return_name or return_index can be provided, not both")
    for i, (arg_name, ann_arg) in enumerate(flat_ann_args.items()):
        if (
            (isinstance(query, int) and query == i)
            or (isinstance(query, str) and query == arg_name)
            or (isinstance(query, Regex) and query.matches(arg_name))
        ):
            if return_name:
                return arg_name
            if return_index:
                return i
            return ann_arg["value"]
    raise KeyError(f"Query '{query}' could not be matched with any argument")


def match_ann_arg(
    ann_args: tp.AnnArgs,
    query: tp.AnnArgQuery,
    return_name: bool = False,
    return_index: bool = False,
) -> tp.Any:
    """Match an argument from annotated arguments by flattening them.

    Args:
        ann_args (AnnArgs): Annotated arguments.

            See `vectorbtpro.utils.parsing.annotate_args`.
        query (AnnArgQuery): Query to identify the argument by position, name, or regular expression.
        return_name (bool): If True, return the argument's name.
        return_index (bool): If True, return the argument's positional index.

    Returns:
        Any: Matched argument value, or its name/index if specified.

    !!! note
        Matching logic is equivalent to that of `match_flat_ann_arg`.
    """
    return match_flat_ann_arg(
        flatten_ann_args(ann_args),
        query,
        return_name=return_name,
        return_index=return_index,
    )


def match_and_set_flat_ann_arg(
    flat_ann_args: tp.FlatAnnArgs,
    query: tp.AnnArgQuery,
    new_value: tp.Any,
) -> None:
    """Match an argument in flattened annotated arguments and update its value.

    Args:
        flat_ann_args (FlatAnnArgs): Flattened annotated arguments.
        query (AnnArgQuery): Query to identify the argument by position, name, or regular expression.
        new_value (Any): New value to assign to the matched argument(s).

    Returns:
        None: Function modifies `flat_ann_args` in place.

    !!! note
        All matching arguments are updated.
    """
    matched = False
    for i, (arg_name, ann_arg) in enumerate(flat_ann_args.items()):
        if (
            (isinstance(query, int) and query == i)
            or (isinstance(query, str) and query == arg_name)
            or (isinstance(query, Regex) and query.matches(arg_name))
        ):
            ann_arg["value"] = new_value
            matched = True
    if not matched:
        raise KeyError(f"Query '{query}' could not be matched with any argument")


def ignore_flat_ann_args(
    flat_ann_args: tp.FlatAnnArgs, ignore_args: tp.Iterable[tp.AnnArgQuery]
) -> tp.FlatAnnArgs:
    """Return flattened annotated arguments excluding those that match specified queries.

    Args:
        flat_ann_args (FlatAnnArgs): Flattened annotated arguments.
        ignore_args (Iterable[AnnArgQuery]): Queries indicating which arguments to ignore.

    Returns:
        FlatAnnArgs: Dictionary of flattened annotated arguments after ignoring specified entries.
    """
    new_flat_ann_args = {}
    for i, (arg_name, arg) in enumerate(flat_ann_args.items()):
        arg_matched = False
        for ignore_arg in ignore_args:
            if (
                (isinstance(ignore_arg, int) and ignore_arg == i)
                or (isinstance(ignore_arg, str) and ignore_arg == arg_name)
                or (isinstance(ignore_arg, Regex) and ignore_arg.matches(arg_name))
            ):
                arg_matched = True
                break
        if not arg_matched:
            new_flat_ann_args[arg_name] = arg
    return new_flat_ann_args


def get_expr_var_names(expression: str) -> tp.List[str]:
    """Extract variable names from a Python expression.

    Args:
        expression (str): Python expression as a string.

    Returns:
        List[str]: List of variable names found in the expression.
    """
    return [node.id for node in ast.walk(ast.parse(expression)) if type(node) is ast.Name]


def get_context_vars(
    var_names: tp.Iterable[str],
    frames_back: int = 0,
    local_dict: tp.Optional[tp.Mapping] = None,
    global_dict: tp.Optional[tp.Mapping] = None,
) -> tp.List[tp.Any]:
    """Retrieve variables from the calling context.

    Args:
        var_names (Iterable[str]): Iterable of variable names to retrieve.
        frames_back (int): Number of frames to go back from the current frame.
        local_dict (Optional[Mapping]): Dictionary of local variables.

            If not provided, uses the calling frame's local variables.
        global_dict (Optional[Mapping]): Dictionary of global variables.

            If not provided, uses the calling frame's global variables.

    Returns:
        List[Any]: List of variable values corresponding to `var_names`.
    """
    call_frame = sys._getframe(frames_back + 1)
    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals
        clear_local_dict = clear_local_dict and frame_globals is not local_dict
        args = []
        for var_name in var_names:
            try:
                a = local_dict[var_name]
            except KeyError:
                a = global_dict[var_name]
            args.append(a)
    finally:
        # See https://github.com/pydata/numexpr/issues/310
        if clear_local_dict:
            local_dict.clear()
    return args


def suppress_stdout(func: tp.Callable) -> tp.Callable:
    """Suppress stdout output when executing the decorated function.

    Args:
        func (Callable): Function whose printed output will be suppressed.

    Returns:
        Callable: Decorated function with stdout redirection.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> tp.Any:
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    return wrapper


PrintsSuppressedT = tp.TypeVar("PrintsSuppressedT", bound="PrintsSuppressed")


class PrintsSuppressed(contextlib.redirect_stdout, Base):
    """Context manager to temporarily suppress printed output.

    Args:
        *args: Positional arguments for `contextlib.redirect_stdout`.
        **kwargs: Keyword arguments for `contextlib.redirect_stdout`.
    """

    def __new__(cls, *args, **kwargs) -> PrintsSuppressedT:
        return cls(io.StringIO(), *args, **kwargs)
