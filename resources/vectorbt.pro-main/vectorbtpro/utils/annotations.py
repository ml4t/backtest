# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for annotations."""

from collections import defaultdict

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.base import Base

__all__ = [
    "Annotatable",
    "VarArgs",
    "VarKwargs",
    "Union",
]

__pdoc__ = {}

try:
    from inspect import get_annotations as get_raw_annotations
except ImportError:
    import functools
    import sys
    import types

    def get_raw_annotations(
        obj: tp.Any,
        *,
        globals: tp.Optional[dict] = None,
        locals: tp.Optional[dict] = None,
        eval_str: bool = False,
    ) -> dict:
        """Return raw annotations for a module, class, or callable.

        Backport of Python 3.10's `inspect.get_annotations` function.
        See https://github.com/python/cpython/blob/main/Lib/inspect.py

        Args:
            obj (Any): Module, class, or callable to retrieve annotations from.
            globals (Optional[dict]): Global namespace for evaluation.
            locals (Optional[dict]): Local namespace for evaluation.
            eval_str (bool): Whether to evaluate string annotations.

        Returns:
            dict: Dictionary containing the annotations.
        """
        if isinstance(obj, type):
            # class
            obj_dict = getattr(obj, "__dict__", None)
            if obj_dict and hasattr(obj_dict, "get"):
                ann = obj_dict.get("__annotations__", None)
                if isinstance(ann, types.GetSetDescriptorType):
                    ann = None
            else:
                ann = None

            obj_globals = None
            module_name = getattr(obj, "__module__", None)
            if module_name:
                module = sys.modules.get(module_name, None)
                if module:
                    obj_globals = getattr(module, "__dict__", None)
            obj_locals = dict(vars(obj))
            unwrap = obj
        elif isinstance(obj, types.ModuleType):
            # module
            ann = getattr(obj, "__annotations__", None)
            obj_globals = obj.__dict__
            obj_locals = None
            unwrap = None
        elif callable(obj):
            # this includes types.Function, types.BuiltinFunctionType,
            # types.BuiltinMethodType, functools.partial, functools.singledispatch,
            # "class funclike" from Lib/test/test_inspect... on and on it goes.
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__globals__", None)
            obj_locals = None
            unwrap = obj
        else:
            raise TypeError(f"{obj!r} is not a module, class, or callable.")

        if ann is None:
            return {}

        if not isinstance(ann, dict):
            raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

        if not ann:
            return {}

        if not eval_str:
            return dict(ann)

        if unwrap is not None:
            while True:
                if hasattr(unwrap, "__wrapped__"):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__

        if globals is None:
            globals = obj_globals
        if locals is None:
            locals = obj_locals

        return_value = {
            key: value if not isinstance(value, str) else eval(value, globals, locals)
            for key, value in ann.items()
        }
        return return_value


def get_annotations(*args, **kwargs) -> tp.Annotations:
    """Return annotations for an object with union types resolved.

    Args:
        *args: Positional arguments for `get_raw_annotations`.
        **kwargs: Keyword arguments for `get_raw_annotations`.

    Returns:
        Annotations: Dictionary of annotations with union types resolved.
    """
    annotations = get_raw_annotations(*args, **kwargs)
    new_annotations = {}
    for k, v in annotations.items():
        if isinstance(v, Union):
            v = v.resolve()
        new_annotations[k] = v
    return new_annotations


def flatten_annotations(
    annotations: tp.Annotations,
    only_var_args: bool = False,
    return_var_arg_maps: bool = False,
) -> tp.Union[tp.Annotations, tp.Tuple[tp.Annotations, tp.Dict[str, str], tp.Dict[str, str]]]:
    """Return flattened annotations by unpacking variable arguments.

    Args:
        annotations (Annotations): Mapping of annotation names to annotation values.
        only_var_args (bool): If True, include only variable argument annotations.
        return_var_arg_maps (bool): If True, also return maps for unpacked variable arguments.

    Returns:
        Union[Annotations, Tuple[Annotations, Dict[str, str], Dict[str, str]]]:
            Flattened annotations and, optionally, maps for variable arguments.
    """
    flat_annotations = {}
    var_args_map = {}
    var_kwargs_map = {}
    for k, v in annotations.items():
        if isinstance(v, VarArgs):
            for i, arg_v in enumerate(v.args):
                if isinstance(arg_v, Union):
                    arg_v = arg_v.resolve()
                new_k = f"{k}_{i}"
                if new_k in annotations:
                    raise ValueError(f"Unpacked key {new_k} already exists in annotations")
                flat_annotations[new_k] = arg_v
                var_args_map[new_k] = k
        elif isinstance(v, VarKwargs):
            for arg_k, arg_v in v.kwargs.items():
                if isinstance(arg_v, Union):
                    arg_v = arg_v.resolve()
                if arg_k in annotations:
                    raise ValueError(f"Unpacked key {arg_k} already exists in annotations")
                flat_annotations[arg_k] = arg_v
                var_kwargs_map[arg_k] = k
        elif not only_var_args:
            flat_annotations[k] = v
    if return_var_arg_maps:
        return flat_annotations, var_args_map, var_kwargs_map
    return flat_annotations


class MetaAnnotatable(type):
    """Metaclass for `Annotatable` supporting union operator overloads."""

    def __or__(cls, other: tp.Annotation) -> tp.Annotation:
        return Union(cls, other).resolve()

    def __ror__(cls, other: tp.Annotation) -> tp.Annotation:
        return Union(cls, other).resolve()


class Annotatable(Base, metaclass=MetaAnnotatable):
    """Class for representing annotatable types supporting union operations via the `|` operator."""

    def __or__(self, other: tp.Annotation) -> tp.Annotation:
        return Union(self, other).resolve()

    def __ror__(self, other: tp.Annotation) -> tp.Annotation:
        return Union(self, other).resolve()


def has_annotatables(func: tp.Callable, target_cls: tp.Type[Annotatable] = Annotatable) -> bool:
    """Determine if a function's signature contains any subclass or instance of `Annotatable`.

    Args:
        func (Callable): Function to inspect.
        target_cls (Type[Annotatable]): Class to check against.

    Returns:
        bool: True if any argument or return type is an instance of `Annotatable`, False otherwise.
    """
    annotations = flatten_annotations(get_annotations(func))
    for k, v in annotations.items():
        if isinstance(v, type) and issubclass(v, target_cls):
            return True
        if not isinstance(v, type) and isinstance(v, target_cls):
            return True
    return False


@define
class VarArgs(Annotatable, DefineMixin):
    """Class for representing annotations for variable positional arguments.

    Args:
        *args: Positional arguments.
    """

    args: tp.Tuple[tp.Annotation, ...] = define.field()
    """Tuple containing annotations for each positional argument."""

    def __init__(self, *args) -> None:
        DefineMixin.__init__(self, args=args)


@define
class VarKwargs(Annotatable, DefineMixin):
    """Class for representing annotations for variable keyword arguments.

    Args:
        **kwargs: Keyword arguments.
    """

    kwargs: tp.Dict[str, tp.Annotation] = define.field()
    """Dictionary mapping argument names to their annotations."""

    def __init__(self, **kwargs) -> None:
        DefineMixin.__init__(self, kwargs=kwargs)


@define
class Union(Annotatable, DefineMixin):
    """Class representing a union of one or more annotations.

    Args:
        *annotations (Annotation): Additional annotations to include in the union.
        resolved (bool): Indicates whether the union is marked as resolved.
    """

    annotations: tp.Tuple[tp.Annotation, ...] = define.field()
    """Tuple of annotations that comprise the union."""

    resolved: bool = define.field(default=False)
    """Indicates if the union is resolved."""

    def __init__(self, *annotations, resolved: bool = False) -> None:
        DefineMixin.__init__(self, annotations=annotations, resolved=resolved)

    def resolve(self) -> tp.Annotation:
        """Return the resolved union by merging nested annotations.

        This method flattens any nested unions and combines their annotations.

        If the union contains both `VarArgs` and `VarKwargs`, or if conflicting annotation
        types are encountered during resolution, a `ValueError` is raised.

        Returns:
            Annotation: Resolved annotation.
        """
        if self.resolved:
            return self
        annotations = []
        for annotation in self.annotations:
            if isinstance(annotation, Union):
                annotation = annotation.resolve()
            if isinstance(annotation, Union):
                for annotation in annotation.annotations:
                    if annotation not in annotations:
                        annotations.append(annotation)
            else:
                if annotation not in annotations:
                    annotations.append(annotation)
        var_args_found = False
        var_kwargs_found = False
        for annotation in annotations:
            if isinstance(annotation, VarArgs):
                var_args_found = True
            if isinstance(annotation, VarKwargs):
                var_kwargs_found = True
        if var_args_found and var_kwargs_found:
            raise ValueError("Cannot make a union of VarArgs and VarKwargs")

        if var_args_found:
            if len(annotations) == 1:
                return annotations[0]
            max_n_args = 0
            for annotation in annotations:
                if isinstance(annotation, VarArgs):
                    if len(annotation.args) > max_n_args:
                        max_n_args = len(annotation.args)
            var_args_annotations = [[] for _ in range(max_n_args)]
            for annotation in annotations:
                if isinstance(annotation, VarArgs):
                    for i, v in enumerate(annotation.args):
                        var_args_annotations[i].append(v)
                else:
                    for i in range(len(var_args_annotations)):
                        var_args_annotations[i].append(annotation)
            var_args_unions = []
            for v in var_args_annotations:
                v_union = Union(*v).resolve()
                if isinstance(v_union, VarArgs):
                    raise ValueError("Found VarArgs inside VarArgs")
                if isinstance(v_union, VarKwargs):
                    raise ValueError("Found VarKwargs inside VarArgs")
                var_args_unions.append(v_union)
            return VarArgs(*var_args_unions)

        if var_kwargs_found:
            if len(annotations) == 1:
                return annotations[0]
            all_keys = set()
            for annotation in annotations:
                if isinstance(annotation, VarKwargs):
                    for k in annotation.kwargs.keys():
                        all_keys.add(k)
            var_kwargs_annotations = defaultdict(list)
            for annotation in annotations:
                if isinstance(annotation, VarKwargs):
                    for k, v in annotation.kwargs.items():
                        var_kwargs_annotations[k].append(v)
                else:
                    for k in all_keys:
                        var_kwargs_annotations[k].append(annotation)
            var_kwargs_unions = {}
            for k, v in var_kwargs_annotations.items():
                v_union = Union(*v).resolve()
                if isinstance(v_union, VarArgs):
                    raise ValueError("Found VarArgs inside VarKwargs")
                if isinstance(v_union, VarKwargs):
                    raise ValueError("Found VarKwargs inside VarKwargs")
                var_kwargs_unions[k] = v_union
            return VarKwargs(**var_kwargs_unions)

        return Union(*annotations, resolved=True)
