# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for merging."""

from functools import partial

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.annotations import Annotatable, Union, get_annotations
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.eval_ import Evaluable
from vectorbtpro.utils.template import substitute_templates

__all__ = [
    "MergeFunc",
]

__pdoc__ = {}

MergeFuncT = tp.TypeVar("MergeFuncT", bound="MergeFunc")


@define
class MergeFunc(Evaluable, Annotatable, DefineMixin):
    """Class for representing a merging function with pre-bound keyword arguments.

    This class supports template substitution in both the merging function and its keyword arguments,
    allowing dynamic resolution and execution when the instance is called.

    Args:
        merge_func (MergeFuncLike): Function to merge the results.

            Resolved using `MergeFunc.resolve_merge_func`.
        merge_kwargs (KwargsLike): Keyword arguments for `MergeFunc.merge_func`.
        context (KwargsLike): Context for template substitution in `MergeFunc.merge_func`
            and `MergeFunc.merge_kwargs`.
        eval_id_prefix (str): Prefix for the substitution identifier.
        eval_id (Optional[MaybeSequence[Hashable]]): Identifier(s) used for evaluation.
        **kwargs: Keyword arguments acting as `merge_kwargs`.
    """

    merge_func: tp.MergeFuncLike = define.field()
    """Merging function used to perform merging operations."""

    merge_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for `MergeFunc.merge_func`."""

    context: tp.KwargsLike = define.field(default=None)
    """Context for performing template substitutions in `MergeFunc.merge_func` and `MergeFunc.merge_kwargs`."""

    eval_id_prefix: str = define.field(default="")
    """Prefix for the substitution identifier used during template substitution."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more evaluation identifiers."""

    def __init__(self, *args, **kwargs) -> None:
        attr_names = [a.name for a in self.fields]
        if attr_names.index("merge_kwargs") < len(args):
            new_args = list(args)
            merge_kwargs = new_args[attr_names.index("merge_kwargs")]
            if merge_kwargs is None:
                merge_kwargs = {}
            else:
                merge_kwargs = dict(merge_kwargs)
            merge_kwargs.update(
                {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names}
            )
            new_args[attr_names.index("merge_kwargs")] = merge_kwargs
            args = tuple(new_args)
        else:
            merge_kwargs = kwargs.pop("merge_kwargs", None)
            if merge_kwargs is None:
                merge_kwargs = {}
            else:
                merge_kwargs = dict(merge_kwargs)
            merge_kwargs.update(
                {k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names}
            )
            kwargs["merge_kwargs"] = merge_kwargs

        DefineMixin.__init__(self, *args, **kwargs)

    def resolve_merge_func(self) -> tp.Optional[tp.Callable]:
        """Return the merging function with pre-bound keyword arguments after applying template substitutions.

        Uses `vectorbtpro.base.merging.resolve_merge_func` for resolving the merging function.
        Uses `MergeFunc.context` for template substitution in `MergeFunc.merge_func` and `MergeFunc.merge_kwargs`.

        Returns:
            Optional[Callable]: Merging function with pre-bound keyword arguments,
                or None if the merging function cannot be resolved.
        """
        from vectorbtpro.base.merging import resolve_merge_func

        merge_func = resolve_merge_func(self.merge_func)
        if merge_func is None:
            return None
        merge_kwargs = self.merge_kwargs
        if merge_kwargs is None:
            merge_kwargs = {}
        merge_func = substitute_templates(
            merge_func, self.context, eval_id=self.eval_id_prefix + "merge_func"
        )
        merge_kwargs = substitute_templates(
            merge_kwargs, self.context, eval_id=self.eval_id_prefix + "merge_kwargs"
        )
        return partial(merge_func, **merge_kwargs)

    def __call__(self, *objs, **kwargs) -> tp.Any:
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        merge_func = self.resolve_merge_func()
        if merge_func is None:
            return objs
        return merge_func(objs, **kwargs)


def parse_merge_func(
    func: tp.Callable, eval_id: tp.Optional[tp.Hashable] = None
) -> tp.Optional[MergeFunc]:
    """Parse the merging function from the provided function's annotations.

    Args:
        func (Callable): Function from which to parse the merging function annotation.
        eval_id (Optional[Hashable]): Evaluation identifier.

    Returns:
        Optional[MergeFunc]: Merging function(s) extracted from the annotations,
            or None if not found.
    """
    annotations = get_annotations(func)
    merge_func = None
    for k, v in annotations.items():
        if k == "return":
            if not isinstance(v, Union):
                v = Union(v)
            for annotation in v.annotations:
                if isinstance(annotation, str):
                    from vectorbtpro.base.merging import merge_func_config

                    if annotation in merge_func_config:
                        annotation = MergeFunc(annotation)
                if checks.is_complex_sequence(annotation):
                    for o in annotation:
                        if (
                            o is None
                            or isinstance(o, str)
                            or (isinstance(o, MergeFunc) and o.meets_eval_id(eval_id))
                        ):
                            if merge_func is None:
                                merge_func = []
                            elif not isinstance(merge_func, list):
                                raise ValueError(
                                    f"Two merging functions found in annotations: {merge_func} and {o}"
                                )
                            merge_func.append(o)
                elif isinstance(annotation, MergeFunc) and annotation.meets_eval_id(eval_id):
                    if merge_func is not None:
                        raise ValueError(
                            f"Two merging functions found in annotations: {merge_func} and {annotation}"
                        )
                    merge_func = annotation
    return merge_func
