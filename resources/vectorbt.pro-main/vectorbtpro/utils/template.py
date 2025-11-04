# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with templates.

!!! info
    For default settings, see `vectorbtpro._settings.template`.
"""

from string import Template

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.eval_ import Evaluable, evaluate, get_free_vars
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.search_ import contains_in_obj, find_and_replace_in_obj

__all__ = [
    "CustomTemplate",
    "Sub",
    "SafeSub",
    "Rep",
    "RepEval",
    "RepFunc",
    "substitute_templates",
]


@define
class CustomTemplate(Evaluable, DefineMixin):
    """Class for substituting templates."""

    template: tp.Any = define.field()
    """Template to be processed."""

    context: tp.KwargsLike = define.field(default=None)
    """Context mapping."""

    strict: tp.Optional[bool] = define.field(default=None)
    """Whether to raise an error if processing template fails.

    If not None, overrides `strict` passed by `substitute_templates`.
    """

    context_merge_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments for `vectorbtpro.utils.config.merge_dicts`."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def resolve_context(
        self,
        context: tp.KwargsLike = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Kwargs:
        """Return a merged context dictionary by combining the default context from
        `vectorbtpro._settings.template`, the instance context, and the provided `context`.

        Also, append `eval_id` and import entries from `vectorbtpro.imported_star` if available.

        Args:
            context (KwargsLike): Additional context to merge.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            Kwargs: Merged context mapping.

        !!! info
            For default settings, see `vectorbtpro._settings.template`.
        """
        from vectorbtpro._settings import settings

        template_cfg = settings["template"]

        context_merge_kwargs = self.context_merge_kwargs
        if context_merge_kwargs is None:
            context_merge_kwargs = {}
        new_context = merge_dicts(
            template_cfg["context"],
            self.context,
            context,
            **context_merge_kwargs,
        )
        if "context" not in new_context:
            new_context["context"] = dict(new_context)
        if "eval_id" not in new_context:
            new_context["eval_id"] = eval_id
        try:
            for k, v in vbt.imported_star.items():
                if k not in new_context:
                    new_context[k] = v
        except AttributeError:
            pass
        return new_context

    def resolve_strict(self, strict: tp.Optional[bool] = None) -> bool:
        """Return the resolved strict flag, combining the provided `strict` argument,
        the instance setting, and the global default from `vectorbtpro._settings.template`.

        Args:
            strict (Optional[bool]): Flag indicating whether to raise an error if evaluation fails.

        Returns:
            bool: True if strict mode is enabled, False otherwise.

        !!! info
            For default settings, see `vectorbtpro._settings.template`.
        """
        if strict is None:
            strict = self.strict
        if strict is None:
            from vectorbtpro._settings import settings

            template_cfg = settings["template"]

            strict = template_cfg["strict"]
        return strict

    def get_context_vars(self) -> tp.List[str]:
        """Return a list of variable names extracted from the template.

        Returns:
            List[str]: Names of the placeholders in the template.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """Perform template substitution by merging the instance context with the provided
        `context` and processing `CustomTemplate.template`.

        Args:
            context (KwargsLike): Additional context mapping for substitution.
            strict (Optional[bool]): Flag indicating whether to raise an error if evaluation fails.
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            Any: Result of the template substitution.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError


class Sub(CustomTemplate):
    """Class for substituting placeholders in a template string using a provided context mapping.

    Uses `string.Template.substitute` to replace placeholders with corresponding values.

    Returns a new string upon successful substitution, or the original instance if substitution is skipped or fails.
    """

    def get_context_vars(self) -> tp.List[str]:
        tmpl = Template(self.template)
        variables = []
        for match in tmpl.pattern.finditer(tmpl.template):
            named = match.group("named")
            braced = match.group("braced")
            if named is not None and named not in variables:
                variables.append(named)
            elif braced is not None and braced not in variables:
                variables.append(braced)
        return variables

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return Template(self.template).substitute(context)
        except KeyError as e:
            if strict:
                raise e
        return self


class SafeSub(CustomTemplate):
    """Class for performing safe placeholder substitution in a template string using a provided context mapping.

    Uses `string.Template.safe_substitute` to replace placeholders, leaving unmatched placeholders unchanged.
    """

    def get_context_vars(self) -> tp.List[str]:
        tmpl = Template(self.template)
        variables = []
        for match in tmpl.pattern.finditer(tmpl.template):
            named = match.group("named")
            braced = match.group("braced")
            if named is not None and named not in variables:
                variables.append(named)
            elif braced is not None and braced not in variables:
                variables.append(braced)
        return variables

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return Template(self.template).safe_substitute(context)
        except KeyError as e:
            if strict:
                raise e
        return self


class Rep(CustomTemplate):
    """Class for replacing a template key with its corresponding value from a context mapping."""

    def get_context_vars(self) -> tp.List[str]:
        return [self.template]

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return context[self.template]
        except KeyError as e:
            if strict:
                raise e
        return self


class RepEval(CustomTemplate):
    """Class for evaluating a template expression using `vectorbtpro.utils.eval_.evaluate`
    with a provided context mapping as local variables."""

    def get_context_vars(self) -> tp.List[str]:
        return get_free_vars(self.template)

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return evaluate(self.template, context)
        except NameError as e:
            if strict:
                raise e
        return self


class RepFunc(CustomTemplate):
    """Class for executing a function provided as a template using parameters extracted from a context mapping."""

    def get_context_vars(self) -> tp.List[str]:
        return get_func_arg_names(self.template)

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        func_arg_names = get_func_arg_names(self.template)
        func_kwargs = dict()
        for k, v in context.items():
            if k in func_arg_names:
                func_kwargs[k] = v

        try:
            return self.template(**func_kwargs)
        except TypeError as e:
            if strict:
                raise e
        return self


def has_templates(obj: tp.Any, **kwargs) -> tp.Any:
    """Check whether the object contains any template instances.

    This function recursively searches for instances of `CustomTemplate` or `Template`
    within the provided object using `vectorbtpro.utils.search_.contains_in_obj`.

    The default search behavior can be customized by passing additional keyword arguments
    that merge with `search_kwargs` from `vectorbtpro._settings.template`.

    Args:
        obj (Any): Object to search for template instances.
        **kwargs: Additional parameters to override default search settings.

    Returns:
        Any: Object containing template instances, or None if none are found.

    !!! info
        For default settings, see `vectorbtpro._settings.template`.
    """
    from vectorbtpro._settings import settings

    template_cfg = settings["template"]

    search_kwargs = merge_dicts(template_cfg["search_kwargs"], kwargs)

    def _match_func(k, v):
        return isinstance(v, (CustomTemplate, Template))

    return contains_in_obj(obj, _match_func, **search_kwargs)


def substitute_templates(
    obj: tp.Any,
    context: tp.KwargsLike = None,
    strict: tp.Optional[bool] = None,
    eval_id: tp.Optional[tp.Hashable] = None,
    **kwargs,
) -> tp.Any:
    """Substitute template instances within the object using a context.

    This function recursively traverses the input object and replaces any instance of
    `CustomTemplate` or `Template` with its substituted value.

    For `CustomTemplate`, the substitution includes `context`, `strict`, and `eval_id`,
    while for `Template` it uses `context` only.

    If `strict` is True, the function raises an error on substitution failure; otherwise,
    it returns the original template instance.

    The default search behavior can be customized by passing additional keyword arguments
    that merge with `search_kwargs` from `vectorbtpro._settings.template`.

    Args:
        obj (Any): Object to traverse for template substitution.
        context (KwargsLike): Context for replacing template placeholders.
        strict (Optional[bool]): Flag indicating whether to raise an error if evaluation fails.
        eval_id (Optional[Hashable]): Evaluation identifier.
        **kwargs: Additional parameters to override default search settings.

    Returns:
        Any: Object with template instances replaced by their substituted values.

    !!! info
        For default settings, see `vectorbtpro._settings.template`.

    Examples:
        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.substitute_templates(vbt.Sub('$key', {'key': 100}))
        100
        >>> vbt.substitute_templates(vbt.Sub('$key', {'key': 100}), {'key': 200})
        200
        >>> vbt.substitute_templates(vbt.Sub('$key$key'), {'key': 100})
        100100
        >>> vbt.substitute_templates(vbt.Rep('key'), {'key': 100})
        100
        >>> vbt.substitute_templates([vbt.Rep('key'), vbt.Sub('$key$key')], {'key': 100}, incl_types=list)
        [100, '100100']
        >>> vbt.substitute_templates(vbt.RepFunc(lambda key: key == 100), {'key': 100})
        True
        >>> vbt.substitute_templates(vbt.RepEval('key == 100'), {'key': 100})
        True
        >>> vbt.substitute_templates(vbt.RepEval('key == 100', strict=True))
        NameError: name 'key' is not defined
        >>> vbt.substitute_templates(vbt.RepEval('key == 100', strict=False))
        <vectorbtpro.utils.template.RepEval at 0x7fe3ad2ab668>
        ```
    """
    from vectorbtpro._settings import settings

    template_cfg = settings["template"]

    search_kwargs = merge_dicts(template_cfg["search_kwargs"], kwargs)

    def _match_func(k, v):
        return isinstance(v, (CustomTemplate, Template))

    def _replace_func(k, v):
        if isinstance(v, CustomTemplate):
            return v.substitute(context=context, strict=strict, eval_id=eval_id)
        if isinstance(v, Template):
            return v.substitute(context=context)

    return find_and_replace_in_obj(obj, _match_func, _replace_func, **search_kwargs)
