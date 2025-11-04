# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with tags."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.parsing import get_expr_var_names
from vectorbtpro.utils.template import RepEval

__all__ = []


def match_tags(tags: tp.MaybeIterable[str], in_tags: tp.MaybeIterable[str]) -> bool:
    """Return True if any tag from `tags` matches identifiers in `in_tags` using an OR rule.

    For valid identifier tags, match them directly within `in_tags`. For tags that are not valid
    identifiers, evaluate the tag as a boolean expression where each variable indicates its
    presence in `in_tags`. A `ValueError` is raised if any tag in `in_tags` is not a valid identifier,
    and a `TypeError` is raised if a tag expression does not produce a boolean.

    Args:
        tags (MaybeIterable[str]): Tag or collection of tags to match.

            If a tag is not a valid identifier, it is evaluated as a boolean expression.
        in_tags (MaybeIterable[str]): Identifier or collection of identifiers to search for matches.

            All elements must be valid Python identifiers.

    Returns:
        bool: True if any tag from `tags` matches identifiers in `in_tags`, False otherwise.

    Examples:
        ```pycon
        >>> from vectorbtpro.utils.tagging import match_tags

        >>> match_tags('hello', 'hello')
        True
        >>> match_tags('hello', 'world')
        False
        >>> match_tags(['hello', 'world'], 'world')
        True
        >>> match_tags('hello', ['hello', 'world'])
        True
        >>> match_tags('hello and world', ['hello', 'world'])
        True
        >>> match_tags('hello and not world', ['hello', 'world'])
        False
        ```
    """
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(in_tags, str):
        in_tags = [in_tags]
    for in_t in in_tags:
        if not in_t.isidentifier():
            raise ValueError(f"Tag '{in_t}' must be an identifier")

    for t in tags:
        if not t.isidentifier():
            var_names = get_expr_var_names(t)
            eval_context = {var_name: var_name in in_tags for var_name in var_names}
            eval_result = RepEval(t).substitute(eval_context)
            if not isinstance(eval_result, bool):
                raise TypeError(f"Tag expression '{t}' must produce a boolean")
            if eval_result:
                return True
        else:
            if t in in_tags:
                return True
    return False
