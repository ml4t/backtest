"""Compatibility fixes for supported Python prereleases."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any


def _apply_python_prerelease_compatibility() -> None:
    """Apply dependency fixes needed by the tested Python prerelease."""
    if sys.version_info < (3, 15):
        return

    import polars as pl
    from polars.series import utils as series_utils

    def empty_docstring_method() -> None:
        """Sentinel matching Polars expression-dispatch methods."""

    if series_utils._is_empty_method(empty_docstring_method):
        return

    original_is_empty_method = series_utils._is_empty_method

    def is_empty_method(function: Callable[..., Any]) -> bool:
        if original_is_empty_method(function):
            return True
        code = function.__code__
        return (
            code.co_code in series_utils._EMPTY_BYTECODE
            and function.__doc__ is not None
            and code.co_consts == (function.__doc__,)
        )

    series_utils._is_empty_method = is_empty_method
    try:
        series_utils.expr_dispatch(pl.Series)
        probe = pl.Series([1])
        for accessor in pl.Series._accessors - {"plot"}:
            series_utils.expr_dispatch(type(getattr(probe, accessor)))
    finally:
        series_utils._is_empty_method = original_is_empty_method


_apply_python_prerelease_compatibility()
