# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for managing and formatting warnings."""

import io
import warnings
from contextlib import contextmanager, redirect_stdout
from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = [
    "warn",
]


@contextmanager
def use_formatwarning(formatwarning: tp.Any) -> tp.Generator[None, None, None]:
    """Temporarily set a custom warning formatter during the context.

    Args:
        formatwarning (Any): Custom function to format warning messages.

    Yields:
        None: Context manager that temporarily sets the custom formatter.
    """
    old_formatter = warnings.formatwarning
    warnings.formatwarning = formatwarning
    try:
        yield
    finally:
        warnings.formatwarning = old_formatter


def custom_formatwarning(
    message: tp.Any,
    category: tp.Type[Warning],
    filename: str,
    lineno: int,
    line: tp.Optional[str] = None,
) -> str:
    """Format warning messages using a custom structure.

    Args:
        message (Any): Warning message.
        category (Type[Warning]): Warning category.
        filename (str): Filename where the warning occurred.
        lineno (int): Line number in the file.
        line (Optional[str]): Source code line triggering the warning.

    Returns:
        str: Formatted warning message.
    """
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


class VBTWarning(Warning):
    """Base warning class for warnings raised by vectorbtpro."""


def warn(message: tp.Any, category: type = VBTWarning, stacklevel: int = 2) -> None:
    """Emit a warning using a custom formatter.

    Args:
        message (Any): Warning message to emit.
        category (type): Warning category to use.
        stacklevel (int): Stack level for the warning; defaults to 2.

    Returns:
        None
    """
    with use_formatwarning(custom_formatwarning):
        warnings.warn(message, category, stacklevel=stacklevel)


def warn_stdout(func: tp.Callable) -> tp.Callable:
    """Suppress standard output from the decorated function and emit it as a warning.

    Args:
        func (Callable): Function whose standard output is to be captured.

    Returns:
        Callable: Decorated function that emits captured output as a warning.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> tp.Any:
        with redirect_stdout(io.StringIO()) as f:
            out = func(*args, **kwargs)
        s = f.getvalue()
        if len(s) > 0:
            warn(s)
        return out

    return wrapper


class WarningsFiltered(warnings.catch_warnings, Base):
    """Context manager to temporarily ignore warnings globally.

    Args:
        entries (Optional[MaybeSequence[Union[str, Kwargs]]]): Simple entries to add to the warnings filters.
        **kwargs: Keyword arguments for `warnings.catch_warnings`.
    """

    def __init__(
        self, entries: tp.Optional[tp.MaybeSequence[tp.Union[str, tp.Kwargs]]] = "ignore", **kwargs
    ) -> None:
        warnings.catch_warnings.__init__(self, **kwargs)
        self._entries = entries

    @property
    def entries(self) -> tp.Optional[tp.MaybeSequence[tp.Union[str, tp.Kwargs]]]:
        """Simple entries to add to the warnings filters.

        Returns:
            Optional[MaybeSequence[Union[str, Kwargs]]]: Entries for the warnings filters.
        """
        return self._entries

    def __enter__(self) -> tp.Self:
        warnings.catch_warnings.__enter__(self)
        if self.entries is not None:
            if isinstance(self.entries, (str, dict)):
                entry = self.entries
                if isinstance(entry, str):
                    entry = dict(action=entry)
                warnings.simplefilter(**entry)
            else:
                for entry in self.entries:
                    if isinstance(entry, str):
                        entry = dict(action=entry)
                    warnings.simplefilter(**entry)
        return self
