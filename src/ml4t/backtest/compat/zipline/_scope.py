"""Callback-local API ownership, including exception and nested-run cleanup."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import MigrationContext

_ACTIVE: ContextVar["MigrationContext | None"] = ContextVar("ml4t_zipline_callback", default=None)


def current() -> "MigrationContext":
    context = _ACTIVE.get()
    if context is None:
        raise RuntimeError("Zipline API requires an active migration callback")
    return context


@contextmanager
def callback_scope(context: "MigrationContext"):
    token = _ACTIVE.set(context)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
