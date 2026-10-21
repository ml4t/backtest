"""Callback-scoped migration API; all account mutations use the native broker."""

from typing import Any

import pandas as pd

from ._scope import current
from .assets import Asset


def symbol(symbol_str: str) -> Asset:
    return current().symbol(symbol_str)


def symbols(*symbol_strs: str) -> tuple[Asset, ...]:
    return current().symbols(*symbol_strs)


def sid(sid: int) -> Asset:
    return current().sid(sid)


def order(asset: Asset, amount: float) -> str | None:
    return current().order(asset, amount)


def record(**values: Any) -> None:
    current().record(**values)


def get_datetime(tz: str | None = None) -> pd.Timestamp:
    return current().get_datetime(tz)


def set_commission(model: Any) -> None:
    current().set_commission(model)


def set_slippage(model: Any) -> None:
    current().set_slippage(model)


def set_benchmark(asset: Asset | None) -> None:
    current().set_benchmark(asset)
