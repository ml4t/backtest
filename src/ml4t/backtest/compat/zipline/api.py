"""Callback-scoped migration API; all account mutations use the native broker."""

from typing import Any

import pandas as pd

from ._scope import current
from .assets import Asset
from .finance.order import OrderView


def symbol(symbol_str: str) -> Asset:
    return current().symbol(symbol_str)


def symbols(*symbol_strs: str) -> tuple[Asset, ...]:
    return current().symbols(*symbol_strs)


def sid(sid: int) -> Asset:
    return current().sid(sid)


def order(
    asset: Asset,
    amount: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    style: Any | None = None,
) -> str | None:
    return current().order(asset, amount, limit_price, stop_price, style)


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


def order_value(
    asset: Asset,
    value: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    style: Any | None = None,
) -> str | None:
    return current().order_value(asset, value, limit_price, stop_price, style)


def order_percent(
    asset: Asset,
    percent: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    style: Any | None = None,
) -> str | None:
    return current().order_percent(asset, percent, limit_price, stop_price, style)


def order_target(
    asset: Asset,
    target: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    style: Any | None = None,
) -> str | None:
    return current().order_target(asset, target, limit_price, stop_price, style)


def order_target_value(
    asset: Asset,
    target: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    style: Any | None = None,
) -> str | None:
    return current().order_target_value(asset, target, limit_price, stop_price, style)


def order_target_percent(
    asset: Asset,
    target: float,
    limit_price: float | None = None,
    stop_price: float | None = None,
    style: Any | None = None,
) -> str | None:
    return current().order_target_percent(asset, target, limit_price, stop_price, style)


def get_order(order_id: str) -> OrderView | None:
    return current().get_order(order_id)


def get_open_orders(asset: Asset | None = None) -> dict[Asset, list[OrderView]] | list[OrderView]:
    return current().get_open_orders(asset)


def cancel_order(order_param: str | OrderView) -> None:
    current().cancel_order(order_param)
