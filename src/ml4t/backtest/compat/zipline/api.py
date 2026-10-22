"""Callback-scoped migration API; all account mutations use the native broker."""

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ._scope import current
from .assets import Asset, Future
from .continuous import ContinuousFuture
from .csv import PreparedCSV
from .events import _DateRule, _TimeRule
from .events import calendars as calendars
from .events import date_rules as date_rules
from .events import time_rules as time_rules
from .finance.asset_restrictions import Restrictions
from .finance.order import OrderView
from .pipeline import PreparedPipeline


def symbol(symbol_str: str, country_code: str | None = None) -> Asset:
    return current().symbol(symbol_str, country_code)


def symbols(*symbol_strs: str, country_code: str | None = None) -> list[Asset]:
    return current().symbols(*symbol_strs, country_code=country_code)


def sid(sid: int) -> Asset:
    return current().sid(sid)


def future_symbol(symbol: str) -> Future:
    return current().future_symbol(symbol)


def continuous_future(
    root_symbol_str: str, offset: int = 0, roll: str = "volume", adjustment: str | None = "mul"
) -> ContinuousFuture:
    return current().continuous_future(root_symbol_str, offset, roll, adjustment)


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


def set_commission(model: Any = None, *, us_equities: Any = None, us_futures: Any = None) -> None:
    current().set_commission(model, us_equities=us_equities, us_futures=us_futures)


def set_slippage(model: Any = None, *, us_equities: Any = None, us_futures: Any = None) -> None:
    current().set_slippage(model, us_equities=us_equities, us_futures=us_futures)


def set_benchmark(asset: Asset | None) -> None:
    current().set_benchmark(asset)


def set_cancel_policy(cancel_policy: Any) -> None:
    current().set_cancel_policy(cancel_policy)


def set_asset_restrictions(restrictions: Restrictions, on_error: str = "fail") -> None:
    current().set_asset_restrictions(restrictions, on_error)


def set_do_not_order_list(restricted_list: Any, on_error: str = "fail") -> None:
    current().set_do_not_order_list(restricted_list, on_error)


def fetch_csv(url: str) -> PreparedCSV:
    return current().fetch_csv(url)


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


def schedule_function(
    func: Callable[..., Any],
    date_rule: _DateRule | _TimeRule | None = None,
    time_rule: _TimeRule | None = None,
    half_days: bool = True,
    calendar: str | None = None,
) -> None:
    current().schedule_function(func, date_rule, time_rule, half_days, calendar)


def set_max_order_count(max_count: int, on_error: str = "fail") -> None:
    current().set_max_order_count(max_count, on_error)


def set_long_only(on_error: str = "fail") -> None:
    current().set_long_only(on_error)


def set_max_order_size(
    asset: Asset | None = None,
    max_shares: float | None = None,
    max_notional: float | None = None,
    on_error: str = "fail",
) -> None:
    current().set_max_order_size(asset, max_shares, max_notional, on_error)


def set_max_position_size(
    asset: Asset | None = None,
    max_shares: float | None = None,
    max_notional: float | None = None,
    on_error: str = "fail",
) -> None:
    current().set_max_position_size(asset, max_shares, max_notional, on_error)


def set_max_leverage(max_leverage: float) -> None:
    current().set_max_leverage(max_leverage)


def set_min_leverage(min_leverage: float, grace_period: timedelta) -> None:
    current().set_min_leverage(min_leverage, grace_period)


def get_environment(field: str = "platform") -> Any:
    return current().get_environment(field)


def set_symbol_lookup_date(dt: date | datetime | str) -> None:
    current().set_symbol_lookup_date(dt)


def attach_pipeline(
    pipeline: PreparedPipeline, name: str, chunks: Any = None, eager: bool = True
) -> PreparedPipeline:
    return current().attach_pipeline(pipeline, name, chunks, eager)


def pipeline_output(name: str) -> pd.DataFrame:
    return current().pipeline_output(name)
