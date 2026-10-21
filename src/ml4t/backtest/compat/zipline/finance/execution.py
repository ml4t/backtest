"""Migration styles translated into native order type and price parameters."""

from dataclasses import dataclass

from ....types import OrderType


@dataclass(frozen=True)
class MarketOrder:
    pass


@dataclass(frozen=True)
class LimitOrder:
    limit_price: float


@dataclass(frozen=True)
class StopOrder:
    stop_price: float


@dataclass(frozen=True)
class StopLimitOrder:
    limit_price: float
    stop_price: float


def _native_style(style, limit_price, stop_price):
    if style is not None:
        if limit_price is not None or stop_price is not None:
            raise ValueError("Specify a style or separate limit/stop prices, not both")
        if isinstance(style, MarketOrder):
            return OrderType.MARKET, None, None
        if isinstance(style, LimitOrder):
            return OrderType.LIMIT, style.limit_price, None
        if isinstance(style, StopOrder):
            return OrderType.STOP, None, style.stop_price
        if isinstance(style, StopLimitOrder):
            return OrderType.STOP_LIMIT, style.limit_price, style.stop_price
        raise TypeError("Use a declared migration execution style")
    kind = (
        OrderType.STOP_LIMIT
        if limit_price is not None and stop_price is not None
        else OrderType.LIMIT
        if limit_price is not None
        else OrderType.STOP
        if stop_price is not None
        else OrderType.MARKET
    )
    return kind, limit_price, stop_price
