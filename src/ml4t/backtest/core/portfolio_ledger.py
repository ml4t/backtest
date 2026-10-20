"""Portfolio/account view helpers extracted from Broker."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .state import MarketState, OrderState

if TYPE_CHECKING:
    from ..accounting import AccountState
    from ..broker import Broker
    from ..types import Position


class PortfolioLedger:
    """Read-model helpers for account/portfolio state."""

    def __init__(
        self,
        broker: Broker,
        *,
        account: AccountState,
        market: MarketState,
        orders: OrderState,
    ) -> None:
        self.broker = broker
        self.account = account
        self.market = market
        self.orders = orders

    def position_mark(self, asset: str, position: Position) -> tuple[float, str, datetime | None]:
        price = self.broker.get_mark_price(asset, quantity=position.quantity)
        if price is not None:
            return price, self.broker.mark_price.value, self.market.time
        price = self.broker.get_last_price(asset)
        if price is not None:
            return price, "last_observed", self.market.last_price_times.get(asset)
        if position.current_price is not None:
            return position.current_price, "position_current", None
        return position.entry_price, "position_entry", position.entry_time

    def snapshot(
        self, *, include_holdings: bool
    ) -> tuple[tuple[float, float, float, float, int], list[tuple[Any, ...]] | None]:
        cash = self.account.cash
        gross = net = 0.0
        holdings = [] if include_holdings else None
        for asset, position in self.account.positions.items():
            price, source, mark_time = self.position_mark(asset, position)
            value = position.quantity * price * position.multiplier
            gross += abs(value)
            net += value
            if holdings is not None:
                spec = self.broker.get_contract_spec(asset)
                holdings.append(
                    (
                        asset,
                        position.quantity,
                        price,
                        position.multiplier,
                        spec.currency if spec else None,
                        value,
                        value,
                        source,
                        mark_time,
                    )
                )
        return (cash + net, cash, gross, net, len(self.account.positions)), holdings

    def get_account_value(self) -> float:
        value = self.account.cash
        for asset, pos in self.account.positions.items():
            price, _, _ = self.position_mark(asset, pos)
            multiplier = self.broker.get_multiplier(asset)
            value += pos.quantity * price * multiplier
        return value

    def get_rejected_orders(self, asset: str | None = None):
        rejected = [o for o in self.orders.orders if o.status.value == "rejected"]
        if asset is not None:
            rejected = [o for o in rejected if o.asset == asset]
        return rejected

    @property
    def last_rejection_reason(self):
        rejected = [o for o in self.orders.orders if o.status.value == "rejected"]
        return rejected[-1].rejection_reason if rejected else None
