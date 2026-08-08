"""Portfolio/account view helpers extracted from Broker."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .state import MarketState, OrderState

if TYPE_CHECKING:
    from ..accounting import AccountState
    from ..broker import Broker


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

    def get_account_value(self) -> float:
        value = self.account.cash
        for asset, pos in self.account.positions.items():
            price = self.broker.get_mark_price(asset, quantity=pos.quantity)
            if price is None:
                price = self.market.last_prices.get(asset)
            if price is None:
                continue
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
