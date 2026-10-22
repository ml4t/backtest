"""Portfolio/account view helpers extracted from Broker."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..controls import AccountControlContext
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
        corporate = self.account.corporate
        adjusted = corporate.marks.get(asset) if corporate is not None else None
        if adjusted is not None and price is not None:
            return price, "action_adjusted_previous_mark", adjusted[2]
        if price is not None:
            return price, self.broker.mark_price.value, self.market.time
        price = self.broker.get_last_price(asset)
        if price is not None:
            times = self.market.last_price_times
            return price, "last_observed", times.get(asset) if times is not None else None
        if position.current_price is not None:
            return position.current_price, "position_current", None
        return position.entry_price, "position_entry", position.entry_time

    def snapshot(
        self, *, include_holdings: bool
    ) -> tuple[
        tuple[float, float, float, float, int],
        list[tuple[Any, ...]] | None,
        AccountControlContext | None,
    ]:
        cash = self.account.cash
        gross = net = 0.0
        variation = self.account.futures_methods is not None
        liquidation = unsettled = initial_margin = maintenance_margin = 0.0
        if variation:
            from ..accounting.policy import UnifiedAccountPolicy

            policy = self.account.policy
            if not isinstance(policy, UnifiedAccountPolicy):
                raise TypeError("Futures variation requires a declared margin policy")
        holdings = [] if include_holdings else None
        for asset, position in self.account.positions.items():
            if include_holdings:
                price, source, mark_time = self.position_mark(asset, position)
            else:
                price = self.broker.get_mark_price(asset, quantity=position.quantity)
                if price is None:
                    price, _, _ = self.position_mark(asset, position)
            value = position.quantity * price * position.multiplier
            gross += abs(value)
            net += value
            liquidation_value = (
                value
                if position.variation_basis is None
                else position.quantity * (price - position.variation_basis) * position.multiplier
            )
            if variation:
                liquidation += liquidation_value
                if position.variation_basis is not None:
                    unsettled += liquidation_value
                initial_margin += policy.get_margin_requirement(
                    asset,
                    position.quantity,
                    price,
                    multiplier=position.multiplier,
                )
                maintenance_margin += policy.get_margin_requirement(
                    asset,
                    position.quantity,
                    price,
                    for_initial=False,
                    multiplier=position.multiplier,
                )
            if holdings is not None:
                holdings.append(
                    (
                        asset,
                        position.quantity,
                        price,
                        position.multiplier,
                        self.broker._get_contract_currency(asset),
                        value,
                        liquidation_value,
                        source,
                        mark_time,
                    )
                )
        equity = cash + (liquidation if variation else net)
        receivables = (
            self.account.corporate.receivable_value if self.account.corporate is not None else 0.0
        )
        equity += receivables
        details = None
        if variation:
            holds = (
                self.account.unsettled_cash if self.broker.settlement_reduces_buying_power else 0.0
            )
            buffer = max(cash - holds, 0.0) * self.broker.cash_buffer_pct
            details = AccountControlContext(
                self.market.time,
                equity,
                cash,
                gross,
                net,
                positions_value=liquidation,
                initial_margin_required=initial_margin,
                maintenance_margin_required=maintenance_margin,
                available_funds=equity - receivables - initial_margin - holds - buffer,
                futures_unsettled_pnl=unsettled,
                settlement_holds=holds,
                cash_buffer=buffer,
                margin_breach=equity < maintenance_margin,
            )
        return (
            (
                equity,
                cash,
                gross,
                net,
                len(self.account.positions),
            ),
            holdings,
            details,
        )

    def get_account_value(self) -> float:
        value = self.account.cash + (
            self.account.corporate.receivable_value if self.account.corporate is not None else 0.0
        )
        for asset, pos in self.account.positions.items():
            price = self.broker.get_mark_price(asset, quantity=pos.quantity)
            if price is None:
                price, _, _ = self.position_mark(asset, pos)
            multiplier = self.broker.get_multiplier(asset)
            value += (
                pos.quantity
                * (price if pos.variation_basis is None else price - pos.variation_basis)
                * multiplier
            )
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
