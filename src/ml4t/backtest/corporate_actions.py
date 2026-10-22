"""Causal raw-action application to the existing broker account and orders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite, trunc, ulp
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from ._temporal import temporal_rows
from .accounting.account import AccountState, CorporateState
from .calendar import get_schedule
from .config import ShareType, ShortCashPolicy
from .types import AssetClass, OrderStatus, Position

if TYPE_CHECKING:
    from .broker import Broker
    from .config import BacktestConfig
    from .core.state import ExecutionJournal, MarketState, OrderState


@dataclass(frozen=True)
class _Action:
    key: tuple[str, str]
    terms: dict[str, Any]
    effective: datetime
    payment: datetime | None


def action_terms_json(terms: dict[str, Any]) -> str:
    def date_value(value: Any) -> str:
        if isinstance(value, date):
            return value.isoformat()
        raise TypeError(f"Nonportable action term: {type(value).__name__}")

    return json.dumps(terms, default=date_value, sort_keys=True, allow_nan=False)


def _number(row: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = row.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"Action {key} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


def _raw_metadata(manifest: dict[str, Any] | None) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Raw corporate actions require declared input_manifest price_basis")
    entries = manifest.get("content", {}).get("tables")
    if entries is None:
        metadata = [manifest]
    else:
        metadata = [
            entry.get("metadata", {})
            for entry in entries.values()
            if entry.get("role") in {"prices", "bars", "daily_bars", "minute_bars"}
        ]
    if not metadata or any(row.get("price_basis") != "raw" for row in metadata):
        raise ValueError(
            "Corporate actions require raw prices; adjusted or unknown basis double counts"
        )


class CorporateActionLedger:
    """Validate all terms before execution; apply each clock group atomically.

    ``declared_terms`` retains fractional shares and recognizes cash/stock receivables.
    ``zipline_3_2`` truncates shares toward zero, uses reference split cash-in-lieu and
    recognizes distributions at payment. Split input is always new shares per old share.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        manifest: dict[str, Any] | None,
        config: BacktestConfig,
        broker: Broker,
        timestamps: list[datetime],
        *,
        account: AccountState,
        market: MarketState,
        orders: OrderState,
        journal: ExecutionJournal,
    ) -> None:
        _raw_metadata(manifest)
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("corporate_actions must be a Polars DataFrame")
        required = {
            "source",
            "event_id",
            "revision_id",
            "asset",
            "action_kind",
            "effective_at",
            "available_at",
        }
        if not required <= set(frame.columns):
            raise ValueError(
                f"corporate_actions missing required fields: {sorted(required - set(frame.columns))}"
            )
        self.broker, self.config = broker, config
        self.account, self.market, self.orders = account, market, orders
        self.policy = config.corporate_action_policy
        self.records: list[dict[str, Any]] = []
        account.corporate = CorporateState()
        journal.completed_corporate_entitlements = []
        self.completed: list[dict[str, Any]] = journal.completed_corporate_entitlements
        self._cursor = 0
        self._events: list[tuple[datetime, int, _Action, str]] = []
        reference = timestamps[0] if timestamps else None
        versions: dict[tuple[str, str], list[_Action]] = {}
        duplicate: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in temporal_rows(frame, named=True):
            for key in ("source", "event_id", "revision_id", "asset"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"Action {key} must be nonempty")
            identity = row["source"], row["event_id"], row["revision_id"]
            if identity in duplicate:
                if duplicate[identity] != row:
                    raise ValueError("Conflicting terms for the same action revision")
                continue
            duplicate[identity] = row
            kind = row["action_kind"]
            if kind not in {
                "split",
                "cash_dividend",
                "cash_distribution",
                "stock_dividend",
                "stock_distribution",
                "merger",
            }:
                raise ValueError(f"Unsupported action_kind: {kind}")
            effective_term = row["effective_at"]
            if kind != "split" and kind != "merger":
                eligibility = (
                    row.get("ex_at") if self.policy == "zipline_3_2" else row.get("entitlement_at")
                )
                if eligibility is None:
                    eligibility = row.get("ex_date")
                if eligibility is None:
                    eligibility = row.get("entitlement_date")
                if eligibility is not None:
                    effective_term = eligibility
            effective = self._clock(effective_term, reference)
            available = row["available_at"]
            if not isinstance(available, datetime):
                raise ValueError("Action available_at must be a known physical instant")
            available = self._clock(available, reference)
            if available > effective:
                raise ValueError(
                    "Post-effective action corrections require an explicit forward event"
                )
            if (
                not isinstance(row.get("currency"), str)
                or not row["currency"]
                or row["currency"] != broker._get_contract_currency(row["asset"])
            ):
                raise ValueError(
                    "Action currency must match the source instrument; FX conversion is not implicit"
                )
            spec = broker.get_contract_spec(row["asset"])
            if spec is not None and spec.asset_class is not AssetClass.EQUITY:
                raise ValueError("Corporate actions require an equity instrument")
            payment = None
            if kind == "split":
                _number(row, "split_ratio", positive=True)
                if row.get("split_ratio_convention") != "new_per_old":
                    raise ValueError(
                        "split_ratio_convention must declare new_per_old; normalize legacy ratios once in Data"
                    )
                fractional = row.get("fractional_policy") or "retain"
                if fractional not in {"retain", "cash_in_lieu"}:
                    raise ValueError("fractional_policy must be retain or cash_in_lieu")
                if fractional == "cash_in_lieu" and self.policy != "zipline_3_2":
                    _number(row, "cash_in_lieu_price", positive=True)
            elif kind in {"cash_dividend", "cash_distribution"}:
                _number(row, "cash_per_share")
            else:
                _number(row, "quantity_ratio", positive=kind != "merger")
                if row["quantity_ratio"] != 0:
                    destination = row.get("destination_asset")
                    if not isinstance(destination, str) or not destination:
                        raise ValueError("Security distribution requires destination_asset")
                    if broker._get_contract_currency(destination) != row["currency"]:
                        raise ValueError("Destination currency requires explicit FX conversion")
                if kind == "merger":
                    _number(row, "cash_component")
            if kind != "split":
                if row.get("share_basis") not in {"pre_event", "post_event"}:
                    raise ValueError("Action share_basis must be pre_event or post_event")
                payment_term = row.get("payment_at")
                if payment_term is None:
                    payment_term = row.get("payment_date")
                if kind == "merger" and payment_term is None:
                    payment_term = row["effective_at"]
                if payment_term is None:
                    raise ValueError(
                        "Distribution requires payment_at or a declared payment_date session label"
                    )
                payment = self._clock(payment_term, reference)
                if payment < effective:
                    raise ValueError("Action payment_date precedes entitlement")
            key = row["source"], row["event_id"]
            action_terms_json(row)
            versions.setdefault(key, []).append(_Action(key, row, effective, payment))
        for candidates in versions.values():
            candidates.sort(
                key=lambda event: (event.terms["available_at"], event.terms["revision_id"])
            )
            if (
                len(candidates) > 1
                and candidates[-1].terms["available_at"] == candidates[-2].terms["available_at"]
            ):
                raise ValueError("Action revisions at the same publication instant are ambiguous")
            event = candidates[-1]
            if event.terms["available_at"] > min(candidate.effective for candidate in candidates):
                raise ValueError(
                    "Post-effective action revisions require a forward correction event"
                )
            kind = event.terms["action_kind"]
            if kind == "split" or kind == "merger":
                self._events.append((event.effective, 1, event, kind))
            else:
                phase = 0 if event.terms["share_basis"] == "pre_event" else 2
                self._events.append((event.effective, phase, event, "entitlement"))
            if event.payment is not None:
                self._events.append((event.payment, 3, event, "payment"))
        self._events.sort(key=lambda item: (item[0], item[1], item[2].key))

    @property
    def state(self) -> CorporateState:
        state = self.account.corporate
        assert state is not None
        return state

    @property
    def entitlements(self) -> dict[tuple[str, str], dict[str, Any]]:
        return self.state.entitlements

    def _clock(self, value: Any, reference: datetime | None) -> datetime:
        if isinstance(value, datetime):
            if reference is not None and (value.tzinfo is None) != (reference.tzinfo is None):
                raise ValueError("Action clock and feed require compatible physical timezones")
            return value
        if not isinstance(value, date) or self.config.resolved_calendar is None:
            raise ValueError(
                "Action date labels require a calendar; calendar=None requires physical instants"
            )
        rows = get_schedule(self.config.resolved_calendar, value, value + timedelta(days=1))
        for row in temporal_rows(rows, named=True):
            if row["session_date"] != value:
                continue
            # A session label applies from its UTC session boundary, before pre-market callbacks.
            opened = pd.Timestamp(value, tz="UTC")
            return (
                opened.tz_localize(None).to_pydatetime()
                if reference is not None and reference.tzinfo is None
                else opened.to_pydatetime()
            )
        raise ValueError("Action date is not a declared calendar session")

    def _record(self, event: _Action, timestamp: datetime, operation: str, **values: Any) -> None:
        self.records.append(
            dict(
                timestamp=timestamp,
                source=event.key[0],
                event_id=event.key[1],
                revision_id=event.terms["revision_id"],
                asset=event.terms["asset"],
                action_kind=event.terms["action_kind"],
                operation=operation,
                available_at=event.terms["available_at"],
                effective_at=event.effective,
                payment_at=event.payment,
                terms_json=action_terms_json(event.terms),
                **({"cash_delta": 0.0, "realized_pnl": 0.0, "allocated_commission": 0.0} | values),
            )
        )

    def advance(self, timestamp: datetime) -> None:
        while self._cursor < len(self._events) and self._events[self._cursor][0] <= timestamp:
            instant = self._events[self._cursor][0]
            end = self._cursor
            while end < len(self._events) and self._events[end][0] == instant:
                end += 1
            broker = self.broker
            state = broker._snapshot_lifecycle_state(
                all_positions=True,
                all_pending_orders=True,
                risk_rules=True,
                all_asset_stats=True,
                target_intents=True,
            )
            length = len(self.records)
            try:
                for _, _, event, operation in self._events[self._cursor : end]:
                    if operation == "split":
                        self._split(event, instant)
                    elif operation == "merger":
                        self._merger(event, instant)
                    elif operation == "entitlement":
                        self._entitle(event, instant)
                    else:
                        self._pay(event, instant)
                self.refresh()
            except Exception:
                broker._restore_lifecycle_state(state)
                del self.records[length:]
                raise
            self._cursor = end

    def _cash(self, delta: float) -> None:
        account = self.account
        cash = account.cash + delta
        free = account._lock_notional_free_cash + delta
        if not isfinite(cash) or not isfinite(free):
            raise ValueError("Corporate action would make the cash ledger nonfinite")
        account.cash, account._lock_notional_free_cash = cash, free

    def _mark(self, asset: str, quantity: float) -> float | None:
        value = self.broker.get_mark_price(asset, quantity=quantity)
        return value if value is not None else self.broker.get_last_price(asset)

    def _rebase_mark(
        self, asset: str, *, ratio: float = 1.0, cash: float = 0.0, short_cash: float | None = None
    ) -> None:
        long_mark, short_mark = self._mark(asset, 1.0), self._mark(asset, -1.0)
        if long_mark is None or short_mark is None:
            return
        long_value = long_mark / ratio - cash
        short_value = short_mark / ratio - (cash if short_cash is None else short_cash)
        if not all(isfinite(value) and value >= 0 for value in [long_value, short_value]):
            raise ValueError(
                "Corporate action requires a finite nonnegative adjusted previous mark"
            )
        market = self.market
        known = self.state.marks.get(asset)
        observed = (
            known[2]
            if known is not None
            else (
                market.last_price_times.get(asset) if market.last_price_times is not None else None
            )
        )
        self.state.marks[asset] = (long_value, short_value, observed)
        position = self.broker.positions.get(asset)
        if position is not None:
            position.current_price = long_value if position.quantity > 0 else short_value

    def _split(self, event: _Action, instant: datetime) -> None:
        broker, row = self.broker, event.terms
        asset, ratio = row["asset"], row["split_ratio"]
        legacy = self.policy == "zipline_3_2"
        position = broker.positions.get(asset)
        old_quantity = position.quantity if position is not None else 0.0
        cash_delta = 0.0
        realized_pnl = allocated_commission = 0.0
        if position is not None:
            exact = old_quantity * ratio
            if not isfinite(exact) or not isfinite(position.entry_price / ratio):
                raise ValueError("split_ratio would make the position nonfinite")
            whole = (
                trunc(exact) if legacy or row.get("fractional_policy") == "cash_in_lieu" else exact
            )
            basis = position.commission_adjusted_cost_basis / ratio
            if legacy:
                basis = round(basis, 2)
            fraction = exact - whole
            cash_delta = (
                round(fraction * basis, 2)
                if legacy
                else fraction * row.get("cash_in_lieu_price", 0.0)
            )
            if fraction:
                allocated_commission = position.entry_commission * abs(fraction / exact)
                realized_pnl = (
                    cash_delta - fraction * (position.entry_price / ratio) - allocated_commission
                )
                position.entry_commission -= allocated_commission
            position.quantity = whole
            position.entry_price /= ratio
            position._commission_adjusted_cost_basis = basis
            if position.initial_quantity is not None:
                position.initial_quantity *= ratio
            position.entry_slippage /= ratio
            for field in ("current_price", "high_water_mark", "low_water_mark"):
                value = getattr(position, field)
                if value is not None:
                    setattr(position, field, value / ratio)
            if not whole:
                del self.account.positions[asset]
            self._cash(cash_delta)
            if old_quantity < 0 and fraction:
                debt = broker.account._lock_notional_short_basis.get(asset, 0.0)
                released = debt * abs(fraction / exact)
                broker.account._lock_notional_short_basis[asset] = debt - released
                broker.account._lock_notional_free_cash += 2 * released
        for order in broker.pending_orders:
            if order.asset != asset or order.status is not OrderStatus.PENDING:
                continue
            before = order.quantity
            rounding = trunc if legacy or broker.share_type is ShareType.INTEGER else float
            total = (
                order.action_adjusted_quantity
                if order.action_adjusted_quantity is not None
                else order.requested_quantity
            )
            assert total is not None
            order.action_adjusted_quantity = float(rounding(total * ratio))
            order.filled_quantity = float(rounding(order.filled_quantity * ratio))
            order.quantity = max(0.0, order.action_adjusted_quantity - order.filled_quantity)
            if order.filled_price is not None:
                order.filled_price /= ratio
            for field in (
                "limit_price",
                "stop_price",
                "trail_amount",
                "_signal_price",
                "_risk_fill_price",
            ):
                value = getattr(order, field)
                if value is not None:
                    setattr(order, field, round(value / ratio, 2) if legacy else value / ratio)
            if order.order_id in self.orders.partial_quantities:
                self.orders.partial_quantities[order.order_id] = max(0.0, order.quantity)
            if order.quantity == 0:
                order.status = OrderStatus.FILLED
                order.filled_at = instant
            self._record(
                event,
                instant,
                "order_adjustment",
                order_id=order.order_id,
                quantity_before=before,
                quantity_after=order.quantity,
            )
        self._rebase_mark(asset, ratio=ratio)
        flat = self.state.flat_deliveries.get(asset)
        if flat is not None:
            index, basis = flat
            self.state.flat_deliveries[asset] = (
                index,
                round(basis / ratio, 2) if legacy else basis / ratio,
            )
        self._record(
            event,
            instant,
            "split",
            quantity_before=old_quantity,
            quantity_after=broker.positions[asset].quantity if asset in broker.positions else 0.0,
            cash_delta=cash_delta,
            realized_pnl=realized_pnl,
            allocated_commission=allocated_commission,
        )

    def _entitle(self, event: _Action, instant: datetime) -> None:
        row = event.terms
        position = self.broker.positions.get(row["asset"])
        quantity = position.quantity if position is not None else 0.0
        stock = row["action_kind"] in {"stock_dividend", "stock_distribution"}
        units = quantity * row["quantity_ratio"] if stock else 0.0
        if stock and self.policy == "zipline_3_2":
            nearest = round(units)
            units = float(trunc(nearest if abs(units - nearest) <= ulp(units) else units))
        self.entitlements[event.key] = {
            "source": event.key[0],
            "event_id": event.key[1],
            "revision_id": row["revision_id"],
            "asset": row["asset"],
            "action_kind": row["action_kind"],
            "entitlement_at": instant,
            "payment_at": event.payment,
            "eligible_quantity": quantity,
            "cash_amount": 0.0 if stock else quantity * row["cash_per_share"],
            "destination_asset": row.get("destination_asset") if stock else None,
            "security_quantity": units,
            "currency": row["currency"],
            "paid": False,
            "destination_basis": 0.0,
            "destination_commission": 0.0,
        }
        self._record(
            event, instant, "entitlement", quantity_before=quantity, quantity_after=quantity
        )
        if row["action_kind"] in {"cash_dividend", "cash_distribution"}:
            adjustment = row["cash_per_share"]
        else:
            if row["destination_asset"] == row["asset"]:
                self._rebase_mark(row["asset"], ratio=1 + row["quantity_ratio"])
                return
            long_mark = self._mark(row["destination_asset"], 1.0)
            short_mark = self._mark(row["destination_asset"], -1.0)
            if long_mark is None or short_mark is None:
                if not quantity:
                    return
                raise ValueError("Distribution requires an observed destination mark")
            self._rebase_mark(
                row["asset"],
                cash=row["quantity_ratio"] * long_mark,
                short_cash=row["quantity_ratio"] * short_mark,
            )
            return
        self._rebase_mark(row["asset"], cash=adjustment)

    def _add_security(
        self, asset: str, quantity: float, basis: float, instant: datetime, commission: float = 0.0
    ) -> tuple[float, float]:
        if not quantity:
            return 0.0, 0.0
        broker = self.broker
        existing = broker.positions.get(asset)
        mark = self._mark(asset, quantity)
        if mark is None:
            raise ValueError("Security distribution requires an observed destination mark")
        total = quantity + (existing.quantity if existing is not None else 0.0)
        reported_basis = existing.commission_adjusted_cost_basis if existing is not None else None
        if not isfinite(total) or not isfinite(basis):
            raise ValueError("Security conversion would create nonfinite position terms")
        self._short_collateral(
            asset, existing.quantity if existing is not None else 0.0, total, basis
        )
        realized = allocated = 0.0
        if existing is not None and existing.quantity * quantity < 0:
            closed = min(abs(existing.quantity), abs(quantity))
            allocated_old = existing.entry_commission * closed / abs(existing.quantity)
            allocated_incoming = commission * closed / abs(quantity)
            allocated = allocated_old + allocated_incoming
            realized = (
                closed * (1 if existing.quantity > 0 else -1) * (basis - existing.entry_price)
                - allocated
            )
            if total == 0:
                del self.account.positions[asset]
                return realized, allocated
            if total * existing.quantity > 0:
                existing.quantity = total
                existing.entry_commission -= allocated_old
                return realized, allocated
            commission -= allocated_incoming
            existing = None
        if existing is None:
            self.account.positions[asset] = Position(
                asset,
                total,
                basis,
                instant,
                current_price=mark,
                multiplier=broker.get_multiplier(asset),
                entry_commission=commission,
            )
        else:
            old_basis = existing.commission_adjusted_cost_basis
            existing.entry_price = (
                existing.quantity * existing.entry_price + quantity * basis
            ) / total
            existing._commission_adjusted_cost_basis = (
                existing.quantity * old_basis + quantity * basis
            ) / total
            existing.entry_commission += commission
            existing.quantity = total
        if self.policy == "zipline_3_2" and reported_basis is not None:
            broker.positions[asset]._commission_adjusted_cost_basis = reported_basis
        return realized, allocated

    def _short_collateral(
        self, asset: str, old_quantity: float, new_quantity: float, basis: float
    ) -> None:
        if (
            self.config.allow_leverage
            or self.broker.short_cash_policy is not ShortCashPolicy.LOCK_NOTIONAL
        ):
            return
        account = self.account
        before = account._lock_notional_short_basis.get(asset, 0.0)
        if new_quantity >= 0:
            after = 0.0
        elif old_quantity < 0:
            after = (
                before * min(1.0, abs(new_quantity / old_quantity))
                + max(0.0, abs(new_quantity) - abs(old_quantity)) * basis
            )
        else:
            after = abs(new_quantity) * basis
        free = account._lock_notional_free_cash + 2 * (before - after)
        if not isfinite(free) or not isfinite(after) or after < 0:
            raise ValueError(
                "Corporate conversion requires finite nonnegative short collateral basis"
            )
        account._lock_notional_free_cash = free
        if after:
            account._lock_notional_short_basis[asset] = after
        else:
            account._lock_notional_short_basis.pop(asset, None)

    def _pay(self, event: _Action, instant: datetime) -> None:
        entitlement = self.entitlements.get(event.key)
        if entitlement is None or entitlement["paid"]:
            return
        self._cash(entitlement["cash_amount"])
        destination = entitlement["destination_asset"]
        prior = self.broker.positions.get(destination)
        prior_reported_basis = prior.commission_adjusted_cost_basis if prior is not None else 0.0
        realized, allocated = self._add_security(
            entitlement["destination_asset"],
            entitlement["security_quantity"],
            entitlement.get("destination_basis", 0.0),
            instant,
            entitlement.get("destination_commission", 0.0),
        )
        if event.terms["action_kind"] == "merger" and self.policy == "zipline_3_2":
            realized += (
                entitlement["cash_amount"]
                + entitlement["security_quantity"] * entitlement["destination_basis"]
                + entitlement.get("destination_commission", 0.0)
            )
        entitlement["paid"] = True
        entitlement["receivable_value"] = 0.0
        if destination is not None:
            remaining = self.broker.positions.get(destination)
            entitlement["delivered_position_quantity"] = (
                remaining.quantity if remaining is not None else 0.0
            )
            entitlement["delivered_position_basis"] = (
                remaining.commission_adjusted_cost_basis
                if remaining is not None
                else prior_reported_basis
            )
            entitlement["delivery_fill_count"] = len(self.broker.fills)
            if remaining is None:
                self.state.flat_deliveries[destination] = (
                    len(self.completed),
                    entitlement["delivered_position_basis"],
                )
            else:
                self.state.flat_deliveries.pop(destination, None)
        self.completed.append(entitlement)
        del self.entitlements[event.key]
        self._record(
            event,
            instant,
            "payment",
            cash_delta=entitlement["cash_amount"],
            quantity_after=entitlement["security_quantity"],
            realized_pnl=realized,
            allocated_commission=allocated,
        )

    def _merger(self, event: _Action, instant: datetime) -> None:
        row, broker = event.terms, self.broker
        position = broker.positions.get(row["asset"])
        quantity = position.quantity if position is not None else 0.0
        realized = commission = 0.0
        for order in tuple(broker.pending_orders):
            if order.asset == row["asset"]:
                broker._cancel_order_at(
                    order.order_id, timestamp=instant, reason="corporate_conversion"
                )
                self._record(
                    event,
                    instant,
                    "order_cancellation",
                    order_id=order.order_id,
                    quantity_before=order.quantity,
                    quantity_after=0.0,
                )
        self.state.converted_assets.add(row["asset"])
        self.state.marks.pop(row["asset"], None)
        self.state.flat_deliveries.pop(row["asset"], None)
        if position is not None:
            self._short_collateral(row["asset"], quantity, 0.0, 0.0)
            units = quantity * row["quantity_ratio"]
            self.entitlements[event.key] = {
                "source": event.key[0],
                "event_id": event.key[1],
                "revision_id": row["revision_id"],
                "asset": row["asset"],
                "action_kind": "merger",
                "entitlement_at": instant,
                "payment_at": event.payment,
                "eligible_quantity": quantity,
                "cash_amount": quantity * row["cash_component"],
                "destination_asset": row.get("destination_asset"),
                "security_quantity": units,
                "currency": row["currency"],
                "paid": False,
                "destination_basis": max(0.0, position.entry_price - row["cash_component"])
                / row["quantity_ratio"]
                if units
                else 0.0,
                "destination_commission": position.entry_commission if units else 0.0,
            }
            if self.policy == "zipline_3_2":
                realized = -quantity * position.entry_price - position.entry_commission
            elif units:
                realized = quantity * max(0.0, row["cash_component"] - position.entry_price)
            if not units:
                commission = position.entry_commission
                if self.policy == "declared_terms":
                    realized = (
                        quantity * (row["cash_component"] - position.entry_price) - commission
                    )
            del self.account.positions[row["asset"]]
        self._record(
            event,
            instant,
            "merger",
            quantity_before=quantity,
            quantity_after=0.0,
            realized_pnl=realized,
            allocated_commission=commission,
        )

    def refresh(self) -> None:
        value = 0.0
        for entitlement in self.entitlements.values():
            entitlement["receivable_value"] = 0.0
        if self.policy == "declared_terms":
            for entitlement in self.entitlements.values():
                if entitlement["paid"]:
                    continue
                recognized = entitlement["cash_amount"]
                if entitlement["security_quantity"]:
                    mark = self._mark(
                        entitlement["destination_asset"], entitlement["security_quantity"]
                    )
                    if mark is None:
                        raise ValueError(
                            "Security receivable requires an observed destination mark"
                        )
                    recognized += (
                        entitlement["security_quantity"]
                        * mark
                        * self.broker.get_multiplier(entitlement["destination_asset"])
                    )
                entitlement["receivable_value"] = recognized
                value += recognized
        self.state.receivable_value = value
