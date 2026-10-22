"""Variation-cash events applied to the existing account and position basis."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import TYPE_CHECKING, Any

import polars as pl

from ._temporal import temporal_rows

if TYPE_CHECKING:
    from .accounting.account import AccountState
    from .broker import Broker
    from .core.state import ExecutionJournal


class FuturesLedger:
    def __init__(
        self,
        broker: Broker,
        events: pl.DataFrame | None,
        timestamps: list[datetime],
        *,
        account: AccountState,
        journal: ExecutionJournal,
    ) -> None:
        self.broker, self.account, self.journal = broker, account, journal
        self._cursor = 0
        self._events: list[dict[str, Any]] = []
        if events is None:
            return
        if not isinstance(events, pl.DataFrame):
            raise TypeError("settlement_events must be a Polars DataFrame")
        required = {
            "timestamp",
            "asset",
            "available_at",
            "source",
            "event_id",
            "revision_id",
            "settlement_price",
            "currency",
        }
        if not required <= set(events.columns):
            raise ValueError(
                f"settlement_events missing fields: {sorted(required - set(events.columns))}"
            )
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        reference = timestamps[0] if timestamps else None
        for row in temporal_rows(events, named=True):
            for name in ["asset", "source", "event_id", "revision_id", "currency"]:
                if not isinstance(row[name], str) or not row[name].strip():
                    raise ValueError(f"Settlement {name} must be a known nonempty identity")
            for name in ["timestamp", "available_at"]:
                value = row[name]
                if (
                    not isinstance(value, datetime)
                    or reference is not None
                    and (value.tzinfo is None) != (reference.tzinfo is None)
                ):
                    raise ValueError("Settlement clocks require feed-compatible physical instants")
            if row["available_at"] > row["timestamp"]:
                raise ValueError("Settlement price cannot be paid before publication")
            price = row["settlement_price"]
            if (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not isfinite(price)
            ):
                raise ValueError("Settlement price must be finite")
            if (account.futures_methods or {}).get(row["asset"]) != "scheduled":
                raise ValueError("Settlement requires a scheduled futures_variation instrument")
            if broker._get_contract_currency(row["asset"]) != row["currency"]:
                raise ValueError(
                    "Settlement currency must match its instrument; FX is not implicit"
                )
            key = row["source"], row["event_id"]
            if key in seen:
                if seen[key] == row:
                    continue
                raise ValueError(
                    "Conflicting or revised settlement requires an explicit forward event"
                )
            seen[key] = row
            self._events.append(row)
        self._events.sort(key=lambda row: (row["timestamp"], row["source"], row["event_id"]))

    @property
    def records(self) -> list[dict[str, Any]]:
        records = self.journal.futures_settlements
        assert records is not None
        return records

    def _settle(self, event: dict[str, Any], *, phase: str) -> None:
        position = self.account.positions.get(event["asset"])
        quantity = position.quantity if position is not None else 0.0
        basis = position.variation_basis if position is not None else None
        multiplier = self.broker.get_multiplier(event["asset"])
        delta = 0.0 if not quantity else quantity * (event["settlement_price"] - basis) * multiplier
        cash = self.account.cash + delta
        if not isfinite(delta) or not isfinite(cash):
            raise ValueError("Settlement would make the cash ledger nonfinite")
        self.account.cash = cash
        self.account._lock_notional_free_cash += delta
        if position is not None:
            position.variation_basis = event["settlement_price"]
        self.records.append(
            {
                **event,
                "phase": phase,
                "quantity": quantity,
                "basis_before": basis,
                "multiplier": multiplier,
                "cash_delta": delta,
            }
        )

    def advance(self, timestamp: datetime) -> None:
        while (
            self._cursor < len(self._events)
            and self._events[self._cursor]["timestamp"] <= timestamp
        ):
            end = self._cursor + 1
            while (
                end < len(self._events)
                and self._events[end]["timestamp"] == self._events[self._cursor]["timestamp"]
            ):
                end += 1
            state = self.broker._snapshot_lifecycle_state(all_positions=True)
            length = len(self.records)
            try:
                for event in self._events[self._cursor : end]:
                    self._settle(event, phase="scheduled")
            except Exception:
                self.broker._restore_lifecycle_state(state)
                del self.records[length:]
                raise
            self._cursor = end

    def on_mark(self, timestamp: datetime, phase: str) -> None:
        planned = []
        cash = self.account.cash
        for asset, mode in (self.account.futures_methods or {}).items():
            position = self.account.positions.get(asset)
            if position is None:
                continue
            mark = self.broker.get_mark_price(asset, quantity=position.quantity)
            if mark is None:
                continue
            if not isfinite(mark):
                raise ValueError("Futures account mark must be finite")
            event = None
            if mode == "on_mark" and mark != position.variation_basis:
                basis = position.variation_basis
                if basis is None or not isfinite(basis):
                    raise ValueError("Futures variation basis must be captured and finite")
                delta = position.quantity * (mark - basis) * position.multiplier
                cash += delta
                if not isfinite(delta) or not isfinite(cash):
                    raise ValueError("Settlement would make the cash ledger nonfinite")
                event = {
                    "timestamp": timestamp,
                    "available_at": timestamp,
                    "asset": asset,
                    "settlement_price": mark,
                    "currency": self.broker._get_contract_currency(asset),
                    "source": "ml4t.backtest/on_mark",
                    "event_id": f"{asset}:{timestamp.isoformat()}:{phase}",
                    "revision_id": "generated",
                }
            planned.append((position, mark, event))
        for position, mark, event in planned:
            position.current_price = mark
            if event is not None:
                self._settle(event, phase=phase)
