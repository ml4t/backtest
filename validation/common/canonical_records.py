"""Canonical record extraction shared by scenario adapters."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _first(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"Record lacks all expected fields: {names}")


def vectorbt_fills(order_records: list[dict[str, Any]], *, asset: str) -> list[dict[str, Any]]:
    """Convert VectorBT's native filled-order records to the canonical fill surface."""
    return [
        {
            "timestamp": _first(record, "Fill Index", "Timestamp"),
            "asset": asset,
            "side": str(record["Side"]).lower(),
            "quantity": abs(float(record["Size"])),
            "price": float(record["Price"]),
            "commission": abs(float(record.get("Fees", 0.0))),
        }
        for record in order_records
    ]


def vectorbt_trades(
    trade_records: list[dict[str, Any]],
    *,
    asset: str,
) -> list[dict[str, Any]]:
    """Convert VectorBT's native closed-trade records to the canonical trade surface."""
    normalized: list[dict[str, Any]] = []
    for record in trade_records:
        status = str(record.get("Status", "Closed")).lower()
        if status != "closed":
            continue
        entry_price = float(_first(record, "Avg Entry Price", "Entry Price"))
        exit_price = float(_first(record, "Avg Exit Price", "Exit Price"))
        size = abs(float(record.get("Size", 0.0)))
        direction = str(record.get("Direction", "Long"))
        commission = abs(float(record.get("Entry Fees", 0.0))) + abs(
            float(record.get("Exit Fees", 0.0))
        )
        direction_sign = -1.0 if direction.lower() == "short" else 1.0
        normalized.append(
            {
                "entry_time": _first(record, "Entry Index", "Entry Timestamp"),
                "exit_time": _first(record, "Exit Index", "Exit Timestamp"),
                "asset": asset,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": (exit_price - entry_price) * size * direction_sign - commission,
                "size": size,
                "direction": direction,
                "commission": commission,
            }
        )
    return normalized


def zipline_fills(
    transactions: pd.DataFrame,
    *,
    asset: str,
    commission_rate: float = 0.0,
    per_share_rate: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert Zipline transactions and the configured fee model to canonical fills."""
    if transactions.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, transaction in transactions.sort_values("dt").iterrows():
        amount = float(transaction["amount"])
        price = float(transaction["price"])
        commission = abs(amount) * (price * commission_rate + per_share_rate)
        records.append(
            {
                "timestamp": transaction["dt"],
                "asset": asset,
                "side": "buy" if amount > 0 else "sell",
                "quantity": abs(amount),
                "price": price,
                "commission": commission,
            }
        )
    return records
