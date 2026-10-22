"""Observed outcomes of paired, non-atomic dated-contract order intents."""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from ._temporal import temporal_frame, timestamp_dtype

_CLOCKS = (
    "decision_at",
    "available_at",
    "submitted_at",
    "first_fill_at",
    "last_fill_at",
    "filled_at",
)


def project_rolls(intents, orders, fills) -> list[dict[str, Any]]:
    by_id = {order.order_id: order for order in orders}
    executed: dict[str, list[tuple[int, Any]]] = {}
    for sequence, fill in enumerate(fills):
        executed.setdefault(fill.order_id, []).append((sequence, fill))
    rows = []
    for intent in intents:
        legs = []
        for leg in ("close", "open"):
            order_id = intent[f"{leg}_order_id"]
            order = by_id.get(order_id)
            actual = executed.get(order_id, [])
            quantity = intent[f"{leg}_quantity"]
            filled = sum(fill.quantity for _, fill in actual)
            legs.append(
                {
                    "roll_id": intent["roll_id"],
                    "leg": leg,
                    "asset": intent[f"{leg}_asset"],
                    "order_id": order_id,
                    "quantity": abs(quantity),
                    "side": "buy" if quantity > 0 else "sell" if quantity < 0 else None,
                    "filled_quantity": filled,
                    "unfilled_requested_quantity": max(abs(quantity) - filled, 0.0),
                    "overfilled_requested_quantity": max(filled - abs(quantity), 0.0),
                    "status": order.status.value if order is not None else "not_submitted",
                    "rejection_reason": order.rejection_reason if order is not None else None,
                    "commission": sum(fill.commission for _, fill in actual),
                    "fill_sequences": [sequence for sequence, _ in actual],
                    "first_fill_at": actual[0][1].timestamp.isoformat() if actual else None,
                    "last_fill_at": actual[-1][1].timestamp.isoformat() if actual else None,
                    "filled_at": (
                        actual[-1][1].timestamp.isoformat()
                        if actual and order is not None and order.status.value == "filled"
                        else None
                    ),
                    **{
                        name: intent[name]
                        for name in (
                            "decision_at",
                            "available_at",
                            "submitted_at",
                            "selection_json",
                        )
                    },
                    "submission_policy": "paired_non_atomic",
                }
            )
        state = (
            "no_position"
            if not intent["close_quantity"]
            else "mismatched"
            if any(row["overfilled_requested_quantity"] for row in legs)
            else "completed"
            if all(row["unfilled_requested_quantity"] == 0 for row in legs)
            else "partial"
            if any(row["filled_quantity"] for row in legs)
            else "pending"
            if any(row["status"] == "pending" for row in legs)
            else "not_executed"
        )
        rows.extend({**row, "roll_state": state} for row in legs)
    return rows


def roll_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "roll_id": pl.String,
        "leg": pl.String,
        "asset": pl.String,
        "order_id": pl.String,
        "quantity": pl.Float64,
        "side": pl.String,
        "filled_quantity": pl.Float64,
        "unfilled_requested_quantity": pl.Float64,
        "overfilled_requested_quantity": pl.Float64,
        "status": pl.String,
        "rejection_reason": pl.String,
        "commission": pl.Float64,
        "fill_sequences": pl.List(pl.UInt32),
        "selection_json": pl.String,
        "submission_policy": pl.String,
        "roll_state": pl.String,
    }
    decoded = [dict(row) for row in rows]
    for row in decoded:
        for name in _CLOCKS:
            if row[name] is not None and isinstance(row[name], str):
                row[name] = pd.Timestamp(row[name])
    for name in _CLOCKS:
        schema[name] = timestamp_dtype(row[name] for row in decoded)
    return temporal_frame(decoded, schema=schema, strict=True)
