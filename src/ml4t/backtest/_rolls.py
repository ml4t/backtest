"""Observed outcomes of paired, non-atomic dated-contract order intents."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from math import isfinite
from typing import Any

import pandas as pd
import polars as pl

from ._temporal import temporal_frame, temporal_rows, timestamp_dtype
from .corporate_actions import action_terms_json
from .types import OrderStatus

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


def roll_digest(frame: pl.DataFrame) -> str:
    return sha256(
        action_terms_json({"rolls": list(temporal_rows(frame, named=True))}).encode()
    ).hexdigest()


def validate_rolls(frame: pl.DataFrame, fills) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    order_ids = set()
    for row in temporal_rows(frame, named=True):
        if any(
            not isinstance(row[name], str) or not row[name].strip()
            for name in (
                "roll_id",
                "asset",
                "leg",
            )
        ) or row["leg"] not in {"close", "open"}:
            raise ValueError("Roll rows require known identities and close/open legs")
        if row["submission_policy"] != "paired_non_atomic":
            raise ValueError("Unknown roll submission policy")
        if any(row[name] is not None and not isinstance(row[name], datetime) for name in _CLOCKS):
            raise ValueError("Roll clocks require physical datetime instants")
        if (
            row["decision_at"] is None
            or row["submitted_at"] is None
            or (
                row["decision_at"] > row["submitted_at"]
                or row["available_at"] is not None
                and row["available_at"] > row["decision_at"]
            )
        ):
            raise ValueError("Roll rows violate their selection and submission clocks")
        if row["selection_json"] is not None and not isinstance(
            json.loads(row["selection_json"]), dict
        ):
            raise ValueError("Roll selection facts must be a JSON object")
        for name in (
            "quantity",
            "filled_quantity",
            "unfilled_requested_quantity",
            "overfilled_requested_quantity",
            "commission",
        ):
            if (
                row[name] is None
                or not isfinite(row[name])
                or name != "commission"
                and row[name] < 0
            ):
                raise ValueError("Roll quantities and actual fees must be finite")
        sequences = row["fill_sequences"]
        if (
            not isinstance(sequences, list)
            or any(
                not isinstance(sequence, int) or not 0 <= sequence < len(fills)
                for sequence in sequences
            )
            or sequences != sorted(set(sequences))
        ):
            raise ValueError("Roll fill sequences must identify actual ordered fills")
        actual = [fills[sequence] for sequence in sequences]
        if row["order_id"] is None:
            if sequences or row["status"] != "not_submitted":
                raise ValueError("An unsubmitted roll leg cannot claim fills")
        else:
            if row["order_id"] in order_ids or row["status"] not in {
                status.value for status in OrderStatus
            }:
                raise ValueError("Roll order identity or status is invalid")
            order_ids.add(row["order_id"])
            if sequences != [i for i, fill in enumerate(fills) if fill.order_id == row["order_id"]]:
                raise ValueError("Roll leg omits actual partial fills")
        if any(
            fill.order_id != row["order_id"]
            or fill.asset != row["asset"]
            or fill.side.value != row["side"]
            for fill in actual
        ):
            raise ValueError("Roll leg disagrees with its actual fills")
        if row["filled_quantity"] != sum(fill.quantity for fill in actual) or row[
            "commission"
        ] != sum(fill.commission for fill in actual):
            raise ValueError("Roll quantities or fees disagree with actual fills")
        if row["unfilled_requested_quantity"] != max(
            row["quantity"] - row["filled_quantity"], 0
        ) or row["overfilled_requested_quantity"] != max(
            row["filled_quantity"] - row["quantity"], 0
        ):
            raise ValueError("Roll requested and executed quantities do not reconcile")
        clocks = (
            actual[0].timestamp if actual else None,
            actual[-1].timestamp if actual else None,
            actual[-1].timestamp if actual and row["status"] == "filled" else None,
        )
        if tuple(row[name] for name in ("first_fill_at", "last_fill_at", "filled_at")) != clocks:
            raise ValueError("Roll fill clocks disagree with actual execution")
        groups.setdefault(row["roll_id"], []).append(row)
    for legs in groups.values():
        if (
            len(legs) != 2
            or [row["leg"] for row in legs] != ["close", "open"]
            or legs[0]["asset"] == legs[1]["asset"]
        ):
            raise ValueError("A roll requires two distinct ordered contract legs")
        if any(
            legs[0][name] != legs[1][name]
            for name in (
                "decision_at",
                "available_at",
                "submitted_at",
                "selection_json",
                "roll_state",
            )
        ):
            raise ValueError("Roll legs disagree on their original intent")
        state = (
            "no_position"
            if not legs[0]["quantity"]
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
        if any(row["roll_state"] != state for row in legs):
            raise ValueError("Roll completion status contradicts actual leg outcomes")
