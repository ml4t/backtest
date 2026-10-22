"""Versioned action tables, input lineage and observed-account reconciliation."""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from typing import Any

import polars as pl

from ._temporal import temporal_frame, temporal_rows, timestamp_dtype
from .corporate_actions import action_terms_json

_ACTION_SCHEMA = {
    "timestamp": pl.Datetime,
    "source": pl.String,
    "event_id": pl.String,
    "revision_id": pl.String,
    "asset": pl.String,
    "action_kind": pl.String,
    "operation": pl.String,
    "available_at": pl.Datetime,
    "effective_at": pl.Datetime,
    "payment_at": pl.Datetime,
    "terms_json": pl.String,
    "order_id": pl.String,
    "quantity_before": pl.Float64,
    "quantity_after": pl.Float64,
    "cash_delta": pl.Float64,
    "realized_pnl": pl.Float64,
    "allocated_commission": pl.Float64,
}
_ENTITLEMENT_SCHEMA = {
    "source": pl.String,
    "event_id": pl.String,
    "revision_id": pl.String,
    "asset": pl.String,
    "action_kind": pl.String,
    "entitlement_at": pl.Datetime,
    "payment_at": pl.Datetime,
    "eligible_quantity": pl.Float64,
    "cash_amount": pl.Float64,
    "destination_asset": pl.String,
    "security_quantity": pl.Float64,
    "currency": pl.String,
    "paid": pl.Boolean,
    "destination_basis": pl.Float64,
    "receivable_value": pl.Float64,
    "destination_commission": pl.Float64,
    "delivered_position_quantity": pl.Float64,
    "delivered_position_basis": pl.Float64,
    "delivery_fill_count": pl.UInt32,
}


def action_frame(rows: list[dict[str, Any]], *, entitlements: bool = False) -> pl.DataFrame:
    schema = dict(_ENTITLEMENT_SCHEMA if entitlements else _ACTION_SCHEMA)
    for name, dtype in schema.items():
        if dtype is pl.Datetime:
            schema[name] = timestamp_dtype(row.get(name) for row in rows)
    return temporal_frame(rows, schema=schema, strict=True)


def action_digest(
    actions: pl.DataFrame, entitlements: pl.DataFrame, receivables: list[float]
) -> str:
    payload = {
        "actions": list(temporal_rows(actions, named=True)),
        "entitlements": list(temporal_rows(entitlements, named=True)),
        "observed_receivables": receivables,
    }
    return sha256(action_terms_json(payload).encode()).hexdigest()


def validate_action_tables(
    actions: pl.DataFrame,
    entitlements: pl.DataFrame,
    accounts: pl.DataFrame,
    receivables: list[float],
    *,
    account_method: str | None = None,
) -> None:
    if len(receivables) != accounts.height or any(not isfinite(value) for value in receivables):
        raise ValueError("Corporate receivables require finite actual account observations")
    for row, value in zip(temporal_rows(accounts, named=True), receivables, strict=True):
        positions = (
            row["positions_value"] if account_method == "per_instrument" else row["net_exposure"]
        )
        components = row["cash"] + positions + value
        if abs(row["equity"] - components) > max(1e-9, abs(row["equity"]) * 1e-14):
            raise ValueError("Account NAV does not reconcile with cash, holdings and receivables")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in temporal_rows(actions, named=True):
        key = row["source"], row["event_id"]
        if any(
            not isinstance(row[name], str) or not row[name]
            for name in ["source", "event_id", "revision_id", "asset"]
        ):
            raise ValueError(
                "Action application requires nonempty source/event/revision/asset identity"
            )
        if row["operation"] not in {
            "split",
            "order_adjustment",
            "order_cancellation",
            "entitlement",
            "payment",
            "merger",
        }:
            raise ValueError("Unknown action application operation")
        terms = json.loads(row["terms_json"])
        if not isinstance(terms, dict):
            raise ValueError("Corporate action terms must be a JSON object")
        if any(
            terms.get(name) != row[name]
            for name in ["source", "event_id", "revision_id", "asset", "action_kind"]
        ):
            raise ValueError("Action input lineage disagrees with its application")
        expected = row["payment_at"] if row["operation"] == "payment" else row["effective_at"]
        if (
            expected is None
            or row["timestamp"] != expected
            or row["available_at"] > row["effective_at"]
        ):
            raise ValueError("Action application violates its declared causal clock")
        for name in ["cash_delta", "quantity_before", "quantity_after", "realized_pnl"]:
            if row[name] is not None and not isfinite(row[name]):
                raise ValueError("Action application contains nonfinite financial values")
        grouped.setdefault(key, []).append(row)
    seen = set()
    for entitlement in temporal_rows(entitlements, named=True):
        key = entitlement["source"], entitlement["event_id"]
        if key in seen or key not in grouped:
            raise ValueError("Entitlement identity must be unique and refer to an actual action")
        seen.add(key)
        records = grouped[key]
        earned = [row for row in records if row["operation"] in {"entitlement", "merger"}]
        paid = [row for row in records if row["operation"] == "payment"]
        if len(earned) != 1 or len(paid) != int(entitlement["paid"]):
            raise ValueError("Entitlement/payment ledger is incomplete or applied more than once")
        if paid and paid[0]["cash_delta"] != entitlement["cash_amount"]:
            raise ValueError("Payment cash does not reconcile with its earned entitlement")
        if any(
            entitlement[name] is None or not isfinite(entitlement[name])
            for name in [
                "eligible_quantity",
                "cash_amount",
                "security_quantity",
                "receivable_value",
            ]
        ):
            raise ValueError("Entitlement values must be finite and captured")
    for key, records in grouped.items():
        for operation in ["split", "entitlement", "merger", "payment"]:
            if sum(row["operation"] == operation for row in records) > 1:
                raise ValueError("Action was applied more than once")
        if (
            any(row["operation"] in {"entitlement", "payment"} for row in records)
            and key not in seen
        ):
            raise ValueError("Distribution applications require a captured entitlement")
