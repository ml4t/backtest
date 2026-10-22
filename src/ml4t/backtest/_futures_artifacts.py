"""Portable observed variation cash and collateral, separate from notional exposure."""

from __future__ import annotations

from math import isfinite
from typing import Any

import polars as pl

from ._temporal import temporal_frame, temporal_rows, timestamp_dtype
from .corporate_actions import action_terms_json

ACCOUNTING_COLUMNS = {
    "positions_value": pl.Float64,
    "initial_margin_required": pl.Float64,
    "maintenance_margin_required": pl.Float64,
    "available_funds": pl.Float64,
    "futures_unsettled_pnl": pl.Float64,
    "settlement_holds": pl.Float64,
    "cash_buffer": pl.Float64,
    "margin_breach": pl.Boolean,
}
_SETTLEMENT_SCHEMA = {
    "timestamp": pl.Datetime,
    "available_at": pl.Datetime,
    "asset": pl.String,
    "source": pl.String,
    "event_id": pl.String,
    "revision_id": pl.String,
    "phase": pl.String,
    "quantity": pl.Float64,
    "basis_before": pl.Float64,
    "settlement_price": pl.Float64,
    "multiplier": pl.Float64,
    "currency": pl.String,
    "cash_delta": pl.Float64,
    "fill_sequence": pl.UInt32,
}


def settlement_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = dict(_SETTLEMENT_SCHEMA)
    for name in ("timestamp", "available_at"):
        schema[name] = timestamp_dtype(row[name] for row in rows)
    return temporal_frame(rows, schema=schema, strict=True)


def validate_instrument_accounting(declarations: Any) -> None:
    if not isinstance(declarations, dict) or not declarations:
        raise ValueError("Per-instrument accounting requires declared variation instruments")
    for asset, terms in declarations.items():
        if not isinstance(asset, str) or not asset.strip() or not isinstance(terms, dict):
            raise ValueError("Invalid variation instrument identity")
        if terms.get("accounting_method") != "futures_variation" or terms.get(
            "settlement_policy"
        ) not in {"scheduled", "on_mark"}:
            raise ValueError("Unknown per-instrument futures accounting method")
        currency, multiplier = terms.get("currency"), terms.get("multiplier")
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError("Variation instruments require a declared currency")
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not isfinite(multiplier)
            or multiplier <= 0
        ):
            raise ValueError("Variation instrument multiplier must be finite and positive")


def futures_payload(declarations, accounts, settlements) -> str:
    return action_terms_json(
        {
            "instruments": declarations,
            "accounts": list(temporal_rows(accounts, named=True)),
            "settlements": list(temporal_rows(settlements, named=True)),
        }
    )


def validate_futures_tables(declarations, accounts, settlements, fills) -> None:
    validate_instrument_accounting(declarations)
    if not set(ACCOUNTING_COLUMNS) <= set(accounts.columns):
        raise ValueError("Variation account requires actual observed liquidation and margin")
    for row in temporal_rows(accounts, named=True):
        for name, dtype in ACCOUNTING_COLUMNS.items():
            value = row[name]
            if dtype == pl.Boolean:
                if not isinstance(value, bool):
                    raise ValueError("Margin breach must be an observed boolean")
            elif value is None or not isfinite(value):
                raise ValueError("Variation account observations must be finite")
        components = row["cash"] + row["positions_value"] + row.get("receivables_value", 0)
        available = (
            row["equity"]
            - row.get("receivables_value", 0)
            - row["initial_margin_required"]
            - row["settlement_holds"]
            - row["cash_buffer"]
        )
        tolerance = max(1e-9, abs(row["equity"]) * 1e-14)
        if (
            abs(row["equity"] - components) > tolerance
            or abs(row["available_funds"] - available) > tolerance
        ):
            raise ValueError("Variation NAV and available funds do not reconcile")
        if any(
            row[name] < 0
            for name in (
                "initial_margin_required",
                "maintenance_margin_required",
                "settlement_holds",
                "cash_buffer",
            )
        ) or row["margin_breach"] != (row["equity"] < row["maintenance_margin_required"]):
            raise ValueError("Variation margin state is inconsistent")
    seen = set()
    for row in temporal_rows(settlements, named=True):
        key = row["source"], row["event_id"]
        if key in seen or any(
            not isinstance(row[name], str) or not row[name].strip()
            for name in ("source", "event_id", "revision_id")
        ):
            raise ValueError("Futures settlement identity is missing or duplicated")
        seen.add(key)
        terms = declarations.get(row["asset"])
        if terms is None or any(row[name] != terms[name] for name in ("multiplier", "currency")):
            raise ValueError("Futures settlement disagrees with its declared instrument")
        if (
            row["available_at"] is None
            or row["timestamp"] is None
            or row["available_at"] > row["timestamp"]
        ):
            raise ValueError("Futures settlement cannot precede publication")
        if (
            any(
                row[name] is None or not isfinite(row[name])
                for name in ("quantity", "settlement_price", "cash_delta")
            )
            or row["basis_before"] is not None
            and not isfinite(row["basis_before"])
        ):
            raise ValueError("Futures settlement values must be finite")
        if row["quantity"] and row["basis_before"] is None:
            raise ValueError("Exposed futures settlement requires an observed basis")
        expected = (
            row["quantity"] * (row["settlement_price"] - row["basis_before"]) * row["multiplier"]
            if row["quantity"]
            else 0
        )
        if row["cash_delta"] != expected:
            raise ValueError(
                "Futures settlement cash does not match its observed quantity and basis"
            )
        phase, sequence = row["phase"], row["fill_sequence"]
        if phase in {"close", "fill_mark"}:
            if not isinstance(sequence, int) or not 0 <= sequence < len(fills):
                raise ValueError("Futures fill settlement requires its actual fill")
            fill = fills[sequence]
            if (
                fill.asset != row["asset"]
                or fill.timestamp != row["timestamp"]
                or fill.price != row["settlement_price"]
            ):
                raise ValueError("Futures payout does not refer to its actual fill")
            if phase != ("fill_mark" if terms["settlement_policy"] == "on_mark" else "close"):
                raise ValueError("Futures fill payout policy disagrees with its instrument")
        elif sequence is not None or (
            phase != "scheduled"
            if terms["settlement_policy"] == "scheduled"
            else phase not in {"before_callbacks", "after_fills"}
        ):
            raise ValueError("Futures settlement phase disagrees with its declared policy")
