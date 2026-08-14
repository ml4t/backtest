"""Declared result surfaces for the scenario comparison adapters."""

from __future__ import annotations

from common.types import CAPABILITY_KEYS

ML4T_CAPABILITIES = {
    "intents": "input_only",
    "orders": "unavailable",
    "rejections": "unavailable",
    "fills": "native",
    "positions": "unavailable",
    "cash_flows": "aggregate_only",
    "open_trades": "unavailable",
    "closed_trades": "native",
    "exit_reason": "native",
    "terminal": "native",
}

FRAMEWORK_CAPABILITIES = {
    "vectorbt_pro": {
        "intents": "input_only",
        "orders": "native_filled_only",
        "rejections": "unavailable",
        "fills": "native",
        "positions": "unavailable",
        "cash_flows": "aggregate_only",
        "open_trades": "unavailable",
        "closed_trades": "native",
        "exit_reason": "unavailable",
        "terminal": "native",
    },
    "vectorbt_oss": {
        "intents": "input_only",
        "orders": "native_filled_only",
        "rejections": "unavailable",
        "fills": "native",
        "positions": "unavailable",
        "cash_flows": "aggregate_only",
        "open_trades": "unavailable",
        "closed_trades": "native",
        "exit_reason": "unavailable",
        "terminal": "native",
    },
    "backtrader": {
        "intents": "input_only",
        "orders": "native_filled_only",
        "rejections": "unavailable",
        "fills": "native",
        "positions": "unavailable",
        "cash_flows": "aggregate_only",
        "open_trades": "unavailable",
        "closed_trades": "native",
        "exit_reason": "unavailable",
        "terminal": "native",
    },
    "zipline": {
        "intents": "input_only",
        "orders": "native_filled_only",
        "rejections": "unavailable",
        "fills": "reconstructed",
        "positions": "unavailable",
        "cash_flows": "aggregate_only",
        "open_trades": "unavailable",
        "closed_trades": "reconstructed",
        "exit_reason": "unavailable",
        "terminal": "native",
    },
}

assert set(ML4T_CAPABILITIES) == set(CAPABILITY_KEYS)
assert all(
    set(capabilities) == set(CAPABILITY_KEYS) for capabilities in FRAMEWORK_CAPABILITIES.values()
)
