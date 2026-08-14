#!/usr/bin/env python3
"""Measure current VectorBT behavior without importing ML4T comparison code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGETS = {
    "vectorbt_oss": {
        "package": "vectorbt",
        "version": "1.1.0",
        "commit": "259d2d89fe2e7638baf3ca76c394937cd32b656d",
        "repository": "https://github.com/polakowo/vectorbt",
    },
    "vectorbt_pro": {
        "package": "vectorbtpro",
        "version": "2026.6.27",
        "commit": "6e18cf0aa37849cfc20848f40f1d26ecfdc771b4",
        "repository": "https://github.com/polakowo/vectorbt.pro",
    },
}


def _bool(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def _orders(portfolio: Any) -> list[dict[str, Any]]:
    return portfolio.orders.records_readable.to_dict("records")


def _trades(portfolio: Any) -> list[dict[str, Any]]:
    return portfolio.trades.records_readable.to_dict("records")


def _cash(portfolio: Any) -> list[float]:
    value = portfolio.cash() if callable(portfolio.cash) else portfolio.cash
    return np.asarray(value, dtype=float).tolist()


def _source_location(package: Any, function: Any, target: dict[str, str]) -> dict:
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise RuntimeError(f"Could not locate source for {function}")
    package_root = Path(package.__file__).resolve().parent.parent
    relative_path = Path(source_file).resolve().relative_to(package_root).as_posix()
    line = inspect.getsourcelines(function)[1]
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "path": relative_path,
        "line": line,
        "source_url": (f"{target['repository']}/blob/{target['commit']}/{relative_path}#L{line}"),
    }


def _defaults(framework: str, vbt: Any) -> dict[str, Any]:
    portfolio = vbt.settings.portfolio
    if framework == "vectorbt_pro":
        signals = portfolio["from_signals"]
        return {
            "accumulate": signals["accumulate"],
            "allow_partial": portfolio["allow_partial"],
            "call_seq": portfolio["call_seq"],
            "cash_sharing": portfolio["cash_sharing"],
            "fees": portfolio["fees"],
            "ffill_val_price": portfolio["ffill_val_price"],
            "fillna_close": portfolio["fillna_close"],
            "fixed_fees": portfolio["fixed_fees"],
            "raise_reject": portfolio["raise_reject"],
            "slippage": portfolio["slippage"],
            "upon_dir_conflict": signals["upon_dir_conflict"],
            "upon_long_conflict": signals["upon_long_conflict"],
            "upon_opposite_entry": signals["upon_opposite_entry"],
            "upon_short_conflict": signals["upon_short_conflict"],
            "update_value": portfolio["update_value"],
        }
    return {
        key: portfolio[key]
        for key in (
            "accumulate",
            "allow_partial",
            "call_seq",
            "cash_sharing",
            "fees",
            "ffill_val_price",
            "fillna_close",
            "fixed_fees",
            "raise_reject",
            "slippage",
            "upon_dir_conflict",
            "upon_long_conflict",
            "upon_opposite_entry",
            "upon_short_conflict",
            "update_value",
        )
    }


def _expected_defaults(framework: str) -> dict[str, Any]:
    return {
        "accumulate": False,
        "allow_partial": True,
        "call_seq": None if framework == "vectorbt_pro" else "default",
        "cash_sharing": False,
        "fees": 0.0,
        "ffill_val_price": True,
        "fillna_close": True,
        "fixed_fees": 0.0,
        "raise_reject": False,
        "slippage": 0.0,
        "upon_dir_conflict": "ignore",
        "upon_long_conflict": "ignore",
        "upon_opposite_entry": "reversereduce",
        "upon_short_conflict": "ignore",
        "update_value": False,
    }


def _behavior_checks(framework: str, vbt: Any) -> list[tuple[str, Any, Any]]:
    index = pd.date_range("2024-01-01", periods=3)
    close = pd.Series([100.0, 110.0, 120.0], index=index)
    entries = _bool([1, 0, 0])
    exits = _bool([0, 1, 0])

    signal_pf = vbt.Portfolio.from_signals(
        close, entries=entries, exits=exits, size=1, init_cash=1_000
    )
    signal_orders = _orders(signal_pf)
    time_key = "Fill Index" if framework == "vectorbt_pro" else "Timestamp"

    explicit_pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        price=pd.Series([99.0, 109.0, 119.0], index=index),
        size=1,
        init_cash=1_000,
    )
    target_pf = vbt.Portfolio.from_orders(
        pd.Series([100.0, 200.0], index=index[:2]),
        size=[0.5, 0.0],
        size_type="targetpercent",
        init_cash=1_000,
    )
    shared_close = pd.DataFrame([[100.0, 100.0]], index=index[:1], columns=pd.Index(["a", "b"]))
    shared_default = vbt.Portfolio.from_orders(
        shared_close,
        size=1,
        init_cash=100,
        cash_sharing=True,
        group_by=True,
    )
    shared_reversed = vbt.Portfolio.from_orders(
        shared_close,
        size=1,
        init_cash=100,
        cash_sharing=True,
        group_by=True,
        call_seq="reversed",
    )
    accumulate_default = vbt.Portfolio.from_signals(
        close, entries=_bool([1, 1, 0]), exits=False, size=1, init_cash=1_000
    )
    accumulate_true = vbt.Portfolio.from_signals(
        close,
        entries=_bool([1, 1, 0]),
        exits=False,
        size=1,
        accumulate=True,
        init_cash=1_000,
    )
    conflict_kwargs = {
        "entries": _bool([1, 1, 0]),
        "exits": _bool([0, 1, 0]),
        "size": 1,
        "accumulate": True,
        "init_cash": 1_000,
    }
    conflict_default = vbt.Portfolio.from_signals(close, **conflict_kwargs)
    conflict_entry = vbt.Portfolio.from_signals(
        close, **conflict_kwargs, upon_long_conflict="entry"
    )
    conflict_exit = vbt.Portfolio.from_signals(close, **conflict_kwargs, upon_long_conflict="exit")
    short_pf = vbt.Portfolio.from_signals(
        close,
        short_entries=entries,
        short_exits=exits,
        size=1,
        init_cash=1_000,
    )
    stop_pf = vbt.Portfolio.from_signals(
        pd.Series([100.0, 95.0, 100.0], index=index),
        open=pd.Series([100.0, 100.0, 100.0], index=index),
        high=pd.Series([101.0, 101.0, 101.0], index=index),
        low=pd.Series([99.0, 85.0, 99.0], index=index),
        entries=entries,
        size=1,
        sl_stop=0.1,
        init_cash=1_000,
    )
    trailing_kwargs: dict[str, Any] = {
        "close": pd.Series(
            [100.0, 110.0, 107.0, 107.0], index=pd.date_range("2024-01-01", periods=4)
        ),
        "open": pd.Series(
            [100.0, 110.0, 109.0, 107.0], index=pd.date_range("2024-01-01", periods=4)
        ),
        "high": pd.Series(
            [101.0, 120.0, 110.0, 108.0], index=pd.date_range("2024-01-01", periods=4)
        ),
        "low": pd.Series([99.0, 109.0, 105.0, 106.0], index=pd.date_range("2024-01-01", periods=4)),
        "entries": _bool([1, 0, 0, 0]),
        "size": 1,
        "init_cash": 1_000,
    }
    if framework == "vectorbt_pro":
        trailing_kwargs["tsl_stop"] = 0.1
    else:
        trailing_kwargs["sl_stop"] = 0.1
        trailing_kwargs["sl_trail"] = True
    trailing_pf = vbt.Portfolio.from_signals(**trailing_kwargs)
    cost_pf = vbt.Portfolio.from_signals(
        pd.Series([100.0, 100.0], index=index[:2]),
        entries=_bool([1, 0]),
        exits=_bool([0, 1]),
        size=1,
        fees=0.01,
        fixed_fees=2,
        slippage=0.01,
        init_cash=1_000,
    )
    missing_pf = vbt.Portfolio.from_signals(
        pd.Series([100.0, np.nan, 110.0], index=index),
        entries=entries,
        exits=exits,
        size=1,
        init_cash=1_000,
    )
    partial_pf = vbt.Portfolio.from_signals(
        pd.Series([100.0]),
        entries=_bool([1]),
        size=2,
        init_cash=100,
    )
    trade = _trades(signal_pf)[0]
    dtype_result: str
    try:
        vbt.Portfolio.from_signals(
            close, entries=np.asarray([1, 0, 0]), exits=np.asarray([0, 1, 0]), size=1
        )
        dtype_result = "accepted"
    except AssertionError:
        dtype_result = "rejected"

    return [
        ("defaults", _defaults(framework, vbt), _expected_defaults(framework)),
        (
            "signal_timing_and_default_fill",
            {
                "timestamps": [str(order[time_key]) for order in signal_orders],
                "prices": [order["Price"] for order in signal_orders],
            },
            {
                "timestamps": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                "prices": [100.0, 110.0],
            },
        ),
        (
            "explicit_fill_price",
            [order["Price"] for order in _orders(explicit_pf)],
            [99.0, 109.0],
        ),
        (
            "target_percent_sizing",
            [order["Size"] for order in _orders(target_pf)],
            [5.0, 5.0],
        ),
        (
            "cash_sharing_and_call_sequence",
            {
                "default": [_orders(shared_default)[0]["Column"]],
                "reversed": [_orders(shared_reversed)[0]["Column"]],
            },
            {"default": ["a"], "reversed": ["b"]},
        ),
        (
            "accumulation",
            {
                "default_orders": len(_orders(accumulate_default)),
                "enabled_orders": len(_orders(accumulate_true)),
            },
            {"default_orders": 1, "enabled_orders": 2},
        ),
        (
            "long_signal_conflict",
            {
                "default": [order["Side"] for order in _orders(conflict_default)],
                "entry": [order["Side"] for order in _orders(conflict_entry)],
                "exit": [order["Side"] for order in _orders(conflict_exit)],
            },
            {"default": ["Buy"], "entry": ["Buy", "Buy"], "exit": ["Buy", "Sell"]},
        ),
        (
            "short_cash",
            {"cash": _cash(short_pf)[:2], "sides": [order["Side"] for order in _orders(short_pf)]},
            {"cash": [1_100.0, 990.0], "sides": ["Sell", "Buy"]},
        ),
        ("stop_fill", [order["Price"] for order in _orders(stop_pf)], [100.0, 90.0]),
        (
            "trailing_stop_extreme_and_intrabar_fill",
            [order["Price"] for order in _orders(trailing_pf)],
            [100.0, 108.0],
        ),
        (
            "fees_and_slippage",
            {
                "fees": [order["Fees"] for order in _orders(cost_pf)],
                "prices": [order["Price"] for order in _orders(cost_pf)],
            },
            {"fees": [3.01, 2.99], "prices": [101.0, 99.0]},
        ),
        (
            "missing_order_price",
            {"order_count": len(_orders(missing_pf)), "cash": _cash(missing_pf)},
            {"order_count": 1, "cash": [900.0, 900.0, 900.0]},
        ),
        (
            "insufficient_cash_partial_fill",
            [order["Size"] for order in _orders(partial_pf)],
            [1.0],
        ),
        (
            "record_construction",
            {
                "direction": trade["Direction"],
                "order_ids": [order["Order Id"] for order in signal_orders],
                "pnl": trade["PnL"],
                "status": trade["Status"],
                "trade_count": len(_trades(signal_pf)),
            },
            {
                "direction": "Long",
                "order_ids": [0, 1],
                "pnl": 10.0,
                "status": "Closed",
                "trade_count": 1,
            },
        ),
        (
            "integer_signal_dtype",
            dtype_result,
            "rejected" if framework == "vectorbt_pro" else "accepted",
        ),
    ]


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, (float, int)):
        return math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_equal(left, right) for left, right in zip(actual, expected, strict=True))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_equal(actual[key], expected[key]) for key in actual)
    return bool(actual == expected)


def run(framework: str) -> dict[str, Any]:
    """Execute the frozen framework directly and return native evidence."""
    target = TARGETS[framework]
    vbt = importlib.import_module(target["package"])
    version = importlib.metadata.version(target["package"])
    if version != target["version"]:
        raise RuntimeError(f"{target['package']} version differs: {version} != {target['version']}")

    source_locations = {
        "from_orders": _source_location(vbt, vbt.Portfolio.from_orders, target),
        "from_signals": _source_location(vbt, vbt.Portfolio.from_signals, target),
    }
    checks: list[dict[str, Any]] = []
    for check_id, actual, expected in _behavior_checks(framework, vbt):
        checks.append(
            {
                "id": check_id,
                "actual": actual,
                "expected": expected,
                "passed": _equal(actual, expected),
                "source": "from_orders"
                if check_id in {"target_percent_sizing", "cash_sharing_and_call_sequence"}
                else "from_signals",
            }
        )
    return {
        "schema_version": 1,
        "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "framework": framework,
        "package": target["package"],
        "version": version,
        "source_commit": target["commit"],
        "source_locations": source_locations,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=tuple(TARGETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = run(args.framework)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"VectorBT native behavior run failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [check["id"] for check in evidence["checks"] if not check["passed"]]
    if failed:
        print(f"VectorBT native behavior differs: {failed}", file=sys.stderr)
        return 1
    print(f"VectorBT native behavior passed: {len(evidence['checks'])} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
