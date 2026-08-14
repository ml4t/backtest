#!/usr/bin/env python3
"""Measure frozen Zipline Reloaded behavior without ML4T comparison code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = "3.1.1"
COMMIT = "09885a2ebc7567d40942c891b3879dc03c745070"
ARTIFACT = "zipline_reloaded-3.1.1.tar.gz"
ARTIFACT_SHA256 = "4a305524616f7aad836f929e5a2ba5afc7db0e238757f47eb49487d9e2457a6f"
REPOSITORY = "https://github.com/stefan-jansen/zipline-reloaded"

zipline: Any = importlib.import_module("zipline")
api: Any = importlib.import_module("zipline.api")
commission: Any = importlib.import_module("zipline.finance.commission")
slippage: Any = importlib.import_module("zipline.finance.slippage")
blotter_module: Any = importlib.import_module("zipline.finance.blotter.simulation_blotter")
algorithm_module: Any = importlib.import_module("zipline.algorithm")
calendar_module: Any = importlib.import_module("exchange_calendars")
bundles_module: Any = importlib.import_module("zipline.data.bundles")


def _source_location(function: Any) -> dict[str, Any]:
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise RuntimeError(f"Could not locate source for {function}")
    package_root = Path(zipline.__file__).resolve().parent.parent
    relative_path = Path(source_file).resolve().relative_to(package_root).as_posix()
    line = inspect.getsourcelines(function)[1]
    return {
        "artifact": ARTIFACT,
        "artifact_sha256": ARTIFACT_SHA256,
        "line": line,
        "module": function.__module__,
        "path": relative_path,
        "qualname": function.__qualname__,
        "source_url": f"{REPOSITORY}/blob/{COMMIT}/{relative_path}#L{line}",
    }


def _frame() -> pd.DataFrame:
    calendar = calendar_module.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2024-01-02", "2024-01-12")[:8]
    return pd.DataFrame(
        {
            "open": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "high": [101.0, 121.0, 131.0, 141.0, 151.0, 161.0, 171.0, 181.0],
            "low": [99.0, 109.0, 119.0, 129.0, 139.0, 149.0, 159.0, 169.0],
            "close": [100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0],
            "volume": [100.0] * 8,
        },
        index=pd.DatetimeIndex(sessions).tz_localize(None),
    )


def _setup_bundle(frame: pd.DataFrame, name: str) -> str:
    def ingest_function(
        _environ: Any,
        asset_db_writer: Any,
        _minute_bar_writer: Any,
        daily_bar_writer: Any,
        adjustment_writer: Any,
        calendar: Any,
        start_session: Any,
        end_session: Any,
        _cache: Any,
        show_progress: bool,
        _output_dir: Any,
    ) -> None:
        sessions = calendar.sessions_in_range(start_session, end_session)
        trading_frame = frame.loc[frame.index.isin(pd.DatetimeIndex(sessions).tz_localize(None))]
        asset_db_writer.write(
            equities=pd.DataFrame(
                {
                    "symbol": ["TEST"],
                    "asset_name": ["Native behavior asset"],
                    "exchange": ["NYSE"],
                }
            )
        )
        daily_bar_writer.write(
            [(0, trading_frame[["open", "high", "low", "close", "volume"]])],
            show_progress=show_progress,
        )
        adjustment_writer.write()

    if name in bundles_module.bundles:
        bundles_module.unregister(name)
    bundles_module.register(
        name,
        ingest_function,
        calendar_name="XNYS",
        start_session=frame.index[0],
        end_session=frame.index[-1],
    )
    bundles_module.ingest(name, show_progress=False)
    return name


def _run(
    bundle: str,
    frame: pd.DataFrame,
    initialize_case: Any,
    handle_case: Any,
) -> pd.DataFrame:
    def initialize(context: Any) -> None:
        context.asset = api.symbol("TEST")
        context.bar_number = 0
        initialize_case(context)

    def handle_data(context: Any, data: Any) -> None:
        handle_case(context, data)
        context.bar_number += 1

    return zipline.run_algorithm(
        start=frame.index[0],
        end=frame.index[-1],
        initialize=initialize,
        handle_data=handle_data,
        analyze=lambda _context, _performance: None,
        capital_base=1_000.0,
        bundle=bundle,
        data_frequency="daily",
    )


class OpenPriceSlippage(slippage.SlippageModel):
    """Explicit comparison protocol: next-session open with no volume cap."""

    @staticmethod
    def process_order(data: Any, order: Any) -> tuple[float, int]:
        return data.current(order.asset, "open"), order.open_amount


def _configure_open(_context: Any) -> None:
    api.set_commission(commission.NoCommission())
    api.set_slippage(OpenPriceSlippage())


def _transactions(results: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for transactions in results["transactions"]:
        for transaction in transactions:
            records.append(
                {
                    "amount": int(transaction["amount"]),
                    "date": pd.Timestamp(transaction["dt"]).date().isoformat(),
                    "price": float(transaction["price"]),
                    "transaction_commission": float(transaction["commission"] or 0.0),
                }
            )
    return records


def _orders(results: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for orders in results["orders"]:
        for order in orders:
            records.append(
                {
                    "amount": int(order["amount"]),
                    "date": pd.Timestamp(order["dt"]).date().isoformat(),
                    "filled": int(order["filled"]),
                    "status": int(order["status"]),
                }
            )
    return records


def _submitted_amounts(results: pd.DataFrame) -> list[int]:
    seen: set[str] = set()
    amounts: list[int] = []
    for orders in results["orders"]:
        for order in orders:
            order_id = str(order["id"])
            if order_id not in seen:
                seen.add(order_id)
                amounts.append(int(order["amount"]))
    return amounts


def _default_models() -> dict[str, Any]:
    broker = blotter_module.SimulationBlotter()
    equity_type = importlib.import_module("zipline.assets").Equity
    default_slippage = broker.slippage_models[equity_type]
    default_commission = broker.commission_models[equity_type]
    volume_share = slippage.VolumeShareSlippage()
    return {
        "commission_class": type(default_commission).__name__,
        "commission_per_share": default_commission.cost_per_share,
        "commission_minimum": default_commission.min_trade_cost,
        "slippage_class": type(default_slippage).__name__,
        "slippage_basis_points": default_slippage.basis_points,
        "slippage_volume_limit": default_slippage.volume_limit,
        "volume_share_price_impact": volume_share.price_impact,
        "volume_share_volume_limit": volume_share.volume_limit,
    }


def _default_fill(bundle: str, frame: pd.DataFrame) -> dict[str, Any]:
    results = _run(
        bundle,
        frame,
        lambda _context: None,
        lambda context, _data: api.order(context.asset, 1) if context.bar_number == 0 else None,
    )
    return {
        "cash_after_fill": float(results["ending_cash"].iloc[1]),
        "transactions": _transactions(results),
    }


def _explicit_minimum_commission(bundle: str, frame: pd.DataFrame) -> dict[str, Any]:
    def configure(_context: Any) -> None:
        api.set_commission(commission.PerShare(cost=0.005, min_trade_cost=1.0))
        api.set_slippage(OpenPriceSlippage())

    results = _run(
        bundle,
        frame,
        configure,
        lambda context, _data: api.order(context.asset, 1) if context.bar_number == 0 else None,
    )
    return {
        "cash_after_fill": float(results["ending_cash"].iloc[1]),
        "transactions": _transactions(results),
    }


def _open_fill(bundle: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    results = _run(
        bundle,
        frame,
        _configure_open,
        lambda context, _data: api.order(context.asset, 1) if context.bar_number == 0 else None,
    )
    return _transactions(results)


def _target_percent(bundle: str, frame: pd.DataFrame) -> dict[str, Any]:
    results = _run(
        bundle,
        frame,
        _configure_open,
        lambda context, _data: api.order_target_percent(context.asset, 0.5)
        if context.bar_number == 0
        else None,
    )
    return {"orders": _orders(results), "transactions": _transactions(results)}


def _target_percent_snapshot(bundle: str, frame: pd.DataFrame) -> dict[str, Any]:
    def handle(context: Any, _data: Any) -> None:
        if context.bar_number == 0:
            api.order_target_percent(context.asset, 0.5)
            api.order_target_percent(context.asset, 0.5)

    results = _run(bundle, frame, _configure_open, handle)
    return {
        "submitted_amounts": _submitted_amounts(results),
        "transactions": _transactions(results),
    }


def _submission_sequence(bundle: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    def handle(context: Any, _data: Any) -> None:
        if context.bar_number == 0:
            api.order(context.asset, 1)
            api.order(context.asset, -1)

    return _transactions(_run(bundle, frame, _configure_open, handle))


def _cash_and_short(bundle: str, frame: pd.DataFrame) -> dict[str, Any]:
    long_results = _run(
        bundle,
        frame,
        _configure_open,
        lambda context, _data: api.order(context.asset, 20) if context.bar_number == 0 else None,
    )
    short_results = _run(
        bundle,
        frame,
        _configure_open,
        lambda context, _data: api.order(context.asset, -1) if context.bar_number == 0 else None,
    )
    return {
        "long": {
            "cash_after_fill": float(long_results["ending_cash"].iloc[1]),
            "position_after_fill": int(long_results["positions"].iloc[1][0]["amount"]),
            "transactions": _transactions(long_results),
        },
        "short": {
            "cash_after_fill": float(short_results["ending_cash"].iloc[1]),
            "position_after_fill": int(short_results["positions"].iloc[1][0]["amount"]),
            "transactions": _transactions(short_results),
        },
    }


def _volume_share_partial(bundle: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    def configure(_context: Any) -> None:
        api.set_commission(commission.NoCommission())
        api.set_slippage(slippage.VolumeShareSlippage())

    results = _run(
        bundle,
        frame,
        configure,
        lambda context, _data: api.order(context.asset, 5) if context.bar_number == 0 else None,
    )
    return _transactions(results)


def _round_trip_records(bundle: str, frame: pd.DataFrame) -> dict[str, Any]:
    def handle(context: Any, _data: Any) -> None:
        if context.bar_number == 0:
            api.order(context.asset, 2)
        elif context.bar_number == 2:
            api.order_target(context.asset, 0)

    results = _run(bundle, frame, _configure_open, handle)
    return {
        "transactions": _transactions(results),
        "native_output_columns": sorted(
            column for column in ("orders", "positions", "transactions") if column in results
        ),
        "native_trade_column": "trades" in results,
    }


def _final_bar_order(bundle: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    results = _run(
        bundle,
        frame,
        _configure_open,
        lambda context, _data: api.order(context.asset, 1)
        if context.bar_number == len(frame) - 1
        else None,
    )
    return _transactions(results)


def _missing_bar(frame: pd.DataFrame) -> dict[str, Any]:
    missing_frame = frame.copy()
    missing_frame.loc[frame.index[2], ["open", "high", "low", "close"]] = float("nan")
    missing_frame.loc[frame.index[2], "volume"] = 0.0
    bundle = _setup_bundle(missing_frame, "ml4t_native_zipline_311_missing")

    def handle(context: Any, data: Any) -> None:
        api.record(
            observed_close=data.current(context.asset, "close"),
            observed_price=data.current(context.asset, "price"),
            observed_stale=data.is_stale(context.asset),
            observed_volume=data.current(context.asset, "volume"),
        )
        if context.bar_number == 1:
            api.order(context.asset, 1)

    results = _run(bundle, frame, _configure_open, handle)
    observations = []
    for date, row in results.iterrows():
        observations.append(
            {
                "close": None if pd.isna(row["observed_close"]) else float(row["observed_close"]),
                "date": pd.Timestamp(str(date)).date().isoformat(),
                "price": None if pd.isna(row["observed_price"]) else float(row["observed_price"]),
                "stale": bool(row["observed_stale"]),
                "volume": float(row["observed_volume"]),
            }
        )
    return {"observations": observations, "transactions": _transactions(results)}


def _expected_missing_bar(frame: pd.DataFrame) -> dict[str, Any]:
    observations = []
    closes = [100.0, 120.0, None, 140.0, 150.0, 160.0, 170.0, 180.0]
    prices = [100.0, 120.0, 120.0, 140.0, 150.0, 160.0, 170.0, 180.0]
    for index, date in enumerate(frame.index):
        observations.append(
            {
                "close": closes[index],
                "date": date.date().isoformat(),
                "price": prices[index],
                "stale": index == 2,
                "volume": 0.0 if index == 2 else 100.0,
            }
        )
    return {
        "observations": observations,
        "transactions": [
            {
                "amount": 1,
                "date": frame.index[3].date().isoformat(),
                "price": 130.0,
                "transaction_commission": 0.0,
            }
        ],
    }


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, (float, int)):
        return math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-10)
    if isinstance(actual, list) and isinstance(expected, list) and len(actual) == len(expected):
        return all(_equal(left, right) for left, right in zip(actual, expected, strict=True))
    if isinstance(actual, dict) and isinstance(expected, dict) and actual.keys() == expected.keys():
        return all(_equal(actual[key], expected[key]) for key in actual)
    return bool(actual == expected)


def run() -> dict[str, Any]:
    actual_version = importlib.metadata.version("zipline-reloaded")
    if actual_version != VERSION:
        raise RuntimeError(f"zipline-reloaded version differs: {actual_version} != {VERSION}")

    frame = _frame()
    bundle = _setup_bundle(frame, "ml4t_native_zipline_311")
    measurements = [
        (
            "defaults",
            _default_models(),
            {
                "commission_class": "PerShare",
                "commission_per_share": 0.001,
                "commission_minimum": 0,
                "slippage_class": "FixedBasisPointsSlippage",
                "slippage_basis_points": 5.0,
                "slippage_volume_limit": 0.1,
                "volume_share_price_impact": 0.1,
                "volume_share_volume_limit": 0.025,
            },
        ),
        (
            "default_next_bar_close_fill",
            _default_fill(bundle, frame),
            {
                "cash_after_fill": 879.9390000000001,
                "transactions": [
                    {
                        "amount": 1,
                        "date": frame.index[1].date().isoformat(),
                        "price": 120.06,
                        "transaction_commission": 0.0,
                    }
                ],
            },
        ),
        (
            "configured_next_bar_open_fill",
            _open_fill(bundle, frame),
            [
                {
                    "amount": 1,
                    "date": frame.index[1].date().isoformat(),
                    "price": 110.0,
                    "transaction_commission": 0.0,
                }
            ],
        ),
        (
            "explicit_minimum_commission",
            _explicit_minimum_commission(bundle, frame),
            {
                "cash_after_fill": 889.0,
                "transactions": [
                    {
                        "amount": 1,
                        "date": frame.index[1].date().isoformat(),
                        "price": 110.0,
                        "transaction_commission": 0.0,
                    }
                ],
            },
        ),
        (
            "integer_target_percent",
            _target_percent(bundle, frame),
            {
                "orders": [
                    {
                        "amount": 5,
                        "date": frame.index[0].date().isoformat(),
                        "filled": 0,
                        "status": 0,
                    },
                    {
                        "amount": 5,
                        "date": frame.index[1].date().isoformat(),
                        "filled": 5,
                        "status": 1,
                    },
                ],
                "transactions": [
                    {
                        "amount": 5,
                        "date": frame.index[1].date().isoformat(),
                        "price": 110.0,
                        "transaction_commission": 0.0,
                    }
                ],
            },
        ),
        (
            "target_percent_snapshot",
            _target_percent_snapshot(bundle, frame),
            {
                "submitted_amounts": [5, 5],
                "transactions": [
                    {
                        "amount": 5,
                        "date": frame.index[1].date().isoformat(),
                        "price": 110.0,
                        "transaction_commission": 0.0,
                    },
                    {
                        "amount": 5,
                        "date": frame.index[1].date().isoformat(),
                        "price": 110.0,
                        "transaction_commission": 0.0,
                    },
                ],
            },
        ),
        (
            "submission_sequence",
            _submission_sequence(bundle, frame),
            [
                {
                    "amount": 1,
                    "date": frame.index[1].date().isoformat(),
                    "price": 110.0,
                    "transaction_commission": 0.0,
                },
                {
                    "amount": -1,
                    "date": frame.index[1].date().isoformat(),
                    "price": 110.0,
                    "transaction_commission": 0.0,
                },
            ],
        ),
        (
            "cash_and_short_proceeds",
            _cash_and_short(bundle, frame),
            {
                "long": {
                    "cash_after_fill": -1_200.0,
                    "position_after_fill": 20,
                    "transactions": [
                        {
                            "amount": 20,
                            "date": frame.index[1].date().isoformat(),
                            "price": 110.0,
                            "transaction_commission": 0.0,
                        }
                    ],
                },
                "short": {
                    "cash_after_fill": 1_110.0,
                    "position_after_fill": -1,
                    "transactions": [
                        {
                            "amount": -1,
                            "date": frame.index[1].date().isoformat(),
                            "price": 110.0,
                            "transaction_commission": 0.0,
                        }
                    ],
                },
            },
        ),
        (
            "volume_share_partial_fills",
            _volume_share_partial(bundle, frame),
            [
                {
                    "amount": 2,
                    "date": frame.index[1].date().isoformat(),
                    "price": 120.0048,
                    "transaction_commission": 0.0,
                },
                {
                    "amount": 2,
                    "date": frame.index[2].date().isoformat(),
                    "price": 130.0052,
                    "transaction_commission": 0.0,
                },
                {
                    "amount": 1,
                    "date": frame.index[3].date().isoformat(),
                    "price": 140.0014,
                    "transaction_commission": 0.0,
                },
            ],
        ),
        (
            "transaction_records_not_native_trades",
            _round_trip_records(bundle, frame),
            {
                "transactions": [
                    {
                        "amount": 2,
                        "date": frame.index[1].date().isoformat(),
                        "price": 110.0,
                        "transaction_commission": 0.0,
                    },
                    {
                        "amount": -2,
                        "date": frame.index[3].date().isoformat(),
                        "price": 130.0,
                        "transaction_commission": 0.0,
                    },
                ],
                "native_output_columns": ["orders", "positions", "transactions"],
                "native_trade_column": False,
            },
        ),
        ("final_bar_market_order", _final_bar_order(bundle, frame), []),
        (
            "session_calendar",
            [date.date().isoformat() for date in frame.index],
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ],
        ),
        ("missing_bar", _missing_bar(frame), _expected_missing_bar(frame)),
    ]
    checks = [
        {"id": check_id, "actual": actual, "expected": expected, "passed": _equal(actual, expected)}
        for check_id, actual, expected in measurements
    ]
    return {
        "schema_version": 1,
        "framework": "zipline",
        "package": "zipline-reloaded",
        "version": actual_version,
        "source_commit": COMMIT,
        "artifact": ARTIFACT,
        "artifact_sha256": ARTIFACT_SHA256,
        "oracle_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_protocol": {
            "commission": "NoCommission",
            "execution": "next-session custom OpenPriceSlippage",
            "risk_rules": "adapter-emulated from daily OHLC; not Zipline native trade records",
            "slippage": "no volume cap unless a scenario explicitly configures slippage",
        },
        "source_locations": {
            "algorithm_target_percent": _source_location(
                algorithm_module.TradingAlgorithm.order_target_percent
            ),
            "commission": _source_location(commission.PerShare.__init__),
            "default_blotter": _source_location(blotter_module.SimulationBlotter.__init__),
            "fixed_basis_points": _source_location(slippage.FixedBasisPointsSlippage.process_order),
            "volume_share": _source_location(slippage.VolumeShareSlippage.process_order),
        },
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    warnings.filterwarnings(
        "ignore",
        message="DataFrameGroupBy.apply operated on the grouping columns.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Ignoring 4 values because they are out of bounds for uint32.*",
        category=UserWarning,
    )
    try:
        evidence = run()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Zipline native behavior run failed: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [check["id"] for check in evidence["checks"] if not check["passed"]]
    if failed:
        print(f"Zipline native behavior differs: {failed}", file=sys.stderr)
        return 1
    print(f"Zipline native behavior passed: {len(evidence['checks'])} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
