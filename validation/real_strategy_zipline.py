#!/usr/bin/env python3
"""Run a frozen real target-allocation workload with Zipline Reloaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from real_strategy_input import filter_comparison_market
from zipline import run_algorithm
from zipline.api import (
    get_datetime,
    order,
    order_target,
    set_commission,
    set_slippage,
    sid,
)
from zipline.data.bundles import ingest, register
from zipline.finance.commission import NoCommission
from zipline.finance.slippage import SlippageModel
from zipline.utils.calendar_utils import get_calendar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targets(frame: pl.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    result: dict[pd.Timestamp, dict[str, float]] = {}
    for row in frame.sort("timestamp", "symbol").iter_rows(named=True):
        timestamp = pd.Timestamp(row["timestamp"]).normalize()
        result.setdefault(timestamp, {})[str(row["symbol"])] = float(row["weight"])
    return result


def _flatten(results: pd.DataFrame, column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if column not in results.columns:
        return pd.DataFrame()
    for timestamp, payload in results[column].items():
        if not isinstance(payload, list):
            continue
        for item in payload:
            record = dict(item) if isinstance(item, dict) else item.to_dict()
            record["timestamp"] = timestamp
            records.append(record)
    return pd.DataFrame(records)


def run(bundle: Path) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["case_study"] != "etfs":
        raise ValueError("This adapter accepts the ETF equity workload")
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    market = filter_comparison_market(
        pl.read_parquet(bundle / "market.parquet"), spec
    ).with_columns(pl.col(column).round(3) for column in ("open", "high", "low", "close"))
    target_lookup = _targets(pl.read_parquet(bundle / "targets.parquet"))
    symbols = market["symbol"].unique().sort().to_list()
    calendar_name = "XNYS"
    calendar = get_calendar(calendar_name)
    first_date = pd.Timestamp(market["timestamp"].min()).normalize()
    last_date = pd.Timestamp(market["timestamp"].max()).normalize()
    sessions = pd.DatetimeIndex(calendar.sessions_in_range(first_date, last_date)).tz_localize(None)
    bundle_name = f"real_etfs_v2_{manifest['bundle_sha256'][:16]}"

    def ingest_bundle(
        _environ,
        asset_db_writer,
        _minute_bar_writer,
        daily_bar_writer,
        adjustment_writer,
        _calendar,
        _start_session,
        _end_session,
        _cache,
        show_progress,
        _output_dir,
    ) -> None:
        metadata_rows = []
        bar_data = []
        for asset_sid, symbol in enumerate(symbols):
            selected = market.filter(pl.col("symbol") == symbol).select(
                "timestamp", "open", "high", "low", "close", "volume"
            )
            frame = pd.DataFrame(selected.to_dict(as_series=False)).set_index("timestamp")
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
            asset_sessions = sessions[
                (sessions >= frame.index.min()) & (sessions <= frame.index.max())
            ]
            observed_sessions = frame.index
            frame = frame.reindex(asset_sessions).ffill()
            frame.loc[~frame.index.isin(observed_sessions), "volume"] = 0
            metadata_rows.append(
                {
                    "sid": asset_sid,
                    "symbol": symbol,
                    "asset_name": symbol,
                    "start_date": frame.index.min(),
                    "end_date": frame.index.max(),
                    "first_traded": frame.index.min(),
                    "exchange": "NYSE",
                }
            )
            bar_data.append((asset_sid, frame))
        metadata = pd.DataFrame(metadata_rows).set_index("sid")
        asset_db_writer.write(equities=metadata)
        daily_bar_writer.write(bar_data, show_progress=show_progress)
        adjustment_writer.write()

    zipline_root = Path(os.environ.get("ZIPLINE_ROOT", Path.home() / ".zipline"))
    bundle_dir = zipline_root / "data" / bundle_name
    register(
        bundle_name,
        ingest_bundle,
        calendar_name=calendar_name,
        start_session=calendar.sessions_in_range(first_date, last_date)[0],
        end_session=calendar.sessions_in_range(first_date, last_date)[-1],
    )
    setup_seconds = 0.0
    if not bundle_dir.exists() or not any(bundle_dir.iterdir()):
        started = time.perf_counter()
        ingest(bundle_name, show_progress=False)
        setup_seconds = time.perf_counter() - started

    class OpenPriceSlippage(SlippageModel):
        @staticmethod
        def process_order(data, pending_order):
            return float(data.current(pending_order.asset, "open")), pending_order.amount

    state = {"targets": target_lookup, "symbols": symbols}

    def initialize(context) -> None:
        context.state = state
        context.assets = {symbol: sid(index) for index, symbol in enumerate(symbols)}
        context.names = {asset: symbol for symbol, asset in context.assets.items()}
        set_commission(us_equities=NoCommission())
        set_slippage(us_equities=OpenPriceSlippage())

    def handle_data(context, data) -> None:
        timestamp = get_datetime()
        timestamp = (
            timestamp.tz_convert(None).normalize() if timestamp.tz else timestamp.normalize()
        )
        requested = context.state["targets"].get(timestamp)
        if requested is None:
            return
        available = {
            symbol: weight
            for symbol, weight in requested.items()
            if data.can_trade(context.assets[symbol])
        }
        if not available:
            return
        value = float(context.portfolio.portfolio_value)
        current = {
            context.names[asset]: float(position.amount)
            * float(data.current(asset, "price"))
            / value
            for asset, position in context.portfolio.positions.items()
            if position.amount != 0 and data.can_trade(asset)
        }
        reductions = sorted(
            symbol for symbol, weight in available.items() if weight < current.get(symbol, 0.0)
        )
        omitted = sorted(symbol for symbol in current if symbol not in available)
        increases = sorted(symbol for symbol in available if symbol not in reductions)
        for symbol in reductions:
            asset = context.assets[symbol]
            price = float(data.current(asset, "price"))
            raw = (available[symbol] - current.get(symbol, 0.0)) * value / price
            amount = int(math.copysign(math.floor(abs(raw) + 0.5), raw))
            if amount:
                order(asset, amount)
        for symbol in omitted:
            order_target(context.assets[symbol], 0)
        for symbol in increases:
            asset = context.assets[symbol]
            price = float(data.current(asset, "price"))
            raw = (available[symbol] - current.get(symbol, 0.0)) * value / price
            amount = int(math.copysign(math.floor(abs(raw) + 0.5), raw))
            if amount:
                order(asset, amount)

    started = time.perf_counter()
    results = run_algorithm(
        start=calendar.sessions_in_range(first_date, last_date)[0],
        end=calendar.sessions_in_range(first_date, last_date)[-1],
        initialize=initialize,
        handle_data=handle_data,
        capital_base=float(spec["backtest_config"]["cash"]["initial"]),
        bundle=bundle_name,
        data_frequency="daily",
        trading_calendar=calendar,
    )
    engine_seconds = time.perf_counter() - started
    transactions = _flatten(results, "transactions")
    fill_rows = []
    for _, row in transactions.iterrows():
        asset = row["sid"]
        amount = float(row["amount"])
        fill_rows.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).tz_localize(None),
                "asset": asset.symbol,
                "side": "buy" if amount > 0 else "sell",
                "quantity": abs(amount),
                "price": float(row["price"]),
                "commission": 0.0,
            }
        )
    fills = pl.DataFrame(fill_rows)
    equity = pl.DataFrame(
        {
            "timestamp": pl.Series(
                pd.DatetimeIndex(results.index).tz_localize(None).to_pydatetime().tolist(),
                dtype=pl.Datetime("us"),
            ),
            "equity": results["portfolio_value"].to_numpy(),
        }
    )
    evidence = {
        "schema_version": 1,
        "case_study": manifest["case_study"],
        "framework": "zipline",
        "comparison_profile": "zipline_strict",
        "comparison_costs": "disabled",
        "comparison_position_rules": "disabled",
        "price_decimals": 3,
        "input_bundle_sha256": manifest["bundle_sha256"],
        "setup_seconds": setup_seconds,
        "engine_seconds": engine_seconds,
        "final_value": float(results["portfolio_value"].iloc[-1]),
        "num_fills": fills.height,
    }
    return evidence, fills, equity


def write_evidence(
    output: Path, evidence: dict[str, Any], fills: pl.DataFrame, equity: pl.DataFrame
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        fills.write_parquet(staging / "fills.parquet", compression="zstd", statistics=True)
        equity.write_parquet(staging / "equity.parquet", compression="zstd", statistics=True)
        evidence["files"] = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(staging.iterdir())
        }
        (staging / "manifest.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"Evidence output already exists: {output}")
        os.replace(staging, output)
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, fills, equity = run(args.bundle.resolve())
    write_evidence(args.output.resolve(), evidence, fills, equity)
    print(
        f"{evidence['case_study']}: {evidence['num_fills']:,} fills, "
        f"{evidence['engine_seconds']:.6f}s engine time"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
