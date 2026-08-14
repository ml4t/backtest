#!/usr/bin/env python3
"""Run the frozen crypto-perpetual strategy with LEAN's Binance futures model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
import tomllib
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl

from ml4t.backtest._validation.lean_runner import (
    copy_lean_artifacts,
    load_lean_artifacts,
    make_lean_env,
    resolve_lean_command,
    run_lean_backtest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symbol_properties(data_root: Path, symbols: list[str]) -> dict[str, dict[str, float]]:
    path = data_root / "symbol-properties" / "symbol-properties-database.csv"
    wanted = set(symbols)
    found: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in rows:
            symbol = row["symbol"]
            if row["market"] == "binance" and row["type"] == "cryptofuture" and symbol in wanted:
                found[symbol] = {
                    "contract_multiplier": float(row["contract_multiplier"]),
                    "lot_size": float(row["lot_size"]),
                    "tick_size": float(row["minimum_price_variation"]),
                }
    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(f"LEAN symbol properties are missing: {missing}")
    return found


def _export_data(bundle: Path, data_root: Path, symbols: list[str]) -> None:
    market = pl.read_parquet(bundle / "market.parquet")
    funding = pl.read_parquet(bundle / "funding.parquet")
    hourly = data_root / "cryptofuture" / "binance" / "hour"
    margin = data_root / "cryptofuture" / "binance" / "margin_interest"
    hourly.mkdir(parents=True, exist_ok=True)
    margin.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        rows = market.filter(pl.col("symbol") == symbol).sort("timestamp").iter_rows(named=True)
        trade_lines: list[str] = []
        quote_lines: list[str] = []
        for row in rows:
            start = pd.Timestamp(row["timestamp"]) - timedelta(hours=1)
            stamp = start.strftime("%Y%m%d %H:%M")
            values = [float(row[name]) for name in ("open", "high", "low", "close")]
            volume = max(1.0, float(row["volume"]))
            o, h, low, close = values
            trade_lines.append(f"{stamp},{o},{h},{low},{close},{volume}")
            quote_lines.append(
                f"{stamp},{o},{h},{low},{close},{volume},{o},{h},{low},{close},{volume}"
            )
        ticker = symbol.lower()
        for kind, lines in (("trade", trade_lines), ("quote", quote_lines)):
            with zipfile.ZipFile(
                hourly / f"{ticker}_{kind}.zip", "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(f"{ticker}.csv", "\n".join(lines))
        funding_rows = funding.filter(pl.col("symbol") == symbol).sort("timestamp")
        margin_lines = [
            f"{pd.Timestamp(row['timestamp']).strftime('%Y%m%d %H:%M:%S')},{float(row['funding_rate']):.12f}"
            for row in funding_rows.iter_rows(named=True)
        ]
        (margin / f"{ticker}.csv").write_text("\n".join(margin_lines) + "\n", encoding="utf-8")


def _prepare_project(bundle: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["case_study"] != "crypto_perps_funding":
        raise ValueError("This adapter accepts the crypto perpetual-funding workload")
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    market = pl.read_parquet(bundle / "market.parquet")
    targets = pl.read_parquet(bundle / "targets.parquet")
    symbols = market["symbol"].unique().sort().to_list()
    workspace = PROJECT_ROOT / "validation" / "lean" / "workspace"
    project = workspace / "real_strategy_crypto_perps_funding"
    project.mkdir(parents=True, exist_ok=True)
    (project / "backtests").mkdir(exist_ok=True)
    for name in (
        "ml4t_daily_equity.csv",
        "ml4t_funding_observed.csv",
        "ml4t_order_events.csv",
        "ml4t_runtime.json",
    ):
        path = project / name
        if path.exists():
            path.unlink()
    (project / "config.json").write_text(
        json.dumps(
            {
                "algorithm-language": "Python",
                "parameters": {},
                "description": "Frozen ML4T crypto-perpetual real-strategy parity workload",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (project / "symbols.csv").write_text("\n".join(symbols) + "\n", encoding="utf-8")
    (project / "ml4t_symbol_map.json").write_text(
        json.dumps({symbol: symbol for symbol in symbols}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    targets.with_columns(pl.col("timestamp").dt.strftime("%Y-%m-%d %H:%M:%S")).write_csv(
        project / "targets.csv"
    )
    _export_data(bundle, workspace / "data", symbols)
    properties = _symbol_properties(workspace / "data", symbols)
    (project / "symbol_properties.json").write_text(
        json.dumps(properties, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    first = pd.Timestamp(cast(datetime, market["timestamp"].min())).date()
    last = pd.Timestamp(cast(datetime, market["timestamp"].max())).date() + timedelta(days=1)
    initial_cash = float(spec["backtest_config"]["cash"]["initial"])
    main_code = f"""from AlgorithmImports import *

import csv
import json
import math
import time
from pathlib import Path


class RealStrategyCryptoPerpsFunding(QCAlgorithm):
    def initialize(self):
        self.set_start_date({first.year}, {first.month}, {first.day})
        self.set_end_date({last.year}, {last.month}, {last.day})
        self.set_time_zone(TimeZones.UTC)
        self.set_brokerage_model(BrokerageName.BINANCE_FUTURES, AccountType.MARGIN)
        self.set_account_currency("USDT", {initial_cash})
        base = Path(__file__).resolve().parent
        self._equity_path = base / "ml4t_daily_equity.csv"
        self._funding_path = base / "ml4t_funding_observed.csv"
        self._events_path = base / "ml4t_order_events.csv"
        self._runtime_path = base / "ml4t_runtime.json"
        self._targets = {{}}
        with (base / "targets.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                self._targets.setdefault(row["timestamp"], {{}})[row["symbol"]] = float(row["weight"])
        self._properties = json.loads((base / "symbol_properties.json").read_text())
        self._symbols = {{}}
        for ticker in (base / "symbols.csv").read_text().splitlines():
            if not ticker:
                continue
            security = self.add_crypto_future(ticker, Resolution.HOUR, fill_forward=False, market=Market.BINANCE)
            security.set_fee_model(ConstantFeeModel(0))
            security.set_slippage_model(ConstantSlippageModel(0))
            self._symbols[ticker] = security.symbol
        with self._equity_path.open("w", newline="") as stream:
            csv.writer(stream).writerow(["timestamp", "equity", "cash", "holdings_value"])
        with self._events_path.open("w", newline="") as stream:
            csv.writer(stream).writerow(["timestamp", "symbol", "status", "direction", "fill_quantity", "fill_price", "fee", "message", "order_id"])
        with self._funding_path.open("w", newline="") as stream:
            csv.writer(stream).writerow(["timestamp", "symbol", "funding_rate"])
        self._started = time.perf_counter()

    def on_data(self, data: Slice):
        key = self.time.strftime("%Y-%m-%d %H:%M:%S")
        with self._funding_path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            for symbol, rate in data.margin_interest_rates.items():
                writer.writerow([key, symbol.value, float(rate.interest_rate)])
        requested = self._targets.get(key)
        if requested is not None:
            active = set(requested)
            active.update(ticker for ticker, symbol in self._symbols.items() if self.portfolio[symbol].quantity != 0)
            value = float(self.portfolio.total_portfolio_value)
            current = {{ticker: float(self.portfolio[symbol].quantity) * float(self.securities[symbol].price) / value for ticker, symbol in self._symbols.items() if self.portfolio[symbol].quantity != 0}}
            reductions = sorted(ticker for ticker in active if requested.get(ticker, 0.0) < current.get(ticker, 0.0))
            increases = sorted(active - set(reductions))
            for ticker in reductions + increases:
                symbol = self._symbols[ticker]
                if symbol not in data.bars:
                    continue
                price = float(data.bars[symbol].close)
                lot = float(self._properties[ticker]["lot_size"])
                raw = (requested.get(ticker, 0.0) - current.get(ticker, 0.0)) * value / price
                delta = math.copysign(math.floor(abs(raw) / lot + 0.5) * lot, raw)
                if abs(delta) >= lot:
                    self.market_order(symbol, delta)
        with self._equity_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([key, float(self.portfolio.total_portfolio_value), float(self.portfolio.cash_book["USDT"].amount), float(self.portfolio.total_holdings_value)])

    def on_order_event(self, event: OrderEvent):
        fee = float(event.order_fee.value.amount) if event.order_fee and event.order_fee.value else 0.0
        with self._events_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([self.time.strftime("%Y-%m-%d %H:%M:%S"), event.symbol.value if event.symbol else "", str(event.status), str(event.direction), float(event.fill_quantity), float(event.fill_price), fee, str(event.message or "").replace("\\n", " "), int(event.order_id)])

    def on_end_of_algorithm(self):
        self._runtime_path.write_text(json.dumps({{"engine_seconds": time.perf_counter() - self._started}}))
"""
    (project / "main.py").write_text(main_code, encoding="utf-8")
    return project, {"manifest": manifest, "properties": properties}


def run(bundle: Path) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    project, inputs = _prepare_project(bundle)
    targets = tomllib.loads((PROJECT_ROOT / "validation/framework_targets.toml").read_text())
    output_dir = project / "backtests" / str(time.time_ns())
    run_lean_backtest(
        lean_cmd=resolve_lean_command(),
        cwd=PROJECT_ROOT,
        project_dir=project,
        lean_config=PROJECT_ROOT / "validation/lean/workspace/lean.json",
        output_dir=output_dir,
        image=targets["framework"]["lean"]["artifact"],
        env=make_lean_env(),
    )
    copy_lean_artifacts(
        project,
        output_dir,
        [
            "ml4t_daily_equity.csv",
            "ml4t_funding_observed.csv",
            "ml4t_order_events.csv",
            "ml4t_runtime.json",
            "ml4t_symbol_map.json",
        ],
    )
    _, final_value, fills_pd, equity_pd, events_pd = load_lean_artifacts(output_dir)
    if fills_pd is None or equity_pd is None or events_pd is None:
        raise RuntimeError("LEAN did not produce the required comparison surfaces")
    fills_pd = fills_pd.rename(columns={"fill_price": "price", "fee": "commission"})
    fills = pl.from_pandas(
        fills_pd[["timestamp", "asset", "side", "quantity", "price", "commission"]]
    )
    equity = pl.from_pandas(equity_pd[["timestamp", "equity", "cash", "holdings_value"]])
    runtime = json.loads((output_dir / "ml4t_runtime.json").read_text(encoding="utf-8"))
    rejected = events_pd[events_pd["status"].astype(str).str.lower().isin({"invalid", "canceled"})]
    evidence = {
        "schema_version": 1,
        "case_study": inputs["manifest"]["case_study"],
        "framework": "lean",
        "comparison_profile": "lean_crypto_future",
        "comparison_costs": "disabled_except_native_funding",
        "comparison_position_rules": "disabled",
        "input_bundle_sha256": inputs["manifest"]["bundle_sha256"],
        "symbol_properties_sha256": _sha256(project / "symbol_properties.json"),
        "engine_seconds": float(runtime["engine_seconds"]),
        "final_value": final_value,
        "num_fills": fills.height,
        "num_rejections": len(rejected),
    }
    rejected_frame = pl.from_pandas(
        rejected[["timestamp", "asset", "fill_quantity", "message", "order_id"]]
    )
    return evidence, fills, equity, rejected_frame


def write_evidence(
    output: Path,
    evidence: dict[str, Any],
    fills: pl.DataFrame,
    equity: pl.DataFrame,
    rejections: pl.DataFrame,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        fills.write_parquet(staging / "fills.parquet", compression="zstd", statistics=True)
        equity.write_parquet(staging / "equity.parquet", compression="zstd", statistics=True)
        rejections.write_parquet(
            staging / "rejected_orders.parquet", compression="zstd", statistics=True
        )
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
    evidence, fills, equity, rejections = run(args.bundle.resolve())
    write_evidence(args.output.resolve(), evidence, fills, equity, rejections)
    print(
        f"{evidence['case_study']}: {evidence['num_fills']:,} fills, "
        f"{evidence['engine_seconds']:.6f}s engine time"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
