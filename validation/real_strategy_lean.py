#!/usr/bin/env python3
"""Run the frozen real ETF target-allocation workload with LEAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl

from ml4t.backtest._validation.lean_runner import (
    build_hashed_ticker_map,
    copy_lean_artifacts,
    export_lean_daily_data,
    load_lean_artifacts,
    make_lean_env,
    resolve_lean_command,
    run_lean_backtest,
)
from ml4t.backtest._validation.real_strategy import filter_comparison_market

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_project(bundle: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["case_study"] != "etfs":
        raise ValueError("The LEAN adapter currently accepts the ETF comparison workload")
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    market = filter_comparison_market(pl.read_parquet(bundle / "market.parquet"), spec)
    targets = pl.read_parquet(bundle / "targets.parquet")
    symbols = market["symbol"].unique().sort().to_list()
    asset_to_ticker = build_hashed_ticker_map("real_etfs", symbols)
    ticker_to_asset = {ticker: asset for asset, ticker in asset_to_ticker.items()}

    workspace = PROJECT_ROOT / "validation" / "lean" / "workspace"
    project = workspace / "real_strategy_etfs"
    project.mkdir(parents=True, exist_ok=True)
    (project / "backtests").mkdir(exist_ok=True)
    for name in ("ml4t_daily_equity.csv", "ml4t_order_events.csv", "ml4t_runtime.json"):
        path = project / name
        if path.exists():
            path.unlink()

    (project / "config.json").write_text(
        json.dumps(
            {
                "algorithm-language": "Python",
                "parameters": {},
                "description": "Frozen ML4T ETF real-strategy parity workload",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (project / "ml4t_symbol_map.json").write_text(
        json.dumps(ticker_to_asset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (project / "symbols.csv").write_text(
        "\n".join(asset_to_ticker[symbol] for symbol in symbols) + "\n", encoding="utf-8"
    )
    target_rows = targets.with_columns(
        pl.col("symbol").replace_strict(asset_to_ticker).alias("ticker"),
        pl.col("timestamp").dt.strftime("%Y-%m-%d"),
    ).select("timestamp", "ticker", "weight")
    target_rows.write_csv(project / "targets.csv")

    prices_by_asset: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        selected = market.filter(pl.col("symbol") == symbol).select(
            "timestamp", "open", "high", "low", "close", "volume"
        )
        prices_by_asset[symbol] = pd.DataFrame(selected.to_dict(as_series=False)).set_index(
            "timestamp"
        )
    data_root = workspace / "data" / "equity" / "usa"
    export_lean_daily_data(
        data_root=data_root,
        prices_by_asset=prices_by_asset,
        asset_to_ticker=asset_to_ticker,
        manifest_path=data_root / "real_strategy_etfs_manifest.json",
        signature_payload={
            "bundle_sha256": manifest["bundle_sha256"],
            "price_decimals": 4,
        },
    )

    first = pd.Timestamp(cast(datetime, market["timestamp"].min())).date()
    last = pd.Timestamp(cast(datetime, market["timestamp"].max())).date() + timedelta(days=7)
    initial_cash = float(spec["backtest_config"]["cash"]["initial"])
    main_code = f"""from AlgorithmImports import *

import csv
import json
import math
import time
from pathlib import Path


class RealStrategyEtfs(QCAlgorithm):
    def initialize(self):
        self.set_start_date({first.year}, {first.month}, {first.day})
        self.set_end_date({last.year}, {last.month}, {last.day})
        self.set_cash({initial_cash})
        self.set_brokerage_model(BrokerageName.DEFAULT, AccountType.MARGIN)
        base = Path(__file__).resolve().parent
        self._equity_path = base / "ml4t_daily_equity.csv"
        self._events_path = base / "ml4t_order_events.csv"
        self._runtime_path = base / "ml4t_runtime.json"
        self._targets = {{}}
        with (base / "targets.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                self._targets.setdefault(row["timestamp"], {{}})[row["ticker"]] = float(row["weight"])
        self._symbols = {{}}
        for ticker in (base / "symbols.csv").read_text().splitlines():
            if not ticker:
                continue
            security = self.add_equity(ticker, Resolution.DAILY)
            security.set_fee_model(ConstantFeeModel(0))
            security.set_slippage_model(ConstantSlippageModel(0))
            self._symbols[ticker] = security.symbol
        with self._equity_path.open("w", newline="") as stream:
            csv.writer(stream).writerow(["timestamp", "equity", "cash", "holdings_value"])
        with self._events_path.open("w", newline="") as stream:
            csv.writer(stream).writerow(["timestamp", "symbol", "status", "direction", "fill_quantity", "fill_price", "fee", "message", "order_id"])
        self._started = time.perf_counter()

    def on_data(self, data: Slice):
        key = self.time.strftime("%Y-%m-%d")
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
                raw = (requested.get(ticker, 0.0) - current.get(ticker, 0.0)) * value / price
                delta = int(math.copysign(math.floor(abs(raw) + 0.5), raw))
                if delta:
                    self.market_on_open_order(symbol, delta)
        with self._equity_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([key, float(self.portfolio.total_portfolio_value), float(self.portfolio.cash), float(self.portfolio.total_holdings_value)])

    def on_order_event(self, event: OrderEvent):
        fee = float(event.order_fee.value.amount) if event.order_fee and event.order_fee.value else 0.0
        with self._events_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([self.time.strftime("%Y-%m-%d %H:%M:%S"), event.symbol.value if event.symbol else "", str(event.status), str(event.direction), float(event.fill_quantity), float(event.fill_price), fee, str(event.message or "").replace("\\n", " "), int(event.order_id)])

    def on_end_of_algorithm(self):
        self._runtime_path.write_text(json.dumps({{"engine_seconds": time.perf_counter() - self._started}}))
"""
    (project / "main.py").write_text(main_code, encoding="utf-8")
    return project, {
        "manifest": manifest,
        "spec": spec,
        "ticker_to_asset": ticker_to_asset,
        "valuation_dates": {
            pd.Timestamp(timestamp).strftime("%Y-%m-%d")
            for timestamp in market["timestamp"].unique().to_list()
        },
    }


def run(bundle: Path) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    project, inputs = _prepare_project(bundle)
    targets = tomllib.loads((PROJECT_ROOT / "validation/framework_targets.toml").read_text())
    image = targets["framework"]["lean"]["artifact"]
    output_dir = project / "backtests" / str(time.time_ns())
    run_lean_backtest(
        lean_cmd=resolve_lean_command(),
        cwd=PROJECT_ROOT,
        project_dir=project,
        lean_config=PROJECT_ROOT / "validation/lean/workspace/lean.json",
        output_dir=output_dir,
        image=image,
        env=make_lean_env(),
    )
    names = [
        "ml4t_daily_equity.csv",
        "ml4t_order_events.csv",
        "ml4t_runtime.json",
        "ml4t_symbol_map.json",
    ]
    copy_lean_artifacts(project, output_dir, names)
    _, final_value, fills_pd, equity_pd, events_pd = load_lean_artifacts(output_dir)
    if fills_pd is None or equity_pd is None or events_pd is None:
        raise RuntimeError("LEAN did not produce the required comparison surfaces")
    equity_pd = equity_pd[
        pd.to_datetime(equity_pd["timestamp"])
        .dt.strftime("%Y-%m-%d")
        .isin(inputs["valuation_dates"])
    ].copy()
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
        "comparison_profile": "lean",
        "comparison_costs": "disabled",
        "comparison_position_rules": "disabled",
        "input_bundle_sha256": inputs["manifest"]["bundle_sha256"],
        "engine_seconds": float(runtime["engine_seconds"]),
        "orchestration_seconds_excluded": None,
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
