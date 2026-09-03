#!/usr/bin/env python3
"""Run a frozen real target-allocation workload with Backtrader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import backtrader as bt
import pandas as pd
import polars as pl
from real_strategy_input import (
    comparison_scope,
    filter_comparison_market,
    filter_comparison_targets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_map(targets: pl.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    result: dict[pd.Timestamp, dict[str, float]] = {}
    for row in targets.sort("timestamp", "symbol").iter_rows(named=True):
        result.setdefault(pd.Timestamp(row["timestamp"]), {})[str(row["symbol"])] = float(
            row["weight"]
        )
    return result


class FrozenTargetStrategy(bt.Strategy):
    params = (("targets", None), ("contracts", None))

    def __init__(self) -> None:
        self.fills: list[dict[str, Any]] = []
        self.equity: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []

    def prenext(self) -> None:
        self.next()

    def nextstart(self) -> None:
        self.next()

    def next(self) -> None:
        timestamp = pd.Timestamp(self.datetime.datetime(0))
        self.equity.append(
            {"timestamp": timestamp.to_pydatetime(), "equity": self.broker.getvalue()}
        )
        requested = self.p.targets.get(timestamp)
        if requested is None:
            return
        available = {
            data._name: weight
            for data in self.datas
            if data._name in requested and pd.Timestamp(data.datetime.datetime(0)) == timestamp
            for weight in (requested[data._name],)
        }
        if not available:
            return
        value = self.broker.getvalue()
        current = {
            data._name: (
                self.getposition(data).size
                * float(data.close[0])
                * float(self.p.contracts.get(data._name, {}).get("multiplier", 1.0))
                / value
            )
            for data in self.datas
            if self.getposition(data).size != 0
        }
        reductions = sorted(
            symbol for symbol, weight in available.items() if weight < current.get(symbol, 0.0)
        )
        omitted = sorted(symbol for symbol in current if symbol not in available)
        increases = sorted(symbol for symbol in available if symbol not in reductions)
        by_name = {data._name: data for data in self.datas}
        for symbol in reductions:
            self._submit_target(by_name[symbol], available[symbol], current, value)
        for symbol in omitted:
            self.order_target_size(data=by_name[symbol], target=0)
        for symbol in increases:
            self._submit_target(by_name[symbol], available[symbol], current, value)

    def _submit_target(
        self, data, target_weight: float, current_weights: dict[str, float], value: float
    ) -> None:
        contract = self.p.contracts.get(data._name)
        if contract is None:
            self.order_target_percent(data=data, target=target_weight)
            return
        price = float(data.close[0])
        multiplier = float(contract["multiplier"])
        delta = int(
            (target_weight - current_weights.get(data._name, 0.0)) * value / (price * multiplier)
        )
        target_size = self.getposition(data).size + delta
        self.order_target_size(data=data, target=target_size)

    def notify_order(self, order) -> None:
        if order.status == order.Completed:
            self.fills.append(
                {
                    "timestamp": bt.num2date(order.executed.dt).replace(tzinfo=None),
                    "asset": order.data._name,
                    "side": "buy" if order.isbuy() else "sell",
                    "quantity": abs(float(order.executed.size)),
                    "price": float(order.executed.price),
                    "commission": abs(float(order.executed.comm)),
                }
            )
        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            self.rejections.append(
                {
                    "timestamp": bt.num2date(order.created.dt).replace(tzinfo=None),
                    "asset": order.data._name,
                    "status": order.getstatusname(),
                    "requested_quantity": abs(float(order.created.size)),
                }
            )


def run(bundle: Path) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    market = filter_comparison_market(pl.read_parquet(bundle / "market.parquet"), spec)
    targets = _target_map(
        filter_comparison_targets(pl.read_parquet(bundle / "targets.parquet"), spec)
    )
    contracts = (
        json.loads((bundle / "contracts.json").read_text(encoding="utf-8"))
        if (bundle / "contracts.json").is_file()
        else {}
    )
    cerebro = bt.Cerebro(stdstats=False)
    initial_cash = float(spec["backtest_config"]["cash"]["initial"])
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0)
    symbols = market["symbol"].unique().sort().to_list()
    market_by_symbol = market.select(
        "symbol", "timestamp", "open", "high", "low", "close", "volume"
    ).partition_by("symbol", as_dict=True, include_key=False)
    for symbol in symbols:
        selected = market_by_symbol[(symbol,)]
        frame = pd.DataFrame(selected.to_dict(as_series=False)).set_index("timestamp")
        cerebro.adddata(bt.feeds.PandasData(dataname=frame), name=symbol)
        contract = contracts.get(symbol)
        if contract is not None:
            initial_margin, _ = contract["margin_pct"]
            fixed_margin = (
                float(frame["close"].iloc[0])
                * float(contract["multiplier"])
                * float(initial_margin)
            )
            cerebro.broker.addcommissioninfo(
                bt.CommInfoBase(
                    commission=0.0,
                    mult=float(contract["multiplier"]),
                    margin=fixed_margin,
                    automargin=False,
                    commtype=bt.CommInfoBase.COMM_FIXED,
                    stocklike=False,
                ),
                name=symbol,
            )
    cerebro.addstrategy(FrozenTargetStrategy, targets=targets, contracts=contracts)
    started = time.perf_counter()
    strategy = cerebro.run()[0]
    engine_seconds = time.perf_counter() - started
    fills = pl.DataFrame(strategy.fills)
    equity = pl.DataFrame(strategy.equity)
    evidence = {
        "schema_version": 1,
        "case_study": manifest["case_study"],
        "framework": "backtrader",
        "comparison_profile": "backtrader_strict",
        "comparison_scope": comparison_scope(spec),
        "comparison_costs": "disabled",
        "comparison_position_rules": "disabled",
        "input_bundle_sha256": manifest["bundle_sha256"],
        "engine_seconds": engine_seconds,
        "final_value": float(cerebro.broker.getvalue()),
        "num_fills": fills.height,
        "num_rejections": len(strategy.rejections),
    }
    rejections = pl.DataFrame(
        strategy.rejections,
        schema={
            "timestamp": pl.Datetime("us"),
            "asset": pl.String,
            "status": pl.String,
            "requested_quantity": pl.Float64,
        },
    )
    return evidence, fills, equity, rejections


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
