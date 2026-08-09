#!/usr/bin/env python3
"""Generate reproducible stable-release performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from ml4t.specs.market_data import FeedSpec

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionPrice, SlippageType
from ml4t.backtest.execution import VolumeParticipationLimit
from ml4t.backtest.types import ExecutionMode, OrderSide

ROOT = Path(__file__).parents[1]
DEFAULT_MANIFEST = ROOT / "validation" / "performance_baselines.json"
MEASUREMENT_CONTRACT = {
    "runtime": "perf_counter around Engine.run only; setup excluded",
    "setup": "deterministic data, DataFeed, strategy, config, and Engine construction",
    "memory": "child-process peak RSS from interpreter start through completed run",
    "sample_deviation_reporting_threshold": 0.10,
    "regression_gate": "instrument-free hotpath benchmark in tests/benchmark",
}
WORKER_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class Workload:
    bars: int
    assets: int
    strategy: str
    quote_aware: bool = False
    volume: float = 1_000_000.0
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR
    commission_rate: float = 0.0
    slippage_rate: float = 0.0

    @property
    def data_points(self) -> int:
        return self.bars * self.assets


WORKLOADS = {
    "single_asset": Workload(
        bars=2_000,
        assets=1,
        strategy="alternating",
        execution_mode=ExecutionMode.NEXT_BAR,
        commission_rate=0.0005,
        slippage_rate=0.0002,
    ),
    "daily_250_assets": Workload(bars=2_520, assets=250, strategy="portfolio_alternating"),
    "quote_aware": Workload(bars=1_000, assets=20, strategy="alternating", quote_aware=True),
    "rebalance": Workload(bars=504, assets=50, strategy="rebalance"),
    "partial_fill": Workload(bars=500, assets=10, strategy="partial_fill", volume=100.0),
}


class _Alternating(Strategy):
    def __init__(self, asset: str, interval: int = 50):
        self.asset = asset
        self.interval = interval
        self.bar = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar % self.interval != 1:
            return
        position = broker.get_position(self.asset)
        if position is None:
            broker.submit_order(self.asset, 100.0)
        else:
            broker.close_position(self.asset)


class _PortfolioAlternating(Strategy):
    def __init__(self, assets: list[str], interval: int = 50):
        self.assets = assets
        self.interval = interval
        self.bar = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar % self.interval != 1:
            return
        for asset in self.assets:
            if broker.get_position(asset) is None:
                broker.submit_order(asset, 100.0)
            else:
                broker.close_position(asset)


class _Rebalance(Strategy):
    def __init__(self, assets: list[str]):
        self.assets = assets
        self.bar = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        self.bar += 1
        if self.bar % 21 != 1:
            return
        offset = (self.bar // 21) % 2 * 10
        selected = self.assets[offset : offset + 10]
        broker.rebalance_to_weights(dict.fromkeys(selected, 0.08))


class _PartialFill(Strategy):
    def __init__(self, asset: str):
        self.asset = asset
        self.submitted = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if not self.submitted:
            broker.submit_order(self.asset, 1_000.0, OrderSide.BUY)
            self.submitted = True


def _build_prices(workload: Workload) -> pl.DataFrame:
    bar_index = np.repeat(np.arange(workload.bars, dtype=np.int32), workload.assets)
    asset_index = np.tile(np.arange(workload.assets, dtype=np.int32), workload.bars)
    start = np.datetime64("2018-01-02", "us")
    timestamps = start + bar_index.astype("timedelta64[D]")
    assets = np.array([f"A{i:03d}" for i in range(workload.assets)], dtype=object)[asset_index]
    close = 100.0 + asset_index * 0.05 + bar_index * 0.002 + (bar_index % 17) * 0.01
    data: dict[str, Any] = {
        "timestamp": timestamps,
        "asset": assets,
        "open": close - 0.02,
        "high": close + 0.08,
        "low": close - 0.08,
        "close": close,
        "volume": np.full(len(close), workload.volume),
    }
    if workload.quote_aware:
        data.update(
            {
                "mid_price": close,
                "bid": close - 0.01,
                "ask": close + 0.01,
                "bid_size": np.full(len(close), 5_000.0),
                "ask_size": np.full(len(close), 5_000.0),
            }
        )
    return pl.DataFrame(data)


def _strategy(workload: Workload) -> Strategy:
    assets = [f"A{i:03d}" for i in range(workload.assets)]
    if workload.strategy == "alternating":
        return _Alternating(assets[0])
    if workload.strategy == "portfolio_alternating":
        return _PortfolioAlternating(assets[:50])
    if workload.strategy == "rebalance":
        return _Rebalance(assets)
    if workload.strategy == "partial_fill":
        return _PartialFill(assets[0])
    raise ValueError(f"Unknown workload strategy: {workload.strategy}")


def _config(workload: Workload) -> BacktestConfig:
    execution_price = ExecutionPrice.QUOTE_SIDE if workload.quote_aware else ExecutionPrice.CLOSE
    return BacktestConfig(
        initial_cash=10_000_000.0,
        execution_mode=workload.execution_mode,
        execution_price=execution_price,
        mark_price=execution_price,
        commission_type=(
            CommissionType.PERCENTAGE if workload.commission_rate else CommissionType.NONE
        ),
        commission_rate=workload.commission_rate,
        slippage_type=(SlippageType.PERCENTAGE if workload.slippage_rate else SlippageType.NONE),
        slippage_rate=workload.slippage_rate,
        partial_fills_allowed=workload.strategy == "partial_fill",
    )


def _feed(workload: Workload, prices: pl.DataFrame) -> DataFeed:
    if not workload.quote_aware:
        return DataFeed(prices_df=prices)
    return DataFeed(
        prices_df=prices,
        feed_spec=FeedSpec(
            price_col="mid_price",
            bid_col="bid",
            ask_col="ask",
            bid_size_col="bid_size",
            ask_size_col="ask_size",
        ),
    )


def _canonical_float(value: float) -> float:
    return float(f"{float(value):.10g}")


def _behavior(result) -> tuple[dict[str, Any], str]:
    fills = [
        {
            "timestamp": fill.timestamp.isoformat(),
            "asset": fill.asset,
            "side": fill.side.value,
            "quantity": _canonical_float(fill.quantity),
            "price": _canonical_float(fill.price),
            "commission": _canonical_float(fill.commission),
        }
        for fill in result.fills
    ]
    trades = [
        {
            "asset": trade.symbol,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "quantity": _canonical_float(trade.quantity),
            "pnl": _canonical_float(trade.pnl),
        }
        for trade in result.trades
    ]
    behavior = {
        "fills": fills,
        "trades": trades,
        "equity": [
            [timestamp.isoformat(), _canonical_float(value)]
            for timestamp, value in result.equity_curve
        ],
        "final_value": _canonical_float(result.metrics["final_value"]),
    }
    payload = json.dumps(behavior, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return behavior, hashlib.sha256(payload.encode()).hexdigest()


def _peak_rss_mb() -> float:
    try:
        import resource
    except ImportError as exc:
        raise RuntimeError("Release performance memory evidence requires a POSIX worker") from exc
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def run_worker(name: str) -> dict[str, Any]:
    workload = WORKLOADS[name]
    setup_start = perf_counter()
    prices = _build_prices(workload)
    feed = _feed(workload, prices)
    limits = (
        VolumeParticipationLimit(max_participation=0.10)
        if workload.strategy == "partial_fill"
        else None
    )
    engine = Engine(
        feed,
        _strategy(workload),
        _config(workload),
        execution_limits=limits,
    )
    setup_seconds = perf_counter() - setup_start
    runtime_start = perf_counter()
    result = engine.run()
    runtime_seconds = perf_counter() - runtime_start
    behavior, checksum = _behavior(result)
    return {
        "workload": name,
        "data_points": workload.data_points,
        "setup_seconds": setup_seconds,
        "runtime_seconds": runtime_seconds,
        "total_measured_seconds": setup_seconds + runtime_seconds,
        "process_peak_rss_mb": _peak_rss_mb(),
        "behavior_sha256": checksum,
        "fill_count": len(behavior["fills"]),
        "trade_count": len(behavior["trades"]),
        "final_value": behavior["final_value"],
    }


def _worker_sample(name: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", name]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        raise RuntimeError(
            f"Performance worker {name!r} exceeded {WORKER_TIMEOUT_SECONDS}s"
            + (f": {stderr.strip()}" if stderr else "")
        ) from exc
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "no worker output"
        raise RuntimeError(f"Performance worker {name!r} exited {completed.returncode}: {details}")
    try:
        sample = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Performance worker {name!r} returned malformed JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(sample, dict):
        raise RuntimeError(f"Performance worker {name!r} returned a non-object JSON value")
    return sample


def _relative_deviation(values: list[float]) -> float:
    center = median(values)
    return max(abs(value - center) / center for value in values) if center > 0 else 0.0


def _workload_metadata(workload: Workload) -> dict[str, Any]:
    return {
        "bars": workload.bars,
        "assets": workload.assets,
        "data_points": workload.data_points,
        "strategy": workload.strategy,
        "quote_aware": workload.quote_aware,
        "execution_mode": workload.execution_mode.value,
        "commission_rate": workload.commission_rate,
        "slippage_rate": workload.slippage_rate,
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("Performance manifest schema_version must be 2")
    if manifest.get("measurement_contract") != MEASUREMENT_CONTRACT:
        raise ValueError("Performance manifest measurement contract differs from the runner")
    configured = manifest.get("workloads")
    if not isinstance(configured, dict):
        raise ValueError("Performance manifest workloads must be an object")
    configured_names = set(configured)
    expected_names = set(WORKLOADS)
    if configured_names != expected_names:
        raise ValueError(
            "Performance manifest workloads differ from the runner: "
            f"missing={sorted(expected_names - configured_names)}, "
            f"extra={sorted(configured_names - expected_names)}"
        )
    for name, workload in WORKLOADS.items():
        expected_metadata = _workload_metadata(workload)
        actual = configured[name]
        drift = {
            key: (actual.get(key), value)
            for key, value in expected_metadata.items()
            if actual.get(key) != value
        }
        if drift:
            raise ValueError(f"Performance manifest metadata drift for {name!r}: {drift}")


def collect_evidence(manifest: dict[str, Any], samples: int) -> dict[str, Any]:
    _validate_manifest(manifest)
    reporting_threshold = float(MEASUREMENT_CONTRACT["sample_deviation_reporting_threshold"])
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "measurement_contract": MEASUREMENT_CONTRACT,
        "samples_per_workload": samples,
        "workloads": {},
        "passed": True,
    }
    for name, expected in manifest["workloads"].items():
        runs = [_worker_sample(name) for _ in range(samples)]
        runtimes = [float(run["runtime_seconds"]) for run in runs]
        setups = [float(run["setup_seconds"]) for run in runs]
        totals = [float(run["total_measured_seconds"]) for run in runs]
        memories = [float(run["process_peak_rss_mb"]) for run in runs]
        checksums = {str(run["behavior_sha256"]) for run in runs}
        runtime_deviation = _relative_deviation(runtimes)
        memory_deviation = _relative_deviation(memories)
        checks = {
            "behavior_checksum": checksums == {expected["behavior_sha256"]},
            "data_points": all(run["data_points"] == expected["data_points"] for run in runs),
            "fill_count": all(run["fill_count"] == expected["expected_fill_count"] for run in runs),
            "trade_count": all(
                run["trade_count"] == expected["expected_trade_count"] for run in runs
            ),
            "final_value": all(
                run["final_value"] == expected["expected_final_value"] for run in runs
            ),
        }
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        passed = not failed_checks
        evidence["workloads"][name] = {
            "passed": passed,
            "failed_checks": failed_checks,
            "behavior_sha256": sorted(checksums),
            "expected_behavior_sha256": expected["behavior_sha256"],
            "runtime_median_seconds": median(runtimes),
            "runtime_max_relative_deviation": runtime_deviation,
            "setup_median_seconds": median(setups),
            "total_median_seconds": median(totals),
            "peak_rss_median_mb": median(memories),
            "peak_rss_max_relative_deviation": memory_deviation,
            "sample_spread_within_reporting_threshold": (
                runtime_deviation <= reporting_threshold and memory_deviation <= reporting_threshold
            ),
            "samples": runs,
        }
        evidence["passed"] = bool(evidence["passed"] and passed)
    return evidence


def update_manifest(path: Path) -> None:
    workloads = {}
    for name, workload in WORKLOADS.items():
        sample = _worker_sample(name)
        workloads[name] = {
            **_workload_metadata(workload),
            "behavior_sha256": sample["behavior_sha256"],
            "expected_fill_count": sample["fill_count"],
            "expected_trade_count": sample["trade_count"],
            "expected_final_value": sample["final_value"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "measurement_contract": MEASUREMENT_CONTRACT,
                "workloads": workloads,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--worker", choices=sorted(WORKLOADS))
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(run_worker(args.worker), sort_keys=True))
        return 0
    if args.update_manifest:
        update_manifest(args.manifest)
        return 0
    if args.samples < 3:
        parser.error("release evidence requires at least three samples")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidence = collect_evidence(manifest, args.samples)
    for name, workload in evidence["workloads"].items():
        state = "PASS" if workload["passed"] else "FAIL"
        failed = ",".join(workload["failed_checks"]) or "none"
        print(
            f"{state} {name}: failed_checks={failed}, "
            f"runtime_deviation={workload['runtime_max_relative_deviation']:.3f}, "
            f"rss_deviation={workload['peak_rss_max_relative_deviation']:.3f}",
            file=sys.stderr,
        )
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
