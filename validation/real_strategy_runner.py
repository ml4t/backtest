#!/usr/bin/env python3
"""Execute a frozen real-strategy input bundle with ml4t.backtest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any

import polars as pl

from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    ContractSpec,
    DataFeed,
    Engine,
    RebalanceConfig,
    Strategy,
    TargetWeightExecutor,
)
from ml4t.backtest.profiles import get_profile_config
from ml4t.backtest.risk import RuleChain, StopLoss, TimeExit, TrailingStop

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class FundingSettlementLedger:
    """Apply frozen perpetual-futures funding during engine mark updates."""

    def __init__(self, funding: pl.DataFrame, *, skip_first_after_entry: bool = False) -> None:
        required = {"symbol", "timestamp", "funding_rate"}
        missing = required - set(funding.columns)
        if missing:
            raise ValueError(f"Funding input lacks columns: {sorted(missing)}")
        selected = funding.select("symbol", "timestamp", "funding_rate")
        if selected.null_count().row(0) != (0, 0, 0):
            raise ValueError("Funding input contains null keys or rates")
        if selected.n_unique(["symbol", "timestamp"]) != selected.height:
            raise ValueError("Funding input contains duplicate settlement keys")
        self._rates: dict[datetime, dict[str, float]] = {}
        for row in selected.sort("timestamp", "symbol").iter_rows(named=True):
            rate = float(row["funding_rate"])
            if not math.isfinite(rate):
                raise ValueError("Funding rates must be finite")
            timestamp = _as_utc(row["timestamp"])
            self._rates.setdefault(timestamp, {})[str(row["symbol"])] = rate
        self._rate_count = selected.height
        self._symbols = sorted(selected["symbol"].unique().to_list())
        self._settled_timestamps: set[datetime] = set()
        self._skip_first_after_entry = skip_first_after_entry
        self._next_application: dict[str, datetime | None] = {}
        self.funding_pnl = 0.0
        self.funding_events = 0
        self.funding_settlements = 0

    def install(self, broker: Any) -> None:
        original_update_time = broker._update_time

        @wraps(original_update_time)
        def update_time_with_funding(timestamp, *args, **kwargs):
            result = original_update_time(timestamp, *args, **kwargs)
            self.settle(timestamp, broker)
            return result

        broker._update_time = update_time_with_funding

    def settle(self, timestamp: datetime, broker: Any) -> None:
        normalized = _as_utc(timestamp)
        rates = self._rates.get(normalized)
        if rates is not None and normalized not in self._settled_timestamps:
            self._settled_timestamps.add(normalized)
            self.funding_settlements += len(rates)
        event_cash = 0.0
        symbols = self._symbols if self._skip_first_after_entry else list(rates or ())
        for symbol in symbols:
            position = broker.positions.get(symbol)
            if position is None or float(position.quantity) == 0.0:
                self._next_application[symbol] = None
                continue
            if self._skip_first_after_entry and self._next_application.get(symbol) is None:
                self._next_application[symbol] = self._next_funding_time(normalized)
                continue
            if rates is None or symbol not in rates:
                continue
            if self._skip_first_after_entry:
                next_application = self._next_application[symbol]
                if next_application is None or normalized < next_application:
                    continue
                while next_application <= normalized:
                    next_application += timedelta(hours=8)
                self._next_application[symbol] = next_application
            rate = rates[symbol]
            mark = broker.get_mark_price(symbol, quantity=position.quantity)
            if mark is None:
                raise RuntimeError(f"Funding settlement lacks a mark for {symbol!r}")
            event_cash -= (
                float(position.quantity)
                * float(mark)
                * float(getattr(position, "multiplier", 1.0))
                * rate
            )
        if event_cash:
            broker.cash = float(broker.cash) + event_cash
            self.funding_pnl += event_cash
            self.funding_events += 1

    @staticmethod
    def _next_funding_time(current: datetime) -> datetime:
        if current.hour >= 16:
            return (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if current.hour >= 8:
            return current.replace(hour=16, minute=0, second=0, microsecond=0)
        return current.replace(hour=8, minute=0, second=0, microsecond=0)

    def metrics(self) -> dict[str, int | float]:
        if self.funding_settlements != self._rate_count:
            raise RuntimeError(
                "Funding coverage is incomplete: "
                f"{self.funding_settlements}/{self._rate_count} settlements"
            )
        return {
            "funding_pnl": self.funding_pnl,
            "funding_events": self.funding_events,
            "funding_settlements": self.funding_settlements,
        }


def _position_rules(risk: Mapping[str, Any]) -> object | None:
    rules = []
    for rule in risk.get("position_rules", []):
        rule_type = rule["type"]
        if rule_type == "stop_loss":
            rules.append(StopLoss(pct=float(rule["threshold"])))
        elif rule_type == "trailing_stop":
            rules.append(TrailingStop(pct=float(rule["threshold"])))
        elif rule_type == "time_exit":
            rules.append(TimeExit(max_bars=int(rule["bars"])))
        else:
            raise ValueError(f"Unsupported frozen position rule: {rule_type}")
    if len(rules) == 1:
        return rules[0]
    return RuleChain(rules) if rules else None


def _contract_specs(path: Path) -> dict[str, ContractSpec] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        symbol: ContractSpec(
            symbol=symbol,
            asset_class=AssetClass(item["asset_class"]),
            multiplier=float(item["multiplier"]),
            tick_size=float(item["tick_size"]),
            margin=None if item["margin"] is None else float(item["margin"]),
            margin_pct=(
                None
                if item["margin_pct"] is None
                else (float(item["margin_pct"][0]), float(item["margin_pct"][1]))
            ),
            currency=str(item["currency"]),
        )
        for symbol, item in payload.items()
    }


def _target_map(targets: pl.DataFrame) -> dict[datetime, dict[str, float]]:
    duplicates = targets.select(pl.struct("timestamp", "symbol").is_duplicated().sum()).item()
    if duplicates:
        raise ValueError(f"Targets contain {duplicates} duplicate timestamp-symbol rows")
    result: dict[datetime, dict[str, float]] = {}
    for row in targets.sort("timestamp", "symbol").iter_rows(named=True):
        result.setdefault(row["timestamp"], {})[str(row["symbol"])] = float(row["weight"])
    return result


def load_bundle(bundle: Path) -> dict[str, Any]:
    """Load and verify one content-addressed bundle before engine timing begins."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        path = bundle / name
        if not path.is_file() or _sha256(path) != identity["sha256"]:
            raise ValueError(f"Bundle file identity mismatch: {name}")
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    return {
        "manifest": manifest,
        "spec": spec,
        "market": pl.read_parquet(bundle / "market.parquet"),
        "targets": pl.read_parquet(bundle / "targets.parquet"),
        "funding": (
            pl.read_parquet(bundle / "funding.parquet")
            if (bundle / "funding.parquet").is_file()
            else None
        ),
        "contracts": _contract_specs(bundle / "contracts.json"),
    }


def _comparison_config(spec: Mapping[str, Any], profile: str) -> BacktestConfig:
    source = spec["backtest_config"]
    profile_data = (
        copy.deepcopy(source) if profile == "lean_crypto_future" else get_profile_config(profile)
    )
    profile_data["account"]["allow_short_selling"] = source["account"]["allow_short_selling"]
    if profile not in {"lean", "lean_crypto_future"}:
        profile_data["account"]["allow_leverage"] = source["account"]["allow_leverage"]
    else:
        profile_data["account"]["allow_leverage"] = True
    if profile == "lean_crypto_future":
        profile_data["account"]["initial_margin"] = 0.04
        profile_data["account"]["long_maintenance_margin"] = 0.02
        profile_data["account"]["short_maintenance_margin"] = 0.02
    profile_data["cash"]["initial"] = source["cash"]["initial"]
    profile_data["calendar"] = source["calendar"]
    profile_data["feed"] = source["feed"]
    profile_data["commission"] = {
        "model": "none",
        "rate": 0.0,
        "per_share": 0.0,
        "per_trade": 0.0,
        "minimum": 0.0,
    }
    profile_data["slippage"] = {"model": "none", "rate": 0.0}
    profile_data["orders"]["rebalance_headroom_pct"] = 1.0
    profile_data["metadata"] = {
        **source.get("metadata", {}),
        "comparison_profile": profile,
        "comparison_costs": "disabled",
        "comparison_position_rules": "disabled",
    }
    return BacktestConfig.from_dict(profile_data, preset_name=profile)


def execute_bundle(
    inputs: Mapping[str, Any],
    *,
    comparison_profile: str | None = None,
    price_decimals: int | None = None,
    execution_specs: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[Any, float, dict[str, int | float]]:
    """Run the engine once and return its native result and engine-only runtime."""
    spec = inputs["spec"]
    config = (
        _comparison_config(spec, comparison_profile)
        if comparison_profile is not None
        else BacktestConfig.from_dict(spec["backtest_config"])
    )
    targets = _target_map(inputs["targets"])
    rebalance = spec["strategy"]["rebalance"]
    rules = (
        None
        if comparison_profile is not None
        else _position_rules(spec["strategy"].get("risk", {}))
    )
    funding = inputs["funding"]
    funding_ledger = (
        FundingSettlementLedger(
            funding,
            skip_first_after_entry=comparison_profile == "lean_crypto_future",
        )
        if funding is not None
        else None
    )

    class FrozenTargetStrategy(Strategy):
        def __init__(self) -> None:
            self._rules_set = False
            self._executor = TargetWeightExecutor(
                RebalanceConfig(
                    min_trade_value=(
                        0.0
                        if comparison_profile is not None
                        else float(rebalance["min_trade_value"])
                    ),
                    min_weight_change=(
                        0.0
                        if comparison_profile is not None
                        else float(rebalance["min_weight_change"])
                    ),
                    allow_fractional=None,
                    allow_short=bool(config.allow_short_selling),
                    rebalance_mode=config.rebalance_mode,
                    share_rounding=config.share_rounding,
                )
            )

        def on_start(self, broker) -> None:
            if funding_ledger is not None:
                funding_ledger.install(broker)

        def on_data(self, timestamp, data, context, broker) -> None:
            if not self._rules_set:
                if rules is not None:
                    broker.set_position_rules(rules)
                self._rules_set = True
            target = targets.get(timestamp)
            if target is not None:
                available = {symbol: weight for symbol, weight in target.items() if symbol in data}
                if available:
                    if execution_specs is None:
                        self._executor.execute(available, data, broker)
                    else:
                        value = broker.equity()
                        current = {
                            symbol: (
                                float(position.quantity) * float(data[symbol]["close"]) / value
                            )
                            for symbol, position in broker.positions.items()
                            if symbol in data and float(position.quantity) != 0.0
                        }
                        active = set(available) | set(current)
                        reductions = sorted(
                            symbol
                            for symbol in active
                            if available.get(symbol, 0.0) < current.get(symbol, 0.0)
                        )
                        increases = sorted(active - set(reductions))
                        for symbol in reductions + increases:
                            if symbol not in data:
                                continue
                            price = float(data[symbol]["close"])
                            lot = float(execution_specs[symbol]["lot_size"])
                            raw = (
                                (available.get(symbol, 0.0) - current.get(symbol, 0.0))
                                * value
                                / price
                            )
                            quantity = math.copysign(
                                math.floor(abs(raw) / lot + 0.5) * lot,
                                raw,
                            )
                            if abs(quantity) >= lot:
                                broker.submit_order(symbol, quantity)

    market = inputs["market"]
    if price_decimals is not None:
        market = market.with_columns(
            pl.col(column).round(price_decimals)
            for column in ("open", "high", "low", "close")
            if column in market.columns
        )
    feed = DataFeed(prices_df=market, feed_spec=config.feed_spec)
    engine = Engine.from_config(
        feed,
        FrozenTargetStrategy(),
        config,
        contract_specs=inputs["contracts"],
    )
    started = time.perf_counter()
    result = engine.run()
    engine_seconds = time.perf_counter() - started
    funding_metrics = funding_ledger.metrics() if funding_ledger is not None else {}
    return result, engine_seconds, funding_metrics


def _write_evidence(
    *,
    output: Path,
    inputs: Mapping[str, Any],
    result: Any,
    engine_seconds: float,
    funding_metrics: Mapping[str, int | float],
    comparison_profile: str | None,
    price_decimals: int | None,
    execution_specs_sha256: str | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        frames = {
            "fills.parquet": result.to_fills_dataframe(),
            "rejected_orders.parquet": result.to_rejected_orders_dataframe(),
            "trades.parquet": result.to_trades_dataframe(),
            "equity.parquet": result.to_equity_dataframe(),
            "portfolio_state.parquet": result.to_portfolio_state_dataframe(),
        }
        for name, frame in frames.items():
            frame.write_parquet(staging / name, compression="zstd", statistics=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "case_study": inputs["manifest"]["case_study"],
            "framework": "ml4t_backtest",
            "comparison_profile": comparison_profile,
            "price_decimals": price_decimals,
            "execution_specs_sha256": execution_specs_sha256,
            "input_bundle_sha256": inputs["manifest"]["bundle_sha256"],
            "engine_seconds": engine_seconds,
            "final_value": float(result.metrics["final_value"]),
            "num_fills": len(result.fills),
            "num_rejections": len(result.rejected_orders),
            "num_trades": len(result.trades),
            "funding": dict(funding_metrics),
            "files": {},
        }
        manifest["files"] = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(staging.iterdir())
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    parser.add_argument("--comparison-profile")
    parser.add_argument("--price-decimals", type=int)
    parser.add_argument("--execution-specs", type=Path)
    args = parser.parse_args()
    inputs = load_bundle(args.bundle.resolve())
    execution_specs = (
        json.loads(args.execution_specs.read_text(encoding="utf-8"))
        if args.execution_specs is not None
        else None
    )
    result, engine_seconds, funding_metrics = execute_bundle(
        inputs,
        comparison_profile=args.comparison_profile,
        price_decimals=args.price_decimals,
        execution_specs=execution_specs,
    )
    _write_evidence(
        output=args.output.resolve(),
        inputs=inputs,
        result=result,
        engine_seconds=engine_seconds,
        funding_metrics=funding_metrics,
        comparison_profile=args.comparison_profile,
        price_decimals=args.price_decimals,
        execution_specs_sha256=(
            _sha256(args.execution_specs) if args.execution_specs is not None else None
        ),
    )
    print(
        f"{inputs['manifest']['case_study']}: {len(result.fills):,} fills, "
        f"{len(result.trades):,} trades, {engine_seconds:.6f}s engine time"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
