#!/usr/bin/env python3
"""Run a frozen real target-allocation workload with VectorBT."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
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


def _wide(frame: pl.DataFrame, value: str) -> pd.DataFrame:
    pivoted = frame.select("timestamp", "symbol", value).pivot(
        on="symbol", index="timestamp", values=value
    )
    result = pd.DataFrame(pivoted.sort("timestamp").to_dict(as_series=False)).set_index("timestamp")
    result.index = pd.DatetimeIndex(result.index)
    return result


def run(bundle: Path, framework: str) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    package = "vectorbtpro" if framework == "vectorbt_pro" else "vectorbt"
    vbt = importlib.import_module(package)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    case_study = manifest["case_study"]
    if case_study == "cme_futures" and framework != "vectorbt_pro":
        raise ValueError("VectorBT OSS lacks the contract multiplier used by this workload")
    if case_study not in {"etfs", "cme_futures", "fx_pairs", "us_equities_panel"}:
        raise ValueError("This adapter accepts the ETF, US equity, FX, and CME futures workloads")
    spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    market = filter_comparison_market(pl.read_parquet(bundle / "market.parquet"), spec)
    close = _wide(market, "close")
    weights = _wide(
        filter_comparison_targets(pl.read_parquet(bundle / "targets.parquet"), spec), "weight"
    ).reindex(index=close.index, columns=close.columns)
    rebalance_rows = weights.notna().any(axis=1)
    weights.loc[rebalance_rows] = weights.loc[rebalance_rows].fillna(0.0)
    kwargs: dict[str, Any] = {
        "close": close,
        "size": weights,
        "size_type": "targetpercent",
        "init_cash": float(spec["backtest_config"]["cash"]["initial"]),
        "fees": 0.0,
        "slippage": 0.0,
        "cash_sharing": True,
        "call_seq": "auto",
    }
    if framework == "vectorbt_oss":
        kwargs["lock_cash"] = True
    elif case_study == "cme_futures":
        contracts = json.loads((bundle / "contracts.json").read_text(encoding="utf-8"))
        kwargs["multiplier"] = np.asarray(
            [float(contracts[symbol]["multiplier"]) for symbol in close.columns]
        )
        kwargs["leverage"] = 100.0
        kwargs["leverage_mode"] = "eager"
    started = time.perf_counter()
    portfolio = vbt.Portfolio.from_orders(**kwargs)
    engine_seconds = time.perf_counter() - started

    orders = portfolio.orders.records_readable
    fill_rows = [
        {
            "timestamp": row.get("Timestamp", row.get("Index")),
            "asset": row.get("Column"),
            "side": "buy" if row.get("Side") == "Buy" else "sell",
            "quantity": abs(float(row.get("Size", 0.0))),
            "price": float(row.get("Price", 0.0)),
            "commission": float(row.get("Fees", 0.0)),
        }
        for _, row in orders.iterrows()
    ]
    value_attr = portfolio.value
    value = value_attr() if callable(value_attr) else value_attr
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0] if value.shape[1] == 1 else value.sum(axis=1)
    equity = pl.DataFrame(
        {
            "timestamp": pl.Series(
                pd.DatetimeIndex(value.index).to_pydatetime().tolist(), dtype=pl.Datetime("us")
            ),
            "equity": value.to_numpy(),
        }
    )
    fills = pl.DataFrame(fill_rows)
    evidence = {
        "schema_version": 1,
        "case_study": manifest["case_study"],
        "framework": framework,
        "comparison_profile": (
            "vectorbt_futures_strict"
            if case_study == "cme_futures"
            else "vectorbt_strict"
            if framework == "vectorbt_pro"
            else "vectorbt_oss_strict"
        ),
        "comparison_scope": comparison_scope(spec),
        "comparison_costs": "disabled",
        "comparison_position_rules": "disabled",
        "input_bundle_sha256": manifest["bundle_sha256"],
        "engine_seconds": engine_seconds,
        "final_value": float(value.iloc[-1]),
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
    parser.add_argument("--framework", choices=("vectorbt_pro", "vectorbt_oss"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, fills, equity = run(args.bundle.resolve(), args.framework)
    write_evidence(args.output.resolve(), evidence, fills, equity)
    print(
        f"{evidence['case_study']}: {evidence['num_fills']:,} fills, "
        f"{evidence['engine_seconds']:.6f}s engine time"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
