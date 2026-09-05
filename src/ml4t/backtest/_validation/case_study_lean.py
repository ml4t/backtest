"""Case-study LEAN parity: logged LEAN artifacts vs ml4t-backtest[lean].

This reconstructs the case-study parity harness used by the Chapter 16
``16_case_study_lean_parity`` notebook. Each ``chapter16_<case_study>`` LEAN
workspace is self-contained (``main.py`` + ``weights.csv`` + ``rebalance_dates.csv``
+ ``asset_symbols.csv`` + LEAN daily zips), so both sides reproduce from it:

* **LEAN side** -- load the committed artifacts produced by the workspace
  algorithm: ``ml4t_order_events.csv`` and ``ml4t_daily_equity.csv``. Regenerate
  them with :func:`validation.benchmark_suite.run_lean_backtest`.
* **ml4t side** -- decode the workspace's own daily zips (so both engines consume
  byte-identical prices), then replay the identical target-weight strategy through
  the ``lean`` profile.

Parity is asserted on the sorted daily fill multiset
``(timestamp, asset, side, quantity, 4-decimal price)`` plus terminal portfolio value.

The ml4t strategy mirrors the workspace ``main.py`` exactly, including LEAN's
fill-forward semantics: a name dropped from the target universe is still sized
(off its last known close) and liquidated at the next real bar, rather than left
untouched once it has no bar on a rebalance day.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import pandas as pd
import polars as pl

from ..config import BacktestConfig, CommissionType, SlippageType
from ..datafeed import DataFeed
from ..engine import Engine
from ..strategy import Strategy

_NUMBER = r"[-+]?(?:\d[\d_]*(?:\.\d[\d_]*)?|\.\d[\d_]*)(?:[eE][-+]?\d[\d_]*)?"
_MONEY_QUANTUM = Decimal("0.0001")
RETAINED_ORDER_EVENT_ARTIFACTS = {
    "chapter16_etfs": ("ml4t_order_events.csv.xz",),
    "chapter16_sp500_equity_option_analytics": ("ml4t_order_events.csv",),
    "chapter16_us_equities_panel": (
        "ml4t_order_events.csv.part01.xz",
        "ml4t_order_events.csv.part02.xz",
    ),
}


def _csv_path(base_dir: Path, name: str) -> Path:
    if name == "ml4t_order_events.csv":
        retained = RETAINED_ORDER_EVENT_ARTIFACTS.get(base_dir.name)
        if retained and len(retained) == 1:
            candidate = base_dir / retained[0]
            if candidate.is_file():
                return candidate
    for suffix in (".xz", ".gz", ""):
        path = base_dir / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(base_dir / name)


def _read_csv_fixture(base_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    parts = sorted(base_dir.glob(f"{name}.part*.xz"))
    if parts:
        return pd.concat((pd.read_csv(part, **kwargs) for part in parts), ignore_index=True)
    return pd.read_csv(_csv_path(base_dir, name), **kwargs)


def parse_workspace_params(workspace_dir: Path) -> dict:
    """Read start/end/cash/fee from a chapter16 LEAN workspace ``main.py``."""
    text = (workspace_dir / "main.py").read_text(encoding="utf-8")

    def _date(macro: str) -> str:
        m = re.search(rf"set_{macro}_date\((\d+),\s*(\d+),\s*(\d+)\)", text)
        if not m:
            raise ValueError(f"no set_{macro}_date in {workspace_dir / 'main.py'}")
        y, mo, d = (int(x) for x in m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d}"

    def _number(pattern: str, name: str) -> float:
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"no {name} in {workspace_dir / 'main.py'}")
        return float(m.group(1).replace("_", ""))

    return {
        "start": _date("start"),
        "end": _date("end"),
        "initial_cash": _number(rf"set_cash\(\s*({_NUMBER})\s*\)", "set_cash"),
        "fee": _number(rf"Ml4tPercentFeeModel\(\s*({_NUMBER})\s*\)", "fee model"),
    }


def _decode_lean_zip(zip_path: Path) -> pd.DataFrame:
    """Decode a LEAN daily OHLCV zip (prices stored as x10000 integers)."""
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        dt, o, h, lo, c, v = line.split(",")
        rows.append(
            {
                "timestamp": pd.Timestamp(dt.split()[0]),
                "open": int(o) / 10000.0,
                "high": int(h) / 10000.0,
                "low": int(lo) / 10000.0,
                "close": int(c) / 10000.0,
                "volume": float(v),
            }
        )
    return pd.DataFrame(rows)


def load_workspace(workspace_dir: Path, data_daily: Path) -> dict:
    """Load weights, rebalance dates, asset map and price panel for one workspace."""
    params = parse_workspace_params(workspace_dir)

    asset_to_ticker: dict[str, str] = {}
    with (workspace_dir / "asset_symbols.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            a, t = row["asset"].strip(), row["ticker"].strip()
            if a and t:
                asset_to_ticker[a] = t
    asset_order = list(asset_to_ticker)

    targets: dict[str, dict[str, float]] = {}
    for row in _read_csv_fixture(workspace_dir, "weights.csv").to_dict("records"):
        targets.setdefault(row["timestamp"], {})[row["asset"]] = float(row["target_weight"])

    rebalance_dates = {
        ln.strip()
        for ln in (workspace_dir / "rebalance_dates.csv").read_text().splitlines()
        if ln.strip()
    }

    start = pd.Timestamp(params["start"]).to_pydatetime()
    end = pd.Timestamp(params["end"]).to_pydatetime()
    frames = []
    for asset, ticker in asset_to_ticker.items():
        df = _decode_lean_zip(data_daily / f"{ticker.lower()}.zip")
        df["symbol"] = asset
        frames.append(df)
    prices_pd = pd.concat(frames, ignore_index=True)
    prices = (
        pl.DataFrame({str(name): series.to_list() for name, series in prices_pd.items()})
        .select("symbol", "timestamp", "open", "high", "low", "close", "volume")
        .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        .sort("timestamp", "symbol")
    )
    return {
        "params": params,
        "asset_order": asset_order,
        "asset_to_ticker": asset_to_ticker,
        "targets": targets,
        "rebalance_dates": rebalance_dates,
        "prices": prices,
    }


class CaseStudyWeightStrategy(Strategy):
    """Target-weight rebalance mirroring the LEAN workspace ``main.py``.

    On each rebalance date, size ``target_qty = signed_floor(weight * equity / close)``
    and submit the delta as a market order. Includes LEAN-style fill-forward: an
    asset with no real bar on a rebalance day is still sized off its last known
    close (so a dropped name liquidates), until its final real bar (delisting).
    """

    def __init__(self, asset_order, targets, rebalance_dates, last_active):
        self.asset_order = asset_order
        self.targets = targets
        self.rebalance_dates = rebalance_dates
        self.last_active = last_active
        self.last_close: dict[str, float] = {}

    @staticmethod
    def _target_quantity(weight: float, portfolio_value: float, price: float) -> int:
        raw = (weight * portfolio_value) / price
        if raw >= 0:
            return int(math.floor(raw + 1e-12))
        return -int(math.floor(abs(raw) + 1e-12))

    def on_data(self, timestamp, data, context, broker):
        for asset in self.asset_order:
            bar = data.get(asset)
            if bar is not None:
                px = float(bar["close"])
                if px > 0:
                    self.last_close[asset] = px

        ts = pd.Timestamp(timestamp).normalize()
        if ts.strftime("%Y-%m-%d") not in self.rebalance_dates:
            return
        targets = self.targets.get(ts.strftime("%Y-%m-%d"), {})
        pv = broker.equity()
        for asset in self.asset_order:
            if ts > self.last_active.get(asset, ts):
                continue
            price = self.last_close.get(asset)
            if price is None or price <= 0:
                continue
            target_qty = self._target_quantity(targets.get(asset, 0.0), pv, price)
            pos = broker.get_position(asset)
            current = int(pos.quantity) if pos else 0
            delta = target_qty - current
            if delta != 0:
                broker.submit_order(asset, delta)


def _surface(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.normalize()
    out["asset"] = out["asset"].astype(str)
    out["side"] = out["side"].astype(str).str.lower()
    out["quantity"] = out["quantity"].astype(float).abs()
    out["price"] = out["price"].astype(float).round(4)
    return out[["timestamp", "asset", "side", "quantity", "price"]].reset_index(drop=True)


def run_ml4t_lean(workspace_dir: Path, data_daily: Path) -> dict:
    """Run ml4t-backtest[lean] on a workspace's weights; return value + fill surface."""
    wd = load_workspace(workspace_dir, data_daily)
    cfg = BacktestConfig.from_preset("lean")
    cfg.initial_cash = wd["params"]["initial_cash"]
    cfg.allow_short_selling = True
    cfg.allow_leverage = True
    cfg.commission_type = CommissionType.PERCENTAGE
    cfg.commission_rate = wd["params"]["fee"]
    cfg.commission_per_share = 0.0
    cfg.commission_minimum = 0.0
    cfg.slippage_type = SlippageType.NONE
    cfg.slippage_rate = 0.0

    last_active = {
        row["symbol"]: pd.Timestamp(row["timestamp"])
        for row in wd["prices"].group_by("symbol").agg(pl.col("timestamp").max()).to_dicts()
    }
    feed = DataFeed(prices_df=wd["prices"])
    strat = CaseStudyWeightStrategy(
        wd["asset_order"], wd["targets"], wd["rebalance_dates"], last_active
    )
    result = Engine.from_config(feed, strat, config=cfg).run()
    fills_pl = result.to_fills_dataframe()
    fills = pd.DataFrame(fills_pl.to_dict(as_series=False)).rename(columns={"symbol": "asset"})
    return {"final_value": float(result["final_value"]), "fills": _surface(fills)}


def lean_side(workspace_dir: Path) -> dict:
    """Load the LEAN-side fills + terminal value the workspace algorithm logged.

    Reads the artifacts the LEAN ``main.py`` writes at the workspace root:
    ``ml4t_order_events.csv`` (fills), ``ml4t_daily_equity.csv`` (terminal value),
    ``ml4t_symbol_map.json`` (obfuscated ticker -> real asset).
    """
    symbol_map = json.loads((workspace_dir / "ml4t_symbol_map.json").read_text(encoding="utf-8"))
    equity = _read_csv_fixture(workspace_dir, "ml4t_daily_equity.csv")
    final_value = float(equity["equity"].iloc[-1])

    events = _read_csv_fixture(workspace_dir, "ml4t_order_events.csv", low_memory=False)
    filled = events[events["status"].astype(str).str.lower() == "filled"].copy()
    filled["asset"] = filled["symbol"].astype(str).map(symbol_map).fillna(filled["symbol"])
    filled = filled.rename(columns={"fill_quantity": "quantity", "fill_price": "price"})
    filled["side"] = filled["direction"].astype(str).str.lower()
    fills = _surface(filled)
    return {"final_value": final_value, "fills": fills, "n_trades": len(fills)}


def compare(lean: dict, ml4t: dict) -> dict:
    """Compare LEAN vs ml4t fill surfaces + terminal value."""
    cols = ["timestamp", "asset", "side", "quantity", "price"]
    ls = lean["fills"].sort_values(cols).reset_index(drop=True)
    ms = ml4t["fills"].sort_values(cols).reset_index(drop=True)
    multiset_match = len(ls) == len(ms) and not (ls[cols] != ms[cols]).any().any()
    raw_match = (
        len(ls) == len(ms)
        and not (lean["fills"][cols].reset_index(drop=True) != ml4t["fills"][cols]).any().any()
    )
    lean_value = Decimal(str(lean["final_value"])).quantize(
        _MONEY_QUANTUM, rounding=ROUND_HALF_EVEN
    )
    ml4t_value = Decimal(str(ml4t["final_value"])).quantize(
        _MONEY_QUANTUM, rounding=ROUND_HALF_EVEN
    )
    return {
        "lean_final_value": lean["final_value"],
        "ml4t_final_value": ml4t["final_value"],
        "final_value_gap_usd": ml4t["final_value"] - lean["final_value"],
        "final_value_gap_pct": ml4t["final_value"] / lean["final_value"] - 1.0,
        "canonical_final_value_gap_usd": float(ml4t_value - lean_value),
        "canonical_final_value_match": ml4t_value == lean_value,
        "final_value_quantum_usd": float(_MONEY_QUANTUM),
        "lean_fills": len(lean["fills"]),
        "ml4t_fills": len(ml4t["fills"]),
        "fill_gap": len(ml4t["fills"]) - len(lean["fills"]),
        "sorted_fill_multiset_match": bool(multiset_match),
        "raw_row_order_match": bool(raw_match),
    }
