#!/usr/bin/env python3
"""Comprehensive Performance Benchmark Suite.

This script benchmarks ml4t.backtest against VectorBT Pro, VectorBT OSS,
Backtrader, and Zipline across realistic trading scenarios.

Key scenarios:
1. Long/short top-N/bottom-N with daily rebalancing
2. Stop-loss and take-profit orders
3. Commission and slippage models
4. Scale from small (100 assets × 1 month) to large (500 assets × 1 year)

Usage:
    # Run with VectorBT Pro
    source .venv-vectorbt-pro/bin/activate
    python validation/benchmark_suite.py --framework vbt-pro

    # Run with Backtrader
    source .venv-backtrader/bin/activate
    python validation/benchmark_suite.py --framework backtrader

    # Run with Nautilus Trader
    source .venv-nautilus/bin/activate
    python validation/benchmark_suite.py --framework nautilus

    # Run ml4t only (any venv)
    python validation/benchmark_suite.py --framework ml4t

    # Run all scenarios for specific framework
    python validation/benchmark_suite.py --framework vbt-pro --all

    # Run specific scenario
    python validation/benchmark_suite.py --framework ml4t --scenario baseline
"""

import argparse
import builtins
import gc
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import tracemalloc
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


_BENCHMARK_LOG_FILE = os.getenv("ML4T_BENCHMARK_LOG_FILE")
DEFAULT_REAL_DATA_PATH = Path("/home/stefan/Dropbox/ml4t/data/equities/us_equities.parquet")
DEFAULT_CACHE_ROOT = Path(os.getenv("ML4T_BENCHMARK_CACHE_DIR", "/tmp/ml4t-benchmark-cache"))


def _log(*args, **kwargs):
    """Write benchmark progress to stdout and optional log file."""
    kwargs = dict(kwargs)
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    flush = kwargs.pop("flush", True)

    builtins.print(*args, sep=sep, end=end, flush=flush, **kwargs)

    if _BENCHMARK_LOG_FILE:
        log_path = Path(_BENCHMARK_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(sep.join(str(arg) for arg in args) + end)


def _to_naive_date(ts_like: pd.Timestamp | datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _get_nyse_sessions(start_date: pd.Timestamp | datetime | str, end_date: pd.Timestamp | datetime | str) -> pd.DatetimeIndex:
    start = _to_naive_date(start_date)
    end = _to_naive_date(end_date)
    if start > end:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")

    try:
        import pandas_market_calendars as mcal

        schedule = mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end)
        if not schedule.empty:
            return pd.DatetimeIndex(schedule.index).tz_localize(None)
    except Exception:
        pass

    try:
        import exchange_calendars as xcals

        nyse = xcals.get_calendar("XNYS")
        first_session = _to_naive_date(nyse.first_session)
        last_session = _to_naive_date(nyse.last_session)
        clipped_start = max(start, first_session)
        clipped_end = min(end, last_session)
        if clipped_start <= clipped_end:
            sessions = nyse.sessions_in_range(clipped_start, clipped_end)
            return pd.DatetimeIndex(sessions).tz_localize(None)
    except Exception:
        pass

    return pd.bdate_range(start=start, end=end)


def _get_trailing_nyse_sessions(n_bars: int, end_date: pd.Timestamp | datetime | str | None = None) -> pd.DatetimeIndex:
    if n_bars <= 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")

    end_session = _to_naive_date(end_date if end_date is not None else pd.Timestamp.today())
    lookback_days = max(366, int(n_bars * 2.2))
    sessions = pd.DatetimeIndex([], dtype="datetime64[ns]")

    for _ in range(8):
        start_session = end_session - pd.Timedelta(days=lookback_days)
        sessions = _get_nyse_sessions(start_session, end_session)
        if len(sessions) >= n_bars:
            return sessions[-n_bars:]
        lookback_days *= 2

    return sessions


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    n_bars: int
    n_assets: int
    frequency: str  # "D" for daily, "1min" for minute
    top_n: int  # Number of assets to go long
    bottom_n: int  # Number of assets to go short (0 = long only)
    rebalance_freq: int  # Bars between rebalancing
    stop_loss: float | None = None  # e.g., 0.02 = 2%
    take_profit: float | None = None  # e.g., 0.05 = 5%
    commission_pct: float = 0.0
    slippage_pct: float = 0.0
    initial_cash: float = 1_000_000.0
    zipline_max_leverage: float | None = None
    lean_order_type: str = "market"
    lean_account_type: str = "default"
    lean_security_leverage: float | None = None
    lean_force_zero_fee: bool | None = None
    lean_force_zero_slippage: bool | None = None

    @property
    def data_points(self) -> int:
        """Total data points (n_bars × n_assets)."""
        return self.n_bars * self.n_assets


# Benchmark scenarios - progressive scaling
SCENARIOS = {
    # === Baseline: Quick sanity check ===
    "baseline": BenchmarkConfig(
        name="Baseline (100×1mo minute)",
        n_bars=8_580,  # ~1 month of minute data (390 min/day × 22 days)
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,  # Daily rebalance
    ),
    # === Scale tests ===
    "scale_1": BenchmarkConfig(
        name="Scale 1 (100×3mo minute)",
        n_bars=25_740,  # ~3 months
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
    ),
    "scale_2": BenchmarkConfig(
        name="Scale 2 (250×6mo minute)",
        n_bars=51_480,  # ~6 months
        n_assets=250,
        frequency="1min",
        top_n=25,
        bottom_n=25,
        rebalance_freq=390,
    ),
    "scale_3": BenchmarkConfig(
        name="Scale 3 (500×1yr minute)",
        n_bars=97_500,  # ~1 year (390 × 250 trading days)
        n_assets=500,
        frequency="1min",
        top_n=25,
        bottom_n=25,
        rebalance_freq=390,
    ),
    # === Daily data (faster, 10 years) ===
    "daily_baseline": BenchmarkConfig(
        name="Daily (500×10yr daily)",
        n_bars=2_520,  # 10 years of daily data
        n_assets=500,
        frequency="D",
        top_n=25,
        bottom_n=25,
        rebalance_freq=1,  # Daily rebalance
    ),
    # === Feature tests ===
    "stop_loss": BenchmarkConfig(
        name="Stop-loss (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        stop_loss=0.02,  # 2% stop-loss
    ),
    "take_profit": BenchmarkConfig(
        name="Take-profit (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        take_profit=0.05,  # 5% take-profit
    ),
    "stop_and_profit": BenchmarkConfig(
        name="Stop+Take (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        stop_loss=0.02,
        take_profit=0.05,
    ),
    "with_costs": BenchmarkConfig(
        name="With costs (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        commission_pct=0.001,  # 10 bps
        slippage_pct=0.0005,  # 5 bps
    ),
    # === Long-only variant ===
    "long_only": BenchmarkConfig(
        name="Long-only (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=25,
        bottom_n=0,  # No shorts
        rebalance_freq=390,
    ),
    # === Single-asset scale tests ===
    "single_10yr": BenchmarkConfig(
        name="Single-asset (1×10yr daily)",
        n_bars=2_520,  # 10 years of daily data
        n_assets=1,
        frequency="D",
        top_n=1,
        bottom_n=0,
        rebalance_freq=1,
    ),
    "single_20yr": BenchmarkConfig(
        name="Single-asset (1×20yr daily)",
        n_bars=5_040,  # 20 years of daily data
        n_assets=1,
        frequency="D",
        top_n=1,
        bottom_n=0,
        rebalance_freq=1,
    ),
    "single_50yr": BenchmarkConfig(
        name="Single-asset (1×50yr daily)",
        n_bars=12_600,  # 50 years of daily data
        n_assets=1,
        frequency="D",
        top_n=1,
        bottom_n=0,
        rebalance_freq=1,
    ),
    # === Multi-asset scale tests ===
    "multi_500_10yr": BenchmarkConfig(
        name="Multi-asset (500×10yr daily)",
        n_bars=2_520,
        n_assets=500,
        frequency="D",
        top_n=25,
        bottom_n=25,
        rebalance_freq=1,
    ),
    "multi_100_10yr": BenchmarkConfig(
        name="Multi-asset (100×10yr daily)",
        n_bars=2_520,
        n_assets=100,
        frequency="D",
        top_n=25,
        bottom_n=25,
        rebalance_freq=1,
    ),
    "multi_250_20yr": BenchmarkConfig(
        name="Multi-asset (250×20yr daily)",
        n_bars=5_040,
        n_assets=250,
        frequency="D",
        top_n=25,
        bottom_n=25,
        rebalance_freq=1,
    ),
    "multi_1000_10yr": BenchmarkConfig(
        name="Multi-asset (1000×10yr daily)",
        n_bars=2_520,
        n_assets=1000,
        frequency="D",
        top_n=50,
        bottom_n=50,
        rebalance_freq=1,
    ),
    # === Parameter sweep simulation ===
    "param_sweep_base": BenchmarkConfig(
        name="Param sweep base (100×1yr)",
        n_bars=252,
        n_assets=100,
        frequency="D",
        top_n=10,
        bottom_n=10,
        rebalance_freq=1,
        stop_loss=0.02,
        take_profit=0.05,
    ),
}


def generate_benchmark_data(config: BenchmarkConfig, seed: int = 42) -> tuple:
    """Generate synthetic market data and signals for benchmarking.

    Returns:
        Tuple of (price_data, signals, dates) where:
        - price_data: dict of asset_name -> DataFrame with OHLCV
        - signals: DataFrame with timestamp, asset, score columns
        - dates: DatetimeIndex
    """
    np.random.seed(seed)

    # Generate dates
    if config.frequency == "1min":
        # Generate minute bars (market hours only: 9:30-16:00 = 390 mins/day)
        start = datetime(2023, 1, 3, 9, 30)  # First trading day
        dates = []
        current = start
        bars_generated = 0
        while bars_generated < config.n_bars:
            # Add minute
            dates.append(current)
            bars_generated += 1
            current += timedelta(minutes=1)
            # Skip to next day at 16:00
            if current.hour == 16 and current.minute == 0:
                current = current.replace(hour=9, minute=30) + timedelta(days=1)
                # Skip weekends
                while current.weekday() >= 5:
                    current += timedelta(days=1)
        dates = pd.DatetimeIndex(dates)
    else:
        dates = _get_trailing_nyse_sessions(config.n_bars)
        if len(dates) < config.n_bars:
            raise ValueError(
                f"Requested {config.n_bars} bars, but only {len(dates)} NYSE sessions are available"
            )

    n_bars = len(dates)

    # Generate price data for each asset
    price_data = {}
    for i in range(config.n_assets):
        asset_name = f"ASSET_{i:03d}"
        base_price = 50.0 + np.random.rand() * 150  # $50-200 starting price

        # Generate realistic returns (different vol/drift per asset)
        daily_vol = 0.01 + np.random.rand() * 0.03  # 1-4% daily vol
        drift = -0.0001 + np.random.rand() * 0.0002  # Small drift

        if config.frequency == "1min":
            # Scale vol for minute bars
            bar_vol = daily_vol / np.sqrt(390)
            bar_drift = drift / 390
        else:
            bar_vol = daily_vol
            bar_drift = drift

        returns = np.random.randn(n_bars) * bar_vol + bar_drift
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV with realistic intraday patterns
        high_mult = 1 + np.abs(np.random.randn(n_bars)) * bar_vol
        low_mult = 1 - np.abs(np.random.randn(n_bars)) * bar_vol
        open_offset = np.random.randn(n_bars) * bar_vol * 0.3

        price_data[asset_name] = pd.DataFrame(
            {
                "open": prices * (1 + open_offset),
                "high": prices * high_mult,
                "low": prices * low_mult,
                "close": prices,
                "volume": np.random.randint(10000, 1000000, n_bars).astype(float),
            },
            index=dates,
        )

    # Generate signals (scores for ranking)
    # Score = momentum + noise, changes each rebalance period
    signal_rows = []
    rebalance_bars = list(range(0, n_bars, config.rebalance_freq))

    for bar_idx in rebalance_bars:
        ts = dates[bar_idx]
        # Generate random scores for all assets
        scores = np.random.randn(config.n_assets)
        for i, score in enumerate(scores):
            signal_rows.append(
                {
                    "timestamp": ts,
                    "asset": f"ASSET_{i:03d}",
                    "score": score,
                }
            )

    signals = pd.DataFrame(signal_rows)

    return price_data, signals, dates


def load_real_benchmark_data(
    config: BenchmarkConfig,
    parquet_path: Path,
    seed: int = 42,
    cache_mode: str = "auto",
) -> tuple:
    """Load benchmark data from real daily OHLCV parquet data."""
    if config.frequency != "D":
        raise ValueError("Real-data mode currently supports daily scenarios only")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Real data parquet not found: {parquet_path}")
    if cache_mode not in {"auto", "off", "refresh"}:
        raise ValueError(f"Invalid cache_mode: {cache_mode}")

    resolved_path = parquet_path.resolve()
    stat = resolved_path.stat()
    cache_sig = hashlib.md5(
        (
            f"schema=real_data_v2_nyse_sessions|"
            f"path={resolved_path}|size={stat.st_size}|mtime={stat.st_mtime_ns}|"
            f"bars={config.n_bars}|assets={config.n_assets}|rebalance={config.rebalance_freq}|seed={seed}"
        ).encode()
    ).hexdigest()
    cache_dir = DEFAULT_CACHE_ROOT / "real_data"
    cache_file = cache_dir / f"{cache_sig}.pkl"

    if cache_mode == "auto" and cache_file.exists():
        with cache_file.open("rb") as fh:
            payload = pickle.load(fh)
        _log(f"  Real data cache hit: {cache_file}")
        return payload["price_data"], payload["signals"], payload["dates"]

    use_columns = [
        "ticker",
        "date",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    ]
    raw = pd.read_parquet(parquet_path, columns=use_columns)
    raw = raw.rename(
        columns={
            "adj_open": "open",
            "adj_high": "high",
            "adj_low": "low",
            "adj_close": "close",
            "adj_volume": "volume",
        }
    )
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.dropna(subset=["ticker", "date", "open", "high", "low", "close", "volume"])

    session_dates = _get_nyse_sessions(raw["date"].min(), raw["date"].max())

    if len(session_dates) < config.n_bars:
        raise ValueError(
            f"Requested {config.n_bars} bars, but real dataset only has {len(session_dates)} sessions"
        )
    dates = session_dates[-config.n_bars:]

    window = raw[raw["date"].isin(dates)].copy()
    counts = window.groupby("ticker")["date"].nunique().sort_values(ascending=False)
    if len(counts) < config.n_assets:
        raise ValueError(
            f"Requested {config.n_assets} assets, but only {len(counts)} have rows in selected window"
        )

    selected_assets = sorted(counts.head(config.n_assets).index.tolist())
    window = window[window["ticker"].isin(selected_assets)]

    panels: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        panel = window.pivot(index="date", columns="ticker", values=field)
        panel = panel.reindex(index=dates, columns=selected_assets).sort_index()
        panel = panel.ffill().bfill()
        panels[field] = panel

    price_data: dict[str, pd.DataFrame] = {}
    for asset in selected_assets:
        df = pd.DataFrame(
            {
                "open": panels["open"][asset].astype(float),
                "high": panels["high"][asset].astype(float),
                "low": panels["low"][asset].astype(float),
                "close": panels["close"][asset].astype(float),
                "volume": panels["volume"][asset].astype(float).clip(lower=1.0),
            },
            index=dates,
        )
        high = df[["open", "high", "low", "close"]].max(axis=1)
        low = df[["open", "high", "low", "close"]].min(axis=1)
        df["high"] = high
        df["low"] = low
        price_data[asset] = df

    np.random.seed(seed)
    signal_rows = []
    rebalance_bars = list(range(0, config.n_bars, config.rebalance_freq))
    for bar_idx in rebalance_bars:
        ts = dates[bar_idx]
        scores = np.random.randn(len(selected_assets))
        for i, score in enumerate(scores):
            signal_rows.append(
                {
                    "timestamp": ts,
                    "asset": selected_assets[i],
                    "score": score,
                }
            )
    signals = pd.DataFrame(signal_rows)

    if cache_mode != "off":
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"price_data": price_data, "signals": signals, "dates": dates}
        with cache_file.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _log(f"  Real data cache saved: {cache_file}")

    return price_data, signals, dates


@dataclass(frozen=True)
class CanonicalTopBottomSpec:
    """Canonical strategy contract shared by all benchmark adapters."""

    signal_column: str = "score"
    long_shares: float = 100.0
    short_shares: float = -100.0
    rank_method: str = "first"


CANONICAL_TOP_BOTTOM_SPEC = CanonicalTopBottomSpec()


def build_canonical_target_shares(
    config: BenchmarkConfig,
    signals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    asset_names: list[str],
    spec: CanonicalTopBottomSpec = CANONICAL_TOP_BOTTOM_SPEC,
) -> pd.DataFrame:
    """Build canonical per-bar target shares from ranked signals."""
    signal_pivot = signals.pivot(index="timestamp", columns="asset", values=spec.signal_column)
    signal_pivot = signal_pivot.reindex(index=dates, columns=asset_names).ffill()

    ranks = signal_pivot.rank(axis=1, ascending=False, method=spec.rank_method)

    long_mask = ranks <= config.top_n
    short_mask = (
        ranks > (config.n_assets - config.bottom_n)
        if config.bottom_n > 0
        else pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
    )

    target_shares = pd.DataFrame(0.0, index=dates, columns=asset_names)
    target_shares[long_mask.reindex(target_shares.index).ffill().fillna(False)] = spec.long_shares
    if config.bottom_n > 0:
        target_shares[short_mask.reindex(target_shares.index).ffill().fillna(False)] = (
            spec.short_shares
        )

    return target_shares


def build_canonical_target_lookup(target_shares: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    """Build sparse timestamp -> non-zero target map."""
    target_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    values = target_shares.to_numpy()
    columns = target_shares.columns.to_numpy()

    for idx, ts in enumerate(target_shares.index):
        row = values[idx]
        nz_idx = np.flatnonzero(row)
        if len(nz_idx) == 0:
            target_lookup[pd.Timestamp(ts)] = {}
            continue
        target_lookup[pd.Timestamp(ts)] = {str(columns[i]): float(row[i]) for i in nz_idx}

    return target_lookup


def build_canonical_target_trace(target_shares: pd.DataFrame) -> pd.DataFrame:
    """Create an event trace of target changes for cross-engine debugging."""
    prev = target_shares.shift(1).fillna(0.0)
    delta = target_shares - prev
    changed = delta != 0.0

    trace_targets = target_shares.where(changed).stack()
    trace_prev = prev.where(changed).stack()
    trace_delta = delta.where(changed).stack()

    if len(trace_targets) == 0:
        return pd.DataFrame(columns=["timestamp", "asset", "prev_target", "target", "delta", "action"])

    trace = pd.DataFrame(
        {
            "prev_target": trace_prev,
            "target": trace_targets,
            "delta": trace_delta,
        }
    ).reset_index()
    trace = trace.rename(columns={"level_0": "timestamp", "level_1": "asset"})

    prev_vals = trace["prev_target"].to_numpy()
    target_vals = trace["target"].to_numpy()
    actions = np.select(
        [
            (prev_vals == 0.0) & (target_vals != 0.0),
            (prev_vals != 0.0) & (target_vals == 0.0),
            (prev_vals * target_vals < 0.0),
        ],
        ["open", "close", "flip"],
        default="resize",
    )
    trace["action"] = actions
    return trace


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    framework: str
    scenario: str
    runtime_sec: float
    num_trades: int
    final_value: float
    memory_mb: float
    error: str | None = None
    trades_df: pd.DataFrame | None = None  # Trade log for validation
    positions_df: pd.DataFrame | None = None  # PyFolio positions (Backtrader/Zipline)
    transactions_df: pd.DataFrame | None = None  # PyFolio transactions (Backtrader/Zipline)
    target_trace_df: pd.DataFrame | None = None  # Canonical target-change trace
    # Enhanced metrics
    setup_time_sec: float = 0.0  # Time for data prep, bundle creation, etc.
    data_points: int = 0  # n_bars × n_assets

    @property
    def bars_per_second(self) -> float:
        """Processing speed in bars per second."""
        if self.runtime_sec > 0 and self.data_points > 0:
            return self.data_points / self.runtime_sec
        return 0.0

    @property
    def trades_per_second(self) -> float:
        """Trade generation speed."""
        if self.runtime_sec > 0 and self.num_trades > 0:
            return self.num_trades / self.runtime_sec
        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "framework": self.framework,
            "scenario": self.scenario,
            "runtime_sec": self.runtime_sec,
            "setup_time_sec": self.setup_time_sec,
            "num_trades": self.num_trades,
            "final_value": self.final_value,
            "memory_mb": self.memory_mb,
            "data_points": self.data_points,
            "bars_per_second": self.bars_per_second,
            "trades_per_second": self.trades_per_second,
            "error": self.error,
        }


def save_trades(result: BenchmarkResult, output_dir: Path):
    """Save trade log to CSV for validation."""
    if result.trades_df is not None and len(result.trades_df) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{result.framework.replace(' ', '_').lower()}_{result.scenario.replace(' ', '_').replace('×', 'x').lower()}.csv"
        filepath = output_dir / filename
        result.trades_df.to_csv(filepath, index=False)
        _log(f"  Saved trades to: {filepath}")


def generate_json_report(
    results: list[BenchmarkResult], output_path: Path, metadata: dict | None = None
):
    """Generate JSON report for CI/CD integration.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path to write JSON file
        metadata: Optional metadata (e.g., git hash, version)
    """
    import json as json_lib

    report = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "num_results": len(results),
            **(metadata or {}),
        },
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_scenarios": len(set(r.scenario for r in results)),
            "total_frameworks": len(set(r.framework for r in results)),
            "errors": sum(1 for r in results if r.error),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json_lib.dump(report, f, indent=2)
    _log(f"JSON report written to: {output_path}")


def generate_markdown_report(
    results: list[BenchmarkResult], output_path: Path, title: str = "Benchmark Results"
):
    """Generate Markdown report for human review.

    Args:
        results: List of BenchmarkResult objects
        output_path: Path to write Markdown file
        title: Report title
    """
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Total scenarios: {len(set(r.scenario for r in results))}",
        f"- Frameworks tested: {', '.join(sorted(set(r.framework for r in results)))}",
        f"- Errors: {sum(1 for r in results if r.error)}",
        "",
        "## Results by Scenario",
        "",
    ]

    # Group by scenario
    scenarios: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        if r.scenario not in scenarios:
            scenarios[r.scenario] = []
        scenarios[r.scenario].append(r)

    for scenario, scenario_results in scenarios.items():
        lines.extend(
            [
                f"### {scenario}",
                "",
                "| Framework | Runtime | Trades | Final Value | Memory | Bars/sec |",
                "|-----------|---------|--------|-------------|--------|----------|",
            ]
        )

        for r in scenario_results:
            if r.error:
                lines.append(f"| {r.framework} | ERROR | - | - | - | - |")
            else:
                runtime = (
                    f"{r.runtime_sec:.3f}s" if r.runtime_sec < 60 else f"{r.runtime_sec / 60:.1f}m"
                )
                bars_sec = f"{r.bars_per_second:,.0f}" if r.bars_per_second > 0 else "-"
                lines.append(
                    f"| {r.framework} | {runtime} | {r.num_trades:,} | "
                    f"${r.final_value:,.2f} | {r.memory_mb:.0f}MB | {bars_sec} |"
                )

        lines.append("")

    # Performance comparison if multiple frameworks
    frameworks = sorted(set(r.framework for r in results if not r.error))
    if len(frameworks) > 1:
        lines.extend(
            [
                "## Performance Comparison",
                "",
                "| Scenario | " + " | ".join(frameworks) + " |",
                "|----------|" + "|".join(["---"] * len(frameworks)) + "|",
            ]
        )

        for scenario, scenario_results in scenarios.items():
            row = [scenario]
            for fw in frameworks:
                fw_result = next((r for r in scenario_results if r.framework == fw), None)
                if fw_result and not fw_result.error:
                    row.append(f"{fw_result.runtime_sec:.3f}s")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    _log(f"Markdown report written to: {output_path}")


def compare_trades(results: list[BenchmarkResult]) -> dict:
    """Compare trades between frameworks to detect compounding errors."""
    if len(results) < 2:
        return {}

    comparisons = {}
    baseline = None
    for r in results:
        if r.framework == "ml4t.backtest" and r.trades_df is not None:
            baseline = r
            break

    if baseline is None:
        return {}

    for r in results:
        if r == baseline or r.trades_df is None:
            continue

        # Compare trade counts
        count_diff = abs(r.num_trades - baseline.num_trades)
        count_pct = count_diff / baseline.num_trades * 100 if baseline.num_trades > 0 else 0

        # Compare final values
        value_diff = abs(r.final_value - baseline.final_value)
        value_pct = value_diff / baseline.final_value * 100 if baseline.final_value > 0 else 0

        comparisons[r.framework] = {
            "trade_count_diff": count_diff,
            "trade_count_pct": count_pct,
            "final_value_diff": value_diff,
            "final_value_pct": value_pct,
        }

    return comparisons


def benchmark_ml4t(
    config: BenchmarkConfig,
    price_data: dict,
    signals: pd.DataFrame,
    dates,
    execution_mode: str = "same_bar",
    profile_override: str | None = None,
) -> BenchmarkResult:
    """Benchmark ml4t.backtest with given configuration.

    Args:
        execution_mode: "same_bar" (default, matches VectorBT) or "next_bar" (matches Backtrader)
    """
    import polars as pl

    from ml4t.backtest._validation_imports import (
        BacktestConfig,
        DataFeed,
        Engine,
        Strategy,
    )
    from ml4t.backtest.config import CommissionModel, SlippageModel

    # Select profile by execution style
    default_profile = "backtrader" if execution_mode == "next_bar" else "vectorbt"
    profile_name = profile_override or default_profile
    framework_name = "ml4t.backtest" if execution_mode == "same_bar" else "ml4t.backtest (backtrader-mode)"
    if profile_override is not None:
        framework_name = f"ml4t.backtest[{profile_name}]"

    # Convert price data to Polars format using vectorized DataFrame ops.
    price_frames: list[pd.DataFrame] = []
    for asset_name, df in price_data.items():
        asset_frame = df.reset_index()
        index_col = asset_frame.columns[0]
        asset_frame = asset_frame.rename(columns={index_col: "timestamp"})
        asset_frame["asset"] = asset_name
        price_frames.append(
            asset_frame[["timestamp", "asset", "open", "high", "low", "close", "volume"]]
        )

    prices_pd = pd.concat(price_frames, ignore_index=True)
    prices_pd["timestamp"] = pd.to_datetime(prices_pd["timestamp"])
    if getattr(prices_pd["timestamp"].dt, "tz", None) is not None:
        prices_pd["timestamp"] = prices_pd["timestamp"].dt.tz_localize(None)
    prices_pl = pl.DataFrame(
        {
            "timestamp": prices_pd["timestamp"].to_numpy(),
            "asset": prices_pd["asset"].to_numpy(),
            "open": prices_pd["open"].to_numpy(),
            "high": prices_pd["high"].to_numpy(),
            "low": prices_pd["low"].to_numpy(),
            "close": prices_pd["close"].to_numpy(),
            "volume": prices_pd["volume"].to_numpy(),
        }
    )

    asset_names = sorted(price_data.keys())
    target_shares = build_canonical_target_shares(config, signals, dates, asset_names)
    target_lookup_ts = build_canonical_target_lookup(target_shares)
    target_lookup_dt = {ts.to_pydatetime(): targets for ts, targets in target_lookup_ts.items()}
    target_trace = build_canonical_target_trace(target_shares)

    signals_pl = pl.DataFrame(
        {
            "timestamp": signals["timestamp"].to_numpy(),
            "asset": signals["asset"].to_numpy(),
            "score": signals["score"].to_numpy(),
        }
    )

    class TopBottomStrategy(Strategy):
        """Canonical target-based strategy adapter for ml4t."""

        def __init__(
            self,
            target_lookup: dict[datetime, dict[str, float]],
            stop_loss: float | None,
            take_profit: float | None,
            zipline_order_target_semantics: bool = False,
        ):
            self.target_lookup = target_lookup
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.zipline_order_target_semantics = zipline_order_target_semantics

        def on_start(self, broker):
            """Set up position rules for stop-loss and take-profit."""
            from ml4t.backtest.risk import RuleChain, StopLoss, TakeProfit

            rules = []
            if self.stop_loss is not None:
                rules.append(StopLoss(pct=self.stop_loss))
            if self.take_profit is not None:
                rules.append(TakeProfit(pct=self.take_profit))

            if rules:
                broker.set_position_rules(RuleChain(rules))

        def on_data(self, timestamp, data, context, broker):
            ts_key = pd.Timestamp(timestamp).to_pydatetime()
            targets = self.target_lookup.get(ts_key, {})
            active_assets = set(targets.keys())
            active_assets.update(broker.positions.keys())

            for asset_name in sorted(active_assets):
                position = broker.get_position(asset_name)
                current_qty = position.quantity if position else 0.0
                target_qty = targets.get(asset_name, 0.0)
                pending_orders = broker.get_pending_orders(asset_name)

                if self.zipline_order_target_semantics:
                    if current_qty != target_qty:
                        if pending_orders:
                            for pending_order in pending_orders:
                                broker.cancel_order(pending_order.order_id)
                        broker.submit_order(asset_name, target_qty - current_qty)
                    continue

                pending_qty = 0.0
                for pending_order in pending_orders:
                    pending_qty += (
                        pending_order.quantity
                        if pending_order.side.value == "buy"
                        else -pending_order.quantity
                    )
                delta = target_qty - (current_qty + pending_qty)
                if delta != 0.0:
                    # Keep one effective intent per asset to avoid pending-order buildup.
                    if pending_orders:
                        for pending_order in pending_orders:
                            broker.cancel_order(pending_order.order_id)
                    broker.submit_order(asset_name, target_qty - current_qty)

    def build_ml4t_config(no_costs: bool) -> BacktestConfig:
        cfg = BacktestConfig.from_preset(profile_name)
        cfg.initial_cash = config.initial_cash
        cfg.allow_short_selling = True
        if no_costs:
            cfg.commission_model = CommissionModel.NONE
            cfg.commission_rate = 0.0
            cfg.slippage_model = SlippageModel.NONE
            cfg.slippage_rate = 0.0
        else:
            if config.commission_pct > 0:
                cfg.commission_model = CommissionModel.PERCENTAGE
                cfg.commission_rate = config.commission_pct
            else:
                cfg.commission_model = CommissionModel.NONE
                cfg.commission_rate = 0.0
            if config.slippage_pct > 0:
                cfg.slippage_model = SlippageModel.PERCENTAGE
                cfg.slippage_rate = config.slippage_pct
            else:
                cfg.slippage_model = SlippageModel.NONE
                cfg.slippage_rate = 0.0
        return cfg

    # Warm-up run (smaller data)
    n_warmup = min(1000, config.n_bars // 10)
    warmup_prices = prices_pl.head(n_warmup * config.n_assets)
    warmup_signals = signals_pl.filter(pl.col("timestamp") <= dates[n_warmup])
    warmup_feed = DataFeed(prices_df=warmup_prices, signals_df=warmup_signals)
    warmup_engine = Engine.from_config(
        warmup_feed,
        TopBottomStrategy(
            target_lookup_dt,
            config.stop_loss,
            config.take_profit,
            zipline_order_target_semantics=profile_name == "zipline_strict",
        ),
        config=build_ml4t_config(no_costs=True),
    )
    _ = warmup_engine.run()

    # Actual benchmark
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = TopBottomStrategy(
        target_lookup_dt,
        config.stop_loss,
        config.take_profit,
        zipline_order_target_semantics=profile_name == "zipline_strict",
    )

    engine = Engine.from_config(
        feed,
        strategy,
        config=build_ml4t_config(no_costs=False),
    )

    results = engine.run()

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trade log for validation
    trades_df = None
    if results.get("trades"):
        trade_records = []
        for t in results["trades"]:
            trade_records.append(
                {
                    "timestamp": t.entry_time,
                    "exit_time": t.exit_time,
                    "asset": t.symbol,
                    "side": "long" if t.quantity > 0 else "short",
                    "quantity": abs(t.quantity),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                }
            )
        trades_df = pd.DataFrame(trade_records)

    return BenchmarkResult(
        framework=framework_name,
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=results["num_trades"],
        final_value=results["final_value"],
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
        target_trace_df=target_trace,
    )


def benchmark_vectorbt_pro(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark VectorBT Pro with given configuration."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        return BenchmarkResult(
            framework="VectorBT Pro",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="VectorBT Pro not installed",
        )

    asset_names = sorted(price_data.keys())
    # Prepare close prices DataFrame
    close_df = pd.DataFrame({name: price_data[name]["close"] for name in asset_names})
    target_shares = build_canonical_target_shares(config, signals, dates, asset_names)
    target_trace = build_canonical_target_trace(target_shares)

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    # Use from_orders with target shares
    # cash_sharing=True ensures single cash pool across all assets (like real portfolio)
    pf = vbt.Portfolio.from_orders(
        close=close_df,
        size=target_shares,
        size_type="targetamount",
        init_cash=config.initial_cash,
        cash_sharing=True,  # Critical: single cash pool, not per-column
        fees=config.commission_pct,
        slippage=config.slippage_pct,
        # Note: VectorBT Pro doesn't have native stop-loss in from_orders
        # Would need from_signals with sl_stop parameter
    )

    # Force computation
    final_value = (
        pf.value.iloc[-1].sum() if hasattr(pf.value.iloc[-1], "sum") else pf.value.iloc[-1]
    )
    trades_readable = pf.trades.records_readable
    num_trades = len(trades_readable)

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trade log for validation
    trades_df = None
    if num_trades > 0:
        # Sort by entry date for proper comparison (VBT returns sorted by column/asset)
        trades_readable = trades_readable.sort_values("Entry Index")
        trade_records = []
        for _, row in trades_readable.iterrows():
            # VectorBT Pro's Entry Index is already a Timestamp
            entry_ts = row.get("Entry Index")
            trade_records.append(
                {
                    "timestamp": entry_ts,
                    "asset": row.get("Column", "unknown"),
                    "side": "long"
                    if str(row.get("Direction", "Long")).lower() == "long"
                    else "short",
                    "quantity": abs(row.get("Size", 0)),
                    "entry_price": row.get("Avg Entry Price", 0),
                    "exit_price": row.get("Avg Exit Price", 0),
                    "pnl": row.get("PnL", 0),
                }
            )
        trades_df = pd.DataFrame(trade_records)

    return BenchmarkResult(
        framework="VectorBT Pro",
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=num_trades,
        final_value=float(final_value),
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
        target_trace_df=target_trace,
    )


def benchmark_vectorbt_oss(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark VectorBT OSS with given configuration."""
    try:
        import vectorbt as vbt
    except ImportError:
        return BenchmarkResult(
            framework="VectorBT OSS",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="VectorBT OSS not installed",
        )

    asset_names = sorted(price_data.keys())
    # Prepare close prices DataFrame
    close_df = pd.DataFrame({name: price_data[name]["close"] for name in asset_names})
    target_shares = build_canonical_target_shares(config, signals, dates, asset_names)
    target_trace = build_canonical_target_trace(target_shares)

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    # VectorBT OSS uses from_orders API
    # cash_sharing=True ensures single cash pool across all assets (like real portfolio)
    # lock_cash=True enforces cash constraints on short selling (default is False in OSS!)
    pf = vbt.Portfolio.from_orders(
        close=close_df,
        size=target_shares,
        size_type="targetamount",
        init_cash=config.initial_cash,
        cash_sharing=True,  # Critical: single cash pool, not per-column
        lock_cash=True,  # Critical: enforce cash constraints (VBT OSS default is False!)
        fees=config.commission_pct,
        slippage=config.slippage_pct,
    )

    # Force computation
    final_value = (
        pf.value().iloc[-1].sum() if hasattr(pf.value().iloc[-1], "sum") else pf.value().iloc[-1]
    )
    trades_readable = pf.trades.records_readable
    num_trades = len(trades_readable)

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trade log for validation
    trades_df = None
    if num_trades > 0:
        # Sort by entry date for proper comparison (VBT returns sorted by column/asset)
        entry_col = (
            "Entry Timestamp" if "Entry Timestamp" in trades_readable.columns else "Entry Index"
        )
        trades_readable = trades_readable.sort_values(entry_col)
        trade_records = []
        for _, row in trades_readable.iterrows():
            trade_records.append(
                {
                    "timestamp": row.get("Entry Timestamp", row.get("Entry Index")),
                    "asset": row.get("Column", "unknown"),
                    "side": "long" if row.get("Direction", "Long") == "Long" else "short",
                    "quantity": abs(row.get("Size", 0)),
                    "entry_price": row.get("Avg Entry Price", row.get("Entry Price", 0)),
                    "exit_price": row.get("Avg Exit Price", row.get("Exit Price", 0)),
                    "pnl": row.get("PnL", 0),
                }
            )
        trades_df = pd.DataFrame(trade_records)

    return BenchmarkResult(
        framework="VectorBT OSS",
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=num_trades,
        final_value=float(final_value),
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
        target_trace_df=target_trace,
    )


def benchmark_zipline(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark Zipline with given configuration.

    Creates a multi-asset bundle with all test data and runs a proper
    top-N/bottom-N ranking strategy.
    """
    try:
        from zipline import run_algorithm
        from zipline.api import (
            get_datetime,
            order_target,
            set_commission,
            set_max_leverage,
            set_slippage,
            sid,
        )
        from zipline.data.bundles import ingest, register
        from zipline.finance.commission import NoCommission, PerDollar
        from zipline.finance.slippage import SlippageModel
        from zipline.utils.calendar_utils import get_calendar as zipline_get_calendar
    except ImportError as e:
        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"Zipline not installed: {e}",
        )

    # Zipline only supports daily data in bundles
    if config.frequency == "1min":
        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="Zipline bundles only support daily data",
        )

    # Get NYSE calendar sessions for proper date alignment
    nyse = zipline_get_calendar("XNYS")

    # Prepare all assets for the bundle
    asset_names = sorted(price_data.keys())
    n_assets = len(asset_names)
    target_shares = build_canonical_target_shares(config, signals, pd.DatetimeIndex(dates), asset_names)
    target_trace = build_canonical_target_trace(target_shares)
    target_lookup_raw = build_canonical_target_lookup(target_shares)
    target_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for ts, targets in target_lookup_raw.items():
        ts_naive = ts.tz_convert(None) if ts.tz is not None else ts
        target_lookup[pd.Timestamp(ts_naive).normalize()] = targets

    # Convert price data to have NYSE-aligned dates
    # Filter to only NYSE trading days
    first_df = price_data[asset_names[0]]
    start_date = first_df.index[0]
    end_date = first_df.index[-1]

    # Get actual NYSE sessions
    nyse_sessions = nyse.sessions_in_range(
        pd.Timestamp(start_date).tz_localize(None) if pd.Timestamp(start_date).tz else start_date,
        pd.Timestamp(end_date).tz_localize(None) if pd.Timestamp(end_date).tz else end_date,
    )

    # Custom slippage model for open-price fills (matching ml4t NEXT_BAR mode)
    class OpenPriceSlippage(SlippageModel):
        @staticmethod
        def process_order(data, order):
            open_px = data.current(order.asset, "open")
            if config.slippage_pct > 0:
                if order.amount > 0:
                    open_px = open_px * (1.0 + config.slippage_pct)
                elif order.amount < 0:
                    open_px = open_px * (1.0 - config.slippage_pct)
            return (open_px, order.amount)

    # Bundle ingest function for multi-asset
    def make_multi_asset_ingest(price_data_dict, asset_list):
        def ingest_func(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            sessions = calendar.sessions_in_range(start_session, end_session)
            sessions = pd.DatetimeIndex(sessions).tz_localize(None)

            # Write equity metadata for all assets
            equities_df = pd.DataFrame(
                {
                    "symbol": asset_list,
                    "asset_name": [f"Asset {name}" for name in asset_list],
                    "exchange": ["NYSE"] * len(asset_list),
                }
            )
            asset_db_writer.write(equities=equities_df)

            # Write daily bars for each asset
            bar_data = []
            for sid, asset_name in enumerate(asset_list):
                df = price_data_dict[asset_name].copy()
                # Convert to tz-naive
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                # Align explicitly to session index to satisfy Zipline writer invariants.
                trading_df = df.reindex(sessions).ffill().bfill()[["open", "high", "low", "close", "volume"]]
                if len(trading_df) > 0:
                    bar_data.append((sid, trading_df))

            daily_bar_writer.write(bar_data, show_progress=show_progress)
            adjustment_writer.write()

        return ingest_func

    # Register and ingest bundle (cached - only created once per config)
    import hashlib

    bundle_sig_input = "|".join(asset_names) + f"|{pd.Timestamp(dates[0]).date()}|{pd.Timestamp(dates[-1]).date()}"
    bundle_sig = hashlib.md5(bundle_sig_input.encode("utf-8")).hexdigest()[:10]
    bundle_name = f"bench_multi_{n_assets}_{config.n_bars}_{bundle_sig}"
    start_session = nyse_sessions[0]
    end_session = (
        nyse_sessions[-1]
        if len(nyse_sessions) <= config.n_bars
        else nyse_sessions[config.n_bars - 1]
    )

    # Check if bundle already exists ON DISK to avoid re-ingestion
    # The in-memory `bundles` registry resets each process - check filesystem instead
    import os
    from pathlib import Path

    zipline_root = Path(os.environ.get("ZIPLINE_ROOT", Path.home() / ".zipline"))
    bundle_dir = zipline_root / "data" / bundle_name  # Zipline stores in data/, not bundles/
    bundle_exists = False
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        bundle_runs = sorted(p for p in bundle_dir.iterdir() if p.is_dir())
        if bundle_runs:
            latest_run = bundle_runs[-1]
            assets_ok = any(latest_run.glob("assets-*.sqlite"))
            if assets_ok:
                bundle_exists = True
            else:
                shutil.rmtree(bundle_dir, ignore_errors=True)

    bundle_time = 0.0
    if not bundle_exists:
        bundle_start = time.perf_counter()
        try:
            register(
                bundle_name,
                make_multi_asset_ingest(price_data, asset_names),
                calendar_name="XNYS",
                start_session=start_session,
                end_session=end_session,
            )
            ingest(bundle_name, show_progress=False)
            bundle_time = time.perf_counter() - bundle_start
            _log(f"  Bundle creation: {bundle_time:.1f}s (one-time setup)")
        except Exception as e:
            return BenchmarkResult(
                framework="Zipline",
                scenario=config.name,
                runtime_sec=0,
                num_trades=0,
                final_value=0,
                memory_mb=0,
                error=f"Bundle setup failed: {e}",
            )
    else:
        # Bundle exists on disk - just re-register (required for run_algorithm in this process)
        _log(f"  Using cached bundle: {bundle_name}")
        register(
            bundle_name,
            make_multi_asset_ingest(price_data, asset_names),
            calendar_name="XNYS",
            start_session=start_session,
            end_session=end_session,
        )

    # Algorithm state
    algo_state = {
        "target_lookup": target_lookup,
        "asset_names": asset_names,
    }

    def initialize(context):
        context.state = algo_state
        # Get all asset objects by sid
        context.assets = [sid(i) for i in range(len(context.state["asset_names"]))]
        context.asset_map = {
            name: context.assets[i] for i, name in enumerate(context.state["asset_names"])
        }
        context.name_by_asset = {asset: name for name, asset in context.asset_map.items()}
        if config.commission_pct > 0:
            set_commission(PerDollar(cost=config.commission_pct))
        else:
            set_commission(NoCommission())
        if config.zipline_max_leverage is not None:
            set_max_leverage(config.zipline_max_leverage)
        set_slippage(OpenPriceSlippage())

    def handle_data(context, data):
        dt = get_datetime()
        # Normalize datetime for lookup
        dt_naive = dt.tz_convert(None) if dt.tz else dt
        dt_normalized = dt_naive.normalize()

        targets = context.state["target_lookup"].get(dt_normalized, {})
        active_names = set(targets.keys())
        for asset_obj, position in context.portfolio.positions.items():
            if position.amount != 0:
                asset_name = context.name_by_asset.get(asset_obj)
                if asset_name is not None:
                    active_names.add(asset_name)

        for asset_name in sorted(active_names):
            asset = context.asset_map[asset_name]
            if not data.can_trade(asset):
                continue

            current_pos = context.portfolio.positions[asset].amount
            target = targets.get(asset_name, 0.0)
            if current_pos != target:
                order_target(asset, target)

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        results = run_algorithm(
            start=start_session,
            end=end_session,
            initialize=initialize,
            handle_data=handle_data,
            capital_base=config.initial_cash,
            bundle=bundle_name,
            data_frequency="daily",
        )

        final_value = results["portfolio_value"].iloc[-1]

        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        def flatten_column(column_name: str) -> pd.DataFrame:
            records = []
            if column_name not in results.columns:
                return pd.DataFrame()
            for dt, payload in results[column_name].items():
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    if isinstance(item, dict):
                        record = dict(item)
                    elif hasattr(item, "to_dict"):
                        record = item.to_dict()
                    elif hasattr(item, "items"):
                        record = dict(item.items())
                    else:
                        continue
                    record["dt"] = dt
                    records.append(record)
            return pd.DataFrame(records)

        positions = flatten_column("positions")
        transactions = flatten_column("transactions")
        orders = flatten_column("orders")

        trades_df = None
        trade_records = []

        if not transactions.empty:
            transactions = transactions.copy()
            transactions["dt"] = pd.to_datetime(transactions["dt"])
            amount_col = "amount" if "amount" in transactions.columns else "filled"
            price_col = "price" if "price" in transactions.columns else "last_sale_price"
            symbol_col = (
                "symbol"
                if "symbol" in transactions.columns
                else "sid"
                if "sid" in transactions.columns
                else "asset"
                if "asset" in transactions.columns
                else None
            )

            if symbol_col and amount_col in transactions.columns and price_col in transactions.columns:
                for symbol in transactions[symbol_col].dropna().unique():
                    symbol_txns = transactions[transactions[symbol_col] == symbol].sort_values("dt")

                    running_pos = 0.0
                    entry_time = None
                    entry_price = None
                    entry_size = 0.0

                    for _, row in symbol_txns.iterrows():
                        amount = float(row[amount_col])
                        price = float(row[price_col])
                        prev_pos = running_pos
                        running_pos += amount

                        if prev_pos == 0 and running_pos != 0:
                            entry_time = row["dt"]
                            entry_price = price
                            entry_size = amount
                        elif prev_pos != 0 and running_pos == 0:
                            exit_price = price
                            pnl = (exit_price - entry_price) * entry_size
                            trade_records.append(
                                {
                                    "entry_date": entry_time,
                                    "exit_date": row["dt"],
                                    "asset": str(symbol),
                                    "side": "long" if entry_size > 0 else "short",
                                    "quantity": abs(entry_size),
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "pnl": pnl,
                                }
                            )
                            entry_time = None
                        elif prev_pos != 0 and running_pos != 0 and (prev_pos > 0) != (running_pos > 0):
                            exit_price = price
                            pnl = (exit_price - entry_price) * entry_size
                            trade_records.append(
                                {
                                    "entry_date": entry_time,
                                    "exit_date": row["dt"],
                                    "asset": str(symbol),
                                    "side": "long" if entry_size > 0 else "short",
                                    "quantity": abs(entry_size),
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "pnl": pnl,
                                }
                            )
                            entry_time = row["dt"]
                            entry_price = price
                            entry_size = running_pos

        num_trades = len(trade_records)
        if num_trades == 0:
            num_trades = int(len(orders)) if not orders.empty else 0
        if trade_records:
            trades_df = pd.DataFrame(trade_records).sort_values("entry_date")

        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=end_time - start_time,
            num_trades=num_trades,
            final_value=float(final_value),
            memory_mb=peak / 1024 / 1024,
            trades_df=trades_df,
            positions_df=positions if not positions.empty else None,
            transactions_df=transactions if not transactions.empty else None,
            target_trace_df=target_trace,
        )
    except Exception as e:
        tracemalloc.stop()
        import traceback

        try:
            tb_text = traceback.format_exc()
        except Exception:
            tb_text = "<traceback unavailable>"

        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"{e}\n{tb_text}",
        )


def benchmark_backtrader(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark Backtrader with given configuration."""
    try:
        import backtrader as bt
    except ImportError:
        return BenchmarkResult(
            framework="Backtrader",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="Backtrader not installed",
        )

    asset_names = sorted(price_data.keys())
    target_shares = build_canonical_target_shares(config, signals, pd.DatetimeIndex(dates), asset_names)
    target_trace = build_canonical_target_trace(target_shares)
    target_lookup_raw = build_canonical_target_lookup(target_shares)
    target_lookup = {ts.strftime("%Y-%m-%d"): targets for ts, targets in target_lookup_raw.items()}

    class TopBottomBTStrategy(bt.Strategy):
        def __init__(self):
            self.target_lookup = target_lookup
            self.data_by_name = {d._name: d for d in self.datas}

        def next(self):
            dt = self.datas[0].datetime.datetime(0)
            dt_key = dt.strftime("%Y-%m-%d")
            targets = self.target_lookup.get(dt_key, {})

            active_names = set(targets.keys())
            for data in self.datas:
                if self.getposition(data).size != 0:
                    active_names.add(data._name)

            for asset_name in sorted(active_names):
                data = self.data_by_name[asset_name]
                current_size = self.getposition(data).size
                target_size = targets.get(asset_name, 0.0)
                if current_size != target_size:
                    self.order_target_size(data=data, target=target_size)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TopBottomBTStrategy)

    # Add data feeds
    for asset_name, df in price_data.items():
        data = bt.feeds.PandasData(dataname=df, name=asset_name)
        cerebro.adddata(data)

    # Align capital base with other frameworks for apples-to-apples comparison
    cerebro.broker.setcash(config.initial_cash)

    if config.commission_pct > 0:
        cerebro.broker.setcommission(commission=config.commission_pct)

    # Add PyFolio analyzer to get standardized positions/transactions output
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract positions and transactions via PyFolio analyzer
    strat = results[0]
    pyfolio_analyzer = strat.analyzers.getbyname("pyfolio")
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()

    # Convert transactions to completed trades by tracking position changes
    # transactions format: index=datetime, columns=[amount, price, sid, symbol, value]
    # A trade completes when position goes to 0 or flips sign
    trades_df = None
    trade_records = []

    # Group transactions by symbol and process sequentially
    if len(transactions) > 0 and "symbol" in transactions.columns:
        for symbol in transactions["symbol"].unique():
            symbol_txns = transactions[transactions["symbol"] == symbol].sort_index()

            running_pos = 0
            entry_time = None
            entry_price = None
            entry_size = 0

            for dt, row in symbol_txns.iterrows():
                amount = row["amount"]
                price = row["price"]
                prev_pos = running_pos
                running_pos += amount

                # Check if position just opened
                if prev_pos == 0 and running_pos != 0:
                    entry_time = dt
                    entry_price = price
                    entry_size = amount

                # Check if position just closed (or flipped)
                elif prev_pos != 0 and running_pos == 0:
                    # Position closed completely
                    exit_price = price
                    pnl = (exit_price - entry_price) * entry_size
                    trade_records.append(
                        {
                            "entry_date": entry_time,
                            "exit_date": dt,
                            "asset": str(symbol),
                            "side": "long" if entry_size > 0 else "short",
                            "quantity": abs(entry_size),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                        }
                    )
                    entry_time = None

                # Handle position flip (e.g., long -> short or short -> long)
                elif prev_pos != 0 and running_pos != 0 and (prev_pos > 0) != (running_pos > 0):
                    # Close old position first
                    exit_price = price
                    pnl = (exit_price - entry_price) * entry_size
                    trade_records.append(
                        {
                            "entry_date": entry_time,
                            "exit_date": dt,
                            "asset": str(symbol),
                            "side": "long" if entry_size > 0 else "short",
                            "quantity": abs(entry_size),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                        }
                    )
                    # Open new position
                    entry_time = dt
                    entry_price = price
                    entry_size = running_pos

    num_trades = len(trade_records)
    if trade_records:
        trades_df = pd.DataFrame(trade_records)
        trades_df = trades_df.sort_values("entry_date")

    return BenchmarkResult(
        framework="Backtrader",
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=num_trades,
        final_value=final_value,
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
        positions_df=positions,
        transactions_df=transactions,
        target_trace_df=target_trace,
    )


def _parse_money_like(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, dict):
        for key in ("total", "amount", "value", "raw", "free", "locked"):
            if key in value:
                parsed = _parse_money_like(value[key])
                if parsed is not None:
                    return parsed
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def benchmark_nautilus(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark Nautilus Trader with canonical target-share strategy."""
    if config.frequency != "D":
        return BenchmarkResult(
            framework="Nautilus Trader",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="Nautilus adapter currently supports daily scenarios only",
        )

    try:
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, RiskEngineConfig
        from nautilus_trader.core.datetime import dt_to_unix_nanos, unix_nanos_to_dt
        from nautilus_trader.model.currencies import USD
        from nautilus_trader.model.data import Bar, BarType
        from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
        from nautilus_trader.model.identifiers import TraderId, Venue
        from nautilus_trader.model.objects import Money, Quantity
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
        from nautilus_trader.trading.strategy import Strategy
    except ImportError:
        return BenchmarkResult(
            framework="Nautilus Trader",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="Nautilus Trader not installed",
        )

    asset_names = sorted(price_data.keys())
    progress_every_assets = max(1, int(os.getenv("ML4T_NAUTILUS_PROGRESS_EVERY_ASSETS", "25")))
    progress_every_bars = max(0, int(os.getenv("ML4T_NAUTILUS_PROGRESS_EVERY_BARS", "50000")))
    processed_bars = {"count": 0}

    target_shares = build_canonical_target_shares(config, signals, pd.DatetimeIndex(dates), asset_names)
    target_trace = build_canonical_target_trace(target_shares)
    target_lookup = build_canonical_target_lookup(target_shares)
    target_lookup_str = {ts.strftime("%Y-%m-%d"): targets for ts, targets in target_lookup.items()}

    class TargetSharesStrategy(Strategy):
        def __init__(
            self,
            instruments: dict[str, object],
            bar_types: dict[str, BarType],
            targets_by_day: dict[str, dict[str, float]],
        ):
            super().__init__()
            self._instruments = instruments
            self._bar_types = bar_types
            self._targets_by_day = targets_by_day
            self._last_rebalance_day: str | None = None

        @staticmethod
        def _qty_to_float(value: object) -> float:
            if value is None:
                return 0.0
            for cast in (float,):
                try:
                    return cast(value)
                except Exception:
                    continue
            return 0.0

        def on_start(self) -> None:
            for bar_type in self._bar_types.values():
                self.subscribe_bars(bar_type)

        def on_bar(self, bar: Bar) -> None:
            day = unix_nanos_to_dt(bar.ts_event).date().isoformat()
            processed_bars["count"] += 1
            if progress_every_bars > 0 and processed_bars["count"] % progress_every_bars == 0:
                _log(f"  [nautilus] processed {processed_bars['count']:,} bars (day={day})")
            if day == self._last_rebalance_day:
                return
            self._last_rebalance_day = day

            targets = self._targets_by_day.get(day, {})
            active_assets = set(targets.keys())
            for asset_name, instrument in self._instruments.items():
                current = self._qty_to_float(self.portfolio.net_position(instrument.id))
                if current != 0.0:
                    active_assets.add(asset_name)

            for asset_name in sorted(active_assets):
                instrument = self._instruments.get(asset_name)
                if instrument is None:
                    continue
                current_qty = self._qty_to_float(self.portfolio.net_position(instrument.id))
                target_qty = float(targets.get(asset_name, 0.0))
                delta = int(round(target_qty - current_qty))
                if delta == 0:
                    continue
                order = self.order_factory.market(
                    instrument_id=instrument.id,
                    order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=instrument.make_qty(abs(delta)),
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-NT"),
            logging=LoggingConfig(log_level="ERROR"),
            risk_engine=RiskEngineConfig(bypass=True),
        ),
    )
    venue = Venue("XNAS")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(float(config.initial_cash), USD)],
    )

    instruments: dict[str, object] = {}
    bar_types: dict[str, BarType] = {}

    try:
        total_assets = len(asset_names)
        for idx, asset_name in enumerate(asset_names, start=1):
            instrument = TestInstrumentProvider.equity(symbol=asset_name, venue="XNAS")
            instruments[asset_name] = instrument
            engine.add_instrument(instrument)

            bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
            bar_types[asset_name] = bar_type

            bars: list[Bar] = []
            for ts, row in price_data[asset_name].sort_index().iterrows():
                ts_utc = pd.Timestamp(ts)
                ts_utc = ts_utc.tz_localize("UTC") if ts_utc.tz is None else ts_utc.tz_convert("UTC")
                ts_ns = dt_to_unix_nanos(ts_utc.to_pydatetime())
                bars.append(
                    Bar(
                        bar_type=bar_type,
                        open=instrument.make_price(float(row["open"])),
                        high=instrument.make_price(float(row["high"])),
                        low=instrument.make_price(float(row["low"])),
                        close=instrument.make_price(float(row["close"])),
                        volume=Quantity.from_int(max(1, int(round(float(row["volume"]))))),
                        ts_event=ts_ns,
                        ts_init=ts_ns,
                    )
                )
            # Avoid repeated full-stream sort per instrument; sort once after all loads.
            engine.add_data(bars, sort=False, validate=False)
            if idx % progress_every_assets == 0 or idx == total_assets:
                _log(f"  [nautilus] loaded data for {idx}/{total_assets} assets")

        engine.sort_data()
        _log("  [nautilus] data streams sorted")

        strategy = TargetSharesStrategy(
            instruments=instruments,
            bar_types=bar_types,
            targets_by_day=target_lookup_str,
        )
        engine.add_strategy(strategy)

        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()
        engine.run()
        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        fills_report = engine.trader.generate_order_fills_report()
        account_report = engine.trader.generate_account_report(venue)
        fills_df = fills_report.copy() if isinstance(fills_report, pd.DataFrame) else pd.DataFrame()
        account_df = account_report.copy() if isinstance(account_report, pd.DataFrame) else pd.DataFrame()

        final_value = float(config.initial_cash)
        if not account_df.empty:
            for col in ("total", "balance_total", "equity", "net_value", "free"):
                if col in account_df.columns:
                    parsed = _parse_money_like(account_df.iloc[-1][col])
                    if parsed is not None:
                        final_value = parsed
                        break

        trades_df = None
        if not fills_df.empty:
            trades_df = fills_df.reset_index()
            if "instrument_id" in trades_df.columns:
                trades_df["asset"] = trades_df["instrument_id"].astype(str).str.split(".").str[0]
            if "side" in trades_df.columns:
                trades_df["side"] = trades_df["side"].astype(str).str.lower()

        return BenchmarkResult(
            framework="Nautilus Trader",
            scenario=config.name,
            runtime_sec=end_time - start_time,
            num_trades=int(len(fills_df)),
            final_value=final_value,
            memory_mb=peak / 1024 / 1024,
            trades_df=trades_df,
            target_trace_df=target_trace,
        )
    except Exception as e:
        return BenchmarkResult(
            framework="Nautilus Trader",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=str(e),
            target_trace_df=target_trace,
        )
    finally:
        with suppress(Exception):
            engine.dispose()


def benchmark_lean(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark LEAN CLI with canonical target-share strategy."""
    if config.frequency != "D":
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="LEAN adapter currently supports daily scenarios only",
        )

    lean_workspace = PROJECT_ROOT / "validation" / "lean" / "workspace"
    lean_config = lean_workspace / "lean.json"
    if not lean_config.exists():
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"LEAN config not found: {lean_config}",
        )

    lean_binary = shutil.which("lean")
    if lean_binary is not None:
        lean_cmd = [lean_binary]
    else:
        uvx_binary = shutil.which("uvx")
        if uvx_binary is None:
            return BenchmarkResult(
                framework="LEAN CLI",
                scenario=config.name,
                runtime_sec=0,
                num_trades=0,
                final_value=0,
                memory_mb=0,
                error="Neither 'lean' nor 'uvx' executable found",
            )
        lean_cmd = [uvx_binary, "--python", "3.11", "--with", "setuptools<81", "lean"]

    def parse_int(value: object) -> int:
        if value is None:
            return 0
        text = str(value).replace(",", "").strip()
        if not text:
            return 0
        token = text.split()[0]
        try:
            return int(float(token))
        except ValueError:
            return 0

    def parse_float(value: object) -> float:
        if value is None:
            return 0.0
        text = str(value).replace(",", "").replace("$", "").replace("%", "").strip()
        if not text:
            return 0.0
        token = text.split()[0]
        try:
            return float(token)
        except ValueError:
            return 0.0

    def encode_ticker(idx: int) -> str:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        val = idx
        chars: list[str] = []
        for _ in range(4):
            chars.append(letters[val % 26])
            val //= 26
        return "".join(reversed(chars))

    asset_names = sorted(price_data.keys())
    asset_to_ticker = {asset_name: encode_ticker(i) for i, asset_name in enumerate(asset_names)}
    tickers = [asset_to_ticker[asset_name] for asset_name in asset_names]

    if config.lean_order_type not in {"market", "market_on_open"}:
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"Unsupported lean_order_type: {config.lean_order_type}",
        )
    if config.lean_account_type not in {"default", "cash", "margin"}:
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"Unsupported lean_account_type: {config.lean_account_type}",
        )
    lean_order_call = (
        "self.market_on_open_order(symbol, delta)"
        if config.lean_order_type == "market_on_open"
        else "self.market_order(symbol, delta)"
    )
    force_zero_fee = (
        config.lean_force_zero_fee
        if config.lean_force_zero_fee is not None
        else config.commission_pct == 0.0
    )
    force_zero_slippage = (
        config.lean_force_zero_slippage
        if config.lean_force_zero_slippage is not None
        else config.slippage_pct == 0.0
    )
    lean_account_stmt = ""
    if config.lean_account_type == "cash":
        lean_account_stmt = "self.set_brokerage_model(BrokerageName.DEFAULT, AccountType.CASH)"
    elif config.lean_account_type == "margin":
        lean_account_stmt = "self.set_brokerage_model(BrokerageName.DEFAULT, AccountType.MARGIN)"

    target_shares = build_canonical_target_shares(config, signals, pd.DatetimeIndex(dates), asset_names)
    target_trace = build_canonical_target_trace(target_shares)
    target_shares_lean = target_shares.rename(columns=asset_to_ticker)

    project_dir = lean_workspace / "ml4t_benchmark"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "backtests").mkdir(parents=True, exist_ok=True)

    config_json = {
        "algorithm-language": "Python",
        "parameters": {},
        "description": "ml4t canonical target-share benchmark",
    }
    (project_dir / "config.json").write_text(json.dumps(config_json, indent=4), encoding="utf-8")

    symbols_path = project_dir / "symbols.csv"
    symbols_path.write_text("\n".join(tickers) + "\n", encoding="utf-8")

    targets_long = target_shares_lean.stack().reset_index()
    targets_long.columns = ["timestamp", "ticker", "target"]
    targets_long = targets_long[targets_long["target"] != 0.0]
    targets_long["timestamp"] = pd.to_datetime(targets_long["timestamp"]).dt.strftime("%Y-%m-%d")
    targets_path = project_dir / "targets.csv"
    targets_long.to_csv(targets_path, index=False)

    first_date = pd.Timestamp(dates[0]).date()
    last_date = pd.Timestamp(dates[-1]).date()
    main_code = f"""# region imports
from AlgorithmImports import *
# endregion

import csv
from pathlib import Path


class Ml4tBenchmark(QCAlgorithm):
    def initialize(self):
        self.set_start_date({first_date.year}, {first_date.month}, {first_date.day})
        self.set_end_date({last_date.year}, {last_date.month}, {last_date.day})
        self.set_cash({float(config.initial_cash)})
        {lean_account_stmt}
        self._targets = {{}}
        base_path = Path(__file__).resolve().parent

        with (base_path / "targets.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row["timestamp"]
                if key not in self._targets:
                    self._targets[key] = {{}}
                self._targets[key][row["ticker"]] = float(row["target"])

        self._symbols = {{}}
        for line in (base_path / "symbols.csv").read_text(encoding="utf-8").splitlines():
            ticker = line.strip()
            if not ticker:
                continue
            security = self.add_equity(ticker, Resolution.DAILY)
            {f"security.set_leverage({float(config.lean_security_leverage)})" if config.lean_security_leverage is not None else ""}
            {"security.set_fee_model(ConstantFeeModel(0))" if force_zero_fee else ""}
            {"security.set_slippage_model(ConstantSlippageModel(0))" if force_zero_slippage else ""}
            self._symbols[ticker] = security.symbol
        if self._symbols:
            first_ticker = sorted(self._symbols.keys())[0]
            self.set_benchmark(self._symbols[first_ticker])

    def on_data(self, data: Slice):
        key = self.time.strftime("%Y-%m-%d")
        targets = self._targets.get(key)
        if targets is None:
            return

        active = set(targets.keys())
        for ticker, symbol in self._symbols.items():
            if self.portfolio[symbol].quantity != 0:
                active.add(ticker)

        bars = data.bars
        for ticker in sorted(active):
            symbol = self._symbols[ticker]
            if bars is None or symbol not in bars:
                continue
            target_qty = targets.get(ticker, 0.0)
            current_qty = self.portfolio[symbol].quantity
            delta = int(round(target_qty - current_qty))
            if delta != 0:
                {lean_order_call}
"""
    (project_dir / "main.py").write_text(main_code, encoding="utf-8")

    data_root = lean_workspace / "data" / "equity" / "usa"
    (data_root / "map_files").mkdir(parents=True, exist_ok=True)
    (data_root / "factor_files").mkdir(parents=True, exist_ok=True)
    (data_root / "daily").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.setdefault("UV_TOOL_DIR", "/tmp/uv-tools")

    lean_check = subprocess.run(
        lean_cmd + ["--version"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if lean_check.returncode != 0:
        error_text = (lean_check.stderr or lean_check.stdout).strip()
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"LEAN CLI unavailable: {error_text}",
        )

    def scaled_price(value: float) -> int:
        return int(round(float(value) * 10000.0))

    close_edge_sum = 0.0
    for asset_name in asset_names:
        asset_close = price_data[asset_name]["close"]
        if len(asset_close) > 0:
            close_edge_sum += float(asset_close.iloc[0]) + float(asset_close.iloc[-1])
    lean_data_sig = hashlib.md5(
        (
            f"assets={','.join(asset_names)}|start={pd.Timestamp(dates[0]).date()}|"
            f"end={pd.Timestamp(dates[-1]).date()}|edges={close_edge_sum:.10f}"
        ).encode()
    ).hexdigest()
    manifest_path = data_root / "ml4t_manifest.json"
    cache_hit = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cache_hit = manifest.get("signature") == lean_data_sig
        except Exception:
            cache_hit = False

    prep_start = time.perf_counter()
    if not cache_hit:
        for asset_name in asset_names:
            ticker = asset_to_ticker[asset_name]
            ticker_lower = ticker.lower()
            asset_df = price_data[asset_name].sort_index()
            if asset_df.empty:
                continue

            lines: list[str] = []
            for ts, row in asset_df.iterrows():
                dt = pd.Timestamp(ts)
                dt = dt.tz_convert(None) if dt.tz is not None else dt
                o = scaled_price(row["open"])
                h = scaled_price(row["high"])
                low_px = scaled_price(row["low"])
                c = scaled_price(row["close"])
                high_i = max(o, h, low_px, c)
                low_i = min(o, h, low_px, c)
                vol = max(1, int(round(float(row["volume"]))))
                lines.append(f"{dt.strftime('%Y%m%d')} 00:00,{o},{high_i},{low_i},{c},{vol}")

            zip_path = data_root / "daily" / f"{ticker_lower}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{ticker_lower}.csv", "\n".join(lines))

            first_dt = pd.Timestamp(asset_df.index[0])
            first_dt = first_dt.tz_convert(None) if first_dt.tz is not None else first_dt
            first_key = first_dt.strftime("%Y%m%d")
            (data_root / "map_files" / f"{ticker_lower}.csv").write_text(
                f"{first_key},{ticker_lower}\n20501231,{ticker_lower}\n",
                encoding="utf-8",
            )
            (data_root / "factor_files" / f"{ticker_lower}.csv").write_text(
                f"{first_key},1,1,1\n20501231,1,1,0\n",
                encoding="utf-8",
            )

        manifest_path.write_text(
            json.dumps(
                {
                    "signature": lean_data_sig,
                    "num_assets": len(asset_names),
                    "start": str(pd.Timestamp(dates[0]).date()),
                    "end": str(pd.Timestamp(dates[-1]).date()),
                }
            ),
            encoding="utf-8",
        )

    prep_elapsed = time.perf_counter() - prep_start
    if cache_hit:
        _log(f"  LEAN data export: cache hit ({len(asset_names)} assets)")
    else:
        _log(f"  LEAN data export: {prep_elapsed:.2f}s ({len(asset_names)} assets)")

    output_dir = project_dir / "backtests" / f"{config.n_assets}_{config.n_bars}_{int(time.time())}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_cmd = lean_cmd + [
        "backtest",
        str(project_dir),
        "--lean-config",
        str(lean_config),
        "--no-update",
        "--output",
        str(output_dir),
    ]

    start_time = time.perf_counter()
    run_result = subprocess.run(
        run_cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    runtime_sec = time.perf_counter() - start_time

    if run_result.returncode != 0:
        error_text = (run_result.stderr or run_result.stdout).strip()
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"LEAN backtest failed: {error_text}",
            target_trace_df=target_trace,
        )

    summary_files = sorted(output_dir.glob("*-summary.json"))
    if not summary_files:
        return BenchmarkResult(
            framework="LEAN CLI",
            scenario=config.name,
            runtime_sec=runtime_sec,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"LEAN summary file not found in {output_dir}",
            target_trace_df=target_trace,
        )

    summary = json.loads(summary_files[-1].read_text(encoding="utf-8"))
    trade_stats = summary.get("totalPerformance", {}).get("tradeStatistics", {})
    stats = summary.get("statistics", {})
    portfolio_stats = summary.get("totalPerformance", {}).get("portfolioStatistics", {})
    state = summary.get("state", {})

    num_trades = parse_int(trade_stats.get("totalNumberOfTrades"))
    if num_trades == 0:
        num_trades = parse_int(state.get("OrderCount"))
    if num_trades == 0:
        num_trades = parse_int(stats.get("Total Orders"))

    final_value = parse_float(portfolio_stats.get("endEquity"))
    if final_value == 0.0:
        final_value = parse_float(stats.get("End Equity"))

    return BenchmarkResult(
        framework="LEAN CLI",
        scenario=config.name,
        runtime_sec=runtime_sec,
        num_trades=num_trades,
        final_value=final_value,
        memory_mb=0.0,
        target_trace_df=target_trace,
    )


def run_scenario(
    scenario_name: str,
    frameworks: list[str],
    data_source: str = "synthetic",
    real_data_path: Path = DEFAULT_REAL_DATA_PATH,
    cache_mode: str = "auto",
) -> list[BenchmarkResult]:
    """Run a benchmark scenario across specified frameworks."""
    config = SCENARIOS[scenario_name]
    _log(f"\n{'=' * 70}")
    _log(f"Scenario: {config.name}")
    _log(f"  Bars: {config.n_bars:,} | Assets: {config.n_assets} | Freq: {config.frequency}")
    _log(f"  Long top {config.top_n}, Short bottom {config.bottom_n}")
    if config.stop_loss:
        _log(f"  Stop-loss: {config.stop_loss * 100:.1f}%")
    if config.take_profit:
        _log(f"  Take-profit: {config.take_profit * 100:.1f}%")
    if config.commission_pct > 0:
        _log(f"  Commission: {config.commission_pct * 100:.2f}%")
    if config.slippage_pct > 0:
        _log(f"  Slippage: {config.slippage_pct * 100:.2f}%")
    _log(f"{'=' * 70}")

    _log("\nGenerating data...")
    if data_source == "real":
        price_data, signals, dates = load_real_benchmark_data(
            config,
            real_data_path,
            cache_mode=cache_mode,
        )
        _log(f"  Source: real parquet ({real_data_path})")
    else:
        price_data, signals, dates = generate_benchmark_data(config)
        _log("  Source: synthetic generator")
    _log(f"  Generated {len(price_data)} assets with {len(dates):,} bars each")
    _log(f"  Date range: {pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()}")

    results = []

    for framework in frameworks:
        _log(f"\nRunning {framework}...")
        try:
            if framework == "ml4t":
                result = benchmark_ml4t(
                    config, price_data, signals, dates, execution_mode="same_bar"
                )
            elif framework == "ml4t-backtrader":
                # ML4T with Backtrader-compatible settings (next-bar execution)
                result = benchmark_ml4t(
                    config, price_data, signals, dates, execution_mode="next_bar"
                )
            elif framework == "ml4t-vbt-strict":
                result = benchmark_ml4t(
                    config,
                    price_data,
                    signals,
                    dates,
                    execution_mode="same_bar",
                    profile_override="vectorbt_strict",
                )
            elif framework == "ml4t-backtrader-strict":
                result = benchmark_ml4t(
                    config,
                    price_data,
                    signals,
                    dates,
                    execution_mode="next_bar",
                    profile_override="backtrader_strict",
                )
            elif framework == "ml4t-zipline-strict":
                result = benchmark_ml4t(
                    config,
                    price_data,
                    signals,
                    dates,
                    execution_mode="next_bar",
                    profile_override="zipline_strict",
                )
            elif framework == "ml4t-lean-strict":
                result = benchmark_ml4t(
                    config,
                    price_data,
                    signals,
                    dates,
                    execution_mode="next_bar",
                    profile_override="lean_strict",
                )
            elif framework == "vbt-pro":
                result = benchmark_vectorbt_pro(config, price_data, signals, dates)
            elif framework == "vbt-oss":
                result = benchmark_vectorbt_oss(config, price_data, signals, dates)
            elif framework == "zipline":
                result = benchmark_zipline(config, price_data, signals, dates)
            elif framework == "backtrader":
                result = benchmark_backtrader(config, price_data, signals, dates)
            elif framework == "nautilus":
                result = benchmark_nautilus(config, price_data, signals, dates)
            elif framework == "lean":
                result = benchmark_lean(config, price_data, signals, dates)
            else:
                result = BenchmarkResult(
                    framework=framework,
                    scenario=config.name,
                    runtime_sec=0,
                    num_trades=0,
                    final_value=0,
                    memory_mb=0,
                    error=f"Unknown framework: {framework}",
                )

            # Add realized data_points to result for metrics (can differ from config on bounded calendars)
            result.data_points = len(dates) * len(price_data)
            results.append(result)

            if result.error:
                _log(f"  ERROR: {result.error}")
            else:
                _log(f"  Runtime: {result.runtime_sec:.3f}s")
                _log(f"  Trades: {result.num_trades:,}")
                _log(f"  Final value: ${result.final_value:,.2f}")
                _log(f"  Memory: {result.memory_mb:.1f} MB")
                if result.bars_per_second > 0:
                    _log(f"  Speed: {result.bars_per_second:,.0f} bars/sec")

        except Exception as e:
            _log(f"  EXCEPTION: {e}")
            import traceback

            _log(traceback.format_exc())
            results.append(
                BenchmarkResult(
                    framework=framework,
                    scenario=config.name,
                    runtime_sec=0,
                    num_trades=0,
                    final_value=0,
                    memory_mb=0,
                    error=str(e),
                )
            )

    return results


def print_summary(all_results: list[BenchmarkResult]):
    """Print summary comparison table."""
    _log("\n" + "=" * 80)
    _log("SUMMARY")
    _log("=" * 80)

    # Group by scenario
    scenarios = {}
    for r in all_results:
        if r.scenario not in scenarios:
            scenarios[r.scenario] = {}
        scenarios[r.scenario][r.framework] = r

    _log(f"{'Scenario':<30} {'Framework':<15} {'Runtime':<12} {'Trades':<10} {'Memory':<10}")
    _log("-" * 80)

    for scenario, frameworks in scenarios.items():
        first = True
        for framework, result in frameworks.items():
            scenario_col = scenario if first else ""
            first = False

            if result.error:
                _log(f"{scenario_col:<30} {framework:<15} {'ERROR':<12} {'-':<10} {'-':<10}")
            else:
                runtime_str = f"{result.runtime_sec:.3f}s"
                if result.runtime_sec > 60:
                    runtime_str = f"{result.runtime_sec / 60:.1f}m"
                _log(
                    f"{scenario_col:<30} {framework:<15} {runtime_str:<12} "
                    f"{result.num_trades:<10} {result.memory_mb:.0f} MB"
                )

    _log("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark suite for backtesting frameworks")
    parser.add_argument(
        "--framework",
        choices=[
            "ml4t",
            "ml4t-backtrader",
            "ml4t-vbt-strict",
            "ml4t-backtrader-strict",
            "ml4t-zipline-strict",
            "ml4t-lean-strict",
            "vbt-pro",
            "vbt-oss",
            "backtrader",
            "nautilus",
            "zipline",
            "lean",
            "all",
        ],
        default="ml4t",
        help="Framework to benchmark (ml4t-backtrader uses next-bar execution to match Backtrader)",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="baseline",
        help="Scenario to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios",
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Price data source for benchmark scenarios",
    )
    parser.add_argument(
        "--real-data-path",
        type=str,
        default=str(DEFAULT_REAL_DATA_PATH),
        help="Path to real-data parquet file (used with --data-source real)",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["auto", "off", "refresh"],
        default="auto",
        help="Cache behavior for real-data preprocessing",
    )
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Save trade logs to CSV for validation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        metavar="PATH",
        help="Write JSON report to specified path",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        metavar="PATH",
        help="Write Markdown report to specified path",
    )
    args = parser.parse_args()

    # Determine frameworks
    if args.framework == "all":
        frameworks = ["ml4t", "vbt-pro", "backtrader"]
    else:
        frameworks = [args.framework]

    # Determine scenarios
    if args.all or args.scenario == "all":
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [args.scenario]

    _log("=" * 70)
    _log("Backtesting Framework Benchmark Suite")
    _log("=" * 70)
    _log(f"Frameworks: {', '.join(frameworks)}")
    _log(f"Scenarios: {', '.join(scenario_names)}")

    all_results = []
    output_dir = PROJECT_ROOT / "validation" / "trade_logs"

    for scenario_name in scenario_names:
        results = run_scenario(
            scenario_name,
            frameworks,
            data_source=args.data_source,
            real_data_path=Path(args.real_data_path),
            cache_mode=args.cache_mode,
        )
        all_results.extend(results)

        # Save trade logs if requested
        if args.save_trades:
            for r in results:
                save_trades(r, output_dir)

    print_summary(all_results)

    # Print trade comparison if multiple frameworks
    if len(frameworks) > 1:
        comparisons = compare_trades(all_results)
        if comparisons:
            _log("\nTRADE COMPARISON (vs ml4t.backtest baseline)")
            _log("-" * 60)
            for framework, comp in comparisons.items():
                _log(f"{framework}:")
                _log(
                    f"  Trade count diff: {comp['trade_count_diff']:,} ({comp['trade_count_pct']:.2f}%)"
                )
                _log(
                    f"  Final value diff: ${comp['final_value_diff']:,.2f} ({comp['final_value_pct']:.2f}%)"
                )

    # Generate reports if requested
    if args.output_json:
        generate_json_report(all_results, Path(args.output_json))

    if args.output_markdown:
        generate_markdown_report(all_results, Path(args.output_markdown))

    return 0


if __name__ == "__main__":
    sys.exit(main())
