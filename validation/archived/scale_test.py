#!/usr/bin/env python3
"""Scale test for backtest implementations.

Tests single-asset performance up to 100M data points.
Uses signal-only mode (no trailing stops) for exact match.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class ScaleResult:
    """Results from a single implementation run."""
    name: str
    n_trades: int
    final_value: float
    elapsed_ms: float
    throughput_bars_per_sec: float


def generate_data(n_bars: int, seed: int = 42, entry_prob: float = 0.01, exit_prob: float = 0.03):
    """Generate single-asset test data."""
    rng = np.random.default_rng(seed)

    # Random walk price
    returns = rng.normal(0.0002, 0.015, n_bars)
    cumret = np.clip(np.cumsum(returns), -5, 5)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.3, n_bars)
    volume = np.ones(n_bars) * 1e6

    # Boolean signals
    entries = rng.random(n_bars) < entry_prob
    exits = rng.random(n_bars) < exit_prob
    entries = entries & ~exits

    return {
        "n_bars": n_bars,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "entries": entries,
        "exits": exits,
    }


def run_vbt_pro(data: dict) -> ScaleResult:
    """Run VectorBT Pro."""
    import vectorbtpro as vbt

    close = pd.Series(data["close"])
    entries = pd.Series(data["entries"])
    exits = pd.Series(data["exits"])

    start = time.perf_counter()

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=100_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
        accumulate=False,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    n_bars = data["n_bars"]
    throughput = n_bars / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

    return ScaleResult(
        name="VBT Pro",
        n_trades=pf.trades.count(),
        final_value=pf.final_value,
        elapsed_ms=elapsed_ms,
        throughput_bars_per_sec=throughput,
    )


def run_backtest_rs(data: dict) -> ScaleResult:
    """Run backtest-rs."""
    from ml4t_backtest_rs import RuleEngine, Signal, backtest

    prices = pl.DataFrame({
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
    })

    signals = pl.DataFrame({
        "entry": data["entries"].astype(np.float64),
        "exit": data["exits"].astype(np.float64),
    })

    strategy = RuleEngine(
        entry=Signal("entry") > 0.5,
        exit=Signal("exit") > 0.5,
        position_size=100.0,
    )

    start = time.perf_counter()

    result = backtest(
        prices,
        signals,
        strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
        n_assets=1,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    n_bars = data["n_bars"]
    throughput = n_bars / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

    return ScaleResult(
        name="backtest-rs",
        n_trades=result.n_trades,
        final_value=result.final_value,
        elapsed_ms=elapsed_ms,
        throughput_bars_per_sec=throughput,
    )


def main():
    print("=" * 80)
    print("SCALE TEST: Single-Asset Performance")
    print("=" * 80)

    # Test configurations
    configs = [
        10_000,        # 10K
        100_000,       # 100K
        1_000_000,     # 1M
        10_000_000,    # 10M
        100_000_000,   # 100M
    ]

    print(f"\n{'Bars':>12} | {'VBT Trades':>10} | {'VBT ms':>10} | {'RS Trades':>10} | {'RS ms':>10} | {'Match':>6} | {'Speedup':>8}")
    print("-" * 90)

    for n_bars in configs:
        data = generate_data(n_bars)

        # Run VBT Pro
        try:
            vbt_result = run_vbt_pro(data)
        except Exception as e:
            print(f"{n_bars:>12,} | VBT error: {e}")
            continue

        # Run backtest-rs
        try:
            rs_result = run_backtest_rs(data)
        except Exception as e:
            print(f"{n_bars:>12,} | RS error: {e}")
            continue

        # Compare
        match = "✓" if vbt_result.n_trades == rs_result.n_trades else "✗"
        if vbt_result.elapsed_ms > 0 and rs_result.elapsed_ms > 0:
            speedup = vbt_result.elapsed_ms / rs_result.elapsed_ms
        else:
            speedup = 0

        print(f"{n_bars:>12,} | {vbt_result.n_trades:>10,} | {vbt_result.elapsed_ms:>10.1f} | {rs_result.n_trades:>10,} | {rs_result.elapsed_ms:>10.1f} | {match:>6} | {speedup:>7.1f}x")

    print()
    print("Note: Trade count difference is due to VBT counting open positions as trades.")


if __name__ == "__main__":
    main()
