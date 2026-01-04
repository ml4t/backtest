#!/usr/bin/env python3
"""Performance comparison: backtest-nb vs backtest-rs vs VectorBT Pro."""

import time
import numpy as np
import polars as pl


def generate_test_data(n_bars: int, seed: int = 42):
    """Generate OHLCV + momentum signal data."""
    rng = np.random.default_rng(seed)

    returns = rng.normal(0.0002, 0.015, n_bars)
    close = 100.0 * np.exp(np.cumsum(returns))

    daily_vol = rng.uniform(0.005, 0.02, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.5, n_bars)
    volume = rng.uniform(1e6, 5e6, n_bars)

    # Mean-reverting momentum
    momentum = np.zeros(n_bars)
    for i in range(1, n_bars):
        momentum[i] = 0.5 * momentum[i - 1] + rng.normal(0, 0.04)

    prices = pl.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    })
    signals = pl.DataFrame({"momentum": momentum})

    return prices, signals, close, high, low, momentum


def run_vbt_pro(close, high, low, momentum, n_bars, trailing_stop=0.03):
    """Run VectorBT Pro backtest."""
    import vectorbtpro as vbt

    entries = momentum > 0.01
    exits = momentum < -0.005

    start = time.perf_counter()
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        fees=0.001,
        slippage=0.0005,
    )
    n_trades = pf.trades.count()
    final_value = pf.final_value
    elapsed = time.perf_counter() - start

    return n_trades, final_value, elapsed


def run_nb(prices, signals, trailing_stop=0.03):
    """Run backtest-nb."""
    from ml4t.backtest_nb import RuleEngine, Signal, backtest, HWM_HIGH

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -0.005,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    start = time.perf_counter()
    result = backtest(prices, signals, strategy, trail_hwm_source=HWM_HIGH)
    elapsed = time.perf_counter() - start

    return result.n_trades, result.final_value, elapsed


def run_rs(prices, signals, trailing_stop=0.03):
    """Run backtest-rs."""
    from ml4t_backtest_rs import RuleEngine, Signal, backtest

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -0.005,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    start = time.perf_counter()
    result = backtest(prices, signals, strategy, trail_hwm_source=1)  # 1=HIGH
    elapsed = time.perf_counter() - start

    return result.n_trades, result.final_value, elapsed


def main():
    sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000]

    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON: backtest-nb vs backtest-rs vs VectorBT Pro")
    print(f"{'='*80}")
    print(f"\n{'Bars':>12} | {'VBT Pro':>12} | {'backtest-nb':>12} | {'backtest-rs':>12} | {'nb vs VBT':>10} | {'rs vs VBT':>10}")
    print("-" * 80)

    # Warm up JIT
    prices, signals, close, high, low, momentum = generate_test_data(1000)
    _ = run_nb(prices, signals)

    for n_bars in sizes:
        prices, signals, close, high, low, momentum = generate_test_data(n_bars)

        # Run each
        vbt_trades, vbt_value, vbt_time = run_vbt_pro(close, high, low, momentum, n_bars)
        nb_trades, nb_value, nb_time = run_nb(prices, signals)
        rs_trades, rs_value, rs_time = run_rs(prices, signals)

        # Speedup ratios (positive = faster than VBT, negative = slower)
        nb_vs_vbt = vbt_time / nb_time if nb_time > 0 else 0
        rs_vs_vbt = vbt_time / rs_time if rs_time > 0 else 0

        print(f"{n_bars:>12,} | {vbt_time:>10.4f}s | {nb_time:>10.4f}s | {rs_time:>10.4f}s | {nb_vs_vbt:>9.1f}x | {rs_vs_vbt:>9.1f}x")

    print(f"\n{'='*80}")
    print("Speedup > 1.0 means faster than VBT Pro")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
