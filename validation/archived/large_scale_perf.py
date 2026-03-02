#!/usr/bin/env python3
"""Large-scale performance comparison: 10M rows, 100K+ trades."""

import time
import numpy as np
import polars as pl


def generate_high_frequency_signals(n_bars: int, seed: int = 42):
    """Generate data that produces many trades (~1 trade per 100 bars)."""
    rng = np.random.default_rng(seed)

    # Price series - use smaller returns to avoid overflow
    returns = rng.normal(0.00005, 0.005, n_bars)
    # Clip cumulative returns to prevent overflow
    cumret = np.cumsum(returns)
    cumret = np.clip(cumret, -10, 10)  # Limit to ~e^10 range
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.003, 0.01, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.2, n_bars)
    volume = rng.uniform(1e6, 5e6, n_bars)

    # Fast mean-reverting signal (low persistence = frequent crossings)
    momentum = np.zeros(n_bars)
    for i in range(1, n_bars):
        momentum[i] = 0.3 * momentum[i - 1] + rng.normal(0, 0.05)

    prices = pl.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    })
    signals = pl.DataFrame({"momentum": momentum})

    return prices, signals, close, high, low, momentum


def run_vbt_pro(close, momentum, trailing_stop=0.02):
    """Run VectorBT Pro."""
    import vectorbtpro as vbt

    # Tight thresholds for more trades
    entries = momentum > 0.005
    exits = momentum < -0.003

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
    elapsed = time.perf_counter() - start
    n_trades = pf.trades.count()

    return n_trades, elapsed


def run_nb(prices, signals, trailing_stop=0.02):
    """Run backtest-nb."""
    from ml4t.backtest_nb import RuleEngine, Signal, backtest, HWM_HIGH

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.005,
        exit=Signal("momentum") < -0.003,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    start = time.perf_counter()
    result = backtest(prices, signals, strategy, trail_hwm_source=HWM_HIGH)
    elapsed = time.perf_counter() - start

    return result.n_trades, elapsed


def run_rs(prices, signals, trailing_stop=0.02):
    """Run backtest-rs."""
    from ml4t_backtest_rs import RuleEngine, Signal, backtest

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.005,
        exit=Signal("momentum") < -0.003,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    start = time.perf_counter()
    result = backtest(prices, signals, strategy, trail_hwm_source=1)
    elapsed = time.perf_counter() - start

    return result.n_trades, elapsed


def main():
    print(f"\n{'='*90}")
    print("LARGE-SCALE PERFORMANCE: 10M rows target")
    print(f"{'='*90}\n")

    # Warm up JIT
    print("Warming up JIT...")
    prices, signals, close, _, _, momentum = generate_high_frequency_signals(10_000)
    _ = run_nb(prices, signals)
    print("JIT warm-up complete.\n")

    sizes = [100_000, 1_000_000, 5_000_000, 10_000_000]

    print(f"{'Bars':>12} | {'Trades':>10} | {'VBT Pro':>10} | {'nb':>10} | {'rs':>10} | {'nb/VBT':>8} | {'rs/VBT':>8}")
    print("-" * 90)

    for n_bars in sizes:
        print(f"Generating {n_bars:,} bars...", end=" ", flush=True)
        prices, signals, close, _, _, momentum = generate_high_frequency_signals(n_bars)
        print("done.")

        # VBT Pro
        print(f"  Running VBT Pro...", end=" ", flush=True)
        vbt_trades, vbt_time = run_vbt_pro(close, momentum)
        print(f"{vbt_trades:,} trades in {vbt_time:.2f}s")

        # backtest-nb
        print(f"  Running backtest-nb...", end=" ", flush=True)
        nb_trades, nb_time = run_nb(prices, signals)
        print(f"{nb_trades:,} trades in {nb_time:.2f}s")

        # backtest-rs
        print(f"  Running backtest-rs...", end=" ", flush=True)
        rs_trades, rs_time = run_rs(prices, signals)
        print(f"{rs_trades:,} trades in {rs_time:.2f}s")

        # Ratios (how many times slower than VBT)
        nb_ratio = nb_time / vbt_time if vbt_time > 0 else 0
        rs_ratio = rs_time / vbt_time if vbt_time > 0 else 0

        print(f"\n{n_bars:>12,} | {vbt_trades:>10,} | {vbt_time:>9.2f}s | {nb_time:>9.2f}s | {rs_time:>9.2f}s | {nb_ratio:>7.1f}x | {rs_ratio:>7.1f}x\n")

    print(f"{'='*90}")
    print("Ratio > 1.0 means SLOWER than VBT Pro")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
