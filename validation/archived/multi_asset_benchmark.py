#!/usr/bin/env python3
"""
Proper multi-asset benchmark: 100-500 assets, scaling bars, proportional trades.

Key requirements:
1. Multi-asset universe (100, 500 assets)
2. Independent random signals per asset
3. Trades scale proportionally with bars AND assets
4. Compare against VBT Pro properly
"""

import time
import numpy as np
import pandas as pd


def generate_multi_asset_data(n_bars: int, n_assets: int, seed: int = 42):
    """
    Generate multi-asset OHLCV + entry/exit signals.

    Each asset has independent price and signal series.
    Signals are random with ~1% entry probability per bar per asset.

    Returns pandas DataFrames for VBT Pro compatibility.
    """
    rng = np.random.default_rng(seed)

    # Generate independent price series for each asset
    # Shape: (n_bars, n_assets)
    returns = rng.normal(0.0001, 0.02, (n_bars, n_assets))
    cumret = np.cumsum(returns, axis=0)
    cumret = np.clip(cumret, -5, 5)  # Prevent overflow
    close = 100.0 * np.exp(cumret)

    # OHLC from close
    daily_range = rng.uniform(0.005, 0.015, (n_bars, n_assets))
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    open_ = close * (1 + rng.normal(0, 0.005, (n_bars, n_assets)))

    # Random entry signals: ~1% probability per bar per asset
    # This gives predictable trade scaling: n_bars * n_assets * 0.01 entries
    entry_prob = 0.01
    entries = rng.random((n_bars, n_assets)) < entry_prob

    # Exit after random holding period (5-20 bars) or on signal
    exits = rng.random((n_bars, n_assets)) < 0.05  # ~5% exit prob per bar

    # Create column names
    asset_names = [f"asset_{i:03d}" for i in range(n_assets)]

    # Create DataFrames
    close_df = pd.DataFrame(close, columns=asset_names)
    high_df = pd.DataFrame(high, columns=asset_names)
    low_df = pd.DataFrame(low, columns=asset_names)
    open_df = pd.DataFrame(open_, columns=asset_names)
    entries_df = pd.DataFrame(entries, columns=asset_names)
    exits_df = pd.DataFrame(exits, columns=asset_names)

    return {
        'open': open_df,
        'high': high_df,
        'low': low_df,
        'close': close_df,
        'entries': entries_df,
        'exits': exits_df,
        'n_bars': n_bars,
        'n_assets': n_assets,
    }


def run_vbt_pro(data, trailing_stop=0.03):
    """Run VectorBT Pro multi-asset backtest."""
    import vectorbtpro as vbt

    start = time.perf_counter()
    pf = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=data['entries'],
        exits=data['exits'],
        tsl_stop=trailing_stop,
        init_cash=1_000_000.0,  # More cash for multi-asset
        fees=0.001,
        slippage=0.0005,
        cash_sharing=True,  # Share cash across assets
    )
    elapsed = time.perf_counter() - start

    n_trades = pf.trades.count()
    if hasattr(n_trades, 'sum'):
        n_trades = int(n_trades.sum())

    return n_trades, elapsed


def run_backtest_nb_multi(data, trailing_stop=0.03):
    """Run backtest-nb with multi-asset data."""
    import polars as pl
    from ml4t.backtest_nb import backtest, HWM_HIGH

    n_bars = data['n_bars']
    n_assets = data['n_assets']

    # Convert to format expected by backtest-nb
    # Prices: need to flatten or iterate per asset
    # backtest-nb expects (n_bars, n_assets) shaped arrays internally

    # Create Polars DataFrames
    prices = pl.DataFrame({
        'open': data['open'].values.flatten('F'),  # Column-major for multi-asset
        'high': data['high'].values.flatten('F'),
        'low': data['low'].values.flatten('F'),
        'close': data['close'].values.flatten('F'),
        'volume': np.ones(n_bars * n_assets) * 1e6,
    })

    # For signals, we need entry/exit as numeric
    entry_signal = data['entries'].values.astype(float).flatten('F')
    exit_signal = data['exits'].values.astype(float).flatten('F')

    signals = pl.DataFrame({
        'entry': entry_signal,
        'exit': exit_signal,
    })

    # This won't work directly - backtest-nb expects a RuleEngine
    # Let me check if it can handle raw boolean signals...
    # Actually, let's use the Signal DSL properly

    from ml4t.backtest_nb import RuleEngine, Signal

    strategy = RuleEngine(
        entry=Signal("entry") > 0.5,  # entry signal is 0 or 1
        exit=Signal("exit") > 0.5,
        trailing_stop=trailing_stop,
        position_size=1000.0,  # Dollars per position
    )

    start = time.perf_counter()
    result = backtest(
        prices, signals, strategy,
        initial_cash=1_000_000.0,
        trail_hwm_source=HWM_HIGH,
    )
    elapsed = time.perf_counter() - start

    return result.n_trades, elapsed


def run_backtest_nb_loop(data, trailing_stop=0.03):
    """Run backtest-nb per asset and sum trades (workaround for multi-asset)."""
    import polars as pl
    from ml4t.backtest_nb import backtest, RuleEngine, Signal, HWM_HIGH

    n_assets = data['n_assets']
    total_trades = 0

    strategy = RuleEngine(
        entry=Signal("entry") > 0.5,
        exit=Signal("exit") > 0.5,
        trailing_stop=trailing_stop,
        position_size=10000.0,
    )

    start = time.perf_counter()

    for i in range(n_assets):
        asset_name = f"asset_{i:03d}"

        prices = pl.DataFrame({
            'open': data['open'][asset_name].values,
            'high': data['high'][asset_name].values,
            'low': data['low'][asset_name].values,
            'close': data['close'][asset_name].values,
            'volume': np.ones(data['n_bars']) * 1e6,
        })

        signals = pl.DataFrame({
            'entry': data['entries'][asset_name].values.astype(float),
            'exit': data['exits'][asset_name].values.astype(float),
        })

        result = backtest(
            prices, signals, strategy,
            initial_cash=1_000_000.0,
            trail_hwm_source=HWM_HIGH,
        )
        total_trades += result.n_trades

    elapsed = time.perf_counter() - start

    return total_trades, elapsed


def main():
    print(f"\n{'='*90}")
    print("MULTI-ASSET BENCHMARK: Proper scaling test")
    print(f"{'='*90}\n")

    # Test configurations: (n_bars, n_assets)
    configs = [
        (10_000, 100),      # 1M data points, ~10K expected trades
        (100_000, 100),     # 10M data points, ~100K expected trades
        (100_000, 500),     # 50M data points, ~500K expected trades
        (1_000_000, 100),   # 100M data points, ~1M expected trades
    ]

    print("Warming up JIT...")
    import polars as pl
    from ml4t.backtest_nb import backtest, RuleEngine, Signal, HWM_HIGH
    warmup_prices = pl.DataFrame({
        'open': [100.0]*100, 'high': [101.0]*100,
        'low': [99.0]*100, 'close': [100.0]*100, 'volume': [1e6]*100
    })
    warmup_signals = pl.DataFrame({'entry': [0.0]*100, 'exit': [0.0]*100})
    warmup_strategy = RuleEngine(entry=Signal("entry") > 0.5, exit=Signal("exit") > 0.5, position_size=100.0)
    _ = backtest(warmup_prices, warmup_signals, warmup_strategy)
    print("JIT warm-up complete.\n")

    print(f"{'Config':>20} | {'Data Points':>12} | {'Expected':>10} | {'VBT Trades':>10} | {'VBT Time':>10} | {'nb Trades':>10} | {'nb Time':>10}")
    print("-" * 110)

    for n_bars, n_assets in configs:
        data_points = n_bars * n_assets
        expected_trades = int(n_bars * n_assets * 0.01 * 0.5)  # ~50% of entries become trades

        config_str = f"{n_bars//1000}K x {n_assets}"

        print(f"\nGenerating {config_str}...", end=" ", flush=True)
        data = generate_multi_asset_data(n_bars, n_assets)
        print("done.")

        # VBT Pro
        print(f"  Running VBT Pro...", end=" ", flush=True)
        try:
            vbt_trades, vbt_time = run_vbt_pro(data)
            print(f"{vbt_trades:,} trades in {vbt_time:.2f}s")
        except Exception as e:
            vbt_trades, vbt_time = 0, 0
            print(f"ERROR: {e}")

        # backtest-nb (loop over assets)
        print(f"  Running backtest-nb (per-asset loop)...", end=" ", flush=True)
        try:
            nb_trades, nb_time = run_backtest_nb_loop(data)
            print(f"{nb_trades:,} trades in {nb_time:.2f}s")
        except Exception as e:
            nb_trades, nb_time = 0, 0
            print(f"ERROR: {e}")

        print(f"{config_str:>20} | {data_points:>12,} | {expected_trades:>10,} | {vbt_trades:>10,} | {vbt_time:>9.2f}s | {nb_trades:>10,} | {nb_time:>9.2f}s")

    print(f"\n{'='*90}")
    print("Expected trades = n_bars * n_assets * entry_prob * ~50% conversion")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
