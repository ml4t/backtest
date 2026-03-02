#!/usr/bin/env python3
"""Multi-asset exact match benchmark across all implementations.

Compares: VBT Pro, backtest-nb, backtest-rs
Goal: Identical trade counts and PnL across all implementations.

Usage:
    source .venv-vectorbt-pro/bin/activate
    python validation/multi_asset_exact_match.py
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class BenchmarkResult:
    """Results from a single implementation run."""
    name: str
    n_trades: int
    total_pnl: float
    final_value: float
    elapsed_ms: float
    trades_per_asset: dict[int, int] | None = None


def generate_multi_asset_data(
    n_bars: int = 1000,
    n_assets: int = 100,
    seed: int = 42,
    entry_prob: float = 0.01,
    exit_prob: float = 0.05,
    per_asset_signals: bool = True,
):
    """Generate multi-asset test data with deterministic signals.

    Args:
        per_asset_signals: If True, each asset has independent signals (VBT Pro style).
                          If False, all assets share the same signal (backtest-nb style).
    """
    rng = np.random.default_rng(seed)

    # Generate prices for each asset
    all_opens = []
    all_highs = []
    all_lows = []
    all_closes = []
    all_volumes = []
    all_entries = []
    all_exits = []

    # Generate shared signals if not per-asset
    if not per_asset_signals:
        shared_entries = rng.random(n_bars) < entry_prob
        shared_exits = rng.random(n_bars) < exit_prob
        shared_entries = shared_entries & ~shared_exits

    for asset_id in range(n_assets):
        # Unique seed per asset for reproducibility
        asset_rng = np.random.default_rng(seed + asset_id * 1000)

        # Random walk price
        returns = asset_rng.normal(0.0002, 0.015, n_bars)
        cumret = np.clip(np.cumsum(returns), -5, 5)
        close = 100.0 * np.exp(cumret)

        daily_vol = asset_rng.uniform(0.005, 0.015, n_bars)
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        open_ = close + asset_rng.normal(0, 0.3, n_bars)
        volume = np.ones(n_bars) * 1e6

        if per_asset_signals:
            # Boolean signals (deterministic based on seed)
            entries = asset_rng.random(n_bars) < entry_prob
            exits = asset_rng.random(n_bars) < exit_prob
            # Can't have both entry and exit on same bar - exit wins
            entries = entries & ~exits
        else:
            # Use shared signals
            entries = shared_entries.copy()
            exits = shared_exits.copy()

        all_opens.append(open_)
        all_highs.append(high)
        all_lows.append(low)
        all_closes.append(close)
        all_volumes.append(volume)
        all_entries.append(entries)
        all_exits.append(exits)

    return {
        "n_bars": n_bars,
        "n_assets": n_assets,
        "opens": np.array(all_opens),      # (n_assets, n_bars)
        "highs": np.array(all_highs),
        "lows": np.array(all_lows),
        "closes": np.array(all_closes),
        "volumes": np.array(all_volumes),
        "entries": np.array(all_entries),  # (n_assets, n_bars) boolean
        "exits": np.array(all_exits),
    }


def run_vbt_pro(data: dict) -> BenchmarkResult:
    """Run VectorBT Pro (ground truth)."""
    import vectorbtpro as vbt

    n_bars = data["n_bars"]
    n_assets = data["n_assets"]

    # VBT expects column-per-asset DataFrames
    close_df = pd.DataFrame(
        data["closes"].T,  # Transpose to (n_bars, n_assets)
        columns=[f"asset_{i:03d}" for i in range(n_assets)]
    )
    entries_df = pd.DataFrame(
        data["entries"].T,
        columns=[f"asset_{i:03d}" for i in range(n_assets)]
    )
    exits_df = pd.DataFrame(
        data["exits"].T,
        columns=[f"asset_{i:03d}" for i in range(n_assets)]
    )

    start = time.perf_counter()

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_df,
        exits=exits_df,
        init_cash=1_000_000.0,
        size=100.0,            # Fixed 100 units per trade
        fees=0.001,
        slippage=0.0005,
        cash_sharing=True,     # Share cash across assets
        accumulate=False,      # No pyramiding
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    n_trades = pf.trades.count()
    total_pnl = pf.trades.pnl.sum() if n_trades > 0 else 0.0
    final_value = pf.final_value

    # Get per-asset trade counts
    trades_per_asset = {}
    if n_trades > 0:
        records = pf.trades.records_readable
        if "Column" in records.columns:
            for col, group in records.groupby("Column"):
                asset_id = int(col.split("_")[1])
                trades_per_asset[asset_id] = len(group)

    return BenchmarkResult(
        name="VBT Pro",
        n_trades=n_trades,
        total_pnl=total_pnl,
        final_value=final_value,
        elapsed_ms=elapsed_ms,
        trades_per_asset=trades_per_asset,
    )


def run_backtest_nb(data: dict) -> BenchmarkResult:
    """Run backtest-nb (Numba implementation)."""
    from ml4t.backtest_nb import backtest as nb_backtest
    from ml4t.backtest_nb import RuleEngine, Signal

    n_bars = data["n_bars"]
    n_assets = data["n_assets"]

    # backtest-nb expects BAR-MAJOR layout: all assets at bar 0, then bar 1, etc.
    # Data layout: [bar0_asset0, bar0_asset1, ..., bar1_asset0, bar1_asset1, ...]
    all_rows = []
    for bar in range(n_bars):
        for asset_id in range(n_assets):
            all_rows.append({
                "open": data["opens"][asset_id, bar],
                "high": data["highs"][asset_id, bar],
                "low": data["lows"][asset_id, bar],
                "close": data["closes"][asset_id, bar],
                "volume": data["volumes"][asset_id, bar],
                "asset": asset_id,
            })

    prices_df = pl.DataFrame(all_rows)

    # Signals: also bar-major
    signal_rows = []
    for bar in range(n_bars):
        for asset_id in range(n_assets):
            signal_rows.append({
                "entry": float(data["entries"][asset_id, bar]),
                "exit": float(data["exits"][asset_id, bar]),
            })

    signals_df = pl.DataFrame(signal_rows)

    strategy = RuleEngine(
        entry=Signal("entry") > 0.5,
        exit=Signal("exit") > 0.5,
        position_size=100.0,
    )

    start = time.perf_counter()

    result = nb_backtest(
        prices_df, signals_df, strategy,
        initial_cash=1_000_000.0,
        commission=0.001,
        slippage=0.0005,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Count per-asset trades
    trades_per_asset = {}
    if result.n_trades > 0:
        trades = result.trades[:result.n_trades]
        for t in trades:
            asset_id = int(t["asset_id"])
            trades_per_asset[asset_id] = trades_per_asset.get(asset_id, 0) + 1

    # Compute total PnL from trades
    total_pnl = 0.0
    if result.n_trades > 0:
        trades = result.trades[:result.n_trades]
        total_pnl = sum(float(t["pnl"]) for t in trades)

    return BenchmarkResult(
        name="backtest-nb",
        n_trades=result.n_trades,
        total_pnl=total_pnl,
        final_value=result.final_value,
        elapsed_ms=elapsed_ms,
        trades_per_asset=trades_per_asset,
    )


def run_backtest_rs(data: dict) -> BenchmarkResult:
    """Run backtest-rs (Rust implementation)."""
    from ml4t_backtest_rs import RuleEngine, Signal, backtest as rs_backtest

    n_bars = data["n_bars"]
    n_assets = data["n_assets"]

    # backtest-rs expects bar-major layout: index = bar * n_assets + asset
    # Input data is (n_assets, n_bars), we need to transpose to (n_bars, n_assets) then flatten
    opens = data["opens"].T.flatten()      # Now: all assets at bar 0, then bar 1, ...
    highs = data["highs"].T.flatten()
    lows = data["lows"].T.flatten()
    closes = data["closes"].T.flatten()
    volumes = data["volumes"].T.flatten()

    prices_df = pl.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Signals: also bar-major
    entries = data["entries"].T.flatten().astype(np.float64)
    exits = data["exits"].T.flatten().astype(np.float64)

    signals_df = pl.DataFrame({
        "entry": entries,
        "exit": exits,
    })

    strategy = RuleEngine(
        entry=Signal("entry") > 0.5,
        exit=Signal("exit") > 0.5,
        position_size=100.0,
    )

    start = time.perf_counter()

    result = rs_backtest(
        prices_df,
        signals_df,
        strategy,
        initial_cash=1_000_000.0,
        commission=0.001,
        slippage=0.0005,
        n_assets=n_assets,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Compute total_pnl from final_value - initial_cash
    total_pnl = result.final_value - 1_000_000.0

    return BenchmarkResult(
        name="backtest-rs",
        n_trades=result.n_trades,
        total_pnl=total_pnl,
        final_value=result.final_value,
        elapsed_ms=elapsed_ms,
        trades_per_asset=None,  # TODO: Add per-asset tracking to rs
    )


def print_results(results: list[BenchmarkResult], data: dict):
    """Print comparison table."""
    n_bars = data["n_bars"]
    n_assets = data["n_assets"]
    data_points = n_bars * n_assets

    print(f"\n{'='*80}")
    print(f"Multi-Asset Exact Match Benchmark")
    print(f"{'='*80}")
    print(f"Configuration: {n_bars:,} bars × {n_assets} assets = {data_points:,} data points")
    print(f"Expected trades: ~{n_bars * n_assets * 0.01 * 0.5:.0f} (1% entry, ~50% complete)")
    print()

    # Header
    print(f"{'Implementation':<15} | {'Trades':>10} | {'Total PnL':>15} | {'Final Value':>15} | {'Time (ms)':>10}")
    print("-" * 80)

    # Reference (VBT Pro)
    ref = results[0]
    print(f"{ref.name:<15} | {ref.n_trades:>10,} | {ref.total_pnl:>15,.2f} | {ref.final_value:>15,.2f} | {ref.elapsed_ms:>10,.1f}")

    # Compare others
    for r in results[1:]:
        trade_diff = r.n_trades - ref.n_trades
        pnl_diff = r.total_pnl - ref.total_pnl
        match = "✓" if trade_diff == 0 and abs(pnl_diff) < 1.0 else "✗"
        print(f"{r.name:<15} | {r.n_trades:>10,} | {r.total_pnl:>15,.2f} | {r.final_value:>15,.2f} | {r.elapsed_ms:>10,.1f} {match}")

    # Summary
    print()
    all_match = all(r.n_trades == ref.n_trades for r in results[1:])
    if all_match:
        print("✅ ALL IMPLEMENTATIONS MATCH")
    else:
        print("❌ MISMATCH DETECTED")
        for r in results[1:]:
            if r.n_trades != ref.n_trades:
                print(f"   {r.name}: {r.n_trades - ref.n_trades:+d} trades difference")


def main():
    """Run multi-asset benchmark."""
    # Test with shared signals (same signal for all assets)
    # This matches backtest-nb architecture
    print("=" * 80)
    print("MODE: SHARED SIGNALS (same entry/exit signal for all assets)")
    print("=" * 80)

    configs = [
        (1000, 10),    # 10K data points - quick sanity check
        (1000, 100),   # 100K data points - small scale
    ]

    for n_bars, n_assets in configs:
        print(f"\n{'#'*80}")
        print(f"# Testing: {n_bars:,} bars × {n_assets} assets (SHARED SIGNALS)")
        print(f"{'#'*80}")

        data = generate_multi_asset_data(
            n_bars=n_bars, n_assets=n_assets, per_asset_signals=False
        )

        results = []

        try:
            results.append(run_vbt_pro(data))
        except Exception as e:
            print(f"VBT Pro error: {e}")

        try:
            results.append(run_backtest_nb(data))
        except Exception as e:
            print(f"backtest-nb error: {e}")

        try:
            results.append(run_backtest_rs(data))
        except Exception as e:
            print(f"backtest-rs error: {e}")

        if results:
            print_results(results, data)

    # Also test per-asset signals for reference
    print("\n" + "=" * 80)
    print("MODE: PER-ASSET SIGNALS (VBT Pro style, backtest-nb won't match)")
    print("=" * 80)

    for n_bars, n_assets in [(1000, 10)]:
        print(f"\n{'#'*80}")
        print(f"# Testing: {n_bars:,} bars × {n_assets} assets (PER-ASSET SIGNALS)")
        print(f"{'#'*80}")

        data = generate_multi_asset_data(
            n_bars=n_bars, n_assets=n_assets, per_asset_signals=True
        )

        results = []

        try:
            results.append(run_vbt_pro(data))
        except Exception as e:
            print(f"VBT Pro error: {e}")

        try:
            results.append(run_backtest_nb(data))
        except Exception as e:
            print(f"backtest-nb error: {e}")

        try:
            results.append(run_backtest_rs(data))
        except Exception as e:
            print(f"backtest-rs error: {e}")

        if results:
            print_results(results, data)


if __name__ == "__main__":
    main()
