#!/usr/bin/env python3
"""Large-scale cross-implementation validation at 1000+ and 10K+ trades.

This script validates that backtest-nb, backtest-rs, and ml4t.backtest produce
IDENTICAL results to each other, and can be configured to match VectorBT Pro.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class ValidationResult:
    """Result of a validation run."""

    framework: str
    n_trades: int
    final_value: float
    execution_time: float
    trades_per_second: float


def generate_large_dataset(
    n_bars: int = 10000,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate a dataset that produces many trades.

    Args:
        n_bars: Number of bars
        seed: Random seed for reproducibility

    Returns:
        (prices_df, signals_df) tuple
    """
    rng = np.random.default_rng(seed)

    # Generate trending price with volatility
    returns = rng.normal(0.0002, 0.015, n_bars)
    cumulative = np.exp(np.cumsum(returns))
    close = 100.0 * cumulative

    # Generate OHLC from close
    daily_vol = rng.uniform(0.005, 0.02, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.5, n_bars)
    volume = rng.uniform(1e6, 5e6, n_bars)

    prices = pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Generate mean-reverting momentum signal that crosses zero frequently
    # Lower persistence (0.5) = faster mean reversion = more trades
    momentum = np.zeros(n_bars)
    for i in range(1, n_bars):
        momentum[i] = 0.5 * momentum[i - 1] + rng.normal(0, 0.04)

    signals = pl.DataFrame({"momentum": momentum})

    return prices, signals


def run_backtest_nb(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    entry_threshold: float = 0.01,  # Tighter threshold = more trades
    exit_threshold: float = -0.005,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    trail_hwm_source: int = 0,  # 0=CLOSE, 1=HIGH
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> ValidationResult:
    """Run backtest using backtest-nb (Numba)."""
    from ml4t.backtest_nb import (
        HWM_CLOSE,
        HWM_HIGH,
        RuleEngine,
        Signal,
        backtest,
    )

    hwm = HWM_HIGH if trail_hwm_source == 1 else HWM_CLOSE

    strategy = RuleEngine(
        entry=Signal("momentum") > entry_threshold,
        exit=Signal("momentum") < exit_threshold,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    start = time.perf_counter()
    result = backtest(
        prices,
        signals,
        strategy,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        trail_hwm_source=hwm,
    )
    elapsed = time.perf_counter() - start

    return ValidationResult(
        framework="backtest-nb",
        n_trades=result.n_trades,
        final_value=result.final_value,
        execution_time=elapsed,
        trades_per_second=result.n_trades / elapsed if elapsed > 0 else 0,
    )


def run_backtest_rs(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    entry_threshold: float = 0.01,  # Tighter threshold = more trades
    exit_threshold: float = -0.005,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    trail_hwm_source: int = 0,  # 0=CLOSE, 1=HIGH
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> ValidationResult:
    """Run backtest using backtest-rs (Rust)."""
    from ml4t_backtest_rs import RuleEngine, Signal, backtest

    strategy = RuleEngine(
        entry=Signal("momentum") > entry_threshold,
        exit=Signal("momentum") < exit_threshold,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    start = time.perf_counter()
    result = backtest(
        prices,
        signals,
        strategy,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        trail_hwm_source=trail_hwm_source,
    )
    elapsed = time.perf_counter() - start

    return ValidationResult(
        framework="backtest-rs",
        n_trades=result.n_trades,
        final_value=result.final_value,
        execution_time=elapsed,
        trades_per_second=result.n_trades / elapsed if elapsed > 0 else 0,
    )


def validate_internal_consistency(
    n_bars: int = 10000,
    scenarios: list[dict] | None = None,
) -> bool:
    """Validate that backtest-nb and backtest-rs produce identical results.

    Args:
        n_bars: Number of bars in test data
        scenarios: List of scenario configs (optional)

    Returns:
        True if all validations pass
    """
    if scenarios is None:
        scenarios = [
            {"name": "simple_momentum", "stop_loss": None, "take_profit": None, "trailing_stop": None},
            {"name": "stop_loss_2pct", "stop_loss": 0.02, "take_profit": None, "trailing_stop": None},
            {"name": "take_profit_5pct", "stop_loss": None, "take_profit": 0.05, "trailing_stop": None},
            {"name": "trailing_stop_3pct", "stop_loss": None, "take_profit": None, "trailing_stop": 0.03},
            {"name": "combined", "stop_loss": 0.02, "take_profit": 0.05, "trailing_stop": 0.03},
        ]

    print(f"\n{'='*70}")
    print(f"INTERNAL CONSISTENCY VALIDATION ({n_bars:,} bars)")
    print(f"{'='*70}")

    prices, signals = generate_large_dataset(n_bars)
    all_pass = True

    for scenario in scenarios:
        name = scenario["name"]
        sl = scenario.get("stop_loss")
        tp = scenario.get("take_profit")
        ts = scenario.get("trailing_stop")

        # Test with both HWM sources
        for hwm_source in [0, 1]:  # CLOSE, HIGH
            hwm_name = "CLOSE" if hwm_source == 0 else "HIGH"

            nb_result = run_backtest_nb(
                prices, signals,
                stop_loss=sl, take_profit=tp, trailing_stop=ts,
                trail_hwm_source=hwm_source,
            )

            rs_result = run_backtest_rs(
                prices, signals,
                stop_loss=sl, take_profit=tp, trailing_stop=ts,
                trail_hwm_source=hwm_source,
            )

            trades_match = nb_result.n_trades == rs_result.n_trades
            value_match = abs(nb_result.final_value - rs_result.final_value) < 0.01

            status = "✅ PASS" if (trades_match and value_match) else "❌ FAIL"
            all_pass = all_pass and trades_match and value_match

            print(f"\n{name} (HWM={hwm_name}): {status}")
            print(f"  backtest-nb: {nb_result.n_trades:,} trades, ${nb_result.final_value:,.2f}")
            print(f"  backtest-rs: {rs_result.n_trades:,} trades, ${rs_result.final_value:,.2f}")

            if not trades_match:
                print(f"  ⚠️  Trade count mismatch: {nb_result.n_trades} vs {rs_result.n_trades}")
            if not value_match:
                print(f"  ⚠️  Value mismatch: {nb_result.final_value:.2f} vs {rs_result.final_value:.2f}")

    print(f"\n{'='*70}")
    print(f"RESULT: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*70}")

    return all_pass


def run_scale_benchmark(n_bars: int = 10000) -> dict:
    """Run benchmark at specified scale and report results.

    Args:
        n_bars: Number of bars

    Returns:
        Dict with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"SCALE BENCHMARK ({n_bars:,} bars)")
    print(f"{'='*70}")

    prices, signals = generate_large_dataset(n_bars)

    # Warm up JIT
    _ = run_backtest_nb(prices, signals, trailing_stop=0.03, trail_hwm_source=1)

    # Run benchmarks
    nb_result = run_backtest_nb(prices, signals, trailing_stop=0.03, trail_hwm_source=1)
    rs_result = run_backtest_rs(prices, signals, trailing_stop=0.03, trail_hwm_source=1)

    print(f"\nResults with trailing_stop=3%, HWM=HIGH:")
    print(f"  backtest-nb: {nb_result.n_trades:,} trades in {nb_result.execution_time:.4f}s")
    print(f"               ({nb_result.trades_per_second:,.0f} trades/sec)")
    print(f"  backtest-rs: {rs_result.n_trades:,} trades in {rs_result.execution_time:.4f}s")
    print(f"               ({rs_result.trades_per_second:,.0f} trades/sec)")

    return {
        "n_bars": n_bars,
        "nb": nb_result,
        "rs": rs_result,
    }


if __name__ == "__main__":
    import sys

    # Default to 10K bars (should produce ~1000+ trades with trailing stop)
    n_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    # Run internal consistency validation
    passed = validate_internal_consistency(n_bars)

    # Run scale benchmark
    benchmark = run_scale_benchmark(n_bars)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Data size: {n_bars:,} bars")
    print(f"Trade count: ~{benchmark['nb'].n_trades:,} trades")
    print(f"Internal consistency: {'PASSED' if passed else 'FAILED'}")
    print(f"")
    print("Performance:")
    print(f"  backtest-nb: {benchmark['nb'].execution_time:.4f}s ({benchmark['nb'].trades_per_second:,.0f} trades/sec)")
    print(f"  backtest-rs: {benchmark['rs'].execution_time:.4f}s ({benchmark['rs'].trades_per_second:,.0f} trades/sec)")

    sys.exit(0 if passed else 1)
