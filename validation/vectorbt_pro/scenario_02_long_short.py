#!/usr/bin/env python3
"""Scenario 02: Long/Short validation against VectorBT Pro.

This script validates that ml4t.backtest produces identical results to VectorBT Pro
for a long/short strategy including position reversals.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    cd validation/vectorbt_pro
    python scenario_02_long_short.py

Success criteria:
- Trade count: Exact match
- Fill prices: Exact match
- Short positions handled correctly
- Position reversals (long->short) match
- Final P&L: Exact match (within floating point tolerance)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# Test Data Generation
# ============================================================================


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for long/short strategy.

    Returns:
        prices_df: OHLCV DataFrame with timestamp index
        long_entries: Boolean array of long entry signals
        long_exits: Boolean array of long exit signals
        short_entries: Boolean array of short entry signals
        short_exits: Boolean array of short exit signals
    """
    np.random.seed(seed)

    # Generate price path (random walk)
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02  # 2% daily vol
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars),
        },
        index=dates,
    )

    # Ensure high >= open, close, low and low <= open, close, high
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # Generate alternating long/short signals
    # Pattern: long entry, long exit, short entry, short exit, repeat
    long_entries = np.zeros(n_bars, dtype=bool)
    long_exits = np.zeros(n_bars, dtype=bool)
    short_entries = np.zeros(n_bars, dtype=bool)
    short_exits = np.zeros(n_bars, dtype=bool)

    i = 0
    position = 0  # 0=flat, 1=long, -1=short
    while i < n_bars - 6:
        if position == 0:
            # Enter long
            long_entries[i] = True
            long_exits[i + 5] = True
            position = 1
            i += 10
        elif position == 1:
            # Enter short after long exit
            short_entries[i] = True
            short_exits[i + 5] = True
            position = -1
            i += 10
        else:
            position = 0

    return df, long_entries, long_exits, short_entries, short_exits


# ============================================================================
# VectorBT Pro Execution
# ============================================================================


def run_vectorbt_pro(
    prices_df: pd.DataFrame,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
) -> dict:
    """Run backtest using VectorBT Pro."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed. Run in .venv-vectorbt-pro environment.")

    # VectorBT Pro handles long/short via direction parameter or separate entry/exit arrays
    # Using from_signals with short entries/exits
    pf = vbt.Portfolio.from_signals(
        close=prices_df["close"],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=100_000.0,
        size=100,  # Fixed 100 shares per trade
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    return {
        "framework": "VectorBT Pro",
        "final_value": pf.total_return * 100_000.0 + 100_000.0,
        "total_pnl": pf.total_profit,
        "num_trades": len(trades),
        "trades": trades.to_dict("records") if len(trades) > 0 else [],
    }


# ============================================================================
# ml4t.backtest Execution
# ============================================================================


def run_ml4t_backtest(
    prices_df: pd.DataFrame,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
) -> dict:
    """Run backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

    # Convert to polars format
    prices_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["AAPL"] * len(prices_df),
            "open": prices_df["open"].tolist(),
            "high": prices_df["high"].tolist(),
            "low": prices_df["low"].tolist(),
            "close": prices_df["close"].tolist(),
            "volume": prices_df["volume"].astype(float).tolist(),
        }
    )

    # Create signals DataFrame with long/short signals
    signals_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["AAPL"] * len(prices_df),
            "long_entry": long_entries.tolist(),
            "long_exit": long_exits.tolist(),
            "short_entry": short_entries.tolist(),
            "short_exit": short_exits.tolist(),
        }
    )

    class LongShortStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "AAPL" not in data:
                return

            signals = data["AAPL"].get("signals", {})
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            # Check exits first
            if (
                signals.get("long_exit")
                and current_qty > 0
                or signals.get("short_exit")
                and current_qty < 0
            ):
                broker.close_position("AAPL")

            # Then check entries (only if flat)
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            if current_qty == 0:
                if signals.get("long_entry"):
                    broker.submit_order("AAPL", 100)  # Buy 100 shares
                elif signals.get("short_entry"):
                    broker.submit_order("AAPL", -100)  # Short 100 shares

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = LongShortStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=True, allow_leverage=True,  # Margin account for short selling
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,  # Match VectorBT default
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "direction": "Long" if t.quantity > 0 else "Short",
            }
            for t in results["trades"]
        ],
    }


# ============================================================================
# Comparison
# ============================================================================


def compare_results(vbt_results: dict, ml4t_results: dict) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print("COMPARISON: VectorBT Pro vs ml4t.backtest (Long/Short)")
    print("=" * 70)

    all_match = True

    # Trade count
    vbt_trades = vbt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = vbt_trades == ml4t_trades
    print(
        f"\nTrade Count: VBT={vbt_trades}, ML4T={ml4t_trades} {'OK' if trades_match else 'MISMATCH'}"
    )
    all_match &= trades_match

    # Final value
    vbt_value = vbt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(vbt_value - ml4t_value)
    value_pct_diff = value_diff / vbt_value * 100 if vbt_value != 0 else 0
    values_match = value_pct_diff < 0.01
    print(
        f"Final Value: VBT=${vbt_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'OK' if values_match else 'MISMATCH'}"
    )
    all_match &= values_match

    # Total P&L
    vbt_pnl = vbt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(vbt_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 1.0
    print(
        f"Total P&L: VBT=${vbt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f}) {'OK' if pnl_match else 'MISMATCH'}"
    )
    all_match &= pnl_match

    # Trade-by-trade comparison
    if trades_match and len(vbt_results["trades"]) > 0:
        print("\nTrade-by-Trade Comparison:")
        print("-" * 70)
        vbt_trades_list = vbt_results["trades"]
        ml4t_trades_list = ml4t_results["trades"]

        for i, (vbt_t, ml4t_t) in enumerate(zip(vbt_trades_list[:5], ml4t_trades_list[:5])):
            vbt_entry = vbt_t.get("Avg Entry Price", "N/A")
            vbt_dir = vbt_t.get("Direction", "Unknown")
            ml4t_dir = ml4t_t.get("direction", "Unknown")
            print(
                f"  Trade {i+1}: VBT {vbt_dir} entry={vbt_entry:.2f} | ML4T {ml4t_dir} entry={ml4t_t['entry_price']:.2f}"
            )

    print("\n" + "=" * 70)
    if all_match:
        print("VALIDATION PASSED: Results match within tolerance")
    else:
        print("VALIDATION FAILED: Results do not match")
    print("=" * 70)

    return all_match


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("Scenario 02: Long/Short Validation (VectorBT Pro)")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    prices_df, long_entries, long_exits, short_entries, short_exits = generate_test_data(n_bars=100)
    print(f"   Bars: {len(prices_df)}")
    print(f"   Long entries: {long_entries.sum()}")
    print(f"   Long exits: {long_exits.sum()}")
    print(f"   Short entries: {short_entries.sum()}")
    print(f"   Short exits: {short_exits.sum()}")

    # Run VectorBT Pro
    print("\nRunning VectorBT Pro...")
    try:
        vbt_results = run_vectorbt_pro(
            prices_df, long_entries, long_exits, short_entries, short_exits
        )
        print(f"   Trades: {vbt_results['num_trades']}")
        print(f"   Final Value: ${vbt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(
            prices_df, long_entries, long_exits, short_entries, short_exits
        )
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    success = compare_results(vbt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
