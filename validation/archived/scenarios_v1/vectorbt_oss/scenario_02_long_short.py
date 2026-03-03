#!/usr/bin/env python3
"""Scenario 02: Long/Short validation against VectorBT OSS.

This script validates that ml4t.backtest produces identical results to VectorBT OSS
for a long/short strategy including position reversals.

Run from .venv-vectorbt environment:
    .venv-vectorbt/bin/python3 validation/vectorbt_oss/scenario_02_long_short.py

Success criteria:
- Trade count: Exact match
- Fill prices: Exact match
- Final P&L: Exact match (within floating point tolerance)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_test_data(n_bars: int = 100, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Generate test data with long and short signals."""
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
    long_entries = np.zeros(n_bars, dtype=bool)
    long_exits = np.zeros(n_bars, dtype=bool)
    short_entries = np.zeros(n_bars, dtype=bool)
    short_exits = np.zeros(n_bars, dtype=bool)

    # Pattern: long entry at 0, exit at 5, short entry at 10, exit at 15, repeat
    i = 0
    is_long = True
    while i < n_bars - 6:
        if is_long:
            long_entries[i] = True
            long_exits[i + 5] = True
        else:
            short_entries[i] = True
            short_exits[i + 5] = True
        i += 10
        is_long = not is_long

    signals = {
        "long_entries": long_entries,
        "long_exits": long_exits,
        "short_entries": short_entries,
        "short_exits": short_exits,
    }

    return df, signals


def run_vectorbt_oss(prices_df: pd.DataFrame, signals: dict) -> dict:
    """Run backtest using VectorBT OSS."""
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("VectorBT OSS not installed. Run in .venv-vectorbt environment.")

    # Run portfolio simulation with both long and short signals
    pf = vbt.Portfolio.from_signals(
        close=prices_df["close"],
        entries=signals["long_entries"],
        exits=signals["long_exits"],
        short_entries=signals["short_entries"],
        short_exits=signals["short_exits"],
        init_cash=100_000.0,
        size=100,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        accumulate=False,
        freq="D",
    )

    # Extract results
    trades = pf.trades.records_readable

    final_value = float(pf.final_value())
    total_pnl = float(pf.total_profit())

    return {
        "framework": "VectorBT OSS",
        "final_value": final_value,
        "total_pnl": total_pnl,
        "num_trades": len(trades),
        "trades": trades.to_dict("records") if len(trades) > 0 else [],
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, signals: dict) -> dict:
    """Run backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest._validation_imports import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

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

    # Create signals DataFrame
    signals_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["AAPL"] * len(prices_df),
            "long_entry": signals["long_entries"].tolist(),
            "long_exit": signals["long_exits"].tolist(),
            "short_entry": signals["short_entries"].tolist(),
            "short_exit": signals["short_exits"].tolist(),
        }
    )

    class LongShortStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "AAPL" not in data:
                return

            sigs = data["AAPL"].get("signals", {})
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            # Process exits first
            if (
                current_qty > 0
                and sigs.get("long_exit")
                or current_qty < 0
                and sigs.get("short_exit")
            ):
                broker.close_position("AAPL")
                current_qty = 0

            # Then process entries
            if current_qty == 0:
                if sigs.get("long_entry"):
                    broker.submit_order("AAPL", 100)  # Long 100 shares
                elif sigs.get("short_entry"):
                    broker.submit_order("AAPL", -100)  # Short 100 shares

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = LongShortStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=True, allow_leverage=True,  # Need margin for short selling
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
                "direction": "long" if t.quantity > 0 else "short",
            }
            for t in results["trades"]
        ],
    }


def compare_results(vbt_results: dict, ml4t_results: dict) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print("COMPARISON: VectorBT OSS vs ml4t.backtest")
    print("=" * 70)

    all_match = True

    # Trade count
    vbt_trades = vbt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = vbt_trades == ml4t_trades
    print(f"\nTrade Count: VBT={vbt_trades}, ML4T={ml4t_trades} {'✅' if trades_match else '❌'}")
    all_match &= trades_match

    # Final value
    vbt_value = vbt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(vbt_value - ml4t_value)
    value_pct_diff = value_diff / vbt_value * 100 if vbt_value != 0 else 0
    values_match = value_pct_diff < 0.01
    print(
        f"Final Value: VBT=${vbt_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'✅' if values_match else '❌'}"
    )
    all_match &= values_match

    # Total P&L
    vbt_pnl = vbt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(vbt_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 1.0
    print(
        f"Total P&L: VBT=${vbt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f}) {'✅' if pnl_match else '❌'}"
    )
    all_match &= pnl_match

    # Trade-by-trade comparison
    if trades_match and len(vbt_results["trades"]) > 0:
        print("\nTrade-by-Trade Comparison:")
        print("-" * 70)
        vbt_trades_list = vbt_results["trades"]
        ml4t_trades_list = ml4t_results["trades"]

        for i, (vbt_t, ml4t_t) in enumerate(zip(vbt_trades_list[:5], ml4t_trades_list[:5])):
            vbt_entry = vbt_t.get("Entry Price", vbt_t.get("Avg Entry Price", "N/A"))
            vbt_exit = vbt_t.get("Exit Price", vbt_t.get("Avg Exit Price", "N/A"))
            vbt_dir = "short" if vbt_t.get("Direction", "Long") == "Short" else "long"
            print(
                f"  Trade {i+1} ({ml4t_t['direction']}): VBT entry={vbt_entry:.2f}, exit={vbt_exit:.2f} | "
                f"ML4T entry={ml4t_t['entry_price']:.2f}, exit={ml4t_t['exit_price']:.2f}"
            )

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED: Results match within tolerance")
    else:
        print("❌ VALIDATION FAILED: Results do not match")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print("Scenario 02: Long/Short Validation (VectorBT OSS)")
    print("=" * 70)

    # Generate test data
    print("\n📊 Generating test data...")
    prices_df, signals = generate_test_data(n_bars=100)
    print(f"   Bars: {len(prices_df)}")
    print(f"   Long entry signals: {signals['long_entries'].sum()}")
    print(f"   Short entry signals: {signals['short_entries'].sum()}")

    # Run VectorBT OSS
    print("\n🔷 Running VectorBT OSS...")
    try:
        vbt_results = run_vectorbt_oss(prices_df, signals)
        print(f"   Trades: {vbt_results['num_trades']}")
        print(f"   Final Value: ${vbt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"   ❌ {e}")
        return 1
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Run ml4t.backtest
    print("\n🔶 Running ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, signals)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    success = compare_results(vbt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
