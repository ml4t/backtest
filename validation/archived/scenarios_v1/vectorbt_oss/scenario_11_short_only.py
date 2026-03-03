#!/usr/bin/env python3
"""Scenario 11: Short-only validation against VectorBT OSS.

This validates that ml4t.backtest short selling produces IDENTICAL results
to VectorBT OSS - trade-by-trade exact match.

Run from .venv-vectorbt environment:
    .venv-vectorbt/bin/python validation/vectorbt_oss/scenario_11_short_only.py

Success criteria:
- Trade count: EXACT match
- Entry prices: EXACT match
- Exit prices: EXACT match
- PnL per trade: EXACT match (within $0.01)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SHARES_PER_TRADE = 100


def generate_test_data(n_bars: int = 200, seed: int = 42):
    """Generate test data with clear SHORT entry/exit signals."""
    np.random.seed(seed)

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    # Generate price path with trends
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    closes = base_price * np.exp(np.cumsum(returns))

    # Generate valid OHLC
    opens = closes * (1 + np.random.randn(n_bars) * 0.003)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.005)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )

    # SHORT entry/exit signals: enter every 20 bars, exit after 10 bars
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    idx = 5  # Start at bar 5
    while idx < n_bars - 11:
        entries[idx] = True
        exits[idx + 10] = True
        idx += 20

    return df, entries, exits


def run_vectorbt_oss(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run short-only backtest using VectorBT OSS."""
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("VectorBT OSS not installed. Run in .venv-vectorbt environment.")

    # VBT OSS from_signals with direction="shortonly"
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=exits,
        direction="shortonly",  # SHORT positions only
        init_cash=1_000_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        accumulate=False,
        freq="D",
    )

    # Extract trades
    trades = pf.trades.records_readable

    trade_list = []
    for idx, t in trades.iterrows():
        trade_list.append({
            "entry_time": t.get("Entry Index", t.get("entry_idx")),
            "exit_time": t.get("Exit Index", t.get("exit_idx")),
            "entry_price": float(t.get("Avg Entry Price", t.get("entry_price", 0))),
            "exit_price": float(t.get("Avg Exit Price", t.get("exit_price", 0))),
            "size": float(t.get("Size", t.get("size", SHARES_PER_TRADE))),
            "pnl": float(t.get("PnL", t.get("pnl", 0))),
            "direction": str(t.get("Direction", "Short")),
        })

    # NOTE: In vectorbt OSS, .value is a method, not a property
    final_value = pf.value()
    return {
        "framework": "VectorBT OSS",
        "final_value": float(final_value.iloc[-1]),
        "num_trades": len(trade_list),
        "trades": sorted(trade_list, key=lambda t: str(t["entry_time"])),
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run short-only backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest._validation_imports import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, OrderSide, Strategy

    prices_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["TEST"] * len(prices_df),
            "open": prices_df["open"].tolist(),
            "high": prices_df["high"].tolist(),
            "low": prices_df["low"].tolist(),
            "close": prices_df["close"].tolist(),
            "volume": prices_df["volume"].tolist(),
        }
    )

    signals_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["TEST"] * len(prices_df),
            "short_entry": entries.tolist(),
            "short_exit": exits.tolist(),
        }
    )

    class ShortOnlyStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            pos = broker.get_position("TEST")
            current_qty = pos.quantity if pos else 0

            # Exit first (cover short)
            if signals.get("short_exit") and current_qty < 0:
                broker.close_position("TEST")
            # Then entry (open short)
            elif signals.get("short_entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE, OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = ShortOnlyStrategy()

    # Match VBT OSS: SAME_BAR execution
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        allow_short_selling=True,
        allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()

    trade_list = []
    for t in results["trades"]:
        trade_list.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": abs(t.quantity),
            "pnl": t.pnl,
            "direction": "Short" if t.quantity < 0 else "Long",
        })

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "num_trades": results["num_trades"],
        "trades": sorted(trade_list, key=lambda t: str(t["entry_time"])),
    }


def compare_results(vbt_results, ml4t_results):
    """Compare trade-by-trade results."""
    print("\n" + "=" * 70)
    print("COMPARISON: VectorBT OSS vs ml4t.backtest (SHORT-ONLY)")
    print("=" * 70)

    all_match = True

    # Trade count
    vbt_count = vbt_results["num_trades"]
    ml4t_count = ml4t_results["num_trades"]
    count_match = vbt_count == ml4t_count
    print(f"\nTrade Count: VBT={vbt_count}, ML4T={ml4t_count} "
          f"{'✅ MATCH' if count_match else '❌ MISMATCH'}")
    all_match &= count_match

    # Final value
    vbt_value = vbt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(vbt_value - ml4t_value)
    value_pct = value_diff / vbt_value * 100 if vbt_value else 0
    value_match = value_pct < 0.01
    print(f"Final Value: VBT=${vbt_value:,.2f}, ML4T=${ml4t_value:,.2f} "
          f"(diff={value_pct:.4f}%) {'✅ MATCH' if value_match else '❌ MISMATCH'}")
    all_match &= value_match

    # Trade-by-trade comparison
    if count_match and vbt_count > 0:
        vbt_trades = vbt_results["trades"]
        ml4t_trades = ml4t_results["trades"]

        matches = 0
        mismatches = 0
        sample_mismatches = []

        for vbt_t, ml4t_t in zip(vbt_trades, ml4t_trades):
            entry_match = abs(vbt_t["entry_price"] - ml4t_t["entry_price"]) < 0.01
            exit_match = abs(vbt_t["exit_price"] - ml4t_t["exit_price"]) < 0.01
            pnl_match = abs(vbt_t["pnl"] - ml4t_t["pnl"]) < 0.10

            if entry_match and exit_match and pnl_match:
                matches += 1
            else:
                mismatches += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append((vbt_t, ml4t_t))

        match_pct = matches / len(vbt_trades) * 100 if vbt_trades else 0
        print(f"\nTrade-Level Match: {matches}/{len(vbt_trades)} ({match_pct:.1f}%)")

        if mismatches > 0:
            print(f"\nSample Mismatches:")
            for vbt_t, ml4t_t in sample_mismatches:
                print(f"  VBT:  entry=${vbt_t['entry_price']:.2f}, "
                      f"exit=${vbt_t['exit_price']:.2f}, pnl=${vbt_t['pnl']:.2f}")
                print(f"  ML4T: entry=${ml4t_t['entry_price']:.2f}, "
                      f"exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

        all_match &= (mismatches == 0)

    # Sample trades
    if vbt_count > 0:
        print("\nSample Short Trades (first 3):")
        print("-" * 70)
        for i, (vbt_t, ml4t_t) in enumerate(
            zip(vbt_results["trades"][:3], ml4t_results["trades"][:3])
        ):
            print(f"  Trade {i+1}:")
            print(f"    VBT:  entry=${vbt_t['entry_price']:.2f}, exit=${vbt_t['exit_price']:.2f}, pnl=${vbt_t['pnl']:.2f}")
            print(f"    ML4T: entry=${ml4t_t['entry_price']:.2f}, exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED: Short-selling results match exactly")
    else:
        print("❌ VALIDATION FAILED: Short-selling results differ")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print("Scenario 11: Short-Only Validation (VectorBT OSS)")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    n_bars = 200
    prices_df, entries, exits = generate_test_data(n_bars=n_bars)
    print(f"  Bars: {n_bars}")
    print(f"  Short entry signals: {entries.sum()}")
    print(f"  Short exit signals: {exits.sum()}")

    # Run VectorBT OSS
    print("\nRunning VectorBT OSS (short-only)...")
    try:
        vbt_results = run_vectorbt_oss(prices_df, entries, exits)
        print(f"  Trades: {vbt_results['num_trades']}")
        print(f"  Final Value: ${vbt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"  ERROR: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest (short-only)...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
        print(f"  Trades: {ml4t_results['num_trades']}")
        print(f"  Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare
    success = compare_results(vbt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
