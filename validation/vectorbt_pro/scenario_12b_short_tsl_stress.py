#!/usr/bin/env python3
"""Scenario 12b: SHORT Trailing Stop STRESS TEST against VectorBT Pro.

Tests edge cases for SHORT trailing stop:
1. Gap-through scenarios (overnight gaps up)
2. Tighter stop (3%) for more frequent triggers
3. Multiple consecutive entries
4. Sharp reversals

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_12b_short_tsl_stress.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SHARES_PER_TRADE = 100


def test_gap_through(trail_pct: float = 0.03):
    """Test gap-through scenario: price gaps above TSL level."""
    print(f"\n{'='*70}")
    print(f"TEST: Gap-Through Scenario (TSL={trail_pct*100:.0f}%)")
    print("="*70)

    np.random.seed(123)
    n_bars = 50

    # Create specific price pattern:
    # Bar 0-10: Downtrend (SHORT profit)
    # Bar 11: Gap UP above TSL level
    prices = [100.0]
    for i in range(1, n_bars):
        if i <= 10:
            # Steady downtrend
            change = -0.005
        elif i == 11:
            # Gap UP: previous close was ~95, gap to ~102 (above 95*1.03=97.85)
            change = 0.07  # 7% gap up - should trigger TSL via gap-through
        else:
            # Sideways
            change = np.random.randn() * 0.005
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.003,
        "low": prices * 0.997,
        "close": prices,
        "volume": np.ones(n_bars) * 100000,
    }, index=dates)

    # For bar 11, make the gap clear: open >> previous close
    df.iloc[11, df.columns.get_loc("open")] = prices[10] * 1.07  # Gap open
    df.iloc[11, df.columns.get_loc("high")] = prices[10] * 1.08
    df.iloc[11, df.columns.get_loc("low")] = prices[10] * 1.06

    # Entry at bar 0
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return run_comparison(df, entries, trail_pct, "Gap-Through")


def test_tight_stop(trail_pct: float = 0.02):
    """Test with tight 2% stop - should exit sooner on reversals."""
    print(f"\n{'='*70}")
    print(f"TEST: Tight Stop (TSL={trail_pct*100:.0f}%)")
    print("="*70)

    np.random.seed(456)
    n_bars = 80

    prices = [100.0]
    for i in range(1, n_bars):
        if i < 20:
            change = -0.003  # Gradual downtrend
        elif i < 25:
            change = 0.012  # Moderate reversal - should trigger 2% TSL
        elif i < 50:
            change = -0.002  # Gradual downtrend
        elif i < 55:
            change = 0.008  # Another reversal
        else:
            change = np.random.randn() * 0.005
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.004),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.004),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
    }, index=dates)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True
    entries[30] = True  # Second entry mid-downtrend

    return run_comparison(df, entries, trail_pct, "Tight Stop")


def test_multiple_entries(trail_pct: float = 0.05):
    """Test with 5 SHORT entries in sequence."""
    print(f"\n{'='*70}")
    print(f"TEST: Multiple Entries (TSL={trail_pct*100:.0f}%)")
    print("="*70)

    np.random.seed(789)
    n_bars = 150

    prices = [100.0]
    for i in range(1, n_bars):
        phase = i // 30  # Changes every 30 bars
        if phase % 2 == 0:
            change = -0.004  # Downtrend
        else:
            change = 0.015  # Uptrend (will trigger TSL)
        change += np.random.randn() * 0.003
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.002),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
    }, index=dates)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Multiple entries at start of each downtrend phase
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True    # Start of downtrend 1
    entries[60] = True   # Start of downtrend 2
    entries[120] = True  # Start of downtrend 3

    return run_comparison(df, entries, trail_pct, "Multiple Entries")


def run_comparison(prices_df: pd.DataFrame, entries: np.ndarray,
                   trail_pct: float, test_name: str) -> bool:
    """Run VBT Pro vs ml4t comparison and return True if matches."""
    import vectorbtpro as vbt
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        OrderSide,
        Strategy,
        WaterMarkSource,
        StopFillMode,
        InitialHwmSource,
    )
    from ml4t.backtest.config import TrailStopTiming
    from ml4t.backtest.risk.position import TrailingStop

    # Run VBT Pro
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=np.zeros_like(entries),
        direction="shortonly",
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        tsl_stop=trail_pct,
        accumulate=False,
        freq="D",
    )

    vbt_trades = pf.trades.records_readable
    vbt_trade_list = []
    for idx, t in vbt_trades.iterrows():
        entry_val = t["Entry Index"]
        if isinstance(entry_val, (pd.Timestamp, np.datetime64)):
            entry_idx = prices_df.index.get_loc(entry_val)
        else:
            entry_idx = int(entry_val)

        exit_val = t.get("Exit Index")
        if pd.notna(exit_val):
            if isinstance(exit_val, (pd.Timestamp, np.datetime64)):
                exit_idx = prices_df.index.get_loc(exit_val)
            else:
                exit_idx = int(exit_val)
        else:
            exit_idx = None

        # Get status - VBT Pro has "Closed" or "Open"
        status = str(t.get("Status", "Unknown"))

        vbt_trade_list.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": float(t["Avg Entry Price"]),
            "exit_price": float(t["Avg Exit Price"]) if pd.notna(t.get("Avg Exit Price")) else None,
            "pnl": float(t["PnL"]) if pd.notna(t.get("PnL")) else 0,
            "status": status,  # "Closed" or "Open"
        })

    # Run ml4t
    prices_pl = pl.DataFrame({
        "timestamp": prices_df.index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "open": prices_df["open"].tolist(),
        "high": prices_df["high"].tolist(),
        "low": prices_df["low"].tolist(),
        "close": prices_df["close"].tolist(),
        "volume": prices_df["volume"].tolist(),
    })

    signals_pl = pl.DataFrame({
        "timestamp": prices_df.index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "short_entry": entries.tolist(),
    })

    class ShortTSLStrategy(Strategy):
        def on_start(self, broker):
            broker.set_position_rules(TrailingStop(pct=trail_pct))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            if signals.get("short_entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE, OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    engine = Engine(
        feed,
        ShortTSLStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=True,
        allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
        trail_hwm_source=WaterMarkSource.BAR_EXTREME,
        initial_hwm_source=InitialHwmSource.BAR_CLOSE,
        stop_fill_mode=StopFillMode.STOP_PRICE,
        trail_stop_timing=TrailStopTiming.INTRABAR,
    )

    results = engine.run()

    ml4t_trade_list = []
    ml4t_open_count = 0
    for t in results.trades:
        entry_idx = prices_df.index.get_loc(t.entry_time) if t.entry_time in prices_df.index else -1
        exit_idx = prices_df.index.get_loc(t.exit_time) if t.exit_time and t.exit_time in prices_df.index else None

        trade_data = {
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price if t.exit_price else None,
            "pnl": t.pnl,
            "status": t.status,  # "closed" or "open"
        }
        ml4t_trade_list.append(trade_data)
        if t.status == "open":
            ml4t_open_count += 1

    # Get final value from equity curve
    ml4t_final_value = results.equity_curve[-1][1] if results.equity_curve else 100000.0

    # Filter to closed trades only for comparison
    # VBT Pro: Status="Closed" vs Status="Open"
    # ml4t: status="closed" vs status="open"
    vbt_closed_trades = [t for t in vbt_trade_list if t.get('status') == 'Closed']
    vbt_open_trades = len(vbt_trade_list) - len(vbt_closed_trades)
    ml4t_closed_trades = [t for t in ml4t_trade_list if t.get('status') == 'closed']

    # Compare
    print(f"\n  VBT Pro:  {len(vbt_trade_list)} trades ({len(vbt_closed_trades)} closed, {vbt_open_trades} open)")
    print(f"  ml4t:     {len(ml4t_trade_list)} trades ({len(ml4t_closed_trades)} closed, {ml4t_open_count} open)")
    print(f"  Final values: VBT=${float(pf.value.iloc[-1]):,.2f}, ml4t=${ml4t_final_value:,.2f}")

    # Compare closed trades only
    if len(vbt_closed_trades) != len(ml4t_closed_trades):
        print(f"  ❌ CLOSED TRADE COUNT MISMATCH: VBT={len(vbt_closed_trades)}, ML4T={len(ml4t_closed_trades)}")
        return False

    # Also check open trade count matches
    if vbt_open_trades != ml4t_open_count:
        print(f"  ⚠️ OPEN TRADE COUNT MISMATCH: VBT={vbt_open_trades}, ML4T={ml4t_open_count}")

    all_match = True
    for i, (vbt_t, ml4t_t) in enumerate(zip(vbt_closed_trades, ml4t_closed_trades)):
        entry_match = vbt_t['entry_idx'] == ml4t_t['entry_idx']
        exit_match = vbt_t['exit_idx'] == ml4t_t['exit_idx']
        price_match = abs(vbt_t['exit_price'] - ml4t_t['exit_price']) < 0.01 if vbt_t['exit_price'] else True

        if not (entry_match and exit_match and price_match):
            print(f"\n  Trade {i+1} MISMATCH:")
            print(f"    VBT:  entry={vbt_t['entry_idx']}, exit={vbt_t['exit_idx']}, exit_price={vbt_t['exit_price']:.4f}")
            print(f"    ML4T: entry={ml4t_t['entry_idx']}, exit={ml4t_t['exit_idx']}, exit_price={ml4t_t['exit_price']:.4f}")
            all_match = False
        else:
            print(f"  Trade {i+1}: entry={vbt_t['entry_idx']}, exit={vbt_t['exit_idx']} ✅")

    if all_match:
        print(f"\n  ✅ {test_name} PASSED")
    else:
        print(f"\n  ❌ {test_name} FAILED")

    return all_match


def main():
    print("=" * 70)
    print("Scenario 12b: SHORT Trailing Stop STRESS TEST")
    print("=" * 70)

    results = []

    results.append(test_gap_through(trail_pct=0.03))
    results.append(test_tight_stop(trail_pct=0.02))
    results.append(test_multiple_entries(trail_pct=0.05))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    tests = ["Gap-Through (3%)", "Tight Stop (2%)", "Multiple Entries (5%)"]
    all_passed = True
    for name, passed in zip(tests, results):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        all_passed &= passed

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL STRESS TESTS PASSED")
    else:
        print("❌ SOME STRESS TESTS FAILED")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
