#!/usr/bin/env python3
"""Scenario 12: SHORT + Trailing Stop validation against VectorBT Pro.

This is the CRITICAL test case: Tests trailing stop behavior for SHORT positions.
Scenario 09 only tested LONG positions (allow_short_selling=False).
Scenario 11 tested SHORT but with explicit entry/exit signals, no trailing stops.

This scenario validates that SHORT + TSL produces IDENTICAL results to VBT Pro.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_12_short_trailing_stop.py

Success criteria:
- Trade count: EXACT match
- Exit bars: EXACT match (within 1 bar tolerance for edge cases)
- Exit prices: Match within 0.5% (for gap-through scenarios)
- PnL per trade: Match within 1%
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
SHARES_PER_TRADE = 100


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate downtrending data that will trigger SHORT trailing stops.

    For SHORT positions:
    - Downtrend = profit (price falls)
    - Reversal (price rises) = trailing stop triggers

    The data generates:
    1. Initial downtrend (SHORT profits)
    2. Sharp reversal upward (triggers TSL)
    3. Another downtrend
    4. Another reversal
    5. Sideways
    """
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            # Initial DOWNtrend (-0.5% per bar on average) - SHORT profits
            change = np.random.randn() * 0.01 - 0.005
        elif i < 35:
            # Sharp reversal UPWARD (+2% per bar) - triggers TSL
            change = 0.02 + np.random.randn() * 0.005
        elif i < 60:
            # Another DOWNtrend - SHORT profits again
            change = np.random.randn() * 0.01 - 0.003
        elif i < 65:
            # Another reversal UPWARD - triggers TSL
            change = 0.015 + np.random.randn() * 0.005
        else:
            # Sideways with slight downward bias
            change = np.random.randn() * 0.01 - 0.001
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    # Ensure high is actually highest and low is actually lowest
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # SHORT entry signals at start of downtrends
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True   # First entry at start (downtrend begins)
    entries[40] = True  # Second entry in middle of second downtrend

    return df, entries


def run_vectorbt_pro(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VectorBT Pro trailing stop for SHORT positions."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed.")

    # VBT Pro from_signals with trailing stop for SHORT
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=np.zeros_like(entries),  # No manual exits, only trailing stop
        direction="shortonly",  # SHORT positions
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        tsl_stop=TRAIL_PCT,  # Trailing stop loss threshold
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    # Build detailed trade list
    trade_list = []
    for idx, t in trades.iterrows():
        # Entry Index might be a timestamp or an integer depending on VBT Pro version
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

        trade_list.append({
            "trade_idx": idx,
            "entry_idx": entry_idx,
            "entry_time": str(entry_val),
            "exit_idx": exit_idx,
            "exit_time": str(exit_val) if pd.notna(exit_val) else "",
            "entry_price": float(t["Avg Entry Price"]),
            "exit_price": float(t["Avg Exit Price"]) if pd.notna(t.get("Avg Exit Price")) else None,
            "size": float(t["Size"]),
            "pnl": float(t["PnL"]) if pd.notna(t.get("PnL")) else 0,
            "direction": str(t.get("Direction", "Short")),
            "status": str(t.get("Status", "Unknown")),
        })

    exit_reasons = {}
    if len(trades) > 0:
        status_cols = ["Status", "status", "Exit Type", "exit_type"]
        for col in status_cols:
            if col in trades.columns:
                exit_reasons = trades[col].value_counts().to_dict()
                break
        if not exit_reasons:
            exit_reasons = {"completed": len(trades)}

    return {
        "framework": "VectorBT Pro",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "trades": trade_list,
        "exit_reasons": exit_reasons,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest trailing stop for SHORT positions."""
    import polars as pl

    from ml4t.backtest import (
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
        }
    )

    class ShortTrailingStopStrategy(Strategy):
        def on_start(self, broker):
            # Set position rules at strategy start
            broker.set_position_rules(TrailingStop(pct=TRAIL_PCT))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            # Only enter SHORT if no position and entry signal
            if signals.get("short_entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE, OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)

    # VBT Pro compatible settings for SHORT trailing stop:
    # - trail_hwm_source: HIGH means we use LOW for LWM (the code handles direction)
    # - initial_hwm_source: BAR_CLOSE (VBT Pro uses bar close, not fill price)
    # - trail_stop_timing: INTRABAR (VBT Pro updates water marks before checking)
    # - stop_fill_mode: STOP_PRICE (VBT Pro fills at trail level)
    engine = Engine(
        feed,
        ShortTrailingStopStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=True,
        allow_leverage=True,  # Need leverage for shorting
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
        trail_hwm_source=WaterMarkSource.BAR_EXTREME,  # Uses HIGH for HWM, LOW for LWM
        initial_hwm_source=InitialHwmSource.BAR_CLOSE,
        stop_fill_mode=StopFillMode.STOP_PRICE,
        trail_stop_timing=TrailStopTiming.INTRABAR,  # VBT Pro live updates
    )

    results = engine.run()

    # Build detailed trade list
    trade_list = []
    for idx, t in enumerate(results["trades"]):
        # Get exit bar index from timestamps
        entry_idx = prices_df.index.get_loc(t.entry_time) if t.entry_time in prices_df.index else -1
        exit_idx = prices_df.index.get_loc(t.exit_time) if t.exit_time and t.exit_time in prices_df.index else None

        trade_list.append({
            "trade_idx": idx,
            "entry_idx": entry_idx,
            "entry_time": str(t.entry_time),
            "exit_idx": exit_idx,
            "exit_time": str(t.exit_time) if t.exit_time else None,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price if t.exit_price else None,
            "size": abs(t.quantity),
            "pnl": t.pnl,
            "direction": "Short" if t.quantity < 0 else "Long",
            "status": "Closed" if t.exit_time else "Open",
        })

    # Count exit reasons
    exit_reasons = {}
    for fill in results["fills"]:
        if fill.quantity > 0:  # Exit fill for SHORT (buying back)
            reason = getattr(fill, "reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_list,
        "exit_reasons": exit_reasons,
    }


def print_trade_details(vbt_trades: list, ml4t_trades: list, prices_df: pd.DataFrame):
    """Print detailed trade-by-trade comparison."""
    print("\n" + "=" * 80)
    print("DETAILED TRADE COMPARISON")
    print("=" * 80)

    max_trades = max(len(vbt_trades), len(ml4t_trades))

    for i in range(max_trades):
        print(f"\n--- Trade {i+1} ---")

        vbt_t = vbt_trades[i] if i < len(vbt_trades) else None
        ml4t_t = ml4t_trades[i] if i < len(ml4t_trades) else None

        if vbt_t:
            print(f"VBT Pro:")
            print(f"  Entry: bar {vbt_t['entry_idx']}, price ${vbt_t['entry_price']:.4f}")
            if vbt_t['exit_idx'] is not None:
                print(f"  Exit:  bar {vbt_t['exit_idx']}, price ${vbt_t['exit_price']:.4f}")
                print(f"  PnL: ${vbt_t['pnl']:.2f}")
            else:
                print(f"  Exit: STILL OPEN")
        else:
            print("VBT Pro: NO TRADE")

        if ml4t_t:
            print(f"ml4t.backtest:")
            print(f"  Entry: bar {ml4t_t['entry_idx']}, price ${ml4t_t['entry_price']:.4f}")
            if ml4t_t['exit_idx'] is not None:
                print(f"  Exit:  bar {ml4t_t['exit_idx']}, price ${ml4t_t['exit_price']:.4f}")
                print(f"  PnL: ${ml4t_t['pnl']:.2f}")
            else:
                print(f"  Exit: STILL OPEN")
        else:
            print("ml4t.backtest: NO TRADE")

        # Compare if both have trades
        if vbt_t and ml4t_t:
            entry_match = abs(vbt_t['entry_price'] - ml4t_t['entry_price']) < 0.01

            if vbt_t['exit_idx'] is not None and ml4t_t['exit_idx'] is not None:
                exit_bar_match = vbt_t['exit_idx'] == ml4t_t['exit_idx']
                exit_price_diff = abs(vbt_t['exit_price'] - ml4t_t['exit_price'])
                exit_price_match = exit_price_diff < vbt_t['exit_price'] * 0.005  # 0.5%
                pnl_diff = abs(vbt_t['pnl'] - ml4t_t['pnl'])
                pnl_match = pnl_diff < abs(vbt_t['pnl']) * 0.01  # 1%

                status = "✅ MATCH" if (entry_match and exit_bar_match and exit_price_match) else "❌ MISMATCH"
                print(f"\n  Comparison: {status}")
                if not exit_bar_match:
                    print(f"    Exit bar: VBT={vbt_t['exit_idx']}, ML4T={ml4t_t['exit_idx']}")
                if not exit_price_match:
                    print(f"    Exit price diff: ${exit_price_diff:.4f}")
                if not pnl_match:
                    print(f"    PnL diff: ${pnl_diff:.2f}")


def debug_trade_bar_by_bar(prices_df: pd.DataFrame, trade_idx: int, entry_bar: int,
                           vbt_exit_bar: int, ml4t_exit_bar: int, entry_price: float):
    """Print bar-by-bar state for a divergent trade."""
    print(f"\n{'='*80}")
    print(f"BAR-BY-BAR DEBUG: Trade {trade_idx+1}")
    print(f"Entry at bar {entry_bar}, price ${entry_price:.4f}")
    print(f"VBT exit: bar {vbt_exit_bar}, ml4t exit: bar {ml4t_exit_bar}")
    print(f"{'='*80}")

    # Track LWM (low water mark) for SHORT
    lwm = entry_price  # Initial LWM is entry price

    start = max(0, entry_bar)
    end = min(len(prices_df), max(vbt_exit_bar or 0, ml4t_exit_bar or 0) + 3)

    print(f"\n{'Bar':<5} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'LWM':>10} {'TSL':>10} {'Trigger?':>10}")
    print("-" * 85)

    for bar in range(start, end):
        row = prices_df.iloc[bar]

        # For SHORT: TSL = LWM * (1 + pct)
        tsl = lwm * (1 + TRAIL_PCT)

        # Check trigger: bar_high >= tsl
        triggered = row["high"] >= tsl
        trigger_mark = "TSL HIT" if triggered else ""

        is_entry = "ENTRY" if bar == entry_bar else ""
        is_vbt_exit = "VBT EXIT" if bar == vbt_exit_bar else ""
        is_ml4t_exit = "ML4T EXIT" if bar == ml4t_exit_bar else ""
        note = " | ".join(filter(None, [is_entry, is_vbt_exit, is_ml4t_exit, trigger_mark]))

        print(f"{bar:<5} {row['open']:>10.4f} {row['high']:>10.4f} {row['low']:>10.4f} "
              f"{row['close']:>10.4f} {lwm:>10.4f} {tsl:>10.4f} {note}")

        # Update LWM for next bar (using bar LOW for SHORT, INTRABAR mode)
        # INTRABAR mode: update LWM = min(lwm, bar_low) BEFORE check
        # But we're showing state AFTER the bar
        if bar >= entry_bar and not triggered:
            lwm = min(lwm, row["low"])


def main():
    print("=" * 70)
    print(f"Scenario 12: SHORT Trailing Stop ({TRAIL_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, SHORT Entry signals: {entries.sum()}")

    # Print price trajectory summary
    print(f"\n  Price trajectory:")
    print(f"    Start: ${prices_df['close'].iloc[0]:.2f}")
    print(f"    Bar 30 (end of downtrend 1): ${prices_df['close'].iloc[30]:.2f}")
    print(f"    Bar 35 (after reversal): ${prices_df['close'].iloc[35]:.2f}")
    print(f"    Bar 60 (end of downtrend 2): ${prices_df['close'].iloc[60]:.2f}")
    print(f"    End: ${prices_df['close'].iloc[-1]:.2f}")

    print("\n  Running VectorBT Pro (SHORT + TSL)...")
    try:
        vbt_results = run_vectorbt_pro(prices_df, entries)
        print(f"   Trades: {vbt_results['num_trades']}")
        print(f"   Exit reasons: {vbt_results['exit_reasons']}")
        print(f"   Final Value: ${vbt_results['final_value']:,.2f}")
        print(f"   Total PnL: ${vbt_results['total_pnl']:.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        print("   Make sure to run in .venv-vectorbt-pro environment")
        return 1
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n  Running ml4t.backtest (SHORT + TSL)...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Exit reasons: {ml4t_results['exit_reasons']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
        print(f"   Total PnL: ${ml4t_results['total_pnl']:.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Detailed trade comparison
    print_trade_details(vbt_results["trades"], ml4t_results["trades"], prices_df)

    # Compare results
    trades_match = vbt_results["num_trades"] == ml4t_results["num_trades"]
    value_diff = abs(vbt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / vbt_results["final_value"] * 100

    # Check trade-by-trade
    trade_mismatches = []
    if trades_match:
        for i, (vbt_t, ml4t_t) in enumerate(zip(vbt_results["trades"], ml4t_results["trades"])):
            if vbt_t['exit_idx'] is not None and ml4t_t['exit_idx'] is not None:
                if vbt_t['exit_idx'] != ml4t_t['exit_idx']:
                    trade_mismatches.append((i, vbt_t, ml4t_t))

    # Debug first mismatch
    if trade_mismatches:
        idx, vbt_t, ml4t_t = trade_mismatches[0]
        debug_trade_bar_by_bar(
            prices_df, idx,
            vbt_t['entry_idx'], vbt_t['exit_idx'], ml4t_t['exit_idx'],
            vbt_t['entry_price']
        )

    values_close = value_pct_diff < 1.0  # 1% tolerance
    no_exit_mismatches = len(trade_mismatches) == 0

    print("\n" + "=" * 70)
    print(f"Trade Count: VBT={vbt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'✅ OK' if trades_match else '❌ FAIL'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'✅ OK' if values_close else '❌ FAIL'}")
    print(f"Exit bar mismatches: {len(trade_mismatches)} {'✅ OK' if no_exit_mismatches else '❌ FAIL'}")

    if trades_match and values_close and no_exit_mismatches:
        print("\n✅ VALIDATION PASSED: SHORT trailing stop matches VBT Pro")
    else:
        print("\n❌ VALIDATION FAILED: SHORT trailing stop differs from VBT Pro")
        if trade_mismatches:
            print(f"\n  First mismatch at trade {trade_mismatches[0][0]+1}:")
            print(f"    VBT exit bar: {trade_mismatches[0][1]['exit_idx']}")
            print(f"    ML4T exit bar: {trade_mismatches[0][2]['exit_idx']}")
    print("=" * 70)

    return 0 if (trades_match and values_close and no_exit_mismatches) else 1


if __name__ == "__main__":
    sys.exit(main())
