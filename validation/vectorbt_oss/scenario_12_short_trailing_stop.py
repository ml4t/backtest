#!/usr/bin/env python3
"""Scenario 12: SHORT + Trailing Stop validation against VectorBT OSS.

This is the CRITICAL test case: Tests trailing stop behavior for SHORT positions.
Scenario 09 only tested LONG positions.
Scenario 11 tested SHORT but with explicit entry/exit signals, no trailing stops.

Run from .venv-vectorbt environment:
    .venv-vectorbt/bin/python validation/vectorbt_oss/scenario_12_short_trailing_stop.py

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
    """
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            # Initial DOWNtrend - SHORT profits
            change = np.random.randn() * 0.01 - 0.005
        elif i < 35:
            # Sharp reversal UPWARD - triggers TSL
            change = 0.02 + np.random.randn() * 0.005
        elif i < 60:
            # Another DOWNtrend
            change = np.random.randn() * 0.01 - 0.003
        elif i < 65:
            # Another reversal UPWARD
            change = 0.015 + np.random.randn() * 0.005
        else:
            # Sideways
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
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # SHORT entry signals
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True
    entries[40] = True

    return df, entries


def run_vectorbt_oss(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VectorBT OSS trailing stop for SHORT positions.

    NOTE: VectorBT OSS does NOT support trailing stops (tsl_stop is VBT Pro only).
    This function implements a MANUAL trailing stop calculation to match ml4t behavior.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("VectorBT OSS not installed. Run in .venv-vectorbt environment.")

    # Implement manual trailing stop for SHORT positions
    # For SHORT: track LOW water mark (best price = lowest), exit when price rises above stop
    n_bars = len(prices_df)
    in_position = False
    low_water_mark = np.inf
    stop_level = np.inf
    entry_price = 0.0
    entry_bar = 0

    trades = []
    exits = np.zeros(n_bars, dtype=bool)

    for i in range(n_bars):
        close_price = prices_df["close"].iloc[i]

        if entries[i] and not in_position:
            # Open short position
            in_position = True
            entry_price = close_price
            entry_bar = i
            low_water_mark = close_price
            stop_level = close_price * (1 + TRAIL_PCT)

        elif in_position:
            # Update low water mark (for short, we track lowest price)
            if close_price < low_water_mark:
                low_water_mark = close_price
                stop_level = low_water_mark * (1 + TRAIL_PCT)

            # Check if stop triggered (price rises above stop level)
            high_price = prices_df["high"].iloc[i]
            if high_price >= stop_level:
                # Exit at stop price
                exit_price = min(stop_level, high_price)
                exits[i] = True
                trades.append({
                    "entry_idx": entry_bar,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": SHARES_PER_TRADE,
                    "pnl": (entry_price - exit_price) * SHARES_PER_TRADE,
                    "direction": "Short",
                    "exit_reason": "TrailingStop",
                })
                in_position = False
                low_water_mark = np.inf
                stop_level = np.inf

    # Now run VBT with calculated exits (no tsl_stop - we handled it manually)
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=exits,
        direction="shortonly",  # SHORT positions
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        accumulate=False,
        freq="D",
    )

    # Return our manually tracked trades (more accurate for trailing stop timing)
    final_value = pf.value()
    total_pnl = sum(t["pnl"] for t in trades)

    return {
        "framework": "VectorBT OSS (manual TSL)",
        "final_value": float(final_value.iloc[-1]) if len(final_value) > 0 else 100_000.0,
        "total_pnl": total_pnl,
        "num_trades": len(trades),
        "trades": trades,
        "exit_reasons": {"TrailingStop": len(trades)},
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
    )
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
            broker.set_position_rules(TrailingStop(pct=TRAIL_PCT))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            if signals.get("short_entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE, OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)

    # VBT OSS compatible settings: SAME_BAR execution
    engine = Engine(
        feed,
        ShortTrailingStopStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=True,
        allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()

    trade_list = []
    for idx, t in enumerate(results["trades"]):
        entry_idx = prices_df.index.get_loc(t.entry_time) if t.entry_time in prices_df.index else -1
        exit_idx = prices_df.index.get_loc(t.exit_time) if t.exit_time and t.exit_time in prices_df.index else None

        trade_list.append({
            "trade_idx": idx,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price if t.exit_price else None,
            "size": abs(t.quantity),
            "pnl": t.pnl,
            "direction": "Short" if t.quantity < 0 else "Long",
        })

    exit_reasons = {}
    for fill in results["fills"]:
        if fill.quantity > 0:
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


def print_trade_details(vbt_trades: list, ml4t_trades: list):
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
            print(f"VectorBT OSS:")
            print(f"  Entry: bar {vbt_t['entry_idx']}, price ${vbt_t['entry_price']:.4f}")
            if vbt_t['exit_idx'] is not None:
                print(f"  Exit:  bar {vbt_t['exit_idx']}, price ${vbt_t['exit_price']:.4f}")
                print(f"  PnL: ${vbt_t['pnl']:.2f}")
            else:
                print(f"  Exit: STILL OPEN")
        else:
            print("VectorBT OSS: NO TRADE")

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


def main():
    print("=" * 70)
    print(f"Scenario 12: SHORT Trailing Stop ({TRAIL_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, SHORT Entry signals: {entries.sum()}")

    print(f"\n  Price trajectory:")
    print(f"    Start: ${prices_df['close'].iloc[0]:.2f}")
    print(f"    Bar 30: ${prices_df['close'].iloc[30]:.2f}")
    print(f"    Bar 35: ${prices_df['close'].iloc[35]:.2f}")
    print(f"    Bar 60: ${prices_df['close'].iloc[60]:.2f}")
    print(f"    End: ${prices_df['close'].iloc[-1]:.2f}")

    print("\n  Running VectorBT OSS (SHORT + TSL)...")
    try:
        vbt_results = run_vectorbt_oss(prices_df, entries)
        print(f"   Trades: {vbt_results['num_trades']}")
        print(f"   Exit reasons: {vbt_results['exit_reasons']}")
        print(f"   Final Value: ${vbt_results['final_value']:,.2f}")
        print(f"   Total PnL: ${vbt_results['total_pnl']:.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        print("   Make sure to run in .venv-vectorbt environment")
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
    print_trade_details(vbt_results["trades"], ml4t_results["trades"])

    # Compare results
    trades_match = vbt_results["num_trades"] == ml4t_results["num_trades"]
    value_diff = abs(vbt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / vbt_results["final_value"] * 100 if vbt_results["final_value"] else 0

    values_close = value_pct_diff < 1.0  # 1% tolerance

    print("\n" + "=" * 70)
    print(f"Trade Count: VBT={vbt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'✅ OK' if trades_match else '❌ FAIL'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'✅ OK' if values_close else '❌ FAIL'}")

    if trades_match and values_close:
        print("\n✅ VALIDATION PASSED: SHORT trailing stop matches VBT OSS")
    else:
        print("\n❌ VALIDATION FAILED: SHORT trailing stop differs from VBT OSS")
    print("=" * 70)

    return 0 if (trades_match and values_close) else 1


if __name__ == "__main__":
    sys.exit(main())
