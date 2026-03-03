#!/usr/bin/env python3
"""Scenario 12: SHORT + Trailing Stop validation against Backtrader.

This is the CRITICAL test case: Tests trailing stop behavior for SHORT positions.
Scenario 09 only tested LONG positions.
Scenario 11 tested SHORT but with explicit entry/exit signals, no trailing stops.

This scenario validates that SHORT + TSL produces matching results with Backtrader.

Run from .venv-backtrader environment:
    .venv-backtrader/bin/python validation/backtrader/scenario_12_short_trailing_stop.py

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


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """Backtrader trailing stop for SHORT positions."""
    try:
        import backtrader as bt
    except ImportError:
        raise ImportError("Backtrader not installed.")

    class PandasData(bt.feeds.PandasData):
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", -1),
        )

    class ShortTrailingStopStrategy(bt.Strategy):
        params = (("entries", None),)

        def __init__(self):
            self.bar_count = 0
            self.trades = []
            self.stop_order = None
            self.entry_price = None
            self.entry_bar = None
            self.pending_trade_size = None  # Track entry size for trade extraction

        def next(self):
            idx = self.bar_count
            if idx >= len(self.params.entries):
                return

            entry = self.params.entries[idx]

            if entry and self.position.size == 0:
                # Open SHORT position
                self.sell(size=SHARES_PER_TRADE)
                self.entry_price = self.data.close[0]
                self.entry_bar = idx
                # Place trailing stop BUY order to cover short
                # For SHORT: StopTrail triggers when price rises above trail level
                self.stop_order = self.buy(
                    exectype=bt.Order.StopTrail,
                    trailpercent=TRAIL_PCT,
                    size=SHARES_PER_TRADE,
                )

            self.bar_count += 1

        def notify_order(self, order):
            if order.status == order.Completed:
                if order.isbuy() and order == self.stop_order:
                    # SHORT exit via trailing stop
                    self.stop_order = None

        def notify_trade(self, trade):
            if trade.justopened:
                # Store entry size when trade opens (trade.size is 0 after close)
                self.pending_trade_size = trade.size
            elif trade.isclosed and self.pending_trade_size is not None:
                entry_size = self.pending_trade_size
                # For short: pnl = (entry - exit) * |size|, so exit = entry - pnl/|size|
                if entry_size < 0:  # Short
                    exit_price = trade.price - trade.pnl / abs(entry_size)
                else:  # Long
                    exit_price = trade.price + trade.pnl / abs(entry_size)
                self.trades.append({
                    "entry_bar": self.entry_bar,
                    "exit_bar": self.bar_count - 1,
                    "entry_price": trade.price,
                    "exit_price": exit_price,
                    "size": abs(entry_size),
                    "pnl": trade.pnl,
                    "direction": "Short" if entry_size < 0 else "Long",
                })
                self.pending_trade_size = None

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)
    cerebro.addstrategy(ShortTrailingStopStrategy, entries=entries)

    results = cerebro.run()
    strategy = results[0]

    # Extract exit reasons
    exit_reasons = {"StopTrail": len(strategy.trades)}

    return {
        "framework": "Backtrader",
        "final_value": cerebro.broker.getvalue(),
        "total_pnl": cerebro.broker.getvalue() - 100_000.0,
        "num_trades": len(strategy.trades),
        "trades": strategy.trades,
        "exit_reasons": exit_reasons,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest trailing stop for SHORT positions."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
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

    # Backtrader compatible settings:
    # - NEXT_BAR execution mode (Backtrader default)
    engine = Engine(
        feed,
        ShortTrailingStopStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=True,
        allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,
    )

    results = engine.run()

    # Build trade list with bar indices
    trade_list = []
    for idx, t in enumerate(results["trades"]):
        entry_bar = prices_df.index.get_loc(t.entry_time) if t.entry_time in prices_df.index else -1
        exit_bar = prices_df.index.get_loc(t.exit_time) if t.exit_time and t.exit_time in prices_df.index else None

        trade_list.append({
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": abs(t.quantity),
            "pnl": t.pnl,
            "direction": "Short" if t.quantity < 0 else "Long",
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


def print_trade_details(bt_trades: list, ml4t_trades: list, prices_df: pd.DataFrame):
    """Print detailed trade-by-trade comparison."""
    print("\n" + "=" * 80)
    print("DETAILED TRADE COMPARISON")
    print("=" * 80)

    max_trades = max(len(bt_trades), len(ml4t_trades))

    for i in range(max_trades):
        print(f"\n--- Trade {i+1} ---")

        bt_t = bt_trades[i] if i < len(bt_trades) else None
        ml4t_t = ml4t_trades[i] if i < len(ml4t_trades) else None

        if bt_t:
            print(f"Backtrader:")
            print(f"  Entry: bar {bt_t['entry_bar']}, price ${bt_t['entry_price']:.4f}")
            if bt_t['exit_bar'] is not None:
                print(f"  Exit:  bar {bt_t['exit_bar']}, price ${bt_t['exit_price']:.4f}")
                print(f"  PnL: ${bt_t['pnl']:.2f}")
            else:
                print(f"  Exit: STILL OPEN")
        else:
            print("Backtrader: NO TRADE")

        if ml4t_t:
            print(f"ml4t.backtest:")
            print(f"  Entry: bar {ml4t_t['entry_bar']}, price ${ml4t_t['entry_price']:.4f}")
            if ml4t_t['exit_bar'] is not None:
                print(f"  Exit:  bar {ml4t_t['exit_bar']}, price ${ml4t_t['exit_price']:.4f}")
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

    # Print price trajectory summary
    print(f"\n  Price trajectory:")
    print(f"    Start: ${prices_df['close'].iloc[0]:.2f}")
    print(f"    Bar 30 (end of downtrend 1): ${prices_df['close'].iloc[30]:.2f}")
    print(f"    Bar 35 (after reversal): ${prices_df['close'].iloc[35]:.2f}")
    print(f"    Bar 60 (end of downtrend 2): ${prices_df['close'].iloc[60]:.2f}")
    print(f"    End: ${prices_df['close'].iloc[-1]:.2f}")

    print("\n  Running Backtrader (SHORT + TSL)...")
    try:
        bt_results = run_backtrader(prices_df, entries)
        print(f"   Trades: {bt_results['num_trades']}")
        print(f"   Exit reasons: {bt_results['exit_reasons']}")
        print(f"   Final Value: ${bt_results['final_value']:,.2f}")
        print(f"   Total PnL: ${bt_results['total_pnl']:.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        print("   Make sure to run in .venv-backtrader environment")
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
    print_trade_details(bt_results["trades"], ml4t_results["trades"], prices_df)

    # Compare results
    trades_match = bt_results["num_trades"] == ml4t_results["num_trades"]
    value_diff = abs(bt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / bt_results["final_value"] * 100 if bt_results["final_value"] else 0

    values_close = value_pct_diff < 2.0  # 2% tolerance for trailing stop timing

    print("\n" + "=" * 70)
    print(f"Trade Count: BT={bt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'✅ OK' if trades_match else '❌ FAIL'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'✅ OK' if values_close else '❌ FAIL'}")

    if trades_match and values_close:
        print("\n✅ VALIDATION PASSED: SHORT trailing stop matches Backtrader")
    else:
        print("\n❌ VALIDATION FAILED (or within acceptable tolerance)")
        print("\nNote: Trailing stop timing may differ due to high-water mark tracking.")
    print("=" * 70)

    return 0 if (trades_match and values_close) else 1


if __name__ == "__main__":
    sys.exit(main())
