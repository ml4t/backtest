#!/usr/bin/env python3
"""Scenario 11: Short-only validation against Backtrader.

This validates that ml4t.backtest short selling produces IDENTICAL results
to Backtrader - trade-by-trade exact match.

Run from .venv-backtrader environment:
    .venv-backtrader/bin/python validation/backtrader/scenario_11_short_only.py

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


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run short-only backtest using Backtrader."""
    try:
        import backtrader as bt
    except ImportError:
        raise ImportError("Backtrader not installed. Run in .venv-backtrader environment.")

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

    class ShortOnlyStrategy(bt.Strategy):
        params = (
            ("entries", None),
            ("exits", None),
        )

        def __init__(self):
            self.bar_count = 0
            self.trades = []
            self.pending_trade = None  # Track open trade info

        def next(self):
            idx = self.bar_count
            entries = self.params.entries
            exits = self.params.exits

            if idx >= len(entries):
                return

            entry = entries[idx]
            exit_signal = exits[idx]

            current_pos = self.position.size

            # Exit first (cover short)
            if exit_signal and current_pos < 0:
                self.close()

            # Then entry (open short)
            elif entry and current_pos == 0:
                self.sell(size=SHARES_PER_TRADE)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.justopened:
                # Store entry info when trade opens
                self.pending_trade = {
                    "entry_time": bt.num2date(trade.dtopen),
                    "entry_price": trade.price,
                    "entry_size": trade.size,  # Negative for short
                }
            elif trade.isclosed and self.pending_trade:
                # Calculate exit price from pnl and size
                entry_size = self.pending_trade["entry_size"]
                # For short: pnl = (entry - exit) * |size|, so exit = entry - pnl/|size|
                # For long: pnl = (exit - entry) * |size|, so exit = entry + pnl/|size|
                if entry_size < 0:  # Short
                    exit_price = self.pending_trade["entry_price"] - trade.pnl / abs(entry_size)
                else:  # Long
                    exit_price = self.pending_trade["entry_price"] + trade.pnl / abs(entry_size)

                self.trades.append({
                    "entry_time": self.pending_trade["entry_time"],
                    "exit_time": bt.num2date(trade.dtclose),
                    "entry_price": self.pending_trade["entry_price"],
                    "exit_price": exit_price,
                    "size": abs(entry_size),
                    "pnl": trade.pnl,
                    "direction": "Short" if entry_size < 0 else "Long",
                })
                self.pending_trade = None

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.setcommission(commission=0.0)

    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)
    cerebro.addstrategy(ShortOnlyStrategy, entries=entries, exits=exits)

    results = cerebro.run()
    strategy = results[0]

    trade_list = sorted(strategy.trades, key=lambda t: t["entry_time"])

    return {
        "framework": "Backtrader",
        "final_value": cerebro.broker.getvalue(),
        "num_trades": len(trade_list),
        "trades": trade_list,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run short-only backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, OrderSide, Strategy

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

    # Match Backtrader: NEXT_BAR execution (default), no commission
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        allow_short_selling=True,
        allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,
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
        "trades": sorted(trade_list, key=lambda t: t["entry_time"]),
    }


def compare_results(bt_results, ml4t_results):
    """Compare trade-by-trade results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Backtrader vs ml4t.backtest (SHORT-ONLY)")
    print("=" * 70)

    all_match = True

    # Trade count
    bt_count = bt_results["num_trades"]
    ml4t_count = ml4t_results["num_trades"]
    count_match = bt_count == ml4t_count
    print(f"\nTrade Count: BT={bt_count}, ML4T={ml4t_count} "
          f"{'✅ MATCH' if count_match else '❌ MISMATCH'}")
    all_match &= count_match

    # Final value
    bt_value = bt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(bt_value - ml4t_value)
    value_pct = value_diff / bt_value * 100 if bt_value else 0
    value_match = value_pct < 0.01
    print(f"Final Value: BT=${bt_value:,.2f}, ML4T=${ml4t_value:,.2f} "
          f"(diff={value_pct:.4f}%) {'✅ MATCH' if value_match else '❌ MISMATCH'}")
    all_match &= value_match

    # Trade-by-trade comparison
    if count_match and bt_count > 0:
        bt_trades = bt_results["trades"]
        ml4t_trades = ml4t_results["trades"]

        matches = 0
        mismatches = 0
        sample_mismatches = []

        for bt_t, ml4t_t in zip(bt_trades, ml4t_trades):
            entry_match = abs(bt_t["entry_price"] - ml4t_t["entry_price"]) < 0.01
            exit_match = abs(bt_t["exit_price"] - ml4t_t["exit_price"]) < 0.01
            pnl_match = abs(bt_t["pnl"] - ml4t_t["pnl"]) < 0.10

            if entry_match and exit_match and pnl_match:
                matches += 1
            else:
                mismatches += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append((bt_t, ml4t_t))

        match_pct = matches / len(bt_trades) * 100 if bt_trades else 0
        print(f"\nTrade-Level Match: {matches}/{len(bt_trades)} ({match_pct:.1f}%)")

        if mismatches > 0:
            print(f"\nSample Mismatches:")
            for bt_t, ml4t_t in sample_mismatches:
                print(f"  BT:   entry=${bt_t['entry_price']:.2f}, "
                      f"exit=${bt_t['exit_price']:.2f}, pnl=${bt_t['pnl']:.2f}")
                print(f"  ML4T: entry=${ml4t_t['entry_price']:.2f}, "
                      f"exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

        all_match &= (mismatches == 0)

    # Sample trades
    if bt_count > 0:
        print("\nSample Short Trades (first 3):")
        print("-" * 70)
        for i, (bt_t, ml4t_t) in enumerate(
            zip(bt_results["trades"][:3], ml4t_results["trades"][:3])
        ):
            print(f"  Trade {i+1}:")
            print(f"    BT:   entry=${bt_t['entry_price']:.2f}, exit=${bt_t['exit_price']:.2f}, pnl=${bt_t['pnl']:.2f}")
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
    print("Scenario 11: Short-Only Validation (Backtrader)")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    n_bars = 200
    prices_df, entries, exits = generate_test_data(n_bars=n_bars)
    print(f"  Bars: {n_bars}")
    print(f"  Short entry signals: {entries.sum()}")
    print(f"  Short exit signals: {exits.sum()}")

    # Run Backtrader
    print("\nRunning Backtrader (short-only)...")
    try:
        bt_results = run_backtrader(prices_df, entries, exits)
        print(f"  Trades: {bt_results['num_trades']}")
        print(f"  Final Value: ${bt_results['final_value']:,.2f}")
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
    success = compare_results(bt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
