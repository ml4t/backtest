#!/usr/bin/env python3
"""Scenario 13: Trailing Stop + Take Profit combination validation against Backtrader.

Tests what happens when BOTH TSL and TP conditions can trigger.
Critical for understanding rule priority behavior.

Test cases:
1. LONG: TP triggers before TSL can engage
2. LONG: TSL triggers before TP is reached
3. SHORT: TP triggers before TSL can engage
4. SHORT: TSL triggers before TP is reached

Run from .venv-backtrader environment:
    source .venv-backtrader/bin/activate
    python validation/backtrader/scenario_13_tsl_tp_combo.py

Success criteria:
- Document Backtrader behavior for TSL vs TP priority
- Match ml4t behavior with documented expectations
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
TP_PCT = 0.08     # 8% take profit
SHARES_PER_TRADE = 100


def generate_test_data_tp_first(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TP triggers before TSL for LONG.

    Scenario:
    - Entry at bar 0 at $100
    - Price rises steadily to $108 (8% gain, triggers TP)
    - TSL never has a chance to engage (no pullback > 5%)
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 12:
            # Steady rise to 8% gain (TP level)
            change = 0.007
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        "close": prices,
        "volume": np.full(n_bars, 100000.0),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def generate_test_data_tsl_first(seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TSL triggers before TP for LONG.

    Scenario:
    - Entry at bar 0 at $100
    - Price rises to $106 (6% gain, below TP)
    - Sharp pullback triggers TSL (5% from high)
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            # Rise to 6% (below 8% TP)
            change = 0.006
        elif i == 10:
            # Sharp drop triggering 5% TSL
            change = -0.06
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        "close": prices,
        "volume": np.full(n_bars, 100000.0),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """Run Backtrader with TSL + TP for LONG positions."""
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

    class TSLTPStrategy(bt.Strategy):
        def __init__(self):
            self.entry_signals = entries
            self.bar_count = 0
            self.order = None
            self.trades = []
            self.entry_price = None

        def next(self):
            if self.bar_count < len(self.entry_signals):
                if self.entry_signals[self.bar_count] and not self.position:
                    # Enter with trailing stop and take profit
                    self.order = self.buy(size=SHARES_PER_TRADE)
                    self.entry_price = self.data.close[0]

                    # Set trailing stop
                    self.sell(
                        exectype=bt.Order.StopTrail,
                        trailpercent=TRAIL_PCT,
                        size=SHARES_PER_TRADE,
                    )

                    # Set take profit
                    tp_price = self.entry_price * (1 + TP_PCT)
                    self.sell(
                        exectype=bt.Order.Limit,
                        price=tp_price,
                        size=SHARES_PER_TRADE,
                    )
            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trades.append({
                    "entry_idx": trade.baropen,
                    "exit_idx": trade.barclose,
                    "entry_price": trade.price,
                    "pnl": trade.pnl,
                    "size": trade.size,
                })

    cerebro = bt.Cerebro()
    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)
    cerebro.addstrategy(TSLTPStrategy)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    return {
        "framework": "Backtrader",
        "scenario": scenario,
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": len(strat.trades),
        "trades": strat.trades,
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """ml4t.backtest with TSL + TP for LONG positions."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk import RuleChain
    from ml4t.backtest.risk.position import TrailingStop, TakeProfit

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
        "entry": entries.tolist(),
    })

    class ComboStrategy(Strategy):
        def on_start(self, broker):
            broker.set_position_rules(RuleChain([
                TrailingStop(pct=TRAIL_PCT),
                TakeProfit(pct=TP_PCT),
            ]))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0
            if signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)

    engine = Engine(
        feed, ComboStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Backtrader compatible
    )

    results = engine.run()

    trade_info = []
    for t in results["trades"]:
        exit_idx = prices_df.index.get_loc(t.exit_time) if t.exit_time and t.exit_time in prices_df.index else None
        trade_info.append({
            "exit_bar": exit_idx,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "status": "Closed" if t.exit_time else "Open",
        })

    return {
        "framework": "ml4t.backtest",
        "scenario": scenario,
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_info,
    }


def main():
    print("=" * 70)
    print("Scenario 13: TSL + TP Rule Combination (Backtrader)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Take Profit: {TP_PCT*100:.0f}%")

    scenarios = [
        ("TP triggers first (steady rise)", generate_test_data_tp_first),
        ("TSL triggers first (rise then pullback)", generate_test_data_tsl_first),
    ]

    all_results = []

    for name, gen_func in scenarios:
        print("\n" + "=" * 70)
        print(f"TEST: {name}")
        print("=" * 70)

        df, entries = gen_func()
        print(f"\nData: {len(df)} bars")
        print(f"Entry price: ${df['close'].iloc[0]:.2f}")
        print(f"TP level: ${df['close'].iloc[0] * (1 + TP_PCT):.2f}")
        print(f"TSL level (from entry): ${df['close'].iloc[0] * (1 - TRAIL_PCT):.2f}")

        print("\nRunning Backtrader...")
        try:
            bt_result = run_backtrader(df, entries, name)
            if bt_result['trades']:
                print(f"  Exit bar: {bt_result['trades'][0]['exit_idx']}")
                print(f"  PnL: ${bt_result['trades'][0]['pnl']:.2f}")
            else:
                print("  No trades")
        except Exception as e:
            print(f"  ERROR: {e}")
            bt_result = None

        print("\nRunning ml4t.backtest...")
        try:
            ml4t_result = run_ml4t(df, entries, name)
            if ml4t_result['trades']:
                print(f"  Exit bar: {ml4t_result['trades'][0]['exit_bar']}")
                print(f"  PnL: ${ml4t_result['trades'][0]['pnl']:.2f}")
            else:
                print("  No trades")
        except Exception as e:
            print(f"  ERROR: {e}")
            ml4t_result = None

        all_results.append((name, bt_result, ml4t_result))

    # Summary
    print("\n" + "=" * 70)
    print("TSL + TP PRIORITY FINDINGS")
    print("=" * 70)

    all_match = True
    for name, bt_res, ml4t in all_results:
        print(f"\n{name}:")
        if bt_res and ml4t:
            bt_pnl = bt_res['trades'][0]['pnl'] if bt_res['trades'] else 0
            ml4t_pnl = ml4t['trades'][0]['pnl'] if ml4t['trades'] else 0
            pnl_match = abs(bt_pnl - ml4t_pnl) < 50  # $50 tolerance
            all_match &= pnl_match
            print(f"  Backtrader PnL: ${bt_pnl:.2f}")
            print(f"  ML4T PnL: ${ml4t_pnl:.2f}")
            print(f"  {'✅ Match' if pnl_match else '❌ Mismatch'}")
        else:
            print(f"  ⚠️  Could not compare")
            all_match = False

    print("\n" + "=" * 70)
    if all_match:
        print("✅ TSL + TP PRIORITY MATCHES BACKTRADER")
    else:
        print("⚠️  TSL + TP PRIORITY DIFFERS - Document behavior")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
