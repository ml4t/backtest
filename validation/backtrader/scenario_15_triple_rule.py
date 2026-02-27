#!/usr/bin/env python3
"""Scenario 15: Triple Rule (TSL + TP + SL) combination validation against Backtrader.

Tests what happens when ALL THREE rules are active:
- Trailing Stop Loss (protects profits)
- Take Profit (locks in gains)
- Stop Loss (limits losses)

Critical for understanding complex rule interactions.

Run from .venv-backtrader environment:
    source .venv-backtrader/bin/activate
    python validation/backtrader/scenario_15_triple_rule.py

Success criteria:
- Document Backtrader behavior for triple rule priority
- Match ml4t behavior with documented expectations
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.03  # 3% trailing stop
TP_PCT = 0.10     # 10% take profit
SL_PCT = 0.05     # 5% stop loss
SHARES_PER_TRADE = 100


def generate_test_data_tp_wins(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TP triggers first (clean win).

    Scenario: Steady rise to TP level, no drawdown.
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 15:
            # Steady rise to 10% gain
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


def generate_test_data_sl_wins(seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where SL triggers first (clean loss).

    Scenario: Immediate drop to SL level, no chance for TSL or TP.
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i == 1:
            # Immediate drop to 5%+ loss
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


def generate_test_data_tsl_wins(seed: int = 44) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TSL triggers first (profit protection).

    Scenario: Rise to 8% gain (below TP), then 3% pullback triggers TSL.
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            # Rise to ~8% gain (below 10% TP)
            change = 0.008
        elif i == 10:
            # Drop 4% from high (triggers 3% TSL)
            change = -0.04
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
    """Run Backtrader with TSL + TP + SL for LONG positions."""
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

    class TripleRuleStrategy(bt.Strategy):
        def __init__(self):
            self.entry_signals = entries
            self.bar_count = 0
            self.order = None
            self.trades = []
            self.entry_price = None

        def next(self):
            if self.bar_count < len(self.entry_signals):
                if self.entry_signals[self.bar_count] and not self.position:
                    # Enter with all three risk rules
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

                    # Set fixed stop loss
                    sl_price = self.entry_price * (1 - SL_PCT)
                    self.sell(
                        exectype=bt.Order.Stop,
                        price=sl_price,
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
    cerebro.addstrategy(TripleRuleStrategy)
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
    """ml4t.backtest with TSL + TP + SL."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk import RuleChain
    from ml4t.backtest.risk.position import TrailingStop, TakeProfit, StopLoss

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

    class TripleRuleStrategy(Strategy):
        def on_start(self, broker):
            # All three rules in chain
            broker.set_position_rules(RuleChain([
                TrailingStop(pct=TRAIL_PCT),
                TakeProfit(pct=TP_PCT),
                StopLoss(pct=SL_PCT),
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
        feed, TripleRuleStrategy(),
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
    print("Scenario 15: Triple Rule (TSL + TP + SL) (Backtrader)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Take Profit: {TP_PCT*100:.0f}%")
    print(f"  Stop Loss: {SL_PCT*100:.0f}%")

    scenarios = [
        ("TP wins (clean uptrend)", generate_test_data_tp_wins),
        ("SL wins (immediate drop)", generate_test_data_sl_wins),
        ("TSL wins (rise then pullback)", generate_test_data_tsl_wins),
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
        print(f"SL level: ${df['close'].iloc[0] * (1 - SL_PCT):.2f}")

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
    print("TRIPLE RULE PRIORITY FINDINGS")
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
        print("✅ TRIPLE RULE PRIORITY MATCHES BACKTRADER")
    else:
        print("⚠️  TRIPLE RULE PRIORITY DIFFERS - Document behavior")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
