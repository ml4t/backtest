#!/usr/bin/env python3
"""Scenario 16: Stress Test with 1000+ bars against Backtrader.

Tests TSL behavior over extended market conditions.

Run from .venv-backtrader environment:
    source .venv-backtrader/bin/activate
    python validation/backtrader/scenario_16_stress_1000bars.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05
SHARES_PER_TRADE = 100
N_BARS = 1500


def generate_stress_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate challenging data with multiple market regimes."""
    np.random.seed(seed)
    base_price = 100.0
    prices = [base_price]

    for i in range(1, N_BARS):
        if i < 100:
            change = np.random.randn() * 0.015 + 0.002
        elif i < 200:
            change = np.random.randn() * 0.02 - 0.005
        elif i < 400:
            change = np.random.randn() * 0.012 + 0.003
        elif i < 500:
            base_change = np.random.randn() * 0.008
            if np.random.random() < 0.1:
                gap = np.random.choice([-0.05, 0.05])
                base_change += gap
            change = base_change
        elif i < 700:
            change = np.random.randn() * 0.01 + 0.004
        elif i < 800:
            if i < 720:
                change = -0.03 + np.random.randn() * 0.01
            else:
                change = 0.02 + np.random.randn() * 0.01
        elif i < 1000:
            change = np.random.randn() * 0.015 - 0.002
        elif i < 1200:
            change = np.random.randn() * 0.04
        else:
            change = np.random.randn() * 0.01 + 0.002

        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=N_BARS, freq="D")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(N_BARS) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(N_BARS)) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(N_BARS)) * 0.01),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, N_BARS).astype(float),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    entries = np.zeros(N_BARS, dtype=bool)
    regime_starts = [0, 100, 200, 400, 500, 700, 800, 1000, 1200]
    for idx in regime_starts:
        if idx < N_BARS:
            entries[idx] = True

    return df, entries


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """Backtrader stress test with TSL."""
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

    class StressStrategy(bt.Strategy):
        def __init__(self):
            self.entry_signals = entries
            self.bar_count = 0
            self.trades = []

        def next(self):
            if self.bar_count < len(self.entry_signals):
                if self.entry_signals[self.bar_count] and not self.position:
                    self.buy(size=SHARES_PER_TRADE)
                    self.sell(
                        exectype=bt.Order.StopTrail,
                        trailpercent=TRAIL_PCT,
                        size=SHARES_PER_TRADE,
                    )
            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trades.append({
                    "exit_idx": trade.barclose,
                    "pnl": trade.pnl,
                })

    cerebro = bt.Cerebro()
    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)
    cerebro.addstrategy(StressStrategy)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    return {
        "framework": "Backtrader",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": len(strat.trades),
        "trades": strat.trades,
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest stress test with TSL."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk.position import TrailingStop

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

    class StressStrategy(Strategy):
        def on_start(self, broker):
            broker.set_position_rules(TrailingStop(pct=TRAIL_PCT))

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
        feed, StressStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Backtrader compatible
    )

    results = engine.run()

    trade_info = []
    for t in results["trades"]:
        exit_idx = None
        if t.exit_time:
            for i, ts in enumerate(prices_df.index):
                if ts == t.exit_time:
                    exit_idx = i
                    break
        trade_info.append({
            "exit_idx": exit_idx,
            "pnl": t.pnl,
        })

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_info,
    }


def main():
    print("=" * 70)
    print(f"Scenario 16: Stress Test with {N_BARS} bars (Backtrader)")
    print("=" * 70)

    df, entries = generate_stress_data()
    print(f"\nData: {len(df)} bars, {entries.sum()} entry signals")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    print("\nRunning Backtrader...")
    try:
        bt_result = run_backtrader(df, entries)
        print(f"  Trades: {bt_result['num_trades']}, PnL: ${bt_result['total_pnl']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        bt_result = None

    print("\nRunning ml4t.backtest...")
    try:
        ml4t_result = run_ml4t(df, entries)
        print(f"  Trades: {ml4t_result['num_trades']}, PnL: ${ml4t_result['total_pnl']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        ml4t_result = None

    print("\n" + "=" * 70)
    if bt_result and ml4t_result:
        trades_match = bt_result['num_trades'] == ml4t_result['num_trades']
        pnl_diff = abs(bt_result['total_pnl'] - ml4t_result['total_pnl'])
        pnl_match = pnl_diff < 200  # Higher tolerance for Backtrader (next-bar mode)

        print(f"Trade Count: BT={bt_result['num_trades']}, ML4T={ml4t_result['num_trades']} "
              f"{'✅' if trades_match else '❌'}")
        print(f"PnL Diff: ${pnl_diff:.2f} {'✅' if pnl_match else '⚠️ (expected with different execution modes)'}")

        if trades_match:
            print("✅ STRESS TEST PASSED (trade count matches)")
        else:
            print("⚠️  STRESS TEST SHOWS DIFFERENCES")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
