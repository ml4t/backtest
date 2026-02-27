#!/usr/bin/env python3
"""Scenario 06: Per-Share Commission validation against Backtrader.

Backtrader supports per-share commission via COMM_FIXED commtype.

Run from .venv-validation environment:
    source .venv-validation/bin/activate
    python validation/backtrader/scenario_06_commission_per_share.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PER_SHARE_RATE = 0.005
SHARES_PER_TRADE = 100


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    i = 0
    while i < n_bars - 6:
        entries[i] = True
        exits[i + 5] = True
        i += 10

    return df, entries, exits


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
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

    class SignalStrategy(bt.Strategy):
        params = (("entries", None), ("exits", None))

        def __init__(self):
            self.bar_count = 0
            self.total_commission = 0.0

        def next(self):
            idx = self.bar_count
            if idx >= len(self.params.entries):
                return

            if self.params.exits[idx] and self.position.size > 0:
                self.close()
            elif self.params.entries[idx] and self.position.size == 0:
                self.buy(size=SHARES_PER_TRADE)

            self.bar_count += 1

        def notify_order(self, order):
            if order.status == order.Completed:
                self.total_commission += order.executed.comm

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    # Backtrader per-share commission: COMM_FIXED applies per share
    cerebro.broker.setcommission(
        commission=PER_SHARE_RATE, commtype=bt.CommInfoBase.COMM_FIXED
    )

    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalStrategy, entries=entries, exits=exits)

    results = cerebro.run()
    strategy = results[0]

    return {
        "framework": "Backtrader",
        "final_value": cerebro.broker.getvalue(),
        "total_pnl": cerebro.broker.getvalue() - 100_000.0,
        "total_commission": strategy.total_commission,
        "num_trades": entries.sum(),
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoSlippage,
        PerShareCommission,
        Strategy,
    )

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
            "entry": entries.tolist(),
            "exit": exits.tolist(),
        }
    )

    class SignalStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            if signals.get("exit") and current_qty > 0:
                broker.close_position("TEST")
            elif signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    engine = Engine(
        feed,
        SignalStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=PerShareCommission(per_share=PER_SHARE_RATE, minimum=0.0),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Backtrader default
    )

    results = engine.run()
    total_commission = sum(f.commission for f in results["fills"])

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "total_commission": total_commission,
        "num_trades": results["num_trades"],
    }


def main():
    print("=" * 70)
    print(f"Scenario 06: Per-Share Commission (${PER_SHARE_RATE}/share)")
    print("=" * 70)

    prices_df, entries, exits = generate_test_data()
    print(f"\n📊 Bars: {len(prices_df)}, Trades: {entries.sum()}")

    print("\n🔷 Running Backtrader...")
    bt_results = run_backtrader(prices_df, entries, exits)
    print(f"   Commission: ${bt_results['total_commission']:.2f}")
    print(f"   Final Value: ${bt_results['final_value']:,.2f}")

    print("\n🔶 Running ml4t.backtest...")
    ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
    print(f"   Commission: ${ml4t_results['total_commission']:.2f}")
    print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")

    # Compare
    all_match = True
    all_match &= bt_results["num_trades"] == ml4t_results["num_trades"]
    all_match &= abs(bt_results["total_commission"] - ml4t_results["total_commission"]) < 0.01
    all_match &= abs(bt_results["final_value"] - ml4t_results["final_value"]) < 1.0

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED")
    else:
        print("❌ VALIDATION FAILED")
        print(f"   Commission diff: ${abs(bt_results['total_commission'] - ml4t_results['total_commission']):.2f}")
        print(f"   Value diff: ${abs(bt_results['final_value'] - ml4t_results['final_value']):.2f}")
    print("=" * 70)

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
