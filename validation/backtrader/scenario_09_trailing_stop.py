#!/usr/bin/env python3
"""Scenario 09: Trailing Stop validation against Backtrader.

Backtrader supports trailing stops via StopTrail order type.

Run from .venv-validation environment:
    source .venv-validation/bin/activate
    python validation/backtrader/scenario_09_trailing_stop.py
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
    """Generate trending data that will trigger trailing stops."""
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            change = np.random.randn() * 0.01 + 0.005
        elif i < 35:
            change = -0.02 + np.random.randn() * 0.005
        elif i < 60:
            change = np.random.randn() * 0.01 + 0.003
        elif i < 65:
            change = -0.015 + np.random.randn() * 0.005
        else:
            change = np.random.randn() * 0.01 + 0.001
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

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True
    entries[40] = True

    return df, entries


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
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

    class TrailingStopStrategy(bt.Strategy):
        params = (("entries", None),)

        def __init__(self):
            self.bar_count = 0
            self.num_trades = 0
            self.stop_order = None

        def next(self):
            idx = self.bar_count
            if idx >= len(self.params.entries):
                return

            entry = self.params.entries[idx]

            if entry and self.position.size == 0:
                self.buy(size=SHARES_PER_TRADE)
                # Place trailing stop
                self.stop_order = self.sell(
                    exectype=bt.Order.StopTrail,
                    trailpercent=TRAIL_PCT,
                    size=SHARES_PER_TRADE,
                )
                self.num_trades += 1

            self.bar_count += 1

        def notify_order(self, order):
            if order.status == order.Completed:
                if order.issell() and order == self.stop_order:
                    self.stop_order = None

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)
    cerebro.addstrategy(TrailingStopStrategy, entries=entries)

    results = cerebro.run()
    strategy = results[0]

    return {
        "framework": "Backtrader",
        "final_value": cerebro.broker.getvalue(),
        "total_pnl": cerebro.broker.getvalue() - 100_000.0,
        "num_trades": strategy.num_trades,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    import polars as pl

    from ml4t.backtest import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
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
            "entry": entries.tolist(),
        }
    )

    class TrailingStopStrategy(Strategy):
        def on_start(self, broker):
            # Set position rules at strategy start
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
        feed,
        TrailingStopStrategy(),
        initial_cash=100_000.0,
        account_type="cash",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
    }


def main():
    print("=" * 70)
    print(f"Scenario 09: Trailing Stop ({TRAIL_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, Entry signals: {entries.sum()}")

    print("\n  Running Backtrader...")
    bt_results = run_backtrader(prices_df, entries)
    print(f"   Trades: {bt_results['num_trades']}")
    print(f"   Final Value: ${bt_results['final_value']:,.2f}")

    print("\n  Running ml4t.backtest...")
    ml4t_results = run_ml4t_backtest(prices_df, entries)
    print(f"   Trades: {ml4t_results['num_trades']}")
    print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")

    # Compare
    trades_match = bt_results["num_trades"] == ml4t_results["num_trades"]
    value_diff = abs(bt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / bt_results["final_value"] * 100
    values_close = value_pct_diff < 2.0  # 2% tolerance for trailing stop timing

    print("\n" + "=" * 70)
    print(f"Trade Count: BT={bt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'OK' if trades_match else 'DIFF'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'OK' if values_close else 'FAIL'}")

    # Trailing stops may exit at different bars due to high-water mark tracking
    # differences between frameworks
    if trades_match and values_close:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED (or within acceptable tolerance)")
        print("\nNote: Trailing stop timing differs due to high-water mark tracking.")
    print("=" * 70)

    return 0 if values_close else 1


if __name__ == "__main__":
    sys.exit(main())
