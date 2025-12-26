#!/usr/bin/env python3
"""Scenario 09: Trailing Stop validation against VectorBT OSS.

Tests 5% trailing stop that tracks high water mark.

Run from .venv-validation environment:
    source .venv-validation/bin/activate
    python validation/vectorbt_oss/scenario_09_trailing_stop.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
SHARES_PER_TRADE = 100


def generate_test_data(n_bars: int = 100, seed: int = 42):
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

    df = pd.DataFrame({
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
    }, index=dates)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True
    entries[40] = True

    return df, entries


def run_vectorbt_oss(prices_df, entries):
    import vectorbt as vbt

    # VectorBT OSS uses sl_stop with sl_trail=True for trailing stops
    pf = vbt.Portfolio.from_signals(
        close=prices_df["close"],
        entries=entries,
        exits=np.zeros_like(entries),
        init_cash=100_000.0, size=SHARES_PER_TRADE, size_type="amount",
        sl_stop=TRAIL_PCT,
        sl_trail=True,  # Makes it a trailing stop
        fees=0.0, accumulate=False, freq="D",
    )

    return {
        "final_value": float(pf.value().iloc[-1]),
        "num_trades": len(pf.trades.records_readable),
    }


def run_ml4t_backtest(prices_df, entries):
    import polars as pl
    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoSlippage, NoCommission, Strategy
    from ml4t.backtest.risk.position import TrailingStop

    prices_pl = pl.DataFrame({
        "timestamp": prices_df.index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "open": prices_df["close"].tolist(),
        "high": prices_df["close"].tolist(),
        "low": prices_df["close"].tolist(),
        "close": prices_df["close"].tolist(),
        "volume": prices_df["volume"].tolist(),
    })
    signals_pl = pl.DataFrame({
        "timestamp": prices_df.index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "entry": entries.tolist(),
    })

    class TrailingStopStrategy(Strategy):
        def on_start(self, broker):
            # Set position rules at strategy start (single rule, not a list)
            broker.set_position_rules(TrailingStop(pct=TRAIL_PCT))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0
            if signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE)

    engine = Engine(
        DataFeed(prices_df=prices_pl, signals_df=signals_pl),
        TrailingStopStrategy(), initial_cash=100_000.0, account_type="cash",
        commission_model=NoCommission(), slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )
    results = engine.run()
    return {"final_value": results["final_value"], "num_trades": results["num_trades"]}


def main():
    print("=" * 70)
    print(f"Scenario 09: Trailing Stop ({TRAIL_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    vbt_results = run_vectorbt_oss(prices_df, entries)
    ml4t_results = run_ml4t_backtest(prices_df, entries)

    print(f"\nVBT Trades: {vbt_results['num_trades']}, Final: ${vbt_results['final_value']:,.2f}")
    print(f"ML4T Trades: {ml4t_results['num_trades']}, Final: ${ml4t_results['final_value']:,.2f}")

    diff_pct = abs(vbt_results["final_value"] - ml4t_results["final_value"]) / vbt_results["final_value"] * 100
    trades_match = vbt_results["num_trades"] == ml4t_results["num_trades"]
    values_close = diff_pct < 1.0

    print("\n" + ("VALIDATION PASSED" if (trades_match and values_close) else "VALIDATION FAILED"))
    return 0 if (trades_match and values_close) else 1


if __name__ == "__main__":
    sys.exit(main())
