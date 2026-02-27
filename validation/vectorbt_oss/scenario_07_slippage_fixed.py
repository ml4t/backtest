#!/usr/bin/env python3
"""Scenario 07: Fixed Slippage validation against VectorBT OSS."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SLIPPAGE_AMOUNT = 0.01
SHARES_PER_TRADE = 100


def generate_test_data(n_bars: int = 100, seed: int = 42):
    np.random.seed(seed)
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame({
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
    }, index=dates)

    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    i = 0
    while i < n_bars - 6:
        entries[i] = True
        exits[i + 5] = True
        i += 10

    return df, entries, exits


def run_vectorbt_oss(prices_df, entries, exits):
    import vectorbt as vbt

    avg_price = prices_df["close"].mean()
    slippage_pct = SLIPPAGE_AMOUNT / avg_price

    pf = vbt.Portfolio.from_signals(
        close=prices_df["close"],
        entries=entries, exits=exits,
        init_cash=100_000.0, size=SHARES_PER_TRADE, size_type="amount",
        slippage=slippage_pct, fees=0.0, accumulate=False, freq="D",
    )

    return {"final_value": float(pf.value().iloc[-1]), "num_trades": len(pf.trades.records_readable)}


def run_ml4t_backtest(prices_df, entries, exits):
    import polars as pl
    from ml4t.backtest._validation_imports import DataFeed, Engine, ExecutionMode, FixedSlippage, NoCommission, Strategy

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
        "exit": exits.tolist(),
    })

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

    engine = Engine(
        DataFeed(prices_df=prices_pl, signals_df=signals_pl),
        SignalStrategy(), initial_cash=100_000.0, allow_short_selling=False,
        commission_model=NoCommission(), slippage_model=FixedSlippage(amount=SLIPPAGE_AMOUNT),
        execution_mode=ExecutionMode.SAME_BAR,
    )
    results = engine.run()
    return {"final_value": results["final_value"], "num_trades": results["num_trades"]}


def main():
    print("=" * 70)
    print(f"Scenario 07: Fixed Slippage (${SLIPPAGE_AMOUNT}/share)")
    print("=" * 70)

    prices_df, entries, exits = generate_test_data()
    vbt_results = run_vectorbt_oss(prices_df, entries, exits)
    ml4t_results = run_ml4t_backtest(prices_df, entries, exits)

    print(f"\nVBT Final: ${vbt_results['final_value']:,.2f}")
    print(f"ML4T Final: ${ml4t_results['final_value']:,.2f}")

    diff_pct = abs(vbt_results["final_value"] - ml4t_results["final_value"]) / vbt_results["final_value"] * 100
    passed = diff_pct < 0.01 and vbt_results["num_trades"] == ml4t_results["num_trades"]

    print("\n" + ("✅ VALIDATION PASSED" if passed else "❌ VALIDATION FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
