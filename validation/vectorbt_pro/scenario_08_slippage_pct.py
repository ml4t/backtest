#!/usr/bin/env python3
"""Scenario 08: Percentage Slippage validation against VectorBT Pro.

Tests 0.1% slippage per trade.
- Buy orders fill at price * (1 + slippage)
- Sell orders fill at price * (1 - slippage)

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_08_slippage_pct.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SLIPPAGE_RATE = 0.001  # 0.1%
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


def run_vectorbt_pro(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """VectorBT Pro uses percentage slippage natively."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed.")

    pf = vbt.Portfolio.from_signals(
        close=prices_df["close"],
        entries=entries,
        exits=exits,
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=SLIPPAGE_RATE,
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable
    return {
        "framework": "VectorBT Pro",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        PercentageSlippage,
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
        commission_model=NoCommission(),
        slippage_model=PercentageSlippage(rate=SLIPPAGE_RATE),
        execution_mode=ExecutionMode.SAME_BAR,
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
    print(f"Scenario 08: Percentage Slippage ({SLIPPAGE_RATE*100:.1f}%)")
    print("=" * 70)

    prices_df, entries, exits = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, Trades: {entries.sum()}")

    print("\n  Running VectorBT Pro...")
    vbt_results = run_vectorbt_pro(prices_df, entries, exits)
    print(f"   Final Value: ${vbt_results['final_value']:,.2f}")

    print("\n  Running ml4t.backtest...")
    ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
    print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")

    # Compare
    value_diff = abs(vbt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / vbt_results["final_value"] * 100
    trades_match = vbt_results["num_trades"] == ml4t_results["num_trades"]
    values_match = value_pct_diff < 0.01  # 0.01% tolerance

    print("\n" + "=" * 70)
    print(f"Trade Count: VBT={vbt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'OK' if trades_match else 'FAIL'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'OK' if values_match else 'FAIL'}")
    if trades_match and values_match:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 70)

    return 0 if (trades_match and values_match) else 1


if __name__ == "__main__":
    sys.exit(main())
