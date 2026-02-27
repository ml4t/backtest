#!/usr/bin/env python3
"""Scenario 09: Trailing Stop validation against VectorBT Pro.

Tests 5% trailing stop that tracks high water mark.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_09_trailing_stop.py
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

    # Create a mix of trends and reversals to trigger trailing stops
    # Up trend, then reversal
    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            # Initial uptrend (+0.5% per bar on average)
            change = np.random.randn() * 0.01 + 0.005
        elif i < 35:
            # Sharp reversal (-2% per bar)
            change = -0.02 + np.random.randn() * 0.005
        elif i < 60:
            # Another uptrend
            change = np.random.randn() * 0.01 + 0.003
        elif i < 65:
            # Another reversal
            change = -0.015 + np.random.randn() * 0.005
        else:
            # Sideways with slight upward bias
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

    # Enter at specific points where trends start
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True  # First entry at start
    entries[40] = True  # Second entry in middle of second trend

    return df, entries


def run_vectorbt_pro(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VectorBT Pro trailing stop via tsl_stop parameter with full OHLC."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed.")

    # VectorBT from_signals with trailing stop
    # CRITICAL: Must provide full OHLC for proper trailing stop behavior
    # tsl_stop (not tsl_th) is the correct parameter
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=np.zeros_like(entries),  # No manual exits, only trailing stop
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        tsl_stop=TRAIL_PCT,  # Trailing stop loss threshold (not tsl_th)
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    # Extract exit reasons - handle different VBT Pro versions
    # Column names may vary: "Status", "status", "Exit Type", etc.
    exit_reasons = {}
    if len(trades) > 0:
        # Try common column names for exit status
        status_cols = ["Status", "status", "Exit Type", "exit_type"]
        for col in status_cols:
            if col in trades.columns:
                exit_reasons = trades[col].value_counts().to_dict()
                break
        # If no status column found, just count total trades
        if not exit_reasons:
            exit_reasons = {"completed": len(trades)}

    return {
        "framework": "VectorBT Pro",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "exit_reasons": exit_reasons,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        Strategy,
        TrailHwmSource,
        StopFillMode,
        InitialHwmSource,
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

            # Only enter if no position and entry signal
            if signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    # VBT Pro compatible settings:
    # - trail_hwm_source: HIGH (VBT Pro updates HWM from bar highs)
    # - initial_hwm_source: BAR_CLOSE (VBT Pro uses bar close, not fill price)
    # - stop_fill_mode: STOP_PRICE (VBT Pro fills at trail level)
    engine = Engine(
        feed,
        TrailingStopStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
        trail_hwm_source=TrailHwmSource.HIGH,
        initial_hwm_source=InitialHwmSource.BAR_CLOSE,
        stop_fill_mode=StopFillMode.STOP_PRICE,
    )

    results = engine.run()

    # Count exit reasons
    exit_reasons = {}
    for fill in results["fills"]:
        if fill.quantity < 0:  # Exit fill
            reason = getattr(fill, "reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "exit_reasons": exit_reasons,
    }


def main():
    print("=" * 70)
    print(f"Scenario 09: Trailing Stop ({TRAIL_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, Entry signals: {entries.sum()}")

    print("\n  Running VectorBT Pro...")
    vbt_results = run_vectorbt_pro(prices_df, entries)
    print(f"   Trades: {vbt_results['num_trades']}")
    print(f"   Exit reasons: {vbt_results['exit_reasons']}")
    print(f"   Final Value: ${vbt_results['final_value']:,.2f}")

    print("\n  Running ml4t.backtest...")
    ml4t_results = run_ml4t_backtest(prices_df, entries)
    print(f"   Trades: {ml4t_results['num_trades']}")
    print(f"   Exit reasons: {ml4t_results['exit_reasons']}")
    print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")

    # Compare
    trades_match = vbt_results["num_trades"] == ml4t_results["num_trades"]
    value_diff = abs(vbt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / vbt_results["final_value"] * 100

    # Note: Trailing stop timing may differ slightly due to high water mark calculation
    # VectorBT uses close prices, ml4t can use intrabar highs
    values_close = value_pct_diff < 1.0  # 1% tolerance

    print("\n" + "=" * 70)
    print(f"Trade Count: VBT={vbt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'OK' if trades_match else 'FAIL'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'OK' if values_close else 'FAIL'}")

    if trades_match and values_close:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
        print("\nNote: Trailing stop behavior can differ due to high water mark timing.")
    print("=" * 70)

    return 0 if (trades_match and values_close) else 1


if __name__ == "__main__":
    sys.exit(main())
