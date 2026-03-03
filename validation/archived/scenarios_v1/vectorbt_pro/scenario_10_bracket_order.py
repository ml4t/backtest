#!/usr/bin/env python3
"""Scenario 10: Bracket Order (OCO) validation against VectorBT Pro.

Bracket order = Stop-loss + Take-profit together.
First to trigger exits the position, cancelling the other.

Tests 5% stop-loss and 10% take-profit.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_10_bracket_order.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STOP_LOSS_PCT = 0.05  # 5% stop-loss
TAKE_PROFIT_PCT = 0.10  # 10% take-profit
SHARES_PER_TRADE = 100


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data that triggers both stop-losses and take-profits."""
    np.random.seed(seed)
    base_price = 100.0

    # Create price movements that trigger both SL and TP
    prices = [base_price]
    for i in range(1, n_bars):
        if i < 15:
            # First trade: gradual rise then sharp drop (triggers SL)
            if i < 8:
                change = np.random.randn() * 0.005 + 0.003
            else:
                change = -0.015 + np.random.randn() * 0.003
        elif i < 30:
            # Recovery
            change = np.random.randn() * 0.005 + 0.002
        elif i < 50:
            # Second trade: strong uptrend (triggers TP)
            change = np.random.randn() * 0.005 + 0.008
        elif i < 70:
            # Third trade: gradual decline (triggers SL)
            change = np.random.randn() * 0.005 - 0.005
        else:
            # Sideways/up
            change = np.random.randn() * 0.008 + 0.001
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

    # Entry signals at specific points
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True   # First trade (will hit SL)
    entries[20] = True  # Second trade (will hit TP)
    entries[55] = True  # Third trade (will hit SL)

    return df, entries


def run_vectorbt_pro(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VectorBT Pro bracket order via sl_th + tp_th."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed.")

    # CRITICAL: Must provide full OHLC for proper stop behavior
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=np.zeros_like(entries),
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        sl_stop=STOP_LOSS_PCT,   # Stop-loss (not sl_th)
        tp_stop=TAKE_PROFIT_PCT, # Take-profit (not tp_th)
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    # Extract exit types - handle different VBT Pro versions
    # Column names may vary: "Status", "status", "Exit Type", etc.
    exit_types = {}
    if len(trades) > 0:
        # Try common column names for exit status
        status_cols = ["Status", "status", "Exit Type", "exit_type"]
        for col in status_cols:
            if col in trades.columns:
                exit_types = trades[col].value_counts().to_dict()
                break
        # If no status column found, just count total trades
        if not exit_types:
            exit_types = {"completed": len(trades)}

    return {
        "framework": "VectorBT Pro",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "exit_types": exit_types,
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
    )
    from ml4t.backtest.risk.position import StopLoss, TakeProfit
    from ml4t.backtest.risk.position.composite import RuleChain

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

    class BracketOrderStrategy(Strategy):
        def on_start(self, broker):
            # Bracket order = StopLoss + TakeProfit in a RuleChain
            # First to trigger wins (OCO behavior)
            bracket = RuleChain([
                StopLoss(pct=STOP_LOSS_PCT),
                TakeProfit(pct=TAKE_PROFIT_PCT),
            ])
            broker.set_position_rules(bracket)

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
        BracketOrderStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()

    # Count exit types
    exit_types = {}
    for fill in results["fills"]:
        if fill.quantity < 0:  # Exit fill
            reason = getattr(fill, "reason", "unknown")
            if "stop_loss" in reason:
                exit_types["StopLoss"] = exit_types.get("StopLoss", 0) + 1
            elif "take_profit" in reason:
                exit_types["TakeProfit"] = exit_types.get("TakeProfit", 0) + 1
            else:
                exit_types[reason] = exit_types.get(reason, 0) + 1

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "exit_types": exit_types,
    }


def main():
    print("=" * 70)
    print(f"Scenario 10: Bracket Order (SL={STOP_LOSS_PCT*100:.0f}%, TP={TAKE_PROFIT_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, Entry signals: {entries.sum()}")

    print("\n  Running VectorBT Pro...")
    vbt_results = run_vectorbt_pro(prices_df, entries)
    print(f"   Trades: {vbt_results['num_trades']}")
    print(f"   Exit types: {vbt_results['exit_types']}")
    print(f"   Final Value: ${vbt_results['final_value']:,.2f}")

    print("\n  Running ml4t.backtest...")
    ml4t_results = run_ml4t_backtest(prices_df, entries)
    print(f"   Trades: {ml4t_results['num_trades']}")
    print(f"   Exit types: {ml4t_results['exit_types']}")
    print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")

    # Compare
    trades_match = vbt_results["num_trades"] == ml4t_results["num_trades"]
    value_diff = abs(vbt_results["final_value"] - ml4t_results["final_value"])
    value_pct_diff = value_diff / vbt_results["final_value"] * 100
    values_close = value_pct_diff < 1.0  # 1% tolerance

    print("\n" + "=" * 70)
    print(f"Trade Count: VBT={vbt_results['num_trades']}, ML4T={ml4t_results['num_trades']} "
          f"{'OK' if trades_match else 'DIFF'}")
    print(f"Final Value diff: {value_pct_diff:.4f}% {'OK' if values_close else 'FAIL'}")

    if trades_match and values_close:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 70)

    return 0 if (trades_match and values_close) else 1


if __name__ == "__main__":
    sys.exit(main())
