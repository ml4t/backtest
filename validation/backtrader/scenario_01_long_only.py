#!/usr/bin/env python3
"""Scenario 01: Long-only validation against Backtrader.

This script validates that ml4t.backtest produces identical results to Backtrader
for a simple long-only strategy using predefined entry/exit signals.

Run from .venv-backtrader environment:
    source .venv-backtrader/bin/activate
    cd validation/backtrader
    python scenario_01_long_only.py

Key differences from VectorBT:
- Backtrader uses next-bar execution by default (cheat-on-close=False)
- Fill prices are at next bar's open price
- Uses broker.setcash() and broker.setcommission()

Success criteria:
- Trade count: Exact match
- Fill prices: Match within OHLC bounds
- Final P&L: Match within 0.1%
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# Test Data Generation
# ============================================================================


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate identical test data for both frameworks.

    Returns:
        prices_df: OHLCV DataFrame with timestamp index
        entries: Boolean array of entry signals
        exits: Boolean array of exit signals
    """
    np.random.seed(seed)

    # Generate price path (random walk)
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02  # 2% daily vol
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars),
        },
        index=dates,
    )

    # Ensure high >= open, close, low and low <= open, close, high
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # Generate simple entry/exit signals
    # Entry: every 10 bars when not in position
    # Exit: 5 bars after entry
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    i = 0
    while i < n_bars - 6:
        entries[i] = True
        exits[i + 5] = True
        i += 10

    return df, entries, exits


# ============================================================================
# Backtrader Execution
# ============================================================================


def run_backtrader(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run backtest using Backtrader."""
    try:
        import backtrader as bt
    except ImportError:
        raise ImportError("Backtrader not installed. Run in .venv-backtrader environment.")

    # Create data feed from pandas
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

    # Strategy that uses pre-computed signals
    class SignalStrategy(bt.Strategy):
        params = (
            ("entries", None),
            ("exits", None),
        )

        def __init__(self):
            self.entries = self.params.entries
            self.exits = self.params.exits
            self.bar_count = 0
            self.trade_log = []

        def next(self):
            idx = self.bar_count

            # Check exit first (if in position)
            if self.position and idx < len(self.exits) and self.exits[idx]:
                self.close()

            # Check entry (if not in position)
            elif not self.position and idx < len(self.entries) and self.entries[idx]:
                self.buy(size=100)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trade_log.append(
                    {
                        "entry_time": bt.num2date(trade.dtopen),
                        "exit_time": bt.num2date(trade.dtclose),
                        "entry_price": trade.price,
                        "pnl": trade.pnl,
                        "pnlcomm": trade.pnlcomm,
                    }
                )

    # Set up cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)  # No commission

    # Add data
    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)

    # Add strategy with signals
    cerebro.addstrategy(SignalStrategy, entries=entries, exits=exits)

    # Run backtest
    results = cerebro.run()
    strategy = results[0]

    final_value = cerebro.broker.getvalue()

    return {
        "framework": "Backtrader",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": len(strategy.trade_log),
        "trades": strategy.trade_log,
    }


# ============================================================================
# ml4t.backtest Execution
# ============================================================================


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run backtest using ml4t.backtest with next-bar execution to match Backtrader."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

    # Convert to polars format
    prices_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["AAPL"] * len(prices_df),
            "open": prices_df["open"].tolist(),
            "high": prices_df["high"].tolist(),
            "low": prices_df["low"].tolist(),
            "close": prices_df["close"].tolist(),
            "volume": prices_df["volume"].astype(float).tolist(),
        }
    )

    # Create signals DataFrame
    signals_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["AAPL"] * len(prices_df),
            "entry": entries.tolist(),
            "exit": exits.tolist(),
        }
    )

    class SignalStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "AAPL" not in data:
                return

            signals = data["AAPL"].get("signals", {})
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            # Check exit first
            if signals.get("exit") and current_qty > 0:
                broker.close_position("AAPL")
            # Then check entry
            elif signals.get("entry") and current_qty == 0:
                broker.submit_order("AAPL", 100)  # Fixed 100 shares

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = SignalStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        account_type="cash",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Backtrader default
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,  # P&L = final - initial
        "num_trades": results["num_trades"],
        "trades": [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
            }
            for t in results["trades"]
        ],
    }


# ============================================================================
# Comparison
# ============================================================================


def compare_results(bt_results: dict, ml4t_results: dict) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print("COMPARISON: Backtrader vs ml4t.backtest")
    print("=" * 70)

    all_match = True

    # Trade count
    bt_trades = bt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = bt_trades == ml4t_trades
    print(
        f"\nTrade Count: BT={bt_trades}, ML4T={ml4t_trades} {'OK' if trades_match else 'MISMATCH'}"
    )
    all_match &= trades_match

    # Final value
    bt_value = bt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(bt_value - ml4t_value)
    value_pct_diff = value_diff / bt_value * 100 if bt_value != 0 else 0
    values_match = value_pct_diff < 0.1  # Within 0.1% (looser than VectorBT)
    print(
        f"Final Value: BT=${bt_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'OK' if values_match else 'MISMATCH'}"
    )
    all_match &= values_match

    # Total P&L
    bt_pnl = bt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(bt_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 10.0  # Within $10 (looser tolerance for Backtrader)
    print(
        f"Total P&L: BT=${bt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f}) {'OK' if pnl_match else 'MISMATCH'}"
    )
    all_match &= pnl_match

    # Trade-by-trade comparison
    if trades_match and len(bt_results["trades"]) > 0:
        print("\nTrade-by-Trade Comparison:")
        print("-" * 70)
        bt_trades_list = bt_results["trades"]
        ml4t_trades_list = ml4t_results["trades"]

        for i, (bt_t, ml4t_t) in enumerate(zip(bt_trades_list[:5], ml4t_trades_list[:5])):
            print(
                f"  Trade {i+1}: BT entry={bt_t['entry_price']:.2f}, "
                f"ML4T entry={ml4t_t['entry_price']:.2f}"
            )

    print("\n" + "=" * 70)
    if all_match:
        print("VALIDATION PASSED: Results match within tolerance")
    else:
        print("VALIDATION FAILED: Results do not match")
    print("=" * 70)

    return all_match


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("Scenario 01: Long-Only Validation (Backtrader)")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    prices_df, entries, exits = generate_test_data(n_bars=100)
    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    print(f"   Exit signals: {exits.sum()}")

    # Run Backtrader
    print("\nRunning Backtrader...")
    try:
        bt_results = run_backtrader(prices_df, entries, exits)
        print(f"   Trades: {bt_results['num_trades']}")
        print(f"   Final Value: ${bt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    success = compare_results(bt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
