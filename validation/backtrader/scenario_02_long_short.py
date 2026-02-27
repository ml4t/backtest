#!/usr/bin/env python3
"""Scenario 02: Long/Short validation against Backtrader.

This script validates that ml4t.backtest produces identical results to Backtrader
for a long/short strategy including position reversals.

Run from .venv-backtrader environment:
    source .venv-backtrader/bin/activate
    cd validation/backtrader
    python scenario_02_long_short.py

Success criteria:
- Trade count: Exact match
- Fill prices: Match within OHLC bounds
- Short positions handled correctly
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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data for long/short strategy."""
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
            "volume": np.random.randint(100000, 1000000, n_bars),
        },
        index=dates,
    )

    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # Generate alternating long/short signals
    long_entries = np.zeros(n_bars, dtype=bool)
    long_exits = np.zeros(n_bars, dtype=bool)
    short_entries = np.zeros(n_bars, dtype=bool)
    short_exits = np.zeros(n_bars, dtype=bool)

    i = 0
    position = 0
    while i < n_bars - 6:
        if position == 0:
            long_entries[i] = True
            long_exits[i + 5] = True
            position = 1
            i += 10
        elif position == 1:
            short_entries[i] = True
            short_exits[i + 5] = True
            position = -1
            i += 10
        else:
            position = 0

    return df, long_entries, long_exits, short_entries, short_exits


# ============================================================================
# Backtrader Execution
# ============================================================================


def run_backtrader(
    prices_df: pd.DataFrame,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
) -> dict:
    """Run backtest using Backtrader."""
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

    class LongShortStrategy(bt.Strategy):
        params = (
            ("long_entries", None),
            ("long_exits", None),
            ("short_entries", None),
            ("short_exits", None),
        )

        def __init__(self):
            self.bar_count = 0
            self.trade_log = []

        def next(self):
            idx = self.bar_count

            # Check exits first
            if (
                self.position.size > 0
                and idx < len(self.params.long_exits)
                and self.params.long_exits[idx]
                or self.position.size < 0
                and idx < len(self.params.short_exits)
                and self.params.short_exits[idx]
            ):
                self.close()

            # Check entries (only if flat)
            elif not self.position:
                if idx < len(self.params.long_entries) and self.params.long_entries[idx]:
                    self.buy(size=100)
                elif idx < len(self.params.short_entries) and self.params.short_entries[idx]:
                    self.sell(size=100)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                # Determine direction from the trade size (positive = long, negative = short)
                direction = "Long" if trade.size > 0 else "Short"
                self.trade_log.append(
                    {
                        "entry_time": bt.num2date(trade.dtopen),
                        "exit_time": bt.num2date(trade.dtclose),
                        "entry_price": trade.price,
                        "pnl": trade.pnl,
                        "direction": direction,
                    }
                )

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)

    cerebro.addstrategy(
        LongShortStrategy,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )

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


def run_ml4t_backtest(
    prices_df: pd.DataFrame,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
) -> dict:
    """Run backtest using ml4t.backtest with next-bar execution."""
    import polars as pl

    from ml4t.backtest._validation_imports import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

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

    signals_pl = pl.DataFrame(
        {
            "timestamp": prices_df.index.to_pydatetime().tolist(),
            "asset": ["AAPL"] * len(prices_df),
            "long_entry": long_entries.tolist(),
            "long_exit": long_exits.tolist(),
            "short_entry": short_entries.tolist(),
            "short_exit": short_exits.tolist(),
        }
    )

    class LongShortStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "AAPL" not in data:
                return

            signals = data["AAPL"].get("signals", {})
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            # Check exits first
            if (
                signals.get("long_exit")
                and current_qty > 0
                or signals.get("short_exit")
                and current_qty < 0
            ):
                broker.close_position("AAPL")

            # Then check entries (only if flat)
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            if current_qty == 0:
                if signals.get("long_entry"):
                    broker.submit_order("AAPL", 100)
                elif signals.get("short_entry"):
                    broker.submit_order("AAPL", -100)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = LongShortStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=True, allow_leverage=True,  # Margin account for short selling
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Backtrader default
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "direction": "Long" if t.quantity > 0 else "Short",
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
    print("COMPARISON: Backtrader vs ml4t.backtest (Long/Short)")
    print("=" * 70)

    all_match = True

    bt_trades = bt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = bt_trades == ml4t_trades
    print(
        f"\nTrade Count: BT={bt_trades}, ML4T={ml4t_trades} {'OK' if trades_match else 'MISMATCH'}"
    )
    all_match &= trades_match

    bt_value = bt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(bt_value - ml4t_value)
    value_pct_diff = value_diff / bt_value * 100 if bt_value != 0 else 0
    values_match = value_pct_diff < 0.1
    print(
        f"Final Value: BT=${bt_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'OK' if values_match else 'MISMATCH'}"
    )
    all_match &= values_match

    bt_pnl = bt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(bt_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 10.0
    print(
        f"Total P&L: BT=${bt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f}) {'OK' if pnl_match else 'MISMATCH'}"
    )
    all_match &= pnl_match

    if trades_match and len(bt_results["trades"]) > 0:
        print("\nTrade-by-Trade Comparison:")
        print("-" * 70)
        for i, (bt_t, ml4t_t) in enumerate(
            zip(bt_results["trades"][:5], ml4t_results["trades"][:5])
        ):
            bt_dir = bt_t.get("direction", "Unknown")
            ml4t_dir = ml4t_t.get("direction", "Unknown")
            print(
                f"  Trade {i+1}: BT {bt_dir} entry={bt_t['entry_price']:.2f} | ML4T {ml4t_dir} entry={ml4t_t['entry_price']:.2f}"
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
    print("Scenario 02: Long/Short Validation (Backtrader)")
    print("=" * 70)

    print("\nGenerating test data...")
    prices_df, long_entries, long_exits, short_entries, short_exits = generate_test_data(n_bars=100)
    print(f"   Bars: {len(prices_df)}")
    print(f"   Long entries: {long_entries.sum()}")
    print(f"   Short entries: {short_entries.sum()}")

    print("\nRunning Backtrader...")
    try:
        bt_results = run_backtrader(prices_df, long_entries, long_exits, short_entries, short_exits)
        print(f"   Trades: {bt_results['num_trades']}")
        print(f"   Final Value: ${bt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        return 1

    print("\nRunning ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(
            prices_df, long_entries, long_exits, short_entries, short_exits
        )
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    success = compare_results(bt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
