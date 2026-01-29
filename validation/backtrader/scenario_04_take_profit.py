#!/usr/bin/env python3
"""Scenario 04: Take-Profit validation against Backtrader.

This script validates that ml4t.backtest take-profit behavior matches Backtrader.

Run from .venv-backtrader environment:
    .venv-backtrader/bin/python3 validation/backtrader/scenario_04_take_profit.py

Backtrader uses limit orders for take-profit.

Key differences from VectorBT:
- Backtrader uses next-bar execution by default
- Limit orders are simulated on next bar's OHLC range
- Fill happens at limit price (not necessarily open)

Success criteria:
- Exit trigger timing: Same bar (after target hit)
- Exit price: At or above target level
- Final P&L: Within 1%
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


def generate_take_profit_data(seed: int = 42) -> pd.DataFrame:
    """Generate test data where price rises to trigger take-profit.

    Creates a scenario where:
    - Enter at bar 0 at $100
    - Price rises steadily
    - Take-profit should trigger when gain exceeds threshold

    Returns:
        prices_df: OHLCV DataFrame with timestamp index
    """
    np.random.seed(seed)

    # Price path: 100 -> gradual rise to trigger 10% take-profit
    n_bars = 20
    closes = np.array(
        [
            100.0,  # Bar 0: Entry
            101.0,  # Bar 1: +1%
            103.0,  # Bar 2: +3%
            105.0,  # Bar 3: +5%
            107.0,  # Bar 4: +7%
            109.0,  # Bar 5: +9%
            111.0,  # Bar 6: +11% -> TAKE-PROFIT TRIGGERED
            113.0,  # Bar 7: +13%
            115.0,  # Bar 8: +15%
            117.0,  # Bar 9: +17%
            119.0,  # Bar 10
            121.0,  # Bar 11
            123.0,  # Bar 12
            125.0,  # Bar 13
            127.0,  # Bar 14
            129.0,  # Bar 15
            131.0,  # Bar 16
            133.0,  # Bar 17
            135.0,  # Bar 18
            137.0,  # Bar 19
        ]
    )

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    # Generate OHLCV - opens are slightly lower than close for rising market
    opens = closes - 0.5
    highs = closes + 0.5
    lows = opens - 0.5

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n_bars, 100000),
        },
        index=dates,
    )

    return df


# ============================================================================
# Backtrader Execution
# ============================================================================


def run_backtrader(prices_df: pd.DataFrame, tp_pct: float) -> dict:
    """Run backtest using Backtrader with take-profit."""
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

    class TakeProfitStrategy(bt.Strategy):
        params = (("tp_pct", 0.10),)

        def __init__(self):
            self.trade_log = []
            self.entry_order = None
            self.limit_order = None
            self.entry_price = None

        def next(self):
            # Only enter once at the beginning
            if len(self) == 1 and not self.position:
                self.entry_price = self.data.close[0]
                self.entry_order = self.buy(size=100)
                # Place take-profit limit order
                limit_price = self.entry_price * (1 + self.params.tp_pct)
                self.limit_order = self.sell(
                    size=100,
                    exectype=bt.Order.Limit,
                    price=limit_price,
                )

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
    cerebro.broker.setcommission(commission=0.0)

    # Add data
    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(TakeProfitStrategy, tp_pct=tp_pct)

    # Run backtest
    results = cerebro.run()
    strategy = results[0]

    final_value = cerebro.broker.getvalue()

    # Get exit price from trades
    exit_price = None
    if strategy.trade_log:
        entry = strategy.trade_log[0]["entry_price"]
        pnl = strategy.trade_log[0]["pnl"]
        exit_price = entry + (pnl / 100)  # 100 shares

    return {
        "framework": "Backtrader",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": len(strategy.trade_log),
        "trades": strategy.trade_log,
        "exit_price": exit_price,
    }


# ============================================================================
# ml4t.backtest Execution
# ============================================================================


def run_ml4t_backtest(prices_df: pd.DataFrame, tp_pct: float) -> dict:
    """Run backtest using ml4t.backtest with take-profit (next-bar mode)."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        StopFillMode,
        StopLevelBasis,
        Strategy,
    )
    from ml4t.backtest.risk import TakeProfit

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

    class TakeProfitStrategy(Strategy):
        def __init__(self, tp_pct: float):
            self.tp_pct = tp_pct
            self.entered = False

        def on_start(self, broker):
            broker.set_position_rules(TakeProfit(pct=self.tp_pct))

        def on_data(self, timestamp, data, context, broker):
            if "AAPL" not in data:
                return

            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            # Entry only on first bar
            if not self.entered and current_qty == 0:
                broker.submit_order("AAPL", 100)
                self.entered = True

    feed = DataFeed(prices_df=prices_pl)
    strategy = TakeProfitStrategy(tp_pct=tp_pct)

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Backtrader default
        stop_fill_mode=StopFillMode.STOP_PRICE,  # Match Backtrader: exact target price
        stop_level_basis=StopLevelBasis.SIGNAL_PRICE,  # Match Backtrader: target from signal close
    )

    results = engine.run()

    exit_price = None
    if results["trades"]:
        exit_price = results["trades"][0].exit_price

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
            }
            for t in results["trades"]
        ],
        "exit_price": exit_price,
    }


# ============================================================================
# Comparison
# ============================================================================


def compare_results(bt_results: dict, ml4t_results: dict, tp_pct: float) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: Backtrader vs ml4t.backtest (Take-Profit={tp_pct:.0%})")
    print("=" * 70)

    all_match = True

    # Trade count
    bt_trades = bt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = bt_trades == ml4t_trades
    print(f"\nTrade Count: BT={bt_trades}, ML4T={ml4t_trades} {'PASS' if trades_match else 'FAIL'}")
    all_match &= trades_match

    # Final value - expect exact match with SIGNAL_PRICE basis
    bt_value = bt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(bt_value - ml4t_value)
    value_pct_diff = value_diff / bt_value * 100 if bt_value != 0 else 0
    values_match = value_pct_diff < 0.01  # Near-exact match
    print(
        f"Final Value: BT=${bt_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'PASS' if values_match else 'FAIL'}"
    )
    all_match &= values_match

    # Total P&L - expect exact match
    bt_pnl = bt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(bt_pnl - ml4t_pnl)
    pnl_pct_diff = pnl_diff / abs(bt_pnl) * 100 if bt_pnl != 0 else 0
    pnl_match = pnl_pct_diff < 1.0  # Near-exact match
    print(
        f"Total P&L: BT=${bt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff={pnl_pct_diff:.1f}%) {'PASS' if pnl_match else 'FAIL'}"
    )
    all_match &= pnl_match

    # Exit price comparison - expect exact match
    bt_exit = bt_results.get("exit_price")
    ml4t_exit = ml4t_results.get("exit_price")
    if bt_exit and ml4t_exit:
        exit_diff = abs(bt_exit - ml4t_exit)
        exit_match = exit_diff < 0.01
        print(
            f"Exit Price: BT=${bt_exit:.2f}, ML4T=${ml4t_exit:.2f} (diff=${exit_diff:.2f}) {'PASS' if exit_match else 'FAIL'}"
        )
        all_match &= exit_match

    print("\n" + "=" * 70)
    print("CONFIGURATION:")
    print("  Using StopFillMode.STOP_PRICE + StopLevelBasis.SIGNAL_PRICE")
    print("  Both frameworks: Target level from signal close price, fill at exact target")
    print("=" * 70)

    if all_match:
        print("\nVALIDATION PASSED: Take-profit produces EXACT MATCH with Backtrader")
    else:
        print("\nVALIDATION FAILED: Take-profit behavior differs")
    print("=" * 70)

    return all_match


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("Scenario 04: Take-Profit Validation (Backtrader)")
    print("=" * 70)

    tp_pct = 0.10  # 10% take-profit

    # Generate test data
    print(f"\nGenerating test data for {tp_pct:.0%} take-profit...")
    prices_df = generate_take_profit_data()
    print(f"   Bars: {len(prices_df)}")
    print(f"   Price at entry: ${prices_df['close'].iloc[0]:.2f}")
    print(f"   Target level: ${prices_df['close'].iloc[0] * (1 + tp_pct):.2f}")

    # Run Backtrader
    print("\nRunning Backtrader...")
    try:
        bt_results = run_backtrader(prices_df, tp_pct)
        print(f"   Trades: {bt_results['num_trades']}")
        print(f"   Final Value: ${bt_results['final_value']:,.2f}")
        if bt_results["exit_price"]:
            print(f"   Exit price: ${bt_results['exit_price']:.2f}")
    except ImportError as e:
        print(f"   ERROR: {e}")
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, tp_pct)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
        if ml4t_results["exit_price"]:
            print(f"   Exit price: ${ml4t_results['exit_price']:.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    success = compare_results(bt_results, ml4t_results, tp_pct)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
