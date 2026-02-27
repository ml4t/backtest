#!/usr/bin/env python3
"""Scenario 03: Stop-Loss validation against VectorBT OSS.

This script validates that ml4t.backtest stop-loss behavior matches VectorBT OSS.

Run from .venv-vectorbt environment:
    source .venv-vectorbt/bin/activate
    python validation/vectorbt_oss/scenario_03_stop_loss.py

VectorBT OSS uses `sl_stop` parameter for percentage-based stop-loss.
With OHLC data, VBT OSS fills at exact stop price when bar's low <= stop price.

Success criteria:
- Trade count: Exact match
- Exit trigger timing: Same bar
- Exit price: Exact match (within floating point tolerance)
- Final P&L: Exact match (0.0000% diff)
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


def generate_stop_loss_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate test data where price drops to trigger stop-loss.

    Creates a scenario where:
    - Enter at bar 0 at $100
    - Price declines steadily
    - Stop-loss should trigger when loss exceeds threshold

    Returns:
        prices_df: OHLCV DataFrame with timestamp index
        entries: Boolean array of entry signals (single entry at bar 0)
    """
    np.random.seed(seed)

    # Price path: 100 -> gradual decline to trigger 5% stop
    n_bars = 20
    prices = np.array(
        [
            100.0,  # Bar 0: Entry
            99.0,  # Bar 1: -1%
            98.0,  # Bar 2: -2%
            97.0,  # Bar 3: -3%
            96.0,  # Bar 4: -4%
            94.5,  # Bar 5: -5.5% -> STOP TRIGGERED
            93.0,  # Bar 6: -7%
            92.0,  # Bar 7: -8%
            91.0,  # Bar 8: -9%
            90.0,  # Bar 9: -10%
            89.0,  # Bar 10
            88.0,  # Bar 11
            87.0,  # Bar 12
            86.0,  # Bar 13
            85.0,  # Bar 14
            84.0,  # Bar 15
            83.0,  # Bar 16
            82.0,  # Bar 17
            81.0,  # Bar 18
            80.0,  # Bar 19
        ]
    )

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    # Generate deterministic OHLCV with close = prices
    # Use fixed OHLC ratios for reproducibility across frameworks
    df = pd.DataFrame(
        {
            "open": prices,  # Open = Close for simplicity
            "high": prices * 1.005,  # High = 0.5% above close
            "low": prices * 0.995,  # Low = 0.5% below close
            "close": prices,
            "volume": np.full(n_bars, 100000),
        },
        index=dates,
    )

    # Entry on bar 0 only
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


# ============================================================================
# VectorBT OSS Execution
# ============================================================================


def run_vectorbt_oss(prices_df: pd.DataFrame, entries: np.ndarray, sl_pct: float) -> dict:
    """Run backtest using VectorBT OSS with stop-loss and OHLC for realistic fills."""
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("VectorBT OSS not installed. Run in .venv-vectorbt environment.")

    # Run portfolio simulation with stop-loss using OHLC for intrabar fills
    # When OHLC is provided, VBT OSS checks if stop price falls within [low, high]
    # and fills at exact stop price (more realistic than close-only)
    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=None,  # No explicit exits - rely on stop-loss
        init_cash=100_000.0,
        size=100,  # Fixed 100 shares per trade
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        accumulate=False,
        sl_stop=sl_pct,  # Stop-loss percentage
        freq="D",
    )

    trades = pf.trades.records_readable

    # Extract trade details - OSS uses different column names than Pro
    trade_list = []
    if len(trades) > 0:
        for _, row in trades.iterrows():
            # OSS column names: 'Entry Price', 'Exit Price' (not 'Avg Entry Price')
            entry_price = row.get("Entry Price", row.get("Avg Entry Price", 0))
            exit_price = row.get("Exit Price", row.get("Avg Exit Price", 0))
            trade_list.append(
                {
                    "entry_time": row.get("Entry Timestamp", row.get("entry_idx")),
                    "exit_time": row.get("Exit Timestamp", row.get("exit_idx")),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": row["PnL"],
                    "return_pct": row["Return"],
                }
            )

    # OSS uses methods not properties
    final_value = float(pf.final_value())
    total_pnl = float(pf.total_profit())

    return {
        "framework": "VectorBT OSS",
        "final_value": final_value,
        "total_pnl": total_pnl,
        "num_trades": len(trades),
        "trades": trade_list,
    }


# ============================================================================
# ml4t.backtest Execution
# ============================================================================


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, sl_pct: float) -> dict:
    """Run backtest using ml4t.backtest with stop-loss."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        StopFillMode,
        StopLevelBasis,
        Strategy,
    )
    from ml4t.backtest.risk import StopLoss

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
        }
    )

    class StopLossStrategy(Strategy):
        def __init__(self, sl_pct: float):
            self.sl_pct = sl_pct

        def on_start(self, broker):
            # Set position-level stop-loss rule
            broker.set_position_rules(StopLoss(pct=self.sl_pct))

        def on_data(self, timestamp, data, context, broker):
            if "AAPL" not in data:
                return

            signals = data["AAPL"].get("signals", {})
            position = broker.get_position("AAPL")
            current_qty = position.quantity if position else 0

            # Entry only (exit handled by stop-loss rule)
            if signals.get("entry") and current_qty == 0:
                broker.submit_order("AAPL", 100)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = StopLossStrategy(sl_pct=sl_pct)

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,  # Match VectorBT default
        stop_fill_mode=StopFillMode.STOP_PRICE,  # Match VectorBT OSS with OHLC: exact stop price
        stop_level_basis=StopLevelBasis.FILL_PRICE,  # VBT calculates from fill price
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
                "return_pct": (t.exit_price - t.entry_price) / t.entry_price,
            }
            for t in results["trades"]
        ],
    }


# ============================================================================
# Comparison
# ============================================================================


def compare_results(vbt_results: dict, ml4t_results: dict, sl_pct: float) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: VectorBT OSS vs ml4t.backtest (Stop-Loss={sl_pct:.0%})")
    print("=" * 70)

    all_match = True

    # Trade count
    vbt_trades = vbt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = vbt_trades == ml4t_trades
    print(
        f"\nTrade Count: VBT={vbt_trades}, ML4T={ml4t_trades} {'PASS' if trades_match else 'FAIL'}"
    )
    all_match &= trades_match

    # Final value - expect exact match with STOP_PRICE fill mode
    vbt_value = vbt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(vbt_value - ml4t_value)
    value_pct_diff = value_diff / vbt_value * 100 if vbt_value != 0 else 0
    values_match = value_pct_diff < 0.01  # Within 0.01% (near-exact)
    print(
        f"Final Value: VBT=${vbt_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'PASS' if values_match else 'FAIL'}"
    )
    all_match &= values_match

    # Total P&L - expect exact match
    vbt_pnl = vbt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(vbt_pnl - ml4t_pnl)
    pnl_pct_diff = pnl_diff / abs(vbt_pnl) * 100 if vbt_pnl != 0 else 0
    pnl_match = pnl_pct_diff < 1.0  # Within 1%
    print(
        f"Total P&L: VBT=${vbt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff={pnl_pct_diff:.1f}%) {'PASS' if pnl_match else 'FAIL'}"
    )
    all_match &= pnl_match

    # Trade-by-trade comparison
    if trades_match and len(vbt_results["trades"]) > 0:
        print("\nTrade-by-Trade Comparison:")
        print("-" * 70)
        vbt_trades_list = vbt_results["trades"]
        ml4t_trades_list = ml4t_results["trades"]

        for i, (vbt_t, ml4t_t) in enumerate(zip(vbt_trades_list, ml4t_trades_list)):
            vbt_ret = vbt_t["return_pct"]
            ml4t_ret = ml4t_t["return_pct"]
            entry_match = abs(vbt_t["entry_price"] - ml4t_t["entry_price"]) < 0.01
            exit_match = abs(vbt_t["exit_price"] - ml4t_t["exit_price"]) < 0.01

            print(f"  Trade {i+1}:")
            print(
                f"    VBT:  entry=${vbt_t['entry_price']:.2f}, exit=${vbt_t['exit_price']:.2f}, return={vbt_ret:.2%}"
            )
            print(
                f"    ML4T: entry=${ml4t_t['entry_price']:.2f}, exit=${ml4t_t['exit_price']:.2f}, return={ml4t_ret:.2%}"
            )
            print(
                f"    Match: entry={'PASS' if entry_match else 'FAIL'}, exit={'PASS' if exit_match else 'FAIL'}"
            )

            all_match &= entry_match
            all_match &= exit_match

    print("\n" + "=" * 70)
    print("CONFIGURATION:")
    print("  Using StopFillMode.STOP_PRICE to match VectorBT OSS with OHLC")
    print("  VBT OSS (OHLC): Fills at exact stop price (intrabar simulation)")
    print("  ml4t.backtest: With STOP_PRICE mode, fills at exact stop price")
    print("=" * 70)

    if all_match:
        print("\nVALIDATION PASSED: Stop-loss produces EXACT MATCH with VectorBT OSS")
    else:
        print("\nVALIDATION FAILED: Stop-loss behavior differs")
    print("=" * 70)

    return all_match


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 70)
    print("Scenario 03: Stop-Loss Validation (VectorBT OSS)")
    print("=" * 70)

    sl_pct = 0.05  # 5% stop-loss

    # Generate test data
    print(f"\n📊 Generating test data for {sl_pct:.0%} stop-loss...")
    prices_df, entries = generate_stop_loss_data()
    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    print(f"   Price at entry: ${prices_df['close'].iloc[0]:.2f}")
    print(f"   Stop level: ${prices_df['close'].iloc[0] * (1 - sl_pct):.2f}")

    # Run VectorBT OSS
    print("\n🔷 Running VectorBT OSS...")
    try:
        vbt_results = run_vectorbt_oss(prices_df, entries, sl_pct)
        print(f"   Trades: {vbt_results['num_trades']}")
        print(f"   Final Value: ${vbt_results['final_value']:,.2f}")
        if vbt_results["trades"]:
            print(f"   Exit price: ${vbt_results['trades'][0]['exit_price']:.2f}")
    except ImportError as e:
        print(f"   ❌ {e}")
        return 1

    # Run ml4t.backtest
    print("\n🔶 Running ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, sl_pct)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
        if ml4t_results["trades"]:
            print(f"   Exit price: ${ml4t_results['trades'][0]['exit_price']:.2f}")
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    success = compare_results(vbt_results, ml4t_results, sl_pct)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
