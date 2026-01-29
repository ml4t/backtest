#!/usr/bin/env python3
"""Scenario 06: Per-Share Commission validation against VectorBT Pro.

This script validates that ml4t.backtest per-share commission calculations
match VectorBT Pro when using fixed 100 shares per trade.

VectorBT doesn't have native per-share commission, so we use fixed_fees
which is equivalent for constant position sizes.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    cd validation/vectorbt_pro
    python scenario_06_commission_per_share.py

Success criteria:
- Trade count: Exact match
- Commission per trade: Exact match (within floating point tolerance)
- Final P&L: Exact match (accounting for commissions)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Per-share commission: $0.005/share (IB-like)
PER_SHARE_RATE = 0.005
SHARES_PER_TRADE = 100
# Fixed fee equivalent for VectorBT: $0.005 × 100 = $0.50 per order
FIXED_FEE_PER_ORDER = PER_SHARE_RATE * SHARES_PER_TRADE


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate identical test data for both frameworks."""
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
    """Run backtest using VectorBT Pro with fixed fee (equivalent to per-share)."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed. Run in .venv-vectorbt-pro environment.")

    pf = vbt.Portfolio.from_signals(
        close=prices_df["close"],
        entries=entries,
        exits=exits,
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,  # No percentage fee
        fixed_fees=FIXED_FEE_PER_ORDER,  # Fixed fee per order
        slippage=0.0,
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    total_fees = 0.0
    if len(trades) > 0:
        total_fees = trades["Entry Fees"].sum() + trades["Exit Fees"].sum()

    return {
        "framework": "VectorBT Pro",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "total_commission": float(total_fees),
        "num_trades": len(trades),
        "trades": trades.to_dict("records") if len(trades) > 0 else [],
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run backtest using ml4t.backtest with per-share commission."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoSlippage,
        PerShareCommission,
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
    strategy = SignalStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=PerShareCommission(per_share=PER_SHARE_RATE, minimum=0.0),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()

    total_commission = sum(f.commission for f in results["fills"])

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "total_commission": total_commission,
        "num_trades": results["num_trades"],
        "trades": [
            {
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "commission": t.commission,
            }
            for t in results["trades"]
        ],
    }


def compare_results(vbt_results: dict, ml4t_results: dict) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: VectorBT Pro vs ml4t.backtest (Per-share=${PER_SHARE_RATE})")
    print("=" * 70)

    all_match = True

    vbt_trades = vbt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = vbt_trades == ml4t_trades
    print(f"\nTrade Count: VBT={vbt_trades}, ML4T={ml4t_trades} {'✅' if trades_match else '❌'}")
    all_match &= trades_match

    vbt_comm = vbt_results["total_commission"]
    ml4t_comm = ml4t_results["total_commission"]
    comm_diff = abs(vbt_comm - ml4t_comm)
    comm_match = comm_diff < 0.01
    print(
        f"Total Commission: VBT=${vbt_comm:.2f}, ML4T=${ml4t_comm:.2f} "
        f"(diff=${comm_diff:.2f}) {'✅' if comm_match else '❌'}"
    )
    all_match &= comm_match

    vbt_value = vbt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(vbt_value - ml4t_value)
    value_pct_diff = value_diff / vbt_value * 100 if vbt_value != 0 else 0
    values_match = value_pct_diff < 0.01
    print(
        f"Final Value: VBT=${vbt_value:,.2f}, ML4T=${ml4t_value:,.2f} "
        f"(diff={value_pct_diff:.4f}%) {'✅' if values_match else '❌'}"
    )
    all_match &= values_match

    vbt_pnl = vbt_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(vbt_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 1.0
    print(
        f"Total P&L: VBT=${vbt_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} "
        f"(diff=${pnl_diff:.2f}) {'✅' if pnl_match else '❌'}"
    )
    all_match &= pnl_match

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED: Per-share commission calculations match")
    else:
        print("❌ VALIDATION FAILED: Per-share commission calculations do not match")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print(f"Scenario 06: Per-Share Commission Validation (${PER_SHARE_RATE}/share)")
    print("=" * 70)

    print("\n📊 Generating test data...")
    prices_df, entries, exits = generate_test_data(n_bars=100)
    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    print(f"   Shares per trade: {SHARES_PER_TRADE}")
    print(f"   Commission per order: ${FIXED_FEE_PER_ORDER:.2f}")

    print("\n🔷 Running VectorBT Pro...")
    try:
        vbt_results = run_vectorbt_pro(prices_df, entries, exits)
        print(f"   Trades: {vbt_results['num_trades']}")
        print(f"   Total Commission: ${vbt_results['total_commission']:.2f}")
        print(f"   Final Value: ${vbt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"   ❌ {e}")
        return 1

    print("\n🔶 Running ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Total Commission: ${ml4t_results['total_commission']:.2f}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    success = compare_results(vbt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
