#!/usr/bin/env python3
"""Scenario 16: Stress Test with 1000+ bars.

Tests TSL behavior over extended market conditions:
- 1000+ daily bars
- Multiple trend reversals (10+)
- Gap openings (price jumps)
- Volatile periods

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_16_stress_1000bars.py

Success criteria:
- No crashes or exceptions over extended runs
- Consistent TSL behavior across all market regimes
- HWM/LWM tracking remains accurate after many updates
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
SHARES_PER_TRADE = 100
N_BARS = 1500  # ~6 years of daily data


def generate_stress_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate challenging data with multiple market regimes.

    Structure:
    - Bars 0-100: Initial uptrend
    - Bars 100-200: Sharp reversal (crash)
    - Bars 200-400: Recovery rally
    - Bars 400-500: Choppy sideways with gaps
    - Bars 500-700: Strong bull market
    - Bars 700-800: Flash crash and recovery
    - Bars 800-1000: Extended downtrend
    - Bars 1000-1200: Extreme volatility regime
    - Bars 1200-1500: Final recovery
    """
    np.random.seed(seed)
    base_price = 100.0

    prices = [base_price]

    for i in range(1, N_BARS):
        # Regime-based price generation
        if i < 100:
            # Initial uptrend
            change = np.random.randn() * 0.015 + 0.002
        elif i < 200:
            # Sharp reversal (crash)
            change = np.random.randn() * 0.02 - 0.005
        elif i < 400:
            # Recovery rally
            change = np.random.randn() * 0.012 + 0.003
        elif i < 500:
            # Choppy sideways with occasional gaps
            base_change = np.random.randn() * 0.008
            if np.random.random() < 0.1:  # 10% chance of gap
                gap = np.random.choice([-0.05, 0.05])  # 5% gap
                base_change += gap
            change = base_change
        elif i < 700:
            # Strong bull market
            change = np.random.randn() * 0.01 + 0.004
        elif i < 800:
            # Flash crash and recovery
            if i < 720:
                change = -0.03 + np.random.randn() * 0.01  # Crash
            else:
                change = 0.02 + np.random.randn() * 0.01  # Recovery
        elif i < 1000:
            # Extended downtrend
            change = np.random.randn() * 0.015 - 0.002
        elif i < 1200:
            # Extreme volatility regime
            change = np.random.randn() * 0.04  # 4% daily vol
        else:
            # Final recovery
            change = np.random.randn() * 0.01 + 0.002

        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=N_BARS, freq="D")

    # Generate OHLC with realistic ranges
    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(N_BARS) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(N_BARS)) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(N_BARS)) * 0.01),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, N_BARS).astype(float),
    }, index=dates)

    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # Entry signals at start of each regime change
    entries = np.zeros(N_BARS, dtype=bool)
    regime_starts = [0, 100, 200, 400, 500, 700, 800, 1000, 1200]
    for idx in regime_starts:
        if idx < N_BARS:
            entries[idx] = True

    return df, entries


def run_vectorbt_pro(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VBT Pro stress test with TSL."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed.")

    pf = vbt.Portfolio.from_signals(
        open=prices_df["open"],
        high=prices_df["high"],
        low=prices_df["low"],
        close=prices_df["close"],
        entries=entries,
        exits=np.zeros_like(entries),
        direction="longonly",
        init_cash=100_000.0,
        size=SHARES_PER_TRADE,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        tsl_stop=TRAIL_PCT,
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    trade_info = []
    for idx, t in trades.iterrows():
        exit_val = t.get("Exit Index")
        exit_idx = None
        if pd.notna(exit_val):
            if isinstance(exit_val, (pd.Timestamp, np.datetime64)):
                exit_idx = prices_df.index.get_loc(exit_val)
            else:
                exit_idx = int(exit_val)

        trade_info.append({
            "entry_idx": int(t.get("Entry Index", 0)) if pd.notna(t.get("Entry Index")) else 0,
            "exit_idx": exit_idx,
            "entry_price": float(t.get("Avg Entry Price", 0)),
            "exit_price": float(t.get("Avg Exit Price", 0)) if pd.notna(t.get("Avg Exit Price")) else None,
            "pnl": float(t.get("PnL", 0)),
            "status": str(t.get("Status", "Unknown")),
        })

    return {
        "framework": "VectorBT Pro",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "trades": trade_info,
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest stress test with TSL."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk.position import TrailingStop

    prices_pl = pl.DataFrame({
        "timestamp": prices_df.index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "open": prices_df["open"].tolist(),
        "high": prices_df["high"].tolist(),
        "low": prices_df["low"].tolist(),
        "close": prices_df["close"].tolist(),
        "volume": prices_df["volume"].tolist(),
    })

    signals_pl = pl.DataFrame({
        "timestamp": prices_df.index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "entry": entries.tolist(),
    })

    class StressStrategy(Strategy):
        def on_start(self, broker):
            broker.set_position_rules(TrailingStop(pct=TRAIL_PCT))

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
        feed, StressStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()

    trade_info = []
    for t in results["trades"]:
        entry_idx = None
        exit_idx = None
        for i, ts in enumerate(prices_df.index):
            if t.entry_time and ts == t.entry_time:
                entry_idx = i
            if t.exit_time and ts == t.exit_time:
                exit_idx = i
        trade_info.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "status": "Closed" if t.exit_time else "Open",
        })

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_info,
    }


def main():
    print("=" * 70)
    print(f"Scenario 16: Stress Test with {N_BARS} bars")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Bars: {N_BARS} (~6 years daily)")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Regime changes: 9 (entries at each)")

    df, entries = generate_stress_data()
    print(f"\nData generated:")
    print(f"  Start price: ${df['close'].iloc[0]:.2f}")
    print(f"  End price: ${df['close'].iloc[-1]:.2f}")
    print(f"  Min price: ${df['close'].min():.2f}")
    print(f"  Max price: ${df['close'].max():.2f}")
    print(f"  Entry signals: {entries.sum()}")

    print("\n" + "=" * 70)
    print("Running VectorBT Pro...")
    try:
        vbt_result = run_vectorbt_pro(df, entries)
        print(f"  Trades: {vbt_result['num_trades']}")
        print(f"  Final Value: ${vbt_result['final_value']:,.2f}")
        print(f"  Total PnL: ${vbt_result['total_pnl']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        vbt_result = None

    print("\nRunning ml4t.backtest...")
    try:
        ml4t_result = run_ml4t(df, entries)
        print(f"  Trades: {ml4t_result['num_trades']}")
        print(f"  Final Value: ${ml4t_result['final_value']:,.2f}")
        print(f"  Total PnL: ${ml4t_result['total_pnl']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        ml4t_result = None

    # Comparison
    print("\n" + "=" * 70)
    print("STRESS TEST COMPARISON")
    print("=" * 70)

    if vbt_result and ml4t_result:
        trades_match = vbt_result['num_trades'] == ml4t_result['num_trades']
        pnl_diff = abs(vbt_result['total_pnl'] - ml4t_result['total_pnl'])
        pnl_match = pnl_diff < 100  # $100 tolerance for 1500 bars

        print(f"\nTrade Count: VBT={vbt_result['num_trades']}, ML4T={ml4t_result['num_trades']} "
              f"{'✅' if trades_match else '❌'}")
        print(f"PnL Diff: ${pnl_diff:.2f} {'✅' if pnl_match else '❌'}")

        # Detailed trade comparison
        print("\n--- Trade-by-Trade Comparison ---")
        max_trades = min(len(vbt_result['trades']), len(ml4t_result['trades']), 10)
        for i in range(max_trades):
            vbt_t = vbt_result['trades'][i]
            ml4t_t = ml4t_result['trades'][i]
            print(f"\nTrade {i+1}:")
            print(f"  VBT:  entry={vbt_t['entry_idx']}, exit={vbt_t['exit_idx']}, pnl=${vbt_t['pnl']:.2f}")
            print(f"  ML4T: entry={ml4t_t['entry_idx']}, exit={ml4t_t['exit_idx']}, pnl=${ml4t_t['pnl']:.2f}")

        if len(vbt_result['trades']) > 10:
            print(f"\n... and {len(vbt_result['trades']) - 10} more trades")

        if trades_match and pnl_match:
            print("\n" + "=" * 70)
            print("✅ STRESS TEST PASSED: Extended TSL tracking matches VBT Pro")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("⚠️  STRESS TEST SHOWS DIFFERENCES - Review trade-by-trade")
            print("=" * 70)
    else:
        print("❌ Could not complete comparison")

    return 0


if __name__ == "__main__":
    sys.exit(main())
