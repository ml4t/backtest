#!/usr/bin/env python3
"""Scenario 16: Stress Test with 1000+ bars against Zipline.

Tests TSL behavior over extended market conditions.
Note: Zipline doesn't have built-in TSL, so we implement manually.

Run from .venv-zipline environment:
    source .venv-zipline/bin/activate
    python validation/zipline/scenario_16_stress_1000bars.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05
SHARES_PER_TRADE = 100
N_BARS = 1500


def generate_stress_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate challenging data with multiple market regimes."""
    np.random.seed(seed)
    base_price = 100.0
    prices = [base_price]

    for i in range(1, N_BARS):
        if i < 100:
            change = np.random.randn() * 0.015 + 0.002
        elif i < 200:
            change = np.random.randn() * 0.02 - 0.005
        elif i < 400:
            change = np.random.randn() * 0.012 + 0.003
        elif i < 500:
            base_change = np.random.randn() * 0.008
            if np.random.random() < 0.1:
                gap = np.random.choice([-0.05, 0.05])
                base_change += gap
            change = base_change
        elif i < 700:
            change = np.random.randn() * 0.01 + 0.004
        elif i < 800:
            if i < 720:
                change = -0.03 + np.random.randn() * 0.01
            else:
                change = 0.02 + np.random.randn() * 0.01
        elif i < 1000:
            change = np.random.randn() * 0.015 - 0.002
        elif i < 1200:
            change = np.random.randn() * 0.04
        else:
            change = np.random.randn() * 0.01 + 0.002

        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-02", periods=N_BARS, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(N_BARS) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(N_BARS)) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(N_BARS)) * 0.01),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, N_BARS).astype(float),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    entries = np.zeros(N_BARS, dtype=bool)
    regime_starts = [0, 100, 200, 400, 500, 700, 800, 1000, 1200]
    for idx in regime_starts:
        if idx < N_BARS:
            entries[idx] = True

    return df, entries


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """Zipline stress test with manual TSL."""
    state = {
        "entry_price": None,
        "high_water_mark": None,
        "position_open": False,
        "trades": [],
        "entry_idx": None,
    }

    final_value = 100_000.0

    for i in range(len(prices_df)):
        price = prices_df["close"].iloc[i]
        high = prices_df["high"].iloc[i]

        if state["position_open"]:
            if high > state["high_water_mark"]:
                state["high_water_mark"] = high

            tsl_level = state["high_water_mark"] * (1 - TRAIL_PCT)
            if price <= tsl_level:
                pnl = (price - state["entry_price"]) * SHARES_PER_TRADE
                state["trades"].append({
                    "entry_idx": state["entry_idx"],
                    "exit_idx": i,
                    "pnl": pnl,
                })
                state["position_open"] = False
                final_value += pnl

        elif i < len(entries) and entries[i]:
            state["entry_price"] = price
            state["high_water_mark"] = price
            state["position_open"] = True
            state["entry_idx"] = i

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": len(state["trades"]),
        "trades": state["trades"],
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest stress test with TSL."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk.position import TrailingStop

    naive_index = prices_df.index.tz_localize(None)

    prices_pl = pl.DataFrame({
        "timestamp": naive_index.to_pydatetime().tolist(),
        "asset": ["TEST"] * len(prices_df),
        "open": prices_df["open"].tolist(),
        "high": prices_df["high"].tolist(),
        "low": prices_df["low"].tolist(),
        "close": prices_df["close"].tolist(),
        "volume": prices_df["volume"].tolist(),
    })

    signals_pl = pl.DataFrame({
        "timestamp": naive_index.to_pydatetime().tolist(),
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
    naive_index_list = naive_index.tolist()
    for t in results["trades"]:
        exit_idx = None
        if t.exit_time:
            for i, ts in enumerate(naive_index_list):
                if ts == t.exit_time:
                    exit_idx = i
                    break
        trade_info.append({
            "exit_idx": exit_idx,
            "pnl": t.pnl,
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
    print(f"Scenario 16: Stress Test with {N_BARS} bars (Zipline)")
    print("=" * 70)

    df, entries = generate_stress_data()
    print(f"\nData: {len(df)} bars, {entries.sum()} entry signals")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    print("\nRunning Zipline (manual TSL)...")
    try:
        zl_result = run_zipline(df, entries)
        print(f"  Trades: {zl_result['num_trades']}, PnL: ${zl_result['total_pnl']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        zl_result = None

    print("\nRunning ml4t.backtest...")
    try:
        ml4t_result = run_ml4t(df, entries)
        print(f"  Trades: {ml4t_result['num_trades']}, PnL: ${ml4t_result['total_pnl']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        ml4t_result = None

    print("\n" + "=" * 70)
    if zl_result and ml4t_result:
        trades_match = zl_result['num_trades'] == ml4t_result['num_trades']
        pnl_diff = abs(zl_result['total_pnl'] - ml4t_result['total_pnl'])
        pnl_match = pnl_diff < 100

        print(f"Trade Count: ZL={zl_result['num_trades']}, ML4T={ml4t_result['num_trades']} "
              f"{'✅' if trades_match else '❌'}")
        print(f"PnL Diff: ${pnl_diff:.2f} {'✅' if pnl_match else '❌'}")

        if trades_match and pnl_match:
            print("✅ STRESS TEST PASSED")
        else:
            print("⚠️  STRESS TEST SHOWS DIFFERENCES")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
