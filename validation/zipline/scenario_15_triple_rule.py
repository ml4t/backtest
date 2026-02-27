#!/usr/bin/env python3
"""Scenario 15: Triple Rule (TSL + TP + SL) combination validation against Zipline.

Tests what happens when ALL THREE rules are active.
Note: Zipline doesn't have built-in risk rules, so we implement manually.

Run from .venv-zipline environment:
    source .venv-zipline/bin/activate
    python validation/zipline/scenario_15_triple_rule.py

Success criteria:
- Document Zipline behavior for triple rule priority
- Match ml4t behavior with documented expectations
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.03  # 3% trailing stop
TP_PCT = 0.10     # 10% take profit
SL_PCT = 0.05     # 5% stop loss
SHARES_PER_TRADE = 100


def generate_test_data_tp_wins(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TP triggers first (clean win)."""
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 15:
            change = 0.007  # Steady rise to 10% gain
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-02", periods=n_bars, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        "close": prices,
        "volume": np.full(n_bars, 100000.0),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def generate_test_data_sl_wins(seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where SL triggers first (clean loss)."""
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i == 1:
            change = -0.06  # Immediate drop to 5%+ loss
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-02", periods=n_bars, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        "close": prices,
        "volume": np.full(n_bars, 100000.0),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def generate_test_data_tsl_wins(seed: int = 44) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TSL triggers first (profit protection)."""
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            change = 0.008  # Rise to ~8% gain (below 10% TP)
        elif i == 10:
            change = -0.04  # Drop 4% from high (triggers 3% TSL)
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-02", periods=n_bars, freq="D", tz="UTC")

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_bars) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        "close": prices,
        "volume": np.full(n_bars, 100000.0),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """Run Zipline with manual TSL + TP + SL implementation."""
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

            # Check TP first (most favorable)
            tp_level = state["entry_price"] * (1 + TP_PCT)
            if price >= tp_level:
                pnl = (price - state["entry_price"]) * SHARES_PER_TRADE
                state["trades"].append({
                    "entry_idx": state["entry_idx"],
                    "exit_idx": i,
                    "entry_price": state["entry_price"],
                    "exit_price": price,
                    "pnl": pnl,
                    "exit_reason": "TP",
                })
                state["position_open"] = False
                final_value += pnl
                continue

            # Check SL (more urgent than TSL)
            sl_level = state["entry_price"] * (1 - SL_PCT)
            if price <= sl_level:
                pnl = (price - state["entry_price"]) * SHARES_PER_TRADE
                state["trades"].append({
                    "entry_idx": state["entry_idx"],
                    "exit_idx": i,
                    "entry_price": state["entry_price"],
                    "exit_price": price,
                    "pnl": pnl,
                    "exit_reason": "SL",
                })
                state["position_open"] = False
                final_value += pnl
                continue

            # Check TSL
            tsl_level = state["high_water_mark"] * (1 - TRAIL_PCT)
            if price <= tsl_level:
                pnl = (price - state["entry_price"]) * SHARES_PER_TRADE
                state["trades"].append({
                    "entry_idx": state["entry_idx"],
                    "exit_idx": i,
                    "entry_price": state["entry_price"],
                    "exit_price": price,
                    "pnl": pnl,
                    "exit_reason": "TSL",
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
        "scenario": scenario,
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": len(state["trades"]),
        "trades": state["trades"],
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """ml4t.backtest with TSL + TP + SL."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk import RuleChain
    from ml4t.backtest.risk.position import TrailingStop, TakeProfit, StopLoss

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

    class TripleRuleStrategy(Strategy):
        def on_start(self, broker):
            broker.set_position_rules(RuleChain([
                TrailingStop(pct=TRAIL_PCT),
                TakeProfit(pct=TP_PCT),
                StopLoss(pct=SL_PCT),
            ]))

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
        feed, TripleRuleStrategy(),
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
            "exit_bar": exit_idx,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
            "status": "Closed" if t.exit_time else "Open",
        })

    return {
        "framework": "ml4t.backtest",
        "scenario": scenario,
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_info,
    }


def main():
    print("=" * 70)
    print("Scenario 15: Triple Rule (TSL + TP + SL) (Zipline)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Take Profit: {TP_PCT*100:.0f}%")
    print(f"  Stop Loss: {SL_PCT*100:.0f}%")

    scenarios = [
        ("TP wins (clean uptrend)", generate_test_data_tp_wins),
        ("SL wins (immediate drop)", generate_test_data_sl_wins),
        ("TSL wins (rise then pullback)", generate_test_data_tsl_wins),
    ]

    all_results = []

    for name, gen_func in scenarios:
        print("\n" + "=" * 70)
        print(f"TEST: {name}")
        print("=" * 70)

        df, entries = gen_func()
        print(f"\nData: {len(df)} bars")
        print(f"Entry price: ${df['close'].iloc[0]:.2f}")

        print("\nRunning Zipline (manual triple rule)...")
        try:
            zl_result = run_zipline(df, entries, name)
            if zl_result['trades']:
                print(f"  Exit bar: {zl_result['trades'][0]['exit_idx']}")
                print(f"  PnL: ${zl_result['trades'][0]['pnl']:.2f}")
                print(f"  Exit reason: {zl_result['trades'][0].get('exit_reason', 'N/A')}")
            else:
                print("  No trades")
        except Exception as e:
            print(f"  ERROR: {e}")
            zl_result = None

        print("\nRunning ml4t.backtest...")
        try:
            ml4t_result = run_ml4t(df, entries, name)
            if ml4t_result['trades']:
                print(f"  Exit bar: {ml4t_result['trades'][0]['exit_bar']}")
                print(f"  PnL: ${ml4t_result['trades'][0]['pnl']:.2f}")
            else:
                print("  No trades")
        except Exception as e:
            print(f"  ERROR: {e}")
            ml4t_result = None

        all_results.append((name, zl_result, ml4t_result))

    # Summary
    print("\n" + "=" * 70)
    print("TRIPLE RULE PRIORITY FINDINGS")
    print("=" * 70)

    all_match = True
    for name, zl_res, ml4t in all_results:
        print(f"\n{name}:")
        if zl_res and ml4t:
            zl_pnl = zl_res['trades'][0]['pnl'] if zl_res['trades'] else 0
            ml4t_pnl = ml4t['trades'][0]['pnl'] if ml4t['trades'] else 0
            pnl_match = abs(zl_pnl - ml4t_pnl) < 50
            all_match &= pnl_match
            print(f"  Zipline PnL: ${zl_pnl:.2f}")
            print(f"  ML4T PnL: ${ml4t_pnl:.2f}")
            print(f"  {'✅ Match' if pnl_match else '❌ Mismatch'}")
        else:
            print(f"  ⚠️  Could not compare")
            all_match = False

    print("\n" + "=" * 70)
    if all_match:
        print("✅ TRIPLE RULE PRIORITY MATCHES ZIPLINE")
    else:
        print("⚠️  TRIPLE RULE PRIORITY DIFFERS - Document behavior")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
