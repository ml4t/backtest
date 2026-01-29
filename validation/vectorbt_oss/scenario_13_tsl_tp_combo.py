#!/usr/bin/env python3
"""Scenario 13: Trailing Stop + Take Profit combination validation against VectorBT OSS.

Tests what happens when BOTH TSL and TP conditions can trigger.

Run from .venv-vectorbt environment:
    source .venv-vectorbt/bin/activate
    python validation/vectorbt_oss/scenario_13_tsl_tp_combo.py

Success criteria:
- Document VBT OSS behavior for TSL vs TP priority
- Match ml4t behavior with documented expectations
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
TP_PCT = 0.08     # 8% take profit
SHARES_PER_TRADE = 100


def generate_test_data_tp_first(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TP triggers first (steady rise)."""
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 12:
            change = 0.007  # Steady rise to 8% gain
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

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


def generate_test_data_tsl_first(seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TSL triggers first (rise then pullback)."""
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            change = 0.006  # Rise to 6% (below 8% TP)
        elif i == 10:
            change = -0.06  # Sharp drop triggering 5% TSL
        else:
            change = np.random.randn() * 0.002
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

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


def run_vectorbt_oss(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """Run VectorBT OSS with TSL + TP."""
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("VectorBT OSS not installed. Run in .venv-vectorbt environment.")

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
        tp_stop=TP_PCT,
        accumulate=False,
        freq="D",
    )

    trades = pf.trades.records_readable

    trade_info = []
    for idx, t in trades.iterrows():
        exit_val = t.get("Exit Index", t.get("exit_idx"))
        exit_idx = None
        if pd.notna(exit_val):
            if isinstance(exit_val, (pd.Timestamp, np.datetime64)):
                exit_idx = prices_df.index.get_loc(exit_val)
            else:
                exit_idx = int(exit_val)

        trade_info.append({
            "exit_bar": exit_idx,
            "exit_price": float(t.get("Avg Exit Price", t.get("exit_price", 0))),
            "pnl": float(t.get("PnL", t.get("pnl", 0))),
            "status": str(t.get("Status", "Unknown")),
        })

    return {
        "framework": "VectorBT OSS",
        "scenario": scenario,
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "trades": trade_info,
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """ml4t.backtest with TSL + TP for LONG positions."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk import RuleChain
    from ml4t.backtest.risk.position import TrailingStop, TakeProfit

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

    class ComboStrategy(Strategy):
        def on_start(self, broker):
            broker.set_position_rules(RuleChain([
                TrailingStop(pct=TRAIL_PCT),
                TakeProfit(pct=TP_PCT),
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
        feed, ComboStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()

    trade_info = []
    for t in results["trades"]:
        exit_idx = prices_df.index.get_loc(t.exit_time) if t.exit_time and t.exit_time in prices_df.index else None
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
    print("Scenario 13: TSL + TP Rule Combination (VBT OSS)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Take Profit: {TP_PCT*100:.0f}%")

    scenarios = [
        ("TP triggers first (steady rise)", generate_test_data_tp_first),
        ("TSL triggers first (rise then pullback)", generate_test_data_tsl_first),
    ]

    all_results = []

    for name, gen_func in scenarios:
        print("\n" + "=" * 70)
        print(f"TEST: {name}")
        print("=" * 70)

        df, entries = gen_func()
        print(f"\nData: {len(df)} bars")
        print(f"Entry price: ${df['close'].iloc[0]:.2f}")
        print(f"TP level: ${df['close'].iloc[0] * (1 + TP_PCT):.2f}")
        print(f"TSL level (from entry): ${df['close'].iloc[0] * (1 - TRAIL_PCT):.2f}")

        print("\nRunning VectorBT OSS...")
        try:
            vbt_result = run_vectorbt_oss(df, entries, name)
            if vbt_result['trades']:
                print(f"  Exit bar: {vbt_result['trades'][0]['exit_bar']}")
                print(f"  Exit price: ${vbt_result['trades'][0]['exit_price']:.2f}")
                print(f"  PnL: ${vbt_result['trades'][0]['pnl']:.2f}")
            else:
                print("  No trades")
        except Exception as e:
            print(f"  ERROR: {e}")
            vbt_result = None

        print("\nRunning ml4t.backtest...")
        try:
            ml4t_result = run_ml4t(df, entries, name)
            if ml4t_result['trades']:
                print(f"  Exit bar: {ml4t_result['trades'][0]['exit_bar']}")
                print(f"  Exit price: ${ml4t_result['trades'][0]['exit_price']:.2f}")
                print(f"  PnL: ${ml4t_result['trades'][0]['pnl']:.2f}")
            else:
                print("  No trades")
        except Exception as e:
            print(f"  ERROR: {e}")
            ml4t_result = None

        all_results.append((name, vbt_result, ml4t_result))

    # Summary
    print("\n" + "=" * 70)
    print("TSL + TP PRIORITY FINDINGS")
    print("=" * 70)

    all_match = True
    for name, vbt, ml4t in all_results:
        print(f"\n{name}:")
        if vbt and ml4t:
            vbt_bar = vbt['trades'][0]['exit_bar'] if vbt['trades'] else None
            ml4t_bar = ml4t['trades'][0]['exit_bar'] if ml4t['trades'] else None
            match = vbt_bar == ml4t_bar
            all_match &= match
            print(f"  VBT OSS: bar {vbt_bar}")
            print(f"  ML4T: bar {ml4t_bar}")
            print(f"  {'✅ Match' if match else '❌ Mismatch'}")
        else:
            print(f"  ⚠️  Could not compare")
            all_match = False

    print("\n" + "=" * 70)
    if all_match:
        print("✅ TSL + TP PRIORITY MATCHES VBT OSS")
    else:
        print("⚠️  TSL + TP PRIORITY DIFFERS - Document behavior")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
