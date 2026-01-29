#!/usr/bin/env python3
"""Scenario 15: Triple Rule (TSL + TP + SL) combination validation.

This tests what happens when ALL THREE rules are active:
- Trailing Stop Loss (protects profits)
- Take Profit (locks in gains)
- Stop Loss (limits losses)

Critical for understanding complex rule interactions.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_15_triple_rule.py

Success criteria:
- Document VBT Pro behavior for triple rule priority
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
    """Generate data where TP triggers first (clean win).

    Scenario: Steady rise to TP level, no drawdown.
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 15:
            # Steady rise to 10% gain
            change = 0.007  # ~10.5% gain by bar 15
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


def generate_test_data_sl_wins(seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where SL triggers first (clean loss).

    Scenario: Immediate drop to SL level, no chance for TSL or TP.
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i == 1:
            # Immediate drop to 5%+ loss
            change = -0.06
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


def generate_test_data_tsl_wins(seed: int = 44) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where TSL triggers first (profit protection).

    Scenario: Rise to 8% gain (below TP), then 3% pullback triggers TSL.
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            # Rise to ~8% gain (below 10% TP)
            change = 0.008
        elif i == 10:
            # Drop 4% from high (triggers 3% TSL)
            change = -0.04
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


def generate_test_data_all_breach(seed: int = 45) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where all three levels breach on same bar.

    Scenario: Rise to 12%, then gap down below SL (breaches all).
    - TP at $110 (10%)
    - TSL at 97% of high (~$109)
    - SL at $95 (5%)
    - Gap from $112 to $94 (breaches all)
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 8:
            # Rise to ~$112 (12% gain, above TP)
            change = 0.015
        elif i == 8:
            # Catastrophic drop (breaches all levels)
            change = -0.16  # From ~112 to ~94
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


def run_vectorbt_pro(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """VBT Pro with TSL + TP + SL."""
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
        tp_stop=TP_PCT,
        sl_stop=SL_PCT,
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
            "exit_bar": exit_idx,
            "exit_price": float(t.get("Avg Exit Price", 0)),
            "pnl": float(t.get("PnL", 0)),
            "status": str(t.get("Status", "Unknown")),
        })

    return {
        "framework": "VectorBT Pro",
        "scenario": scenario,
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "trades": trade_info,
    }


def run_ml4t(prices_df: pd.DataFrame, entries: np.ndarray, scenario: str) -> dict:
    """ml4t.backtest with TSL + TP + SL."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk import RuleChain
    from ml4t.backtest.risk.position import TrailingStop, TakeProfit, StopLoss

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

    class TripleRuleStrategy(Strategy):
        def on_start(self, broker):
            # All three rules in chain
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
    print("Scenario 15: Triple Rule (TSL + TP + SL)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Take Profit: {TP_PCT*100:.0f}%")
    print(f"  Stop Loss: {SL_PCT*100:.0f}%")

    scenarios = [
        ("TP wins (clean uptrend)", generate_test_data_tp_wins),
        ("SL wins (immediate drop)", generate_test_data_sl_wins),
        ("TSL wins (rise then pullback)", generate_test_data_tsl_wins),
        ("All levels breached (gap down)", generate_test_data_all_breach),
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
        print(f"SL level: ${df['close'].iloc[0] * (1 - SL_PCT):.2f}")

        print("\nRunning VectorBT Pro...")
        try:
            vbt_result = run_vectorbt_pro(df, entries, name)
            print(f"  Exit bar: {vbt_result['trades'][0]['exit_bar'] if vbt_result['trades'] else 'N/A'}")
            print(f"  Exit price: ${vbt_result['trades'][0]['exit_price']:.2f if vbt_result['trades'] else 0}")
            print(f"  PnL: ${vbt_result['trades'][0]['pnl']:.2f if vbt_result['trades'] else 0}")
            print(f"  Status: {vbt_result['trades'][0]['status'] if vbt_result['trades'] else 'N/A'}")
        except Exception as e:
            print(f"  ERROR: {e}")
            vbt_result = None

        print("\nRunning ml4t.backtest...")
        try:
            ml4t_result = run_ml4t(df, entries, name)
            print(f"  Exit bar: {ml4t_result['trades'][0]['exit_bar'] if ml4t_result['trades'] else 'N/A'}")
            print(f"  Exit price: ${ml4t_result['trades'][0]['exit_price']:.2f if ml4t_result['trades'] else 0}")
            print(f"  PnL: ${ml4t_result['trades'][0]['pnl']:.2f if ml4t_result['trades'] else 0}")
        except Exception as e:
            print(f"  ERROR: {e}")
            ml4t_result = None

        all_results.append((name, vbt_result, ml4t_result))

    # Summary
    print("\n" + "=" * 70)
    print("TRIPLE RULE PRIORITY FINDINGS")
    print("=" * 70)

    all_match = True
    for name, vbt, ml4t in all_results:
        print(f"\n{name}:")
        if vbt and ml4t:
            vbt_bar = vbt['trades'][0]['exit_bar'] if vbt['trades'] else None
            ml4t_bar = ml4t['trades'][0]['exit_bar'] if ml4t['trades'] else None
            vbt_status = vbt['trades'][0]['status'] if vbt['trades'] else 'N/A'
            match = vbt_bar == ml4t_bar
            all_match &= match
            print(f"  VBT: bar {vbt_bar}, status {vbt_status}")
            print(f"  ML4T: bar {ml4t_bar}")
            print(f"  {'✅ Match' if match else '❌ Mismatch'}")
        else:
            print(f"  ⚠️  Could not compare")
            all_match = False

    print("\n" + "=" * 70)
    print("\nEXPECTED BEHAVIOR (based on VBT Pro):")
    print("  1. Rules are evaluated in order: TSL -> TP -> SL")
    print("  2. Whichever triggers FIRST wins")
    print("  3. In same-bar multiple-breach: most favorable exit price fills")
    print("=" * 70)

    if all_match:
        print("✅ TRIPLE RULE PRIORITY MATCHES VBT PRO")
    else:
        print("⚠️  TRIPLE RULE PRIORITY DIFFERS - Document behavior in LIMITATIONS.md")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
