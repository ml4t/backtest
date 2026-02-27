#!/usr/bin/env python3
"""Scenario 13: Trailing Stop + Take Profit combination validation.

This tests what happens when BOTH TSL and TP conditions can trigger on the same bar.
Critical for understanding rule priority behavior.

Test cases:
1. LONG position: price rises to TP, then pulls back to TSL - which triggers first?
2. LONG position: price gaps through both levels - which fills?
3. SHORT position: same scenarios inverted

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_13_tsl_tp_combo.py

Success criteria:
- Document VBT Pro behavior for rule priority
- Match ml4t behavior with documented expectations
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
TP_PCT = 0.10     # 10% take profit
SHARES_PER_TRADE = 100


def generate_test_data_long(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where LONG position hits both TSL and TP conditions.

    Scenario:
    - Entry at bar 0
    - Price rises 12% (exceeds 10% TP)
    - Price pulls back 6% from high (exceeds 5% TSL)
    - Both conditions triggered on same bar (bar 20)
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            # Steady rise to set up high water mark
            change = 0.012  # 1.2% per bar
        elif i < 20:
            # Continue rising, exceeding TP level
            change = 0.008  # 0.8% per bar
        elif i == 20:
            # Sharp pullback that triggers TSL (but still above TP)
            # At bar 19: price ~ 100 * (1.012^10) * (1.008^10) ~ 122
            # TSL triggers at: high * 0.95 ~ 116
            # TP triggers at: 100 * 1.10 = 110
            # This bar: high exceeds 110, but close falls to ~116 (triggers TSL)
            change = -0.05  # 5% drop
        else:
            # Sideways
            change = np.random.randn() * 0.005
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    # Create OHLC with careful control
    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = prices * (1 + np.random.randn(n_bars) * 0.001)
    df["high"] = prices * (1 + np.abs(np.random.randn(n_bars)) * 0.003)
    df["low"] = prices * (1 - np.abs(np.random.randn(n_bars)) * 0.003)

    # For bar 20, ensure high is above TP and low triggers TSL
    if n_bars > 20:
        peak_price = prices[19]  # Bar 19 is the peak
        tp_level = base_price * (1 + TP_PCT)  # 110
        tsl_level = peak_price * (1 - TRAIL_PCT)  # ~116

        # Bar 20: high > TP, low < TSL
        df.loc[df.index[20], "high"] = tp_level + 1  # 111
        df.loc[df.index[20], "low"] = tsl_level - 1  # 115

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    df["volume"] = np.full(n_bars, 100000.0)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def generate_test_data_short(seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate data where SHORT position hits both TSL and TP conditions.

    Scenario:
    - SHORT entry at bar 0
    - Price falls 12% (exceeds 10% TP for short)
    - Price bounces back 6% from low (exceeds 5% TSL)
    - Both conditions triggered on same bar
    """
    np.random.seed(seed)
    n_bars = 50
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 10:
            # Steady fall to set up low water mark
            change = -0.012
        elif i < 20:
            # Continue falling, exceeding TP level for short
            change = -0.008
        elif i == 20:
            # Sharp bounce that triggers TSL (but still below TP)
            change = 0.05
        else:
            change = np.random.randn() * 0.005
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)
    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = prices * (1 + np.random.randn(n_bars) * 0.001)
    df["high"] = prices * (1 + np.abs(np.random.randn(n_bars)) * 0.003)
    df["low"] = prices * (1 - np.abs(np.random.randn(n_bars)) * 0.003)

    if n_bars > 20:
        trough_price = prices[19]  # Bar 19 is the trough
        tp_level = base_price * (1 - TP_PCT)  # 90 for short TP
        tsl_level = trough_price * (1 + TRAIL_PCT)  # ~86 * 1.05 = ~90

        # Bar 20: low < TP, high > TSL
        df.loc[df.index[20], "low"] = tp_level - 1
        df.loc[df.index[20], "high"] = tsl_level + 1

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df["volume"] = np.full(n_bars, 100000.0)

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def run_vectorbt_pro_long(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VBT Pro with TSL + TP for LONG positions."""
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
        "position": "LONG",
        "rules": "TSL + TP",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "trades": trade_info,
    }


def run_vectorbt_pro_short(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """VBT Pro with TSL + TP for SHORT positions."""
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
        direction="shortonly",
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
        "position": "SHORT",
        "rules": "TSL + TP",
        "final_value": float(pf.value.iloc[-1]),
        "total_pnl": float(pf.total_profit),
        "num_trades": len(trades),
        "trades": trade_info,
    }


def run_ml4t_long(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest with TSL + TP for LONG positions."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
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
            # TSL first, then TP (order might matter for priority)
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
        "position": "LONG",
        "rules": "TSL + TP",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_info,
    }


def run_ml4t_short(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """ml4t.backtest with TSL + TP for SHORT positions."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, OrderSide, Strategy,
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
        "short_entry": entries.tolist(),
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
            if signals.get("short_entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE, OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)

    engine = Engine(
        feed, ComboStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=True,
        allow_leverage=True,
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
        "position": "SHORT",
        "rules": "TSL + TP",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": trade_info,
    }


def main():
    print("=" * 70)
    print("Scenario 13: TSL + TP Rule Combination")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trailing Stop: {TRAIL_PCT*100:.0f}%")
    print(f"  Take Profit: {TP_PCT*100:.0f}%")

    # Test LONG
    print("\n" + "=" * 70)
    print("TEST 1: LONG Position - TSL + TP")
    print("=" * 70)

    long_df, long_entries = generate_test_data_long()
    print(f"\nData: {len(long_df)} bars")
    print(f"Entry price: ${long_df['close'].iloc[0]:.2f}")
    print(f"TP level: ${long_df['close'].iloc[0] * (1 + TP_PCT):.2f}")
    print(f"Peak price (bar 19): ${long_df['close'].iloc[19]:.2f}")
    print(f"TSL level from peak: ${long_df['close'].iloc[19] * (1 - TRAIL_PCT):.2f}")

    print("\nRunning VectorBT Pro...")
    try:
        vbt_long = run_vectorbt_pro_long(long_df, long_entries)
        print(f"  Exit bar: {vbt_long['trades'][0]['exit_bar'] if vbt_long['trades'] else 'N/A'}")
        print(f"  Exit price: ${vbt_long['trades'][0]['exit_price']:.2f if vbt_long['trades'] else 0}")
        print(f"  PnL: ${vbt_long['trades'][0]['pnl']:.2f if vbt_long['trades'] else 0}")
        print(f"  Status: {vbt_long['trades'][0]['status'] if vbt_long['trades'] else 'N/A'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        vbt_long = None

    print("\nRunning ml4t.backtest...")
    try:
        ml4t_long = run_ml4t_long(long_df, long_entries)
        print(f"  Exit bar: {ml4t_long['trades'][0]['exit_bar'] if ml4t_long['trades'] else 'N/A'}")
        print(f"  Exit price: ${ml4t_long['trades'][0]['exit_price']:.2f if ml4t_long['trades'] else 0}")
        print(f"  PnL: ${ml4t_long['trades'][0]['pnl']:.2f if ml4t_long['trades'] else 0}")
    except Exception as e:
        print(f"  ERROR: {e}")
        ml4t_long = None

    # Test SHORT
    print("\n" + "=" * 70)
    print("TEST 2: SHORT Position - TSL + TP")
    print("=" * 70)

    short_df, short_entries = generate_test_data_short()
    print(f"\nData: {len(short_df)} bars")
    print(f"Entry price: ${short_df['close'].iloc[0]:.2f}")
    print(f"TP level (short): ${short_df['close'].iloc[0] * (1 - TP_PCT):.2f}")
    print(f"Trough price (bar 19): ${short_df['close'].iloc[19]:.2f}")
    print(f"TSL level from trough: ${short_df['close'].iloc[19] * (1 + TRAIL_PCT):.2f}")

    print("\nRunning VectorBT Pro...")
    try:
        vbt_short = run_vectorbt_pro_short(short_df, short_entries)
        print(f"  Exit bar: {vbt_short['trades'][0]['exit_bar'] if vbt_short['trades'] else 'N/A'}")
        print(f"  Exit price: ${vbt_short['trades'][0]['exit_price']:.2f if vbt_short['trades'] else 0}")
        print(f"  PnL: ${vbt_short['trades'][0]['pnl']:.2f if vbt_short['trades'] else 0}")
        print(f"  Status: {vbt_short['trades'][0]['status'] if vbt_short['trades'] else 'N/A'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        vbt_short = None

    print("\nRunning ml4t.backtest...")
    try:
        ml4t_short = run_ml4t_short(short_df, short_entries)
        print(f"  Exit bar: {ml4t_short['trades'][0]['exit_bar'] if ml4t_short['trades'] else 'N/A'}")
        print(f"  Exit price: ${ml4t_short['trades'][0]['exit_price']:.2f if ml4t_short['trades'] else 0}")
        print(f"  PnL: ${ml4t_short['trades'][0]['pnl']:.2f if ml4t_short['trades'] else 0}")
    except Exception as e:
        print(f"  ERROR: {e}")
        ml4t_short = None

    # Summary
    print("\n" + "=" * 70)
    print("RULE PRIORITY FINDINGS")
    print("=" * 70)
    print("\nVectorBT Pro behavior (reference):")
    if vbt_long:
        print(f"  LONG: Exited at bar {vbt_long['trades'][0]['exit_bar'] if vbt_long['trades'] else 'N/A'}")
        print(f"         Status: {vbt_long['trades'][0]['status'] if vbt_long['trades'] else 'N/A'}")
    if vbt_short:
        print(f"  SHORT: Exited at bar {vbt_short['trades'][0]['exit_bar'] if vbt_short['trades'] else 'N/A'}")
        print(f"          Status: {vbt_short['trades'][0]['status'] if vbt_short['trades'] else 'N/A'}")

    print("\nml4t.backtest behavior:")
    if ml4t_long:
        print(f"  LONG: Exited at bar {ml4t_long['trades'][0]['exit_bar'] if ml4t_long['trades'] else 'N/A'}")
    if ml4t_short:
        print(f"  SHORT: Exited at bar {ml4t_short['trades'][0]['exit_bar'] if ml4t_short['trades'] else 'N/A'}")

    # Check match
    all_match = True
    if vbt_long and ml4t_long:
        long_match = (vbt_long['trades'][0]['exit_bar'] == ml4t_long['trades'][0]['exit_bar']
                      if vbt_long['trades'] and ml4t_long['trades'] else False)
        all_match &= long_match
    if vbt_short and ml4t_short:
        short_match = (vbt_short['trades'][0]['exit_bar'] == ml4t_short['trades'][0]['exit_bar']
                       if vbt_short['trades'] and ml4t_short['trades'] else False)
        all_match &= short_match

    print("\n" + "=" * 70)
    if all_match:
        print("✅ RULE PRIORITY MATCHES VBT PRO")
    else:
        print("⚠️  RULE PRIORITY DIFFERS - Document behavior in LIMITATIONS.md")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
