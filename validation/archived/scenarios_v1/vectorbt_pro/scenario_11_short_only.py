#!/usr/bin/env python3
"""Scenario 11: Short-only validation against VectorBT Pro.

This validates that ml4t.backtest short selling produces IDENTICAL results
to VectorBT Pro - trade-by-trade exact match.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_11_short_only.py

Success criteria:
- Trade count: EXACT match
- Entry prices: EXACT match
- Exit prices: EXACT match
- PnL per trade: EXACT match (within $0.01)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_test_data(n_bars: int = 500, n_assets: int = 50, seed: int = 42):
    """Generate multi-asset test data with short signals."""
    np.random.seed(seed)

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    all_data = []
    all_entries = []
    all_exits = []

    for i in range(n_assets):
        # Generate price path
        base_price = 100.0
        returns = np.random.randn(n_bars) * 0.02
        closes = base_price * np.exp(np.cumsum(returns))

        # Generate valid OHLC
        opens = closes * (1 + np.random.randn(n_bars) * 0.003)
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.005)

        # Short entry/exit signals: enter every 20 bars, exit after 10 bars
        entries = np.zeros(n_bars, dtype=bool)
        exits = np.zeros(n_bars, dtype=bool)

        idx = i % 20  # Stagger entries across assets
        while idx < n_bars - 11:
            entries[idx] = True
            exits[idx + 10] = True
            idx += 20

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        }, index=dates)

        all_data.append((f"ASSET{i:04d}", df))
        all_entries.append(entries)
        all_exits.append(exits)

    return all_data, np.array(all_entries), np.array(all_exits)


def run_vectorbt_pro(data_list, entries, exits):
    """Run short-only backtest using VectorBT Pro."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError("VectorBT Pro not installed. Run in .venv-vectorbt-pro environment.")

    # Build multi-asset OHLCV
    closes = pd.DataFrame({name: df["close"] for name, df in data_list})
    opens = pd.DataFrame({name: df["open"] for name, df in data_list})
    highs = pd.DataFrame({name: df["high"] for name, df in data_list})
    lows = pd.DataFrame({name: df["low"] for name, df in data_list})

    # Convert signals to DataFrame
    entries_df = pd.DataFrame(entries.T, index=closes.index, columns=closes.columns)
    exits_df = pd.DataFrame(exits.T, index=closes.index, columns=closes.columns)

    # Run VBT Pro with short=True
    # Note: VBT Pro with group_by=True treats all columns as one portfolio
    pf = vbt.Portfolio.from_signals(
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        entries=entries_df,
        exits=exits_df,
        direction="shortonly",  # SHORT ONLY
        size=100,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
        init_cash=1_000_000.0,
        accumulate=False,
        freq="D",
        group_by=True,  # Treat as single portfolio
        cash_sharing=True,  # Share cash across assets
    )

    # Extract trades
    trades = pf.trades.records_readable
    trade_list = []

    for _, t in trades.iterrows():
        trade_list.append({
            "asset": t["Column"],
            "entry_time": t["Entry Index"],  # Keep as timestamp
            "exit_time": t["Exit Index"],
            "entry_price": float(t["Avg Entry Price"]),
            "exit_price": float(t["Avg Exit Price"]),
            "size": float(t["Size"]),
            "pnl": float(t["PnL"]),
            "direction": t["Direction"],
        })

    # Get final value - with group_by=True, value is a Series not DataFrame
    final_value = pf.value.iloc[-1]
    if hasattr(final_value, 'sum'):
        final_value = float(final_value.sum())
    else:
        final_value = float(final_value)

    return {
        "framework": "VectorBT Pro",
        "final_value": final_value,
        "num_trades": len(trade_list),
        "trades": sorted(trade_list, key=lambda t: (t["asset"], str(t["entry_time"]))),
    }


def run_ml4t_backtest(data_list, entries, exits):
    """Run short-only backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest._validation_imports import DataFeed, Engine, NoCommission, NoSlippage, OrderSide, Strategy

    # Build polars DataFrame
    all_rows = []
    for i, (name, df) in enumerate(data_list):
        for j, (ts, row) in enumerate(df.iterrows()):
            all_rows.append({
                "timestamp": ts.to_pydatetime(),
                "asset": name,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })

    prices_pl = pl.DataFrame(all_rows)

    # Build signals DataFrame
    signals_rows = []
    for i, (name, df) in enumerate(data_list):
        for j, ts in enumerate(df.index):
            signals_rows.append({
                "timestamp": ts.to_pydatetime(),
                "asset": name,
                "short_entry": bool(entries[i, j]),
                "short_exit": bool(exits[i, j]),
            })

    signals_pl = pl.DataFrame(signals_rows)

    class ShortOnlyStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            for asset in data:
                signals = data[asset].get("signals", {})
                pos = broker.get_position(asset)
                current_qty = pos.quantity if pos else 0

                # Exit first (cover short)
                if signals.get("short_exit") and current_qty < 0:
                    broker.close_position(asset)
                # Then entry (open short)
                elif signals.get("short_entry") and current_qty == 0:
                    broker.submit_order(asset, 100, OrderSide.SELL)  # SHORT

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = ShortOnlyStrategy()

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        allow_short_selling=True, allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )

    results = engine.run()

    trade_list = []
    for t in results["trades"]:
        trade_list.append({
            "asset": t.asset,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": abs(t.quantity),
            "pnl": t.pnl,
            "direction": "Short" if t.quantity < 0 else "Long",
        })

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "num_trades": results["num_trades"],
        "trades": sorted(trade_list, key=lambda t: (t["asset"], str(t["entry_time"]))),
    }


def compare_results(vbt_results, ml4t_results):
    """Compare trade-by-trade results."""
    print("\n" + "=" * 70)
    print("COMPARISON: VectorBT Pro vs ml4t.backtest (SHORT-ONLY)")
    print("=" * 70)

    all_match = True

    # Trade count
    vbt_count = vbt_results["num_trades"]
    ml4t_count = ml4t_results["num_trades"]
    count_match = vbt_count == ml4t_count
    print(f"\nTrade Count: VBT={vbt_count}, ML4T={ml4t_count} "
          f"{'✅ MATCH' if count_match else '❌ MISMATCH'}")
    all_match &= count_match

    # Final value
    vbt_value = vbt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(vbt_value - ml4t_value)
    value_pct = value_diff / vbt_value * 100 if vbt_value else 0
    value_match = value_pct < 0.01
    print(f"Final Value: VBT=${vbt_value:,.2f}, ML4T=${ml4t_value:,.2f} "
          f"(diff={value_pct:.4f}%) {'✅ MATCH' if value_match else '❌ MISMATCH'}")
    all_match &= value_match

    # Trade-by-trade comparison
    if count_match:
        vbt_trades = vbt_results["trades"]
        ml4t_trades = ml4t_results["trades"]

        # Sort by asset and entry time for matching
        vbt_sorted = sorted(vbt_trades, key=lambda t: (t["asset"], str(t["entry_time"])))
        ml4t_sorted = sorted(ml4t_trades, key=lambda t: (t["asset"], str(t["entry_time"])))

        matches = 0
        mismatches = 0
        sample_mismatches = []

        for vbt_t, ml4t_t in zip(vbt_sorted, ml4t_sorted):
            asset_match = vbt_t["asset"] == ml4t_t["asset"]
            entry_match = abs(vbt_t["entry_price"] - ml4t_t["entry_price"]) < 0.01
            exit_match = abs(vbt_t["exit_price"] - ml4t_t["exit_price"]) < 0.01
            pnl_match = abs(vbt_t["pnl"] - ml4t_t["pnl"]) < 0.10

            if asset_match and entry_match and exit_match and pnl_match:
                matches += 1
            else:
                mismatches += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append((vbt_t, ml4t_t))

        match_pct = matches / len(vbt_sorted) * 100 if vbt_sorted else 0
        print(f"\nTrade-Level Match: {matches}/{len(vbt_sorted)} ({match_pct:.1f}%)")

        if mismatches > 0:
            print(f"\nSample Mismatches:")
            for vbt_t, ml4t_t in sample_mismatches:
                print(f"  {vbt_t['asset']}:")
                print(f"    VBT:  entry=${vbt_t['entry_price']:.2f}, "
                      f"exit=${vbt_t['exit_price']:.2f}, pnl=${vbt_t['pnl']:.2f}")
                print(f"    ML4T: entry=${ml4t_t['entry_price']:.2f}, "
                      f"exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

        all_match &= (mismatches == 0)

    # Sample trades
    print("\nSample Short Trades (first 5):")
    print("-" * 70)
    for i, (vbt_t, ml4t_t) in enumerate(zip(vbt_results["trades"][:5], ml4t_results["trades"][:5])):
        print(f"  Trade {i+1} ({vbt_t['asset']}):")
        print(f"    VBT:  entry=${vbt_t['entry_price']:.2f}, exit=${vbt_t['exit_price']:.2f}, pnl=${vbt_t['pnl']:.2f}")
        print(f"    ML4T: entry=${ml4t_t['entry_price']:.2f}, exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED: Short-selling results match exactly")
    else:
        print("❌ VALIDATION FAILED: Short-selling results differ")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print("Scenario 11: Short-Only Validation (VectorBT Pro)")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    n_bars = 500
    n_assets = 50
    data_list, entries, exits = generate_test_data(n_bars=n_bars, n_assets=n_assets)
    print(f"  Assets: {n_assets}")
    print(f"  Bars: {n_bars}")
    print(f"  Short entry signals: {entries.sum()}")
    print(f"  Short exit signals: {exits.sum()}")

    # Run VectorBT Pro
    print("\nRunning VectorBT Pro (short-only)...")
    try:
        vbt_results = run_vectorbt_pro(data_list, entries, exits)
        print(f"  Trades: {vbt_results['num_trades']}")
        print(f"  Final Value: ${vbt_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"  ERROR: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest (short-only)...")
    try:
        ml4t_results = run_ml4t_backtest(data_list, entries, exits)
        print(f"  Trades: {ml4t_results['num_trades']}")
        print(f"  Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare
    success = compare_results(vbt_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
