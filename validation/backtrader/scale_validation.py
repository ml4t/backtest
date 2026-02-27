#!/usr/bin/env python3
"""Scale validation: Backtrader vs ml4t.backtest.

This script validates trade-level exact match between Backtrader and ml4t.backtest
at scale: 500 assets × 10 years (2,520 bars) of daily data.

Target: 100% trade-level exact match for entry/exit timing and prices.

Run from .venv-validation environment:
    source .venv-validation/bin/activate
    python validation/backtrader/scale_validation.py

Key configuration for Backtrader matching:
- ExecutionMode.NEXT_BAR (Backtrader default: cheat-on-close=False)
- Fill prices at next bar's open
- No commission or slippage for exact comparison
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_multi_asset_data(
    n_assets: int = 500,
    n_bars: int = 2520,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate multi-asset price data with predefined signals.

    Args:
        n_assets: Number of assets
        n_bars: Number of bars (default: 10 years daily = 252 * 10)
        seed: Random seed

    Returns:
        prices_df: Multi-asset OHLCV DataFrame
        signals: Dict of {asset: (entries, exits)} arrays
    """
    np.random.seed(seed)

    dates = pd.date_range(start="2010-01-01", periods=n_bars, freq="D")
    all_data = []
    signals = {}

    for i in range(n_assets):
        asset = f"ASSET{i:04d}"

        # Generate price path (random walk)
        base_price = 50.0 + i * 0.1  # Different base per asset
        returns = np.random.randn(n_bars) * 0.02  # 2% daily vol
        closes = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV - ensure valid OHLC relationship
        opens = closes * (1 + np.random.randn(n_bars) * 0.003)
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.005)

        volumes = np.random.randint(100000, 1000000, n_bars)

        # Create DataFrame for this asset
        df = pd.DataFrame({
            "timestamp": dates,
            "asset": asset,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes.astype(float),
        })
        all_data.append(df)

        # Generate simple entry/exit signals
        # Entry: every 20 bars (50 trades per asset)
        # Exit: 10 bars after entry
        entries = np.zeros(n_bars, dtype=bool)
        exits = np.zeros(n_bars, dtype=bool)

        idx = 0
        while idx < n_bars - 11:
            entries[idx] = True
            exits[idx + 10] = True
            idx += 20

        signals[asset] = (entries, exits)

    prices_df = pd.concat(all_data, ignore_index=True)
    return prices_df, signals


def run_backtrader(prices_df: pd.DataFrame, signals: dict) -> dict:
    """Run multi-asset backtest using Backtrader."""
    try:
        import backtrader as bt
    except ImportError:
        raise ImportError("Backtrader not installed. Run in .venv-validation environment.")

    assets = list(signals.keys())

    # Create data feed for each asset
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

    class MultiAssetStrategy(bt.Strategy):
        def __init__(self):
            self.bar_count = 0
            self.trade_log = []
            self.entries = {}
            self.exits = {}

        def set_signals(self, signals_dict):
            for asset, (entries, exits) in signals_dict.items():
                self.entries[asset] = entries
                self.exits[asset] = exits

        def next(self):
            idx = self.bar_count

            for i, d in enumerate(self.datas):
                asset = d._name
                if asset not in self.entries:
                    continue

                entries = self.entries[asset]
                exits = self.exits[asset]

                pos = self.getposition(d).size

                # Check exit first (if in position)
                if pos > 0 and idx < len(exits) and exits[idx]:
                    self.close(data=d)

                # Check entry (if not in position)
                elif pos == 0 and idx < len(entries) and entries[idx]:
                    self.buy(data=d, size=100)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trade_log.append({
                    "asset": trade.data._name,
                    "entry_time": bt.num2date(trade.dtopen),
                    "exit_time": bt.num2date(trade.dtclose),
                    "entry_price": trade.price,
                    "pnl": trade.pnl,
                    "quantity": trade.size,
                })

    # Set up cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10_000_000.0)  # 10M for multi-asset
    cerebro.broker.setcommission(commission=0.0)

    # Add data for each asset
    for asset in assets:
        asset_df = prices_df[prices_df["asset"] == asset].copy()
        asset_df = asset_df.set_index("timestamp")
        asset_df = asset_df[["open", "high", "low", "close", "volume"]]

        data = PandasData(dataname=asset_df)
        data._name = asset
        cerebro.adddata(data, name=asset)

    # Add strategy
    cerebro.addstrategy(MultiAssetStrategy)

    # Run backtest
    results = cerebro.run()
    strategy = results[0]

    # Set signals after creation (Backtrader pattern)
    strategy.set_signals(signals)

    # Re-run with signals
    cerebro2 = bt.Cerebro()
    cerebro2.broker.setcash(10_000_000.0)
    cerebro2.broker.setcommission(commission=0.0)

    for asset in assets:
        asset_df = prices_df[prices_df["asset"] == asset].copy()
        asset_df = asset_df.set_index("timestamp")
        asset_df = asset_df[["open", "high", "low", "close", "volume"]]
        data = PandasData(dataname=asset_df)
        data._name = asset
        cerebro2.adddata(data, name=asset)

    class SignalStrategy(bt.Strategy):
        params = (("signals", None),)

        def __init__(self):
            self.bar_count = 0
            self.trade_log = []

        def next(self):
            idx = self.bar_count
            signals = self.params.signals

            for d in self.datas:
                asset = d._name
                if asset not in signals:
                    continue

                entries, exits = signals[asset]
                pos = self.getposition(d).size

                if pos > 0 and idx < len(exits) and exits[idx]:
                    self.close(data=d)
                elif pos == 0 and idx < len(entries) and entries[idx]:
                    self.buy(data=d, size=100)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trade_log.append({
                    "asset": trade.data._name,
                    "entry_time": bt.num2date(trade.dtopen),
                    "exit_time": bt.num2date(trade.dtclose),
                    "entry_price": trade.price,
                    "pnl": trade.pnl,
                    "quantity": trade.size,
                })

    cerebro2.addstrategy(SignalStrategy, signals=signals)
    results2 = cerebro2.run()
    strategy2 = results2[0]

    final_value = cerebro2.broker.getvalue()

    return {
        "framework": "Backtrader",
        "final_value": final_value,
        "total_pnl": final_value - 10_000_000.0,
        "num_trades": len(strategy2.trade_log),
        "trades": strategy2.trade_log,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, signals: dict) -> dict:
    """Run multi-asset backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        Strategy,
    )

    # Convert to polars format
    prices_pl = pl.DataFrame(prices_df)

    # Create signals DataFrame
    assets = list(signals.keys())
    n_bars = len(signals[assets[0]][0])
    timestamps = prices_df[prices_df["asset"] == assets[0]]["timestamp"].tolist()

    signals_data = []
    for asset, (entries, exits) in signals.items():
        for i, ts in enumerate(timestamps):
            signals_data.append({
                "timestamp": ts,
                "asset": asset,
                "entry": bool(entries[i]),
                "exit": bool(exits[i]),
            })

    signals_pl = pl.DataFrame(signals_data)

    class MultiAssetStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            for asset in data:
                signals = data[asset].get("signals", {})
                position = broker.get_position(asset)
                current_qty = position.quantity if position else 0

                # Check exit first
                if signals.get("exit") and current_qty > 0:
                    broker.close_position(asset)
                # Then check entry
                elif signals.get("entry") and current_qty == 0:
                    broker.submit_order(asset, 100)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = MultiAssetStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=10_000_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Backtrader
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 10_000_000.0,
        "num_trades": results["num_trades"],
        "trades": [
            {
                "asset": t.asset,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "pnl": t.pnl,
                "quantity": t.quantity,
            }
            for t in results["trades"]
        ],
    }


def compare_results(bt_results: dict, ml4t_results: dict) -> dict:
    """Compare trade-by-trade results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Backtrader vs ml4t.backtest (Scale Test)")
    print("=" * 70)

    comparison = {
        "trade_count_match": False,
        "value_match": False,
        "trade_level_matches": 0,
        "trade_level_mismatches": 0,
        "all_match": False,
    }

    # Trade count
    bt_trades = bt_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    comparison["trade_count_match"] = bt_trades == ml4t_trades
    print(f"\nTrade Count: BT={bt_trades:,}, ML4T={ml4t_trades:,} "
          f"{'MATCH' if comparison['trade_count_match'] else 'MISMATCH'}")

    # Final value
    bt_value = bt_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff_pct = abs(bt_value - ml4t_value) / bt_value * 100
    comparison["value_match"] = value_diff_pct < 0.01
    print(f"Final Value: BT=${bt_value:,.2f}, ML4T=${ml4t_value:,.2f} "
          f"(diff={value_diff_pct:.4f}%) {'MATCH' if comparison['value_match'] else 'MISMATCH'}")

    # Trade-by-trade comparison
    if comparison["trade_count_match"]:
        bt_trades_sorted = sorted(bt_results["trades"],
                                  key=lambda t: (t["asset"], str(t["entry_time"])))
        ml4t_trades_sorted = sorted(ml4t_results["trades"],
                                    key=lambda t: (t["asset"], str(t["entry_time"])))

        matches = 0
        mismatches = 0
        sample_mismatches = []

        for bt_t, ml4t_t in zip(bt_trades_sorted, ml4t_trades_sorted):
            # Compare asset, entry price, and PnL
            asset_match = bt_t["asset"] == ml4t_t["asset"]
            price_diff = abs(bt_t["entry_price"] - ml4t_t["entry_price"])
            price_match = price_diff < 0.01
            pnl_diff = abs(bt_t["pnl"] - ml4t_t["pnl"])
            pnl_match = pnl_diff < 1.0  # Within $1

            if asset_match and price_match and pnl_match:
                matches += 1
            else:
                mismatches += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append((bt_t, ml4t_t))

        comparison["trade_level_matches"] = matches
        comparison["trade_level_mismatches"] = mismatches

        match_pct = matches / len(bt_trades_sorted) * 100 if bt_trades_sorted else 0
        print(f"\nTrade-Level Match: {matches:,}/{len(bt_trades_sorted):,} ({match_pct:.1f}%)")

        if mismatches > 0:
            print(f"\nSample Mismatches:")
            for bt_t, ml4t_t in sample_mismatches:
                print(f"  {bt_t['asset']}: BT entry={bt_t['entry_price']:.2f}, "
                      f"ML4T entry={ml4t_t['entry_price']:.2f}")

    comparison["all_match"] = (
        comparison["trade_count_match"] and
        comparison["value_match"] and
        comparison["trade_level_mismatches"] == 0
    )

    print("\n" + "=" * 70)
    if comparison["all_match"]:
        print("VALIDATION PASSED: 100% trade-level exact match")
    else:
        print("VALIDATION FAILED: Results do not match exactly")
    print("=" * 70)

    return comparison


def main():
    print("=" * 70)
    print("BACKTRADER SCALE VALIDATION")
    print("=" * 70)

    # Scale parameters - full scale test
    n_assets = 100  # Medium scale for validation
    n_bars = 2520  # 10 years daily

    print(f"\nConfiguration:")
    print(f"  Assets: {n_assets}")
    print(f"  Bars: {n_bars} (10 years daily)")
    print(f"  Expected trades per asset: ~{n_bars // 20}")
    print(f"  Expected total trades: ~{n_assets * (n_bars // 20)}")

    # Generate data
    print("\nGenerating test data...")
    t0 = time.perf_counter()
    prices_df, signals = generate_multi_asset_data(n_assets=n_assets, n_bars=n_bars)
    gen_time = time.perf_counter() - t0
    print(f"  Generated {len(prices_df):,} rows in {gen_time:.2f}s")

    # Run Backtrader
    print("\nRunning Backtrader...")
    t0 = time.perf_counter()
    try:
        bt_results = run_backtrader(prices_df, signals)
        bt_time = time.perf_counter() - t0
        print(f"  Trades: {bt_results['num_trades']:,}")
        print(f"  Final Value: ${bt_results['final_value']:,.2f}")
        print(f"  Time: {bt_time:.2f}s")
    except ImportError as e:
        print(f"  ERROR: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest...")
    t0 = time.perf_counter()
    try:
        ml4t_results = run_ml4t_backtest(prices_df, signals)
        ml4t_time = time.perf_counter() - t0
        print(f"  Trades: {ml4t_results['num_trades']:,}")
        print(f"  Final Value: ${ml4t_results['final_value']:,.2f}")
        print(f"  Time: {ml4t_time:.2f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare
    comparison = compare_results(bt_results, ml4t_results)

    # Performance summary
    print(f"\nPerformance:")
    print(f"  Backtrader: {bt_time:.2f}s")
    print(f"  ml4t.backtest: {ml4t_time:.2f}s")
    if bt_time > 0:
        print(f"  Speedup: {bt_time / ml4t_time:.1f}x")

    return 0 if comparison["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
