#!/usr/bin/env python3
"""Scale test: ml4t.backtest vs VBT Pro with trailing stops.

Tests:
- 100 assets
- 100K bars (1000 bars per asset)
- Trailing stop loss
- Trade-by-trade comparison

Goal: 100% match on entry/exit bars, prices, and PnL.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Normalized trade record for comparison."""
    asset: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    pnl: float
    size: float


def generate_multi_asset_data(n_bars: int, n_assets: int, seed: int = 42):
    """Generate multi-asset OHLCV data with entry signals.

    IMPORTANT: Generates VALID OHLC data where:
    - low <= open <= high
    - low <= close <= high
    This is required for accurate comparison with VBT Pro.
    """
    rng = np.random.default_rng(seed)

    # Per-asset data
    all_data = {}

    for asset_id in range(n_assets):
        # Unique seed per asset
        asset_rng = np.random.default_rng(seed + asset_id * 1000)

        # Random walk price
        returns = asset_rng.normal(0.0003, 0.02, n_bars)
        cumret = np.clip(np.cumsum(returns), -3, 3)
        close = 100.0 * np.exp(cumret)

        # OHLC from close
        daily_vol = asset_rng.uniform(0.005, 0.015, n_bars)
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)

        # Generate VALID open prices: must be within [low, high] range
        # Use close plus small noise, then clamp to valid range
        open_raw = close + asset_rng.normal(0, 0.2, n_bars)
        open_ = np.clip(open_raw, low, high)

        # Entry signals: ~1% probability per bar
        entries = asset_rng.random(n_bars) < 0.01

        all_data[f"asset_{asset_id:03d}"] = {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "entries": entries,
        }

    return all_data


def run_vbt_pro(data: dict, n_bars: int, trail_pct: float = 0.03) -> list[TradeRecord]:
    """Run VBT Pro and extract trades."""
    import vectorbtpro as vbt

    assets = list(data.keys())
    n_assets = len(assets)

    # Build DataFrames with one column per asset
    close_df = pd.DataFrame({a: data[a]["close"] for a in assets})
    high_df = pd.DataFrame({a: data[a]["high"] for a in assets})
    low_df = pd.DataFrame({a: data[a]["low"] for a in assets})
    open_df = pd.DataFrame({a: data[a]["open"] for a in assets})
    entries_df = pd.DataFrame({a: data[a]["entries"] for a in assets})
    exits_df = pd.DataFrame({a: [False] * n_bars for a in assets})

    print(f"VBT Pro: {n_bars:,} bars × {n_assets} assets = {n_bars * n_assets:,} data points")

    start = time.perf_counter()

    pf = vbt.Portfolio.from_signals(
        open=open_df,
        high=high_df,
        low=low_df,
        close=close_df,
        entries=entries_df,
        exits=exits_df,
        tsl_stop=trail_pct,
        init_cash=10_000_000.0,  # Large cash to avoid rejections
        size=100.0,
        fees=0.001,
        slippage=0.0005,
        cash_sharing=True,
        accumulate=False,
    )

    elapsed = time.perf_counter() - start
    print(f"VBT Pro completed in {elapsed:.2f}s")

    # Extract trades
    trades = []
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            # Only count completed trades
            status = row.get("Status", "")
            if status == "Open":
                continue

            asset = row.get("Column", "unknown")
            entry_idx = int(row.get("Entry Index", 0))
            exit_idx = int(row.get("Exit Index", 0))
            entry_price = float(row.get("Avg Entry Price", 0))
            exit_price = float(row.get("Avg Exit Price", 0))
            pnl = float(row.get("PnL", 0))
            size = float(row.get("Size", 0))

            trades.append(TradeRecord(
                asset=asset,
                entry_bar=entry_idx,
                exit_bar=exit_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                size=size,
            ))

    print(f"VBT Pro trades: {len(trades):,}")
    return trades


def run_ml4t(data: dict, n_bars: int, trail_pct: float = 0.03) -> list[TradeRecord]:
    """Run ml4t.backtest and extract trades."""
    from ml4t.backtest import Broker, OrderSide, Strategy, TrailHwmSource, StopFillMode, InitialHwmSource
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    assets = list(data.keys())
    n_assets = len(assets)

    print(f"ml4t.backtest: {n_bars:,} bars × {n_assets} assets = {n_bars * n_assets:,} data points")

    # Create broker with VBT Pro-compatible settings
    # CRITICAL for VBT Pro matching (ALL via configuration):
    # - trail_hwm_source: HIGH (VBT Pro updates HWM from bar highs)
    # - initial_hwm_source: BAR_CLOSE (VBT Pro uses bar close for initial HWM)
    # - stop_fill_mode: STOP_PRICE (VBT Pro fills at trail level)
    # - Gap-through: handled in TrailingStop (fill at open when gap through stop)
    # See framework_behavior_reference.md for details
    broker = Broker(
        10_000_000.0,  # Same large cash
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH,  # VBT Pro updates HWM from HIGH
        initial_hwm_source=InitialHwmSource.BAR_CLOSE,  # VBT Pro uses bar CLOSE for initial HWM
        stop_fill_mode=StopFillMode.STOP_PRICE,  # VBT Pro fills at trail level
    )
    broker.set_position_rules(TrailingStop(pct=trail_pct))

    start = time.perf_counter()

    # Simulate bar by bar
    base_time = datetime(2020, 1, 1, 9, 30)
    for bar_idx in range(n_bars):
        ts = base_time + timedelta(days=bar_idx)

        # Build price dicts for this bar
        prices = {}
        opens = {}
        highs = {}
        lows = {}
        volumes = {}

        for asset in assets:
            prices[asset] = data[asset]["close"][bar_idx]
            opens[asset] = data[asset]["open"][bar_idx]
            highs[asset] = data[asset]["high"][bar_idx]
            lows[asset] = data[asset]["low"][bar_idx]
            volumes[asset] = 1_000_000

        # Update broker state
        broker._update_time(
            timestamp=ts,
            prices=prices,
            opens=opens,
            highs=highs,
            lows=lows,
            volumes=volumes,
            signals={},
        )

        # Process pending exits
        broker._process_pending_exits()

        # Evaluate position rules (trailing stops) using HWM from previous bar
        broker.evaluate_position_rules()

        # Process exit orders first
        broker._process_orders()

        # Check entry signals and submit orders
        for asset in assets:
            if data[asset]["entries"][bar_idx]:
                pos = broker.get_position(asset)
                if pos is None or pos.quantity == 0:
                    broker.submit_order(asset, 100.0, OrderSide.BUY)

        # Process entry orders
        broker._process_orders()

        # Update water marks at END of bar, AFTER all orders processed
        # VBT Pro behavior: new positions keep CLOSE as HWM on entry bar,
        # HWM is only updated from HIGHs starting on the NEXT bar after entry
        broker._update_water_marks()

    elapsed = time.perf_counter() - start
    print(f"ml4t.backtest completed in {elapsed:.2f}s")

    # Extract trades
    trades = []
    for t in broker.trades:
        # Convert timestamp to bar index
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1

        trades.append(TradeRecord(
            asset=t.asset,
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            pnl=t.pnl,
            size=t.quantity,
        ))

    print(f"ml4t.backtest trades: {len(trades):,}")
    return trades


def compare_trades(vbt_trades: list[TradeRecord], ml4t_trades: list[TradeRecord]) -> dict:
    """Compare trades between implementations."""
    # Sort both by (asset, entry_bar)
    vbt_sorted = sorted(vbt_trades, key=lambda t: (t.asset, t.entry_bar))
    ml4t_sorted = sorted(ml4t_trades, key=lambda t: (t.asset, t.entry_bar))

    results = {
        "vbt_count": len(vbt_trades),
        "ml4t_count": len(ml4t_trades),
        "count_match": len(vbt_trades) == len(ml4t_trades),
        "entry_bar_matches": 0,
        "exit_bar_matches": 0,
        "entry_price_matches": 0,
        "exit_price_matches": 0,
        "pnl_matches": 0,
        "full_matches": 0,
        "mismatches": [],
    }

    # Match trades by (asset, entry_bar)
    vbt_by_key = {(t.asset, t.entry_bar): t for t in vbt_sorted}
    ml4t_by_key = {(t.asset, t.entry_bar): t for t in ml4t_sorted}

    all_keys = set(vbt_by_key.keys()) | set(ml4t_by_key.keys())

    for key in sorted(all_keys):
        vbt_t = vbt_by_key.get(key)
        ml4t_t = ml4t_by_key.get(key)

        if vbt_t is None:
            results["mismatches"].append(f"ml4t has trade {key} not in VBT")
            continue
        if ml4t_t is None:
            results["mismatches"].append(f"VBT has trade {key} not in ml4t")
            continue

        # Compare fields
        entry_bar_match = vbt_t.entry_bar == ml4t_t.entry_bar
        exit_bar_match = vbt_t.exit_bar == ml4t_t.exit_bar
        entry_price_match = abs(vbt_t.entry_price - ml4t_t.entry_price) < 0.01
        exit_price_match = abs(vbt_t.exit_price - ml4t_t.exit_price) < 0.01
        pnl_match = abs(vbt_t.pnl - ml4t_t.pnl) < 1.0  # $1 tolerance

        if entry_bar_match:
            results["entry_bar_matches"] += 1
        if exit_bar_match:
            results["exit_bar_matches"] += 1
        if entry_price_match:
            results["entry_price_matches"] += 1
        if exit_price_match:
            results["exit_price_matches"] += 1
        if pnl_match:
            results["pnl_matches"] += 1

        if entry_bar_match and exit_bar_match and entry_price_match and exit_price_match and pnl_match:
            results["full_matches"] += 1
        else:
            if len(results["mismatches"]) < 10:  # Limit output
                results["mismatches"].append(
                    f"{key}: VBT(exit={vbt_t.exit_bar}, exit_px={vbt_t.exit_price:.2f}, pnl={vbt_t.pnl:.2f}) "
                    f"vs ml4t(exit={ml4t_t.exit_bar}, exit_px={ml4t_t.exit_price:.2f}, pnl={ml4t_t.pnl:.2f})"
                )

    return results


def main():
    print("=" * 80)
    print("SCALE TEST: ml4t.backtest vs VBT Pro with Trailing Stops")
    print("=" * 80)

    # Configuration
    n_bars = 1000  # bars per asset
    n_assets = 100  # number of assets
    trail_pct = 0.03  # 3% trailing stop

    print(f"\nConfiguration:")
    print(f"  Bars per asset: {n_bars:,}")
    print(f"  Assets: {n_assets}")
    print(f"  Total data points: {n_bars * n_assets:,}")
    print(f"  Trailing stop: {trail_pct:.1%}")
    print()

    # Generate data
    print("Generating data...")
    data = generate_multi_asset_data(n_bars, n_assets)
    print(f"Generated {len(data)} assets")
    print()

    # Run VBT Pro
    print("-" * 40)
    vbt_trades = run_vbt_pro(data, n_bars, trail_pct)
    print()

    # Run ml4t.backtest
    print("-" * 40)
    ml4t_trades = run_ml4t(data, n_bars, trail_pct)
    print()

    # Compare
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    results = compare_trades(vbt_trades, ml4t_trades)

    print(f"\nTrade counts:")
    print(f"  VBT Pro: {results['vbt_count']:,}")
    print(f"  ml4t.backtest: {results['ml4t_count']:,}")
    print(f"  Count match: {'✅' if results['count_match'] else '❌'}")

    if results['vbt_count'] > 0:
        n_compared = min(results['vbt_count'], results['ml4t_count'])
        print(f"\nField matches (out of {n_compared:,} comparable trades):")
        print(f"  Entry bar: {results['entry_bar_matches']:,} ({100*results['entry_bar_matches']/n_compared:.1f}%)")
        print(f"  Exit bar: {results['exit_bar_matches']:,} ({100*results['exit_bar_matches']/n_compared:.1f}%)")
        print(f"  Entry price: {results['entry_price_matches']:,} ({100*results['entry_price_matches']/n_compared:.1f}%)")
        print(f"  Exit price: {results['exit_price_matches']:,} ({100*results['exit_price_matches']/n_compared:.1f}%)")
        print(f"  PnL: {results['pnl_matches']:,} ({100*results['pnl_matches']/n_compared:.1f}%)")
        print(f"\n  FULL MATCHES: {results['full_matches']:,} ({100*results['full_matches']/n_compared:.1f}%)")

    if results["mismatches"]:
        print(f"\nSample mismatches (first {len(results['mismatches'])}):")
        for m in results["mismatches"]:
            print(f"  {m}")

    # Final verdict
    print("\n" + "=" * 80)
    if results['count_match'] and results['full_matches'] == results['vbt_count']:
        print("✅ 100% EXACT MATCH")
    else:
        match_pct = 100 * results['full_matches'] / max(results['vbt_count'], 1)
        print(f"❌ PARTIAL MATCH: {match_pct:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
