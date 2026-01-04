#!/usr/bin/env python3
"""Analyze the remaining 0.7% mismatches in detail."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_multi_asset_data(n_bars: int, n_assets: int, seed: int = 42):
    """Generate multi-asset OHLCV data with entry signals."""
    all_data = {}
    for asset_id in range(n_assets):
        asset_rng = np.random.default_rng(seed + asset_id * 1000)
        returns = asset_rng.normal(0.0003, 0.02, n_bars)
        cumret = np.clip(np.cumsum(returns), -3, 3)
        close = 100.0 * np.exp(cumret)
        daily_vol = asset_rng.uniform(0.005, 0.015, n_bars)
        high = close * (1 + daily_vol)
        low = close * (1 - daily_vol)
        open_ = close + asset_rng.normal(0, 0.2, n_bars)
        entries = asset_rng.random(n_bars) < 0.01
        all_data[f"asset_{asset_id:03d}"] = {
            "open": open_, "high": high, "low": low, "close": close, "entries": entries,
        }
    return all_data


def run_vbt_pro(data: dict, n_bars: int, trail_pct: float = 0.03):
    import vectorbtpro as vbt
    assets = list(data.keys())
    close_df = pd.DataFrame({a: data[a]["close"] for a in assets})
    high_df = pd.DataFrame({a: data[a]["high"] for a in assets})
    low_df = pd.DataFrame({a: data[a]["low"] for a in assets})
    open_df = pd.DataFrame({a: data[a]["open"] for a in assets})
    entries_df = pd.DataFrame({a: data[a]["entries"] for a in assets})
    exits_df = pd.DataFrame({a: [False] * n_bars for a in assets})

    pf = vbt.Portfolio.from_signals(
        open=open_df, high=high_df, low=low_df, close=close_df,
        entries=entries_df, exits=exits_df, tsl_stop=trail_pct,
        init_cash=10_000_000.0, size=100.0, fees=0.001, slippage=0.0005,
        cash_sharing=True, accumulate=False,
    )

    trades = {}
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            if row.get("Status", "") == "Open":
                continue
            key = (row.get("Column", "unknown"), int(row.get("Entry Index", 0)))
            trades[key] = {
                "exit_bar": int(row.get("Exit Index", 0)),
                "entry_price": float(row.get("Avg Entry Price", 0)),
                "exit_price": float(row.get("Avg Exit Price", 0)),
                "pnl": float(row.get("PnL", 0)),
            }
    return trades


def run_ml4t(data: dict, n_bars: int, trail_pct: float = 0.03):
    from ml4t.backtest import Broker, OrderSide, TrailHwmSource, StopFillMode
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    assets = list(data.keys())
    broker = Broker(
        10_000_000.0, PercentageCommission(0.001), PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH, stop_fill_mode=StopFillMode.CLOSE_PRICE,
    )
    broker.set_position_rules(TrailingStop(pct=trail_pct))

    base_time = datetime(2020, 1, 1, 9, 30)
    for bar_idx in range(n_bars):
        ts = base_time + timedelta(days=bar_idx)
        prices = {a: data[a]["close"][bar_idx] for a in assets}
        opens = {a: data[a]["open"][bar_idx] for a in assets}
        highs = {a: data[a]["high"][bar_idx] for a in assets}
        lows = {a: data[a]["low"][bar_idx] for a in assets}
        volumes = {a: 1_000_000 for a in assets}

        broker._update_time(ts, prices, opens, highs, lows, volumes, {})
        broker._process_pending_exits()
        broker.evaluate_position_rules()
        broker._process_orders()
        broker._update_water_marks()

        for asset in assets:
            if data[asset]["entries"][bar_idx]:
                pos = broker.get_position(asset)
                if pos is None or pos.quantity == 0:
                    broker.submit_order(asset, 100.0, OrderSide.BUY)
        broker._process_orders()

    trades = {}
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
        key = (t.asset, entry_bar)
        trades[key] = {
            "exit_bar": exit_bar,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
        }
    return trades


def main():
    print("=" * 70)
    print("DETAILED ANALYSIS OF REMAINING MISMATCHES")
    print("=" * 70)

    n_bars, n_assets = 1000, 100
    data = generate_multi_asset_data(n_bars, n_assets)

    vbt_trades = run_vbt_pro(data, n_bars)
    ml4t_trades = run_ml4t(data, n_bars)

    print(f"\nTotal trades: VBT={len(vbt_trades)}, ml4t={len(ml4t_trades)}")

    # Find mismatches
    mismatches = []
    for key in vbt_trades:
        if key not in ml4t_trades:
            mismatches.append({"key": key, "type": "missing_in_ml4t", "vbt": vbt_trades[key], "ml4t": None})
            continue
        vbt = vbt_trades[key]
        ml4t = ml4t_trades[key]

        exit_bar_match = vbt["exit_bar"] == ml4t["exit_bar"]
        exit_price_match = abs(vbt["exit_price"] - ml4t["exit_price"]) < 0.01
        pnl_match = abs(vbt["pnl"] - ml4t["pnl"]) < 1.0

        if not (exit_bar_match and exit_price_match and pnl_match):
            mismatches.append({
                "key": key,
                "type": "value_mismatch",
                "vbt": vbt,
                "ml4t": ml4t,
                "exit_bar_diff": ml4t["exit_bar"] - vbt["exit_bar"],
                "exit_price_diff": ml4t["exit_price"] - vbt["exit_price"],
                "pnl_diff": ml4t["pnl"] - vbt["pnl"],
            })

    for key in ml4t_trades:
        if key not in vbt_trades:
            mismatches.append({"key": key, "type": "missing_in_vbt", "vbt": None, "ml4t": ml4t_trades[key]})

    print(f"Total mismatches: {len(mismatches)}")
    print(f"Match rate: {100 * (len(vbt_trades) - len(mismatches)) / len(vbt_trades):.2f}%")
    print()

    # Categorize mismatches
    exit_bar_mismatches = [m for m in mismatches if m["type"] == "value_mismatch" and m["exit_bar_diff"] != 0]
    price_only_mismatches = [m for m in mismatches if m["type"] == "value_mismatch" and m["exit_bar_diff"] == 0]
    missing_ml4t = [m for m in mismatches if m["type"] == "missing_in_ml4t"]
    missing_vbt = [m for m in mismatches if m["type"] == "missing_in_vbt"]

    print(f"Exit bar mismatches: {len(exit_bar_mismatches)}")
    print(f"Price-only mismatches (same exit bar): {len(price_only_mismatches)}")
    print(f"Missing in ml4t: {len(missing_ml4t)}")
    print(f"Missing in VBT: {len(missing_vbt)}")
    print()

    # Analyze exit bar mismatches
    if exit_bar_mismatches:
        print("=" * 70)
        print("EXIT BAR MISMATCHES (Different exit timing)")
        print("=" * 70)
        total_pnl_diff = 0
        for m in exit_bar_mismatches:
            asset, entry = m["key"]
            print(f"\n{asset}/entry={entry}:")
            print(f"  VBT:  exit_bar={m['vbt']['exit_bar']}, exit_px={m['vbt']['exit_price']:.4f}, pnl={m['vbt']['pnl']:.2f}")
            print(f"  ml4t: exit_bar={m['ml4t']['exit_bar']}, exit_px={m['ml4t']['exit_price']:.4f}, pnl={m['ml4t']['pnl']:.2f}")
            print(f"  Diff: exit_bar={m['exit_bar_diff']:+d}, pnl={m['pnl_diff']:+.2f}")
            total_pnl_diff += m['pnl_diff']

            # Analyze the bars around exit
            exit_vbt = m['vbt']['exit_bar']
            exit_ml4t = m['ml4t']['exit_bar']
            min_exit = min(exit_vbt, exit_ml4t)
            max_exit = max(exit_vbt, exit_ml4t)

            print(f"  Price data around exits (bars {min_exit-1} to {max_exit+1}):")
            for b in range(min_exit-1, max_exit+2):
                if 0 <= b < n_bars:
                    print(f"    Bar {b}: H={data[asset]['high'][b]:.4f}, L={data[asset]['low'][b]:.4f}, C={data[asset]['close'][b]:.4f}")

        print(f"\nTotal PnL difference from exit bar mismatches: {total_pnl_diff:+.2f}")

    # Analyze price-only mismatches
    if price_only_mismatches:
        print("\n" + "=" * 70)
        print("PRICE-ONLY MISMATCHES (Same exit bar, different price)")
        print("=" * 70)
        price_diffs = [abs(m['exit_price_diff']) for m in price_only_mismatches]
        pnl_diffs = [m['pnl_diff'] for m in price_only_mismatches]
        print(f"Count: {len(price_only_mismatches)}")
        print(f"Exit price diff: min={min(price_diffs):.6f}, max={max(price_diffs):.6f}, mean={np.mean(price_diffs):.6f}")
        print(f"PnL diff: min={min(pnl_diffs):.2f}, max={max(pnl_diffs):.2f}, sum={sum(pnl_diffs):.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_vbt_pnl = sum(t["pnl"] for t in vbt_trades.values())
    total_ml4t_pnl = sum(t["pnl"] for t in ml4t_trades.values())
    print(f"Total VBT PnL:  {total_vbt_pnl:,.2f}")
    print(f"Total ml4t PnL: {total_ml4t_pnl:,.2f}")
    print(f"PnL difference: {total_ml4t_pnl - total_vbt_pnl:+,.2f}")
    print(f"PnL diff %:     {100 * (total_ml4t_pnl - total_vbt_pnl) / abs(total_vbt_pnl):.4f}%")


if __name__ == "__main__":
    main()
