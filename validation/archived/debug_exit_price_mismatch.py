#!/usr/bin/env python3
"""Debug small exit price differences for asset_000/117."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_asset_000_data(seed: int = 42):
    """Generate data matching asset_000 from scale test."""
    rng = np.random.default_rng(seed + 0 * 1000)  # asset_id = 0
    n_bars = 1000

    returns = rng.normal(0.0003, 0.02, n_bars)
    cumret = np.clip(np.cumsum(returns), -3, 3)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.2, n_bars)

    entries = rng.random(n_bars) < 0.01

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "entries": entries,
        "n_bars": n_bars,
    }


def run_vbt(data, trail_pct=0.03, target_entry=117):
    """Run VBT Pro and get specific trade."""
    import vectorbtpro as vbt

    n_bars = data["n_bars"]

    pf = vbt.Portfolio.from_signals(
        open=pd.Series(data["open"]),
        high=pd.Series(data["high"]),
        low=pd.Series(data["low"]),
        close=pd.Series(data["close"]),
        entries=pd.Series(data["entries"]),
        exits=pd.Series([False] * n_bars),
        tsl_stop=trail_pct,
        init_cash=10_000_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
        accumulate=False,
    )

    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            if row.get("Status", "") == "Open":
                continue
            if int(row['Entry Index']) == target_entry:
                return {
                    "entry_bar": int(row['Entry Index']),
                    "exit_bar": int(row['Exit Index']),
                    "entry_price": float(row['Avg Entry Price']),
                    "exit_price": float(row['Avg Exit Price']),
                    "pnl": float(row['PnL']),
                }
    return None


def run_ml4t_with_trace(data, trail_pct=0.03, target_entry=117):
    """Run ml4t with HWM trace for specific trade."""
    from ml4t.backtest._validation_imports import Broker, OrderSide, TrailHwmSource, StopFillMode
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    n_bars = data["n_bars"]

    broker = Broker(
        10_000_000.0,
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH,
        stop_fill_mode=StopFillMode.CLOSE_PRICE,
    )
    broker.set_position_rules(TrailingStop(pct=trail_pct))

    base_time = datetime(2020, 1, 1, 9, 30)
    hwm_log = []
    target_pos_active = False

    for bar_idx in range(n_bars):
        ts = base_time + timedelta(days=bar_idx)

        broker._update_time(
            timestamp=ts,
            prices={"TEST": data["close"][bar_idx]},
            opens={"TEST": data["open"][bar_idx]},
            highs={"TEST": data["high"][bar_idx]},
            lows={"TEST": data["low"][bar_idx]},
            volumes={"TEST": 1_000_000},
            signals={},
        )

        broker._process_pending_exits()
        broker.evaluate_position_rules()
        broker._process_orders()

        # Log HWM for target position
        pos = broker.get_position("TEST")
        if pos and pos.quantity > 0:
            entry_bar = (pos.entry_time - base_time).days
            if entry_bar == target_entry:
                target_pos_active = True
                trail_level = pos.high_water_mark * (1 - trail_pct)
                hwm_log.append({
                    "bar": bar_idx,
                    "close": data["close"][bar_idx],
                    "high": data["high"][bar_idx],
                    "low": data["low"][bar_idx],
                    "hwm": pos.high_water_mark,
                    "trail_level": trail_level,
                    "triggered": data["low"][bar_idx] <= trail_level,
                })
        elif target_pos_active:
            # Position just closed
            break

        if data["entries"][bar_idx]:
            pos = broker.get_position("TEST")
            if pos is None or pos.quantity == 0:
                broker.submit_order("TEST", 100.0, OrderSide.BUY)

        broker._process_orders()

    # Get trade
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        if entry_bar == target_entry:
            exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
            return {
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
            }, hwm_log
    return None, hwm_log


def main():
    print("=" * 70)
    print("DEBUG: Exit Price Mismatch for asset_000/117")
    print("=" * 70)
    print()

    data = generate_asset_000_data()
    target_entry = 117

    print(f"Entry at bar {target_entry}? {data['entries'][target_entry]}")
    print()

    # Run VBT
    vbt_trade = run_vbt(data, target_entry=target_entry)
    print("=== VBT Pro ===")
    if vbt_trade:
        print(f"entry={vbt_trade['entry_bar']}, exit={vbt_trade['exit_bar']}")
        print(f"entry_px={vbt_trade['entry_price']:.6f}, exit_px={vbt_trade['exit_price']:.6f}")
        print(f"pnl={vbt_trade['pnl']:.4f}")
    print()

    # Run ml4t
    ml4t_trade, hwm_log = run_ml4t_with_trace(data, target_entry=target_entry)
    print("=== ml4t ===")
    if ml4t_trade:
        print(f"entry={ml4t_trade['entry_bar']}, exit={ml4t_trade['exit_bar']}")
        print(f"entry_px={ml4t_trade['entry_price']:.6f}, exit_px={ml4t_trade['exit_price']:.6f}")
        print(f"pnl={ml4t_trade['pnl']:.4f}")
    print()

    if vbt_trade and ml4t_trade and vbt_trade['exit_bar'] == ml4t_trade['exit_bar']:
        exit_bar = vbt_trade['exit_bar']
        print(f"=== Price data around exit bar {exit_bar} ===")
        print(f"{'Bar':>5} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}")
        for b in range(exit_bar - 3, exit_bar + 2):
            if 0 <= b < data["n_bars"]:
                print(f"{b:5d} {data['open'][b]:10.6f} {data['high'][b]:10.6f} "
                      f"{data['low'][b]:10.6f} {data['close'][b]:10.6f}")
        print()

        print("=== HWM trace (last 5 bars) ===")
        print(f"{'Bar':>5} {'Close':>10} {'High':>10} {'Low':>10} {'HWM':>10} {'Trail':>10} {'Trig':>5}")
        for log in hwm_log[-5:]:
            print(f"{log['bar']:5d} {log['close']:10.6f} {log['high']:10.6f} "
                  f"{log['low']:10.6f} {log['hwm']:10.6f} {log['trail_level']:10.6f} "
                  f"{'YES' if log['triggered'] else 'no':>5}")
        print()

        # Calculate expected exit price
        last_log = hwm_log[-1] if hwm_log else None
        if last_log:
            trail = last_log['trail_level']
            close = last_log['close']
            base_fill = min(trail, close)
            exit_with_slip = base_fill * 0.9995
            print("=== Fill calculation ===")
            print(f"Trail level: {trail:.6f}")
            print(f"Close price: {close:.6f}")
            print(f"Base fill (min): {base_fill:.6f}")
            print(f"With 0.05% slippage: {exit_with_slip:.6f}")
            print()
            print(f"VBT exit_px:  {vbt_trade['exit_price']:.6f}")
            print(f"ml4t exit_px: {ml4t_trade['exit_price']:.6f}")
            print(f"Calculated:   {exit_with_slip:.6f}")


if __name__ == "__main__":
    main()
