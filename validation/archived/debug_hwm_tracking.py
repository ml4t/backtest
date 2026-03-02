#!/usr/bin/env python3
"""Debug HWM tracking difference between VBT and ml4t."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_asset_001_data(seed: int = 42):
    """Generate data matching asset_001 from scale test."""
    rng = np.random.default_rng(seed + 1 * 1000)  # asset_id = 1
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


def run_vbt(data, trail_pct=0.03):
    """Run VBT Pro and trace HWM."""
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

    # Get trade at entry 89
    trades = []
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            if row.get("Status", "") == "Open":
                continue
            entry_bar = int(row['Entry Index'])
            if entry_bar == 89:
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": int(row['Exit Index']),
                    "entry_price": float(row['Avg Entry Price']),
                    "exit_price": float(row['Avg Exit Price']),
                    "pnl": float(row['PnL']),
                })

    return trades


def run_ml4t_with_trace(data, trail_pct=0.03, use_high=True):
    """Run ml4t.backtest with HWM tracing."""
    from ml4t.backtest._validation_imports import Broker, OrderSide, TrailHwmSource, StopFillMode
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    n_bars = data["n_bars"]

    hwm_source = TrailHwmSource.HIGH if use_high else TrailHwmSource.CLOSE
    broker = Broker(
        10_000_000.0,
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=hwm_source,
        stop_fill_mode=StopFillMode.CLOSE_PRICE,
    )
    broker.set_position_rules(TrailingStop(pct=trail_pct))

    base_time = datetime(2020, 1, 1, 9, 30)

    # Track HWM for position starting at bar 89
    hwm_log = []
    target_entry_bar = 89

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

        # Log HWM after rule evaluation if we have position from bar 89
        pos = broker.get_position("TEST")
        if pos and pos.quantity > 0:
            entry_bar = (pos.entry_time - base_time).days
            if entry_bar == target_entry_bar:
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

        if data["entries"][bar_idx]:
            pos = broker.get_position("TEST")
            if pos is None or pos.quantity == 0:
                broker.submit_order("TEST", 100.0, OrderSide.BUY)

        broker._process_orders()

    # Get trade at entry 89
    trades = []
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        if entry_bar == target_entry_bar:
            exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
            trades.append({
                "entry_bar": entry_bar,
                "exit_bar": exit_bar,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
            })

    return trades, hwm_log


def main():
    print("=" * 70)
    print("DEBUG: HWM Tracking for asset_001, entry bar 89")
    print("=" * 70)
    print()

    data = generate_asset_001_data()

    # Find entry bar 89
    print(f"Entry at bar 89? {data['entries'][89]}")
    print()

    # Print price data around bar 89
    print("=== Price data around bar 89-100 ===")
    print(f"{'Bar':>5} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Entry':>6}")
    for b in range(89, 100):
        print(f"{b:5d} {data['open'][b]:10.4f} {data['high'][b]:10.4f} "
              f"{data['low'][b]:10.4f} {data['close'][b]:10.4f} {data['entries'][b]!s:>6}")
    print()

    # Run VBT Pro
    print("=== VBT Pro ===")
    vbt_trades = run_vbt(data)
    if vbt_trades:
        t = vbt_trades[0]
        print(f"Trade: entry={t['entry_bar']}, exit={t['exit_bar']}, "
              f"entry_px={t['entry_price']:.4f}, exit_px={t['exit_price']:.4f}")
    else:
        print("No trade found at bar 89")
    print()

    # Run ml4t with HIGH
    print("=== ml4t with HWM=HIGH ===")
    ml4t_trades_high, hwm_log_high = run_ml4t_with_trace(data, use_high=True)
    if ml4t_trades_high:
        t = ml4t_trades_high[0]
        print(f"Trade: entry={t['entry_bar']}, exit={t['exit_bar']}, "
              f"entry_px={t['entry_price']:.4f}, exit_px={t['exit_price']:.4f}")
        print("\nHWM trace (first 10 bars after entry):")
        print(f"{'Bar':>5} {'Close':>10} {'High':>10} {'Low':>10} {'HWM':>10} {'Trail':>10} {'Trig':>5}")
        for log in hwm_log_high[:10]:
            print(f"{log['bar']:5d} {log['close']:10.4f} {log['high']:10.4f} "
                  f"{log['low']:10.4f} {log['hwm']:10.4f} {log['trail_level']:10.4f} "
                  f"{'YES' if log['triggered'] else 'no':>5}")
    print()

    # Run ml4t with CLOSE
    print("=== ml4t with HWM=CLOSE ===")
    ml4t_trades_close, hwm_log_close = run_ml4t_with_trace(data, use_high=False)
    if ml4t_trades_close:
        t = ml4t_trades_close[0]
        print(f"Trade: entry={t['entry_bar']}, exit={t['exit_bar']}, "
              f"entry_px={t['entry_price']:.4f}, exit_px={t['exit_price']:.4f}")
        print("\nHWM trace (first 10 bars after entry):")
        print(f"{'Bar':>5} {'Close':>10} {'High':>10} {'Low':>10} {'HWM':>10} {'Trail':>10} {'Trig':>5}")
        for log in hwm_log_close[:10]:
            print(f"{log['bar']:5d} {log['close']:10.4f} {log['high']:10.4f} "
                  f"{log['low']:10.4f} {log['hwm']:10.4f} {log['trail_level']:10.4f} "
                  f"{'YES' if log['triggered'] else 'no':>5}")
    print()


if __name__ == "__main__":
    main()
