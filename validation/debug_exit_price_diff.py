#!/usr/bin/env python3
"""Debug small exit price differences."""

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
    """Run VBT Pro."""
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

    trades = []
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            if row.get("Status", "") == "Open":
                continue
            trades.append({
                "entry_bar": int(row['Entry Index']),
                "exit_bar": int(row['Exit Index']),
                "entry_price": float(row['Avg Entry Price']),
                "exit_price": float(row['Avg Exit Price']),
                "pnl": float(row['PnL']),
            })

    return trades


def run_ml4t(data, trail_pct=0.03):
    """Run ml4t.backtest."""
    from ml4t.backtest import Broker, OrderSide, TrailHwmSource
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    n_bars = data["n_bars"]

    broker = Broker(
        10_000_000.0,
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.CLOSE,  # VBT Pro uses CLOSE for HWM
    )
    broker.set_position_rules(TrailingStop(pct=trail_pct))

    base_time = datetime(2020, 1, 1, 9, 30)

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

        if data["entries"][bar_idx]:
            pos = broker.get_position("TEST")
            if pos is None or pos.quantity == 0:
                broker.submit_order("TEST", 100.0, OrderSide.BUY)

        broker._process_orders()

    trades = []
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
        trades.append({
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
        })

    return trades


def main():
    print("=" * 70)
    print("DEBUG: Exit Price Differences (asset_001)")
    print("=" * 70)
    print()

    data = generate_asset_001_data()

    vbt_trades = run_vbt(data)
    ml4t_trades = run_ml4t(data)

    # Find the trade at entry_bar=442
    vbt_442 = next((t for t in vbt_trades if t["entry_bar"] == 442), None)
    ml4t_442 = next((t for t in ml4t_trades if t["entry_bar"] == 442), None)

    if vbt_442 and ml4t_442:
        print("Trade at entry_bar=442:")
        print(f"  VBT:  exit_bar={vbt_442['exit_bar']}, exit_px={vbt_442['exit_price']:.4f}, pnl={vbt_442['pnl']:.2f}")
        print(f"  ml4t: exit_bar={ml4t_442['exit_bar']}, exit_px={ml4t_442['exit_price']:.4f}, pnl={ml4t_442['pnl']:.2f}")

        # Look at the data around exit bar
        exit_bar = vbt_442["exit_bar"]
        print(f"\n=== Price data around exit bar {exit_bar} ===")
        for b in range(exit_bar - 2, exit_bar + 2):
            if 0 <= b < data["n_bars"]:
                print(f"  Bar {b}: O={data['open'][b]:.4f}, H={data['high'][b]:.4f}, "
                      f"L={data['low'][b]:.4f}, C={data['close'][b]:.4f}")

        # Calculate what trail level should be
        # Need to find HWM before exit bar
        entry_bar = vbt_442["entry_bar"]
        hwm = data["high"][entry_bar]  # HWM starts at entry bar high
        for b in range(entry_bar + 1, exit_bar + 1):
            if data["high"][b] > hwm:
                hwm = data["high"][b]

        trail_level = hwm * 0.97
        print(f"\n=== Trail Calculation ===")
        print(f"  HWM before bar {exit_bar} = {hwm:.4f}")
        print(f"  3% trail level = {trail_level:.4f}")
        print(f"  Bar {exit_bar} open = {data['open'][exit_bar]:.4f}")
        print(f"  Bar {exit_bar} low = {data['low'][exit_bar]:.4f}")

        if data["open"][exit_bar] < trail_level:
            print(f"  Gap through! Open < trail, should fill at open with slippage")
            expected_fill = data["open"][exit_bar] * 0.9995
        else:
            print(f"  Normal trigger, should fill at trail level with slippage")
            expected_fill = trail_level * 0.9995

        print(f"\n  Expected fill (with 0.05% slippage): {expected_fill:.4f}")
        print(f"  VBT actual: {vbt_442['exit_price']:.4f}")
        print(f"  ml4t actual: {ml4t_442['exit_price']:.4f}")
    else:
        print("Trade not found!")
        print(f"VBT trades: {len(vbt_trades)}")
        print(f"ml4t trades: {len(ml4t_trades)}")


if __name__ == "__main__":
    main()
