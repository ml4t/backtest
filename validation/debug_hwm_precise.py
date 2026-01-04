#!/usr/bin/env python3
"""Debug HWM timing precisely for asset_038 case."""

import numpy as np
import pandas as pd

def debug_vbt_pro():
    """Debug VBT Pro HWM behavior for asset_038."""
    import vectorbtpro as vbt

    # Generate same data as scale test
    seed = 42 + 38 * 1000  # asset_038
    rng = np.random.default_rng(seed)
    n_bars = 1000

    returns = rng.normal(0.0003, 0.02, n_bars)
    cumret = np.clip(np.cumsum(returns), -3, 3)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.2, n_bars)

    entries = rng.random(n_bars) < 0.01

    # Focus on the entry at bar 485
    print(f"Entry at bar 485: {entries[485]}")

    # Check bars 485-495
    print("\nBar data around entry (485) and exits (486 vs 492):")
    print(f"{'Bar':>5} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Entry':>6}")
    for i in range(480, 500):
        print(f"{i:>5} {open_[i]:>10.4f} {high[i]:>10.4f} {low[i]:>10.4f} {close[i]:>10.4f} {'YES' if entries[i] else '':>6}")

    # Run VBT Pro
    pf = vbt.Portfolio.from_signals(
        open=pd.Series(open_),
        high=pd.Series(high),
        low=pd.Series(low),
        close=pd.Series(close),
        entries=pd.Series(entries),
        exits=pd.Series([False] * n_bars),
        tsl_stop=0.03,
        init_cash=10_000_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
        accumulate=False,
    )

    # Get trades for this asset
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        print("\n\nVBT Pro trades:")
        for _, row in records.iterrows():
            entry_idx = int(row.get("Entry Index", 0))
            exit_idx = int(row.get("Exit Index", 0))
            entry_price = float(row.get("Avg Entry Price", 0))
            exit_price = float(row.get("Avg Exit Price", 0))
            pnl = float(row.get("PnL", 0))
            status = row.get("Status", "")
            if status == "Open":
                continue
            if entry_idx >= 480 and entry_idx <= 490:
                print(f"Entry bar {entry_idx}, Exit bar {exit_idx}")
                print(f"  Entry price: {entry_price:.4f}")
                print(f"  Exit price: {exit_price:.4f}")
                print(f"  PnL: {pnl:.2f}")

                # Calculate what HWM should have been
                # If exit at 486 with price ~39.83, then 3% trail means HWM = 39.83 / 0.97 = 41.06
                implied_hwm = exit_price / 0.97
                print(f"  Implied HWM at exit: {implied_hwm:.4f}")

                # Check if this matches bar 485's close or high
                print(f"  Bar 485 close: {close[485]:.4f}")
                print(f"  Bar 485 high: {high[485]:.4f}")

                # Check subsequent highs
                for b in range(entry_idx, exit_idx + 1):
                    print(f"    Bar {b}: high={high[b]:.4f}, low={low[b]:.4f}, close={close[b]:.4f}")


def debug_ml4t():
    """Debug ml4t HWM behavior for asset_038."""
    from datetime import datetime, timedelta
    from ml4t.backtest import Broker, OrderSide, TrailHwmSource, StopFillMode, InitialHwmSource
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    # Same data generation
    seed = 42 + 38 * 1000
    rng = np.random.default_rng(seed)
    n_bars = 1000

    returns = rng.normal(0.0003, 0.02, n_bars)
    cumret = np.clip(np.cumsum(returns), -3, 3)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.2, n_bars)

    entries = rng.random(n_bars) < 0.01

    broker = Broker(
        10_000_000.0,
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH,
        initial_hwm_source=InitialHwmSource.BAR_CLOSE,  # VBT Pro uses CLOSE
        stop_fill_mode=StopFillMode.STOP_PRICE,
    )
    broker.set_position_rules(TrailingStop(pct=0.03))

    base_time = datetime(2020, 1, 1, 9, 30)
    asset = "asset_038"

    print("\n\nml4t HWM debug:")
    for bar_idx in range(n_bars):
        ts = base_time + timedelta(days=bar_idx)

        broker._update_time(
            timestamp=ts,
            prices={asset: close[bar_idx]},
            opens={asset: open_[bar_idx]},
            highs={asset: high[bar_idx]},
            lows={asset: low[bar_idx]},
            volumes={asset: 1_000_000},
            signals={},
        )

        broker._process_pending_exits()
        broker.evaluate_position_rules()
        broker._process_orders()

        # Check for entry
        if entries[bar_idx]:
            pos = broker.get_position(asset)
            if pos is None or pos.quantity == 0:
                broker.submit_order(asset, 100.0, OrderSide.BUY)

        broker._process_orders()

        # Print HWM status for relevant bars
        pos = broker.get_position(asset)
        if bar_idx >= 484 and bar_idx <= 495:
            if pos and pos.quantity > 0:
                trail_level = pos.high_water_mark * 0.97
                print(f"Bar {bar_idx}: HWM={pos.high_water_mark:.4f}, trail={trail_level:.4f}, low={low[bar_idx]:.4f}, close={close[bar_idx]:.4f}, high={high[bar_idx]:.4f}")
                is_new = asset in broker._positions_created_this_bar
                print(f"         is_new={is_new}")
            else:
                print(f"Bar {bar_idx}: No position (exited)")

        broker._update_water_marks()

    # Print trades
    print("\nml4t trades around bar 485:")
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
        if entry_bar >= 480 and entry_bar <= 490:
            print(f"Entry bar {entry_bar}, Exit bar {exit_bar}")
            print(f"  Entry price: {t.entry_price:.4f}")
            print(f"  Exit price: {t.exit_price:.4f}")
            print(f"  PnL: {t.pnl:.2f}")


if __name__ == "__main__":
    debug_vbt_pro()
    debug_ml4t()
