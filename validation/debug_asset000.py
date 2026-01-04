#!/usr/bin/env python3
"""Debug asset_000 bar 145 case."""

import numpy as np
import pandas as pd

def main():
    # Same data generation as scale test
    seed = 42 + 0 * 1000  # asset_000
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

    print("Bar data around entry (145) and exits (146 vs 151):")
    print(f"{'Bar':>5} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Entry':>6}")
    for i in range(143, 160):
        print(f"{i:>5} {open_[i]:>10.4f} {high[i]:>10.4f} {low[i]:>10.4f} {close[i]:>10.4f} {'YES' if entries[i] else '':>6}")

    # Run VBT Pro
    import vectorbtpro as vbt
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

    print("\nVBT Pro trades around bar 145:")
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            entry_idx = int(row.get("Entry Index", 0))
            exit_idx = int(row.get("Exit Index", 0))
            if 140 <= entry_idx <= 150:
                entry_price = float(row.get("Avg Entry Price", 0))
                exit_price = float(row.get("Avg Exit Price", 0))
                pnl = float(row.get("PnL", 0))
                print(f"Entry bar {entry_idx}, Exit bar {exit_idx}")
                print(f"  Entry price: {entry_price:.4f}")
                print(f"  Exit price: {exit_price:.4f}")
                print(f"  PnL: {pnl:.2f}")

                # Calculate implied HWM
                implied_hwm = exit_price / 0.9995 / 0.97
                print(f"  Implied HWM (if gap-through): {implied_hwm:.4f}")
                print(f"  Bar {exit_idx} open: {open_[exit_idx]:.4f}")
                print(f"  Bar {exit_idx} low: {low[exit_idx]:.4f}")

                # Check if gap-through
                trail = exit_price / 0.9995  # Remove slippage to get trail level
                print(f"  Trail level (no slippage): {trail:.4f}")
                if open_[exit_idx] < trail:
                    print(f"  GAP-THROUGH: open < trail")
                else:
                    print(f"  NO GAP-THROUGH: open >= trail")

    # Run ml4t
    from datetime import datetime, timedelta
    from ml4t.backtest import Broker, OrderSide, TrailHwmSource, StopFillMode, InitialHwmSource
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    broker = Broker(
        10_000_000.0,
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH,
        initial_hwm_source=InitialHwmSource.BAR_CLOSE,
        stop_fill_mode=StopFillMode.STOP_PRICE,
    )
    broker.set_position_rules(TrailingStop(pct=0.03))

    base_time = datetime(2020, 1, 1, 9, 30)
    asset = "asset_000"

    print("\n\nml4t debug:")
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

        if entries[bar_idx]:
            pos = broker.get_position(asset)
            if pos is None or pos.quantity == 0:
                broker.submit_order(asset, 100.0, OrderSide.BUY)

        broker._process_orders()

        pos = broker.get_position(asset)
        if 143 <= bar_idx <= 160:
            if pos and pos.quantity > 0:
                trail_level = pos.high_water_mark * 0.97
                triggered = low[bar_idx] <= trail_level
                gap_through = open_[bar_idx] < trail_level
                print(f"Bar {bar_idx}: HWM={pos.high_water_mark:.4f}, trail={trail_level:.4f}")
                print(f"         open={open_[bar_idx]:.4f}, low={low[bar_idx]:.4f}, close={close[bar_idx]:.4f}, high={high[bar_idx]:.4f}")
                print(f"         triggered={triggered}, gap_through={gap_through}")
            else:
                if bar_idx == 145:
                    print(f"Bar {bar_idx}: Position just created or exited")
                elif bar_idx > 145:
                    print(f"Bar {bar_idx}: No position")

        broker._update_water_marks()

    print("\n\nml4t trades around bar 145:")
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
        if 140 <= entry_bar <= 150:
            print(f"Entry bar {entry_bar}, Exit bar {exit_bar}")
            print(f"  Entry price: {t.entry_price:.4f}")
            print(f"  Exit price: {t.exit_price:.4f}")
            print(f"  PnL: {t.pnl:.2f}")


if __name__ == "__main__":
    main()
