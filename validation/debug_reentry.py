#!/usr/bin/env python3
"""Debug same-bar re-entry behavior."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_reentry_data():
    """Generate data where stop exit and re-entry could happen same bar."""
    # Bar 0: Entry signal
    # Bar 1: Price rises
    # Bar 2: Price drops, stop triggers AND entry signal is true
    # Question: Does VBT allow re-entry on bar 2?

    n_bars = 10
    close = np.array([100.0, 105.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()

    # Entry signal on bar 0 AND bar 2
    entries = np.array([True, False, True, False, False, False, False, False, False, False])

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

    print("=== Price Data ===")
    for i in range(n_bars):
        sig = "ENTRY" if data["entries"][i] else ""
        print(f"  Bar {i}: O={data['open'][i]:.1f}, H={data['high'][i]:.1f}, "
              f"L={data['low'][i]:.1f}, C={data['close'][i]:.1f} {sig}")

    # Calculate what should happen
    print("\n=== Expected Behavior ===")
    print("  Bar 0: Entry at open (100.0)")
    print("  Bar 1: HWM = 106.0 (high), trail = 102.82")
    print("  Bar 2: low = 101.0 < 102.82 => STOP triggers")
    print("  Bar 2: Also has entry signal - does re-entry happen?")

    pf = vbt.Portfolio.from_signals(
        open=pd.Series(data["open"]),
        high=pd.Series(data["high"]),
        low=pd.Series(data["low"]),
        close=pd.Series(data["close"]),
        entries=pd.Series(data["entries"]),
        exits=pd.Series([False] * n_bars),
        tsl_stop=trail_pct,
        init_cash=100_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
        accumulate=False,
    )

    print("\n=== VBT Pro Trades ===")
    if pf.trades.count() > 0:
        for _, row in pf.trades.records_readable.iterrows():
            print(f"  Entry bar: {row['Entry Index']}, Exit bar: {row['Exit Index']}")
            print(f"  Entry price: {row['Avg Entry Price']:.4f}, Exit price: {row['Avg Exit Price']:.4f}")
            print()
    else:
        print("  No trades!")

    print(f"Total trades: {pf.trades.count()}")
    return pf


def run_ml4t(data, trail_pct=0.03):
    """Run ml4t.backtest."""
    from ml4t.backtest import Broker, OrderSide, TrailHwmSource
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    n_bars = data["n_bars"]

    broker = Broker(
        100_000.0,
        PercentageCommission(0.001),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH,
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

    print("\n=== ml4t.backtest Trades ===")
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1
        print(f"  Entry bar: {entry_bar}, Exit bar: {exit_bar}")
        print(f"  Entry price: {t.entry_price:.4f}, Exit price: {t.exit_price:.4f}")
        print()

    print(f"Total trades: {len(broker.trades)}")

    # Check open positions
    pos = broker.get_position("TEST")
    if pos and pos.quantity > 0:
        print(f"\nOPEN POSITION: qty={pos.quantity}, entry_bar=?, entry_price={pos.entry_price:.4f}")

    return broker


def main():
    print("=" * 70)
    print("DEBUG: Same-Bar Re-Entry After Stop Exit")
    print("=" * 70)
    print()

    data = generate_reentry_data()
    vbt_pf = run_vbt(data)
    ml4t_broker = run_ml4t(data)


if __name__ == "__main__":
    main()
