#!/usr/bin/env python3
"""Debug PnL mismatch between ml4t and VBT Pro.

When exit bar and exit price match, why does PnL differ?
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_simple_data(n_bars: int = 200, seed: int = 42):
    """Generate single-asset data."""
    rng = np.random.default_rng(seed)

    returns = rng.normal(0.0003, 0.02, n_bars)
    cumret = np.clip(np.cumsum(returns), -3, 3)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.2, n_bars)

    # Single entry at bar 50
    entries = np.zeros(n_bars, dtype=bool)
    entries[50] = True

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "entries": entries,
    }


def run_vbt(data, trail_pct=0.03):
    """Run VBT Pro."""
    import vectorbtpro as vbt

    n_bars = len(data["close"])

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
    )

    print("=== VBT Pro ===")
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        for _, row in records.iterrows():
            print(f"  Entry bar: {row['Entry Index']}")
            print(f"  Entry price: {row['Avg Entry Price']:.6f}")
            print(f"  Entry fees: {row['Entry Fees']:.6f}")
            print(f"  Exit bar: {row['Exit Index']}")
            print(f"  Exit price: {row['Avg Exit Price']:.6f}")
            print(f"  Exit fees: {row['Exit Fees']:.6f}")
            print(f"  PnL: {row['PnL']:.6f}")
            print(f"  Size: {row['Size']}")

            # Calculate expected PnL
            size = row['Size']
            entry_px = row['Avg Entry Price']
            exit_px = row['Avg Exit Price']
            entry_fees = row['Entry Fees']
            exit_fees = row['Exit Fees']

            gross_pnl = (exit_px - entry_px) * size
            net_pnl = gross_pnl - entry_fees - exit_fees
            print(f"\n  Calculated:")
            print(f"    Gross PnL: ({exit_px:.4f} - {entry_px:.4f}) * {size} = {gross_pnl:.4f}")
            print(f"    Net PnL: {gross_pnl:.4f} - {entry_fees:.4f} - {exit_fees:.4f} = {net_pnl:.4f}")
    else:
        print("  No trades!")

    return pf


def run_ml4t(data, trail_pct=0.03):
    """Run ml4t.backtest."""
    from ml4t.backtest._validation_imports import Broker, OrderSide, TrailHwmSource
    from ml4t.backtest.models import PercentageCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    n_bars = len(data["close"])

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

    print("\n=== ml4t.backtest ===")
    for t in broker.trades:
        entry_bar = (t.entry_time - base_time).days
        exit_bar = (t.exit_time - base_time).days if t.exit_time else n_bars - 1

        print(f"  Entry bar: {entry_bar}")
        print(f"  Entry price: {t.entry_price:.6f}")
        print(f"  Exit bar: {exit_bar}")
        print(f"  Exit price: {t.exit_price:.6f}")
        print(f"  Total commission: {t.commission:.6f}")
        print(f"  Slippage: {t.slippage:.6f}")
        print(f"  PnL: {t.pnl:.6f}")
        print(f"  Size: {t.quantity}")

        # Calculate expected PnL
        gross_pnl = (t.exit_price - t.entry_price) * t.quantity
        net_pnl = gross_pnl - t.commission
        print(f"\n  Calculated:")
        print(f"    Gross PnL: ({t.exit_price:.4f} - {t.entry_price:.4f}) * {t.quantity} = {gross_pnl:.4f}")
        print(f"    Net PnL (gross - commission): {gross_pnl:.4f} - {t.commission:.4f} = {net_pnl:.4f}")

    return broker


def main():
    print("=" * 70)
    print("DEBUG: PnL Mismatch Investigation")
    print("=" * 70)
    print()

    data = generate_simple_data()

    # Print relevant bars
    print("=== Price Data ===")
    for bar in range(50, 54):
        print(f"  Bar {bar}: open={data['open'][bar]:.4f}, high={data['high'][bar]:.4f}, "
              f"low={data['low'][bar]:.4f}, close={data['close'][bar]:.4f}")

    # Calculate trail levels
    print("\n=== Trail Stop Analysis ===")
    # Entry at bar 50
    hwm = data["high"][50]  # Initial HWM
    trail_pct = 0.03
    print(f"  After bar 50: HWM = {hwm:.4f}, trail = {hwm * (1 - trail_pct):.4f}")

    hwm = max(hwm, data["high"][51])
    print(f"  After bar 51: HWM = {hwm:.4f}, trail = {hwm * (1 - trail_pct):.4f}")

    hwm = max(hwm, data["high"][52])
    trail_level = hwm * (1 - trail_pct)
    print(f"  Bar 52: HWM = {hwm:.4f}, trail = {trail_level:.4f}")
    print(f"  Bar 52 low = {data['low'][52]:.4f}, triggers? {data['low'][52] <= trail_level}")

    print()
    vbt_pf = run_vbt(data)
    ml4t_broker = run_ml4t(data)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if vbt_pf.trades.count() > 0 and len(ml4t_broker.trades) > 0:
        vbt_rec = vbt_pf.trades.records_readable.iloc[0]
        ml4t_t = ml4t_broker.trades[0]

        print(f"\nEntry price diff: {abs(vbt_rec['Avg Entry Price'] - ml4t_t.entry_price):.6f}")
        print(f"Exit price diff: {abs(vbt_rec['Avg Exit Price'] - ml4t_t.exit_price):.6f}")
        print(f"PnL diff: {abs(vbt_rec['PnL'] - ml4t_t.pnl):.6f}")


if __name__ == "__main__":
    main()
