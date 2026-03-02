#!/usr/bin/env python3
"""Debug trailing stop fill price issue.

Problem: ml4t.backtest fills at BAR_LOW instead of STOP_PRICE for trailing stops.
Expected: Fill at the trail level (stop_price), not at bar low.
"""

from datetime import datetime

import numpy as np
import pandas as pd


def run_ml4t_debug():
    """Run ml4t.backtest with debug output."""
    from ml4t.backtest._validation_imports import Broker, Order, OrderSide, OrderStatus, OrderType, TrailHwmSource
    from ml4t.backtest.models import NoCommission, PercentageSlippage
    from ml4t.backtest.risk.position import TrailingStop

    print("=" * 70)
    print("DEBUG: ml4t.backtest Trailing Stop Fill Price")
    print("=" * 70)

    # Create broker with trailing stop rule
    # Use TrailHwmSource.HIGH to match VBT Pro behavior
    broker = Broker(
        100_000.0,
        NoCommission(),
        PercentageSlippage(0.0005),
        trail_hwm_source=TrailHwmSource.HIGH,  # VBT Pro uses HIGH for HWM
    )
    broker.set_position_rules(TrailingStop(pct=0.03))  # 3% trailing stop

    # Simulate price movement: entry at 100, peak at 110, drop triggers trail
    # NOTE: We need tight lows to avoid premature triggering
    prices = [
        # Entry phase: bar 0-2
        {"o": 100.0, "h": 100.5, "l": 99.8, "c": 100.0},  # bar 0
        {"o": 100.0, "h": 101.0, "l": 99.8, "c": 100.5},  # bar 1
        {"o": 100.5, "h": 101.5, "l": 100.3, "c": 101.0}, # bar 2 - entry here

        # Price rises to peak - keep lows above trail level as HWM rises
        {"o": 101.0, "h": 103.0, "l": 101.0, "c": 102.5}, # bar 3, HWM=103, trail=99.91
        {"o": 102.5, "h": 105.0, "l": 102.5, "c": 104.5}, # bar 4, HWM=105, trail=101.85
        {"o": 104.5, "h": 108.0, "l": 105.0, "c": 107.5}, # bar 5, HWM=108, trail=104.76 (low 105 > 104.76 OK)
        {"o": 107.5, "h": 110.5, "l": 108.0, "c": 110.0}, # bar 6, HWM=110.5, trail=107.185 (low 108 > 107.185 OK)

        # Price drops - trail triggers when LOW <= 107.185 (110.5 * 0.97)
        {"o": 110.0, "h": 110.5, "l": 108.0, "c": 109.5}, # bar 7 - no trigger (low 108 > 107.185)
        {"o": 109.5, "h": 110.0, "l": 107.5, "c": 108.5}, # bar 8 - no trigger (low 107.5 > 107.185)
        {"o": 108.5, "h": 109.0, "l": 107.0, "c": 108.0}, # bar 9 - TRIGGERS! low=107 < 107.185
    ]

    print("\nPrice path:")
    print("  Bar 6: HWM = 110.5 (high)")
    print("  3% trail level = 110.5 * 0.97 = 107.185")
    print("  Bar 9: low=107.0 < 107.185 => TRIGGERS")
    print("  Expected fill: 107.185 (trail level) with slippage = 107.131")
    print()

    # Run simulation bar by bar
    entry_bar = 2
    for bar_idx, p in enumerate(prices):
        ts = datetime(2020, 1, 1 + bar_idx, 9, 30)
        broker._update_time(
            timestamp=ts,
            prices={"TEST": p["c"]},
            opens={"TEST": p["o"]},
            highs={"TEST": p["h"]},
            lows={"TEST": p["l"]},
            volumes={"TEST": 1_000_000},
            signals={},
        )

        pos = broker.get_position("TEST")

        # Log position state
        if pos and pos.quantity > 0:
            hwm = pos.high_water_mark
            trail_level = hwm * 0.97
            print(f"Bar {bar_idx}: HWM={hwm:.2f}, trail={trail_level:.3f}, "
                  f"low={p['l']:.2f}, close={p['c']:.2f}")
            if p['l'] <= trail_level:
                print(f"  ** Trail triggered: low={p['l']:.2f} <= trail={trail_level:.3f}")

        # Process pending exits first
        broker._process_pending_exits()

        # Evaluate position rules and create exit orders
        exit_orders = broker.evaluate_position_rules()
        for order in exit_orders:
            risk_fill = getattr(order, "_risk_fill_price", None)
            print(f"  Exit order created: _risk_fill_price={risk_fill}")

        # Process orders
        broker._process_orders()

        # Entry signal at bar 2
        if bar_idx == entry_bar:
            if pos is None or pos.quantity == 0:
                order = broker.submit_order("TEST", 100.0, OrderSide.BUY)
                print(f"Bar {bar_idx}: Entry order submitted")

        # Process entry order
        broker._process_orders()

    print("\n" + "=" * 70)
    print("TRADE RESULTS")
    print("=" * 70)

    trades = broker.trades
    for t in trades:
        print(f"  Entry: bar ~{entry_bar}, price {t.entry_price:.4f}")
        print(f"  Exit price: {t.exit_price:.4f}")
        reason = getattr(t, 'exit_reason', getattr(t, 'reason', 'N/A'))
        print(f"  Reason: {reason}")
        print(f"  PnL: {t.pnl:.2f}")

        # Calculate what fill SHOULD have been
        # HWM = 110.5, trail = 107.185, with 0.05% slippage = 107.185 * 0.9995 = 107.131
        expected_fill = 110.5 * 0.97 * (1 - 0.0005)
        print(f"\n  Expected fill (trail level with slippage): {expected_fill:.4f}")
        print(f"  Actual fill: {t.exit_price:.4f}")
        if abs(t.exit_price - expected_fill) < 0.01:
            print("  ✅ MATCH!")
        else:
            bar_low_fill = 107.0 * (1 - 0.0005)
            if abs(t.exit_price - bar_low_fill) < 0.01:
                print(f"  ❌ MISMATCH: Filled at BAR_LOW ({bar_low_fill:.4f}), not trail level!")
            else:
                print(f"  ❌ MISMATCH: Unknown fill logic")

    return trades


def run_vbt_debug():
    """Run VBT Pro for comparison."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        print("\nVBT Pro not available, skipping comparison")
        return None

    # Same price data as ml4t test
    prices = [
        {"o": 100.0, "h": 100.5, "l": 99.8, "c": 100.0},  # bar 0
        {"o": 100.0, "h": 101.0, "l": 99.8, "c": 100.5},  # bar 1
        {"o": 100.5, "h": 101.5, "l": 100.3, "c": 101.0}, # bar 2 - entry here
        {"o": 101.0, "h": 103.0, "l": 101.0, "c": 102.5}, # bar 3
        {"o": 102.5, "h": 105.0, "l": 102.5, "c": 104.5}, # bar 4
        {"o": 104.5, "h": 108.0, "l": 105.0, "c": 107.5}, # bar 5
        {"o": 107.5, "h": 110.5, "l": 108.0, "c": 110.0}, # bar 6 - HWM = 110.5
        {"o": 110.0, "h": 110.5, "l": 108.0, "c": 109.5}, # bar 7
        {"o": 109.5, "h": 110.0, "l": 107.5, "c": 108.5}, # bar 8
        {"o": 108.5, "h": 109.0, "l": 107.0, "c": 108.0}, # bar 9 - TRIGGERS
    ]

    n_bars = len(prices)
    close = pd.Series([p["c"] for p in prices])
    high = pd.Series([p["h"] for p in prices])
    low = pd.Series([p["l"] for p in prices])
    open_ = pd.Series([p["o"] for p in prices])

    entries = pd.Series([False] * n_bars)
    entries.iloc[2] = True
    exits = pd.Series([False] * n_bars)

    print("\n" + "=" * 70)
    print("VBT Pro Reference")
    print("=" * 70)

    pf = vbt.Portfolio.from_signals(
        open=open_,
        high=high,
        low=low,
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=0.03,
        init_cash=100_000.0,
        size=100.0,
        fees=0.0,  # No commission for easier comparison
        slippage=0.0005,
    )

    print(f"\nTrades: {pf.trades.count()}")
    if pf.trades.count() > 0:
        records = pf.trades.records_readable
        print(f"  Columns: {list(records.columns)}")
        for _, row in records.iterrows():
            # Try various column name formats
            entry_idx = row.get("Entry Index", row.get("entry_idx", row.get("Entry Idx", 0)))
            exit_idx = row.get("Exit Index", row.get("exit_idx", row.get("Exit Idx", 0)))
            entry_price = row.get("Avg Entry Price", row.get("Entry Price", row.get("entry_price", 0)))
            exit_price = row.get("Avg Exit Price", row.get("Exit Price", row.get("exit_price", 0)))
            pnl = row.get("PnL", row.get("pnl", 0))
            print(f"  Entry: bar {entry_idx}, price {entry_price:.4f}")
            print(f"  Exit: bar {exit_idx}, price {exit_price:.4f}")
            print(f"  PnL: {pnl:.2f}")

    return pf


if __name__ == "__main__":
    ml4t_trades = run_ml4t_debug()
    vbt_pf = run_vbt_debug()
