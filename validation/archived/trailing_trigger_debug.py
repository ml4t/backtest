#!/usr/bin/env python3
"""Debug trailing stop trigger price (LOW vs CLOSE)."""

import numpy as np
import polars as pl


def generate_data():
    """Generate data where LOW triggers but CLOSE doesn't."""
    n_bars = 20
    # Price rises to peak, then drops such that LOW triggers but CLOSE doesn't
    close = np.array([
        100.0, 101.0, 102.0, 103.0, 104.0,  # Rise bars 0-4
        105.0, 106.0, 107.0, 108.0, 109.0,  # Rise bars 5-9
        110.0,  # Peak bar 10
        109.0, 108.0,  # Drop bars 11-12
        107.0,  # Bar 13: close stays above trail, but low might dip below
        106.0, 105.0, 104.0, 103.0, 102.0, 101.0  # Continue drop
    ])

    # Set up specific OHLC for bar 13 to test the trigger
    # HWM at bar 10 = 110.5, 3% trail = 107.185
    # Set bar 13 close = 107.5 (above trail), low = 106.5 (below trail)
    high = close + 0.5
    low = close - 0.5

    # Bar 13 specific: close 107, high 107.5, low 106.5
    # If using LOW: 106.5 < 107.185 -> triggers
    # If using CLOSE: 107.0 < 107.185? No, 107.0 is NOT < 107.185 (within 0.2% tolerance)
    # Actually 107.0 IS less than 107.185!

    # Let me set bar 13 more carefully:
    # HWM = 110.5 (from bar 10 high)
    # 3% trail = 110.5 * 0.97 = 107.185
    # Set bar 13 close = 108.0 (above 107.185), low = 106.5 (below 107.185)
    close[13] = 108.0
    high[13] = 108.5
    low[13] = 106.5

    open_ = close.copy()

    momentum = np.zeros(n_bars)
    momentum[2:18] = 0.05  # Entry signal

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
        "momentum": momentum,
    }


def run_vbt(data, trailing_stop=0.03):
    """Run VBT Pro."""
    import vectorbtpro as vbt

    close = data["close"]
    high = data["high"]
    low = data["low"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = np.zeros(len(momentum), dtype=bool)

    print("\n=== VBT Pro ===")
    print("Bar 10 (peak): close=110.0, high=110.5")
    print(f"Bar 13: close={close[13]:.1f}, high={high[13]:.1f}, low={low[13]:.1f}")
    print("3% trail from HWM 110.5 = 107.185")
    print(f"  Bar 13 close ({close[13]:.1f}) < 107.185? {close[13] < 107.185}")
    print(f"  Bar 13 low ({low[13]:.1f}) < 107.185? {low[13] < 107.185}")

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
    )

    print(f"\nTrades: {pf.trades.count()}")
    if pf.trades.count() > 0:
        for _, row in pf.trades.records_readable.iterrows():
            entry_idx = int(row.get("Entry Index", row.get("entry_idx", 0)))
            exit_idx = int(row.get("Exit Index", row.get("exit_idx", 0)))
            print(f"  bar {entry_idx}->{exit_idx}")


def run_nb(data, trailing_stop=0.03):
    """Run backtest-nb."""
    from ml4t.backtest_nb import HWM_HIGH, RuleEngine, Signal, backtest

    prices = pl.DataFrame({
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
    })
    signals = pl.DataFrame({"momentum": data["momentum"]})

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -999,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    result = backtest(
        prices, signals, strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
        trail_hwm_source=HWM_HIGH,
    )

    print("\n=== backtest-nb (HWM_HIGH) ===")
    print(f"Trades: {result.n_trades}")
    if result.n_trades > 0:
        trades = result.trades[:result.n_trades]
        for t in trades:
            print(f"  bar {t['entry_bar']}->{t['exit_bar']}")


def main():
    data = generate_data()

    print("=== Trailing Stop Trigger Debug ===")
    print("\nThis tests: Does trailing stop trigger on LOW price or CLOSE price?")
    print("Bar 13 is set up so LOW < trail level but CLOSE > trail level")

    run_vbt(data)
    run_nb(data)


if __name__ == "__main__":
    main()
