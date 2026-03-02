#!/usr/bin/env python3
"""Debug trailing stop behavior differences."""

import numpy as np
import polars as pl


def generate_data(n_bars: int = 50, seed: int = 42):
    """Generate data with specific trailing stop test case."""
    rng = np.random.default_rng(seed)

    # Simple upward then downward price to trigger trailing stop
    close = np.zeros(n_bars)
    close[0] = 100.0
    for i in range(1, n_bars):
        if i < 20:
            close[i] = close[i - 1] + rng.uniform(0, 0.5)  # Upward
        else:
            close[i] = close[i - 1] - rng.uniform(0.2, 0.8)  # Downward

    high = close + rng.uniform(0.1, 0.3, n_bars)
    low = close - rng.uniform(0.1, 0.3, n_bars)
    open_ = close + rng.normal(0, 0.1, n_bars)

    # Entry at bar 5, no explicit exit signal
    momentum = np.zeros(n_bars)
    momentum[5] = 0.02  # Entry signal

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
        "momentum": momentum,
    }


def debug_vbt_trailing(data, trailing_stop=0.03):
    """Debug VBT Pro trailing stop."""
    import vectorbtpro as vbt

    print("\n" + "=" * 60)
    print("VBT Pro Trailing Stop Debug")
    print("=" * 60)

    close = data["close"]
    high = data["high"]
    low = data["low"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = np.zeros(len(momentum), dtype=bool)  # No signal exits

    print(f"\nEntry bar: {np.where(entries)[0]}")

    # Show price data around entry
    print("\nPrice data after entry (bar 5):")
    for i in range(5, min(35, len(close))):
        print(f"  Bar {i:2d}: close={close[i]:.2f}, high={high[i]:.2f}, low={low[i]:.2f}")

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
        trades = pf.trades.records_readable
        print("\nVBT Pro trades:")
        print(trades.to_string())

    return pf


def debug_nb_trailing(data, trailing_stop=0.03):
    """Debug backtest-nb trailing stop."""
    from ml4t.backtest_nb import HWM_CLOSE, HWM_HIGH, RuleEngine, Signal, backtest

    print("\n" + "=" * 60)
    print("backtest-nb Trailing Stop Debug")
    print("=" * 60)

    prices = pl.DataFrame(
        {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        }
    )
    signals = pl.DataFrame({"momentum": data["momentum"]})

    # No exit signal, only trailing stop
    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -999,  # Never triggers
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    # With HWM_CLOSE
    print("\n--- HWM_CLOSE ---")
    result = backtest(
        prices,
        signals,
        strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
        trail_hwm_source=HWM_CLOSE,
    )
    print(f"Trades: {result.n_trades}")
    if result.n_trades > 0:
        trades = result.trades[: result.n_trades]
        for i, t in enumerate(trades):
            print(f"  Trade {i}: bar {t['entry_bar']}->{t['exit_bar']}, price {t['entry_price']:.2f}->{t['exit_price']:.2f}")

    # With HWM_HIGH
    print("\n--- HWM_HIGH ---")
    result2 = backtest(
        prices,
        signals,
        strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
        trail_hwm_source=HWM_HIGH,
    )
    print(f"Trades: {result2.n_trades}")
    if result2.n_trades > 0:
        trades2 = result2.trades[: result2.n_trades]
        for i, t in enumerate(trades2):
            print(f"  Trade {i}: bar {t['entry_bar']}->{t['exit_bar']}, price {t['entry_price']:.2f}->{t['exit_price']:.2f}")

    # Calculate HWM manually
    print("\n--- Manual HWM calculation from entry bar 5 ---")
    close = data["close"]
    high = data["high"]
    low = data["low"]

    entry_price = close[5] * 1.0005  # With slippage
    hwm_close = close[5]
    hwm_high = high[5]

    for i in range(6, 30):
        hwm_close = max(hwm_close, close[i])
        hwm_high = max(hwm_high, high[i])

        trail_level_close = hwm_close * (1 - trailing_stop)
        trail_level_high = hwm_high * (1 - trailing_stop)

        triggered_close = low[i] <= trail_level_close
        triggered_high = low[i] <= trail_level_high

        print(
            f"  Bar {i:2d}: close={close[i]:.2f}, high={high[i]:.2f}, low={low[i]:.2f}, "
            f"hwm_c={hwm_close:.2f}, hwm_h={hwm_high:.2f}, "
            f"trail_c={trail_level_close:.2f}, trail_h={trail_level_high:.2f}, "
            f"trig_c={'Y' if triggered_close else 'N'}, trig_h={'Y' if triggered_high else 'N'}"
        )

        if triggered_close or triggered_high:
            break


def main():
    print("Generating trailing stop test data...")
    data = generate_data(50)

    debug_vbt_trailing(data, trailing_stop=0.03)
    debug_nb_trailing(data, trailing_stop=0.03)


if __name__ == "__main__":
    main()
