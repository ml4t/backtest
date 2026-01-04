#!/usr/bin/env python3
"""Debug re-entry behavior after trailing stop."""

import numpy as np
import polars as pl


def generate_reentry_data(n_bars: int = 100, seed: int = 42):
    """Generate data to test re-entry after trailing stop."""
    rng = np.random.default_rng(seed)

    # Price rises then drops then rises again
    close = np.zeros(n_bars)
    close[0] = 100.0
    for i in range(1, n_bars):
        if i < 15:
            close[i] = close[i - 1] + rng.uniform(0.1, 0.3)  # Rise
        elif i < 25:
            close[i] = close[i - 1] - rng.uniform(0.3, 0.6)  # Drop (triggers stop)
        elif i < 50:
            close[i] = close[i - 1] + rng.uniform(0.1, 0.3)  # Rise again
        else:
            close[i] = close[i - 1] - rng.uniform(0.3, 0.6)  # Drop again

    high = close + rng.uniform(0.1, 0.3, n_bars)
    low = close - rng.uniform(0.1, 0.3, n_bars)
    open_ = close + rng.normal(0, 0.1, n_bars)

    # Multiple entry signals
    momentum = np.zeros(n_bars)
    momentum[5] = 0.02  # Entry
    momentum[22] = 0.02  # Re-entry signal (during price drop - might not trigger due to position)
    momentum[30] = 0.02  # Another entry signal (after first stop)
    momentum[55] = 0.02  # Another entry (after second drop)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
        "momentum": momentum,
    }


def debug_vbt_reentry(data, trailing_stop=0.03):
    """Debug VBT Pro re-entry behavior."""
    import vectorbtpro as vbt

    print("\n" + "=" * 60)
    print("VBT Pro Re-entry Debug")
    print("=" * 60)

    close = data["close"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = np.zeros(len(momentum), dtype=bool)  # No signal exits

    print(f"Entry signals at bars: {np.where(entries)[0].tolist()}")

    # Show key prices
    print("\nKey price points:")
    for bar in [5, 10, 15, 20, 22, 25, 30, 35, 40, 50, 55]:
        if bar < len(close):
            print(f"  Bar {bar:2d}: close={close[bar]:.2f}")

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
        for _, row in trades.iterrows():
            entry_idx = row.get("Entry Index", row.get("entry_idx", 0))
            exit_idx = row.get("Exit Index", row.get("exit_idx", 0))
            pnl = row.get("PnL", row.get("pnl", 0))
            print(f"  bar {entry_idx}->{exit_idx}, pnl=${pnl:.2f}")


def debug_nb_reentry(data, trailing_stop=0.03):
    """Debug backtest-nb re-entry behavior."""
    from ml4t.backtest_nb import HWM_HIGH, RuleEngine, Signal, backtest

    print("\n" + "=" * 60)
    print("backtest-nb Re-entry Debug")
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

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -999,  # Never triggers
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    result = backtest(
        prices,
        signals,
        strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
        trail_hwm_source=HWM_HIGH,
    )

    print(f"\nTrades: {result.n_trades}")
    if result.n_trades > 0:
        trades = result.trades[: result.n_trades]
        print("\nbacktest-nb trades:")
        for i, t in enumerate(trades):
            print(f"  bar {t['entry_bar']}->{t['exit_bar']}, pnl=${t['pnl']:.2f}")


def main():
    print("Generating re-entry test data...")
    data = generate_reentry_data(100)

    debug_vbt_reentry(data, trailing_stop=0.03)
    debug_nb_reentry(data, trailing_stop=0.03)


if __name__ == "__main__":
    main()
