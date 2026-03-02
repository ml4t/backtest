#!/usr/bin/env python3
"""Debug behavior when signal stays above threshold for multiple bars."""

import numpy as np
import polars as pl


def generate_continuous_signal_data(n_bars: int = 100, seed: int = 42):
    """Generate data where signal stays above entry threshold."""
    rng = np.random.default_rng(seed)

    # Simple trending price
    close = 100.0 + np.cumsum(rng.normal(0.1, 0.3, n_bars))
    high = close + rng.uniform(0.1, 0.3, n_bars)
    low = close - rng.uniform(0.1, 0.3, n_bars)
    open_ = close + rng.normal(0, 0.1, n_bars)

    # Momentum that rises and stays high
    momentum = np.zeros(n_bars)
    momentum[10:20] = 0.05  # High for 10 bars
    momentum[30:35] = 0.05  # High for 5 bars
    momentum[50:60] = 0.05  # High for 10 bars

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
        "momentum": momentum,
    }


def debug_vbt_continuous(data, trailing_stop=0.03):
    """Debug VBT Pro with continuous signals."""
    import vectorbtpro as vbt

    print("\n" + "=" * 60)
    print("VBT Pro Continuous Signal Debug")
    print("=" * 60)

    close = data["close"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = momentum < -0.005

    print(f"Entry bars (momentum > 0.01): {np.where(entries)[0].tolist()}")
    print(f"Exit bars (momentum < -0.005): {np.where(exits)[0].tolist()}")

    # Default settings
    print("\n--- Default VBT settings ---")
    pf1 = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
    )
    print(f"Trades: {pf1.trades.count()}")
    if pf1.trades.count() > 0:
        for _, row in pf1.trades.records_readable.iterrows():
            entry_idx = row.get("Entry Index", row.get("entry_idx", 0))
            exit_idx = row.get("Exit Index", row.get("exit_idx", 0))
            print(f"  bar {entry_idx}->{exit_idx}")

    # With accumulate=False explicitly
    print("\n--- With accumulate=False ---")
    pf2 = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
        accumulate=False,
    )
    print(f"Trades: {pf2.trades.count()}")
    if pf2.trades.count() > 0:
        for _, row in pf2.trades.records_readable.iterrows():
            entry_idx = row.get("Entry Index", row.get("entry_idx", 0))
            exit_idx = row.get("Exit Index", row.get("exit_idx", 0))
            print(f"  bar {entry_idx}->{exit_idx}")


def debug_nb_continuous(data, trailing_stop=0.03):
    """Debug backtest-nb with continuous signals."""
    from ml4t.backtest_nb import HWM_HIGH, RuleEngine, Signal, backtest

    print("\n" + "=" * 60)
    print("backtest-nb Continuous Signal Debug")
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
        exit=Signal("momentum") < -0.005,
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
        for i, t in enumerate(trades):
            print(f"  bar {t['entry_bar']}->{t['exit_bar']}")


def main():
    print("Generating continuous signal test data...")
    data = generate_continuous_signal_data(100)

    debug_vbt_continuous(data, trailing_stop=0.03)
    debug_nb_continuous(data, trailing_stop=0.03)


if __name__ == "__main__":
    main()
