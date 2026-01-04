#!/usr/bin/env python3
"""Debug script to understand VBT Pro vs backtest-nb differences."""

import numpy as np
import pandas as pd
import polars as pl


def generate_simple_data(n_bars: int = 100, seed: int = 42):
    """Generate simple data for debugging."""
    rng = np.random.default_rng(seed)

    # Simple upward trending price
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    high = close + rng.uniform(0.1, 0.5, n_bars)
    low = close - rng.uniform(0.1, 0.5, n_bars)
    open_ = close + rng.normal(0, 0.1, n_bars)

    # Simple momentum signal with clear crossings
    momentum = np.zeros(n_bars)
    momentum[10] = 0.02  # Entry at bar 10
    momentum[20] = -0.01  # Exit at bar 20
    momentum[30] = 0.02  # Entry at bar 30
    momentum[50] = -0.01  # Exit at bar 50

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
        "momentum": momentum,
    }


def debug_vbt_pro(data, trailing_stop=0.03):
    """Debug VBT Pro behavior."""
    import vectorbtpro as vbt

    print("\n" + "=" * 60)
    print("VBT Pro Debug")
    print("=" * 60)

    close = data["close"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = momentum < -0.005

    print(f"Entry bars: {np.where(entries)[0]}")
    print(f"Exit bars: {np.where(exits)[0]}")

    # Run with different settings
    print("\n--- Default settings ---")
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        fees=0.001,
        slippage=0.0005,
    )
    print(f"Trades: {pf.trades.count()}")
    print(f"Final value: ${pf.final_value:,.2f}")
    if pf.trades.count() > 0:
        print("\nTrades detail:")
        trades = pf.trades.records_readable
        print(trades.to_string())

    # Try with fixed size
    print("\n--- Fixed size=100 ---")
    pf2 = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        size=100.0,  # Fixed size
        fees=0.001,
        slippage=0.0005,
    )
    print(f"Trades: {pf2.trades.count()}")
    print(f"Final value: ${pf2.final_value:,.2f}")
    if pf2.trades.count() > 0:
        print("\nTrades detail:")
        trades2 = pf2.trades.records_readable
        print(trades2.to_string())

    return pf2


def debug_backtest_nb(data, trailing_stop=0.03):
    """Debug backtest-nb behavior."""
    from ml4t.backtest_nb import HWM_CLOSE, HWM_HIGH, RuleEngine, Signal, backtest

    print("\n" + "=" * 60)
    print("backtest-nb Debug")
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

    # Test with HWM_CLOSE
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
    print(f"Final value: ${result.final_value:,.2f}")
    if result.n_trades > 0:
        trades = result.trades[: result.n_trades]
        print("\nTrades detail:")
        for i, t in enumerate(trades):
            print(f"  Trade {i}: bar {t['entry_bar']}->{t['exit_bar']}, price {t['entry_price']:.2f}->{t['exit_price']:.2f}, pnl=${t['pnl']:.2f}")

    # Test with HWM_HIGH
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
    print(f"Final value: ${result2.final_value:,.2f}")
    if result2.n_trades > 0:
        trades2 = result2.trades[: result2.n_trades]
        print("\nTrades detail:")
        for i, t in enumerate(trades2):
            print(f"  Trade {i}: bar {t['entry_bar']}->{t['exit_bar']}, price {t['entry_price']:.2f}->{t['exit_price']:.2f}, pnl=${t['pnl']:.2f}")

    return result


def main():
    print("Generating simple test data...")
    data = generate_simple_data(100)

    print(f"\nFirst 15 bars:")
    for i in range(15):
        print(f"  Bar {i:2d}: close={data['close'][i]:.2f}, high={data['high'][i]:.2f}, low={data['low'][i]:.2f}, mom={data['momentum'][i]:.3f}")

    debug_vbt_pro(data, trailing_stop=0.03)
    debug_backtest_nb(data, trailing_stop=0.03)


if __name__ == "__main__":
    main()
