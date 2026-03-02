#!/usr/bin/env python3
"""Compare VBT Pro vs backtest-nb with signal-only exits (no stops)."""

import numpy as np
import polars as pl


def generate_data(n_bars: int = 500, seed: int = 42):
    """Generate test data with frequent signal crossings."""
    rng = np.random.default_rng(seed)

    returns = rng.normal(0.0002, 0.015, n_bars)
    cumret = np.clip(np.cumsum(returns), -5, 5)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.3, n_bars)

    # Mean-reverting momentum
    momentum = np.zeros(n_bars)
    for i in range(1, n_bars):
        momentum[i] = 0.5 * momentum[i - 1] + rng.normal(0, 0.04)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
        "momentum": momentum,
    }


def run_vbt_signal_only(data):
    """Run VBT Pro with signal-only entries/exits."""
    import vectorbtpro as vbt

    close = data["close"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = momentum < -0.005

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=100_000.0,
        size=100.0,
        fees=0.001,
        slippage=0.0005,
    )

    return pf.trades.count(), pf.final_value


def run_nb_signal_only(data):
    """Run backtest-nb with signal-only entries/exits."""
    from ml4t.backtest_nb import RuleEngine, Signal, backtest

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
        exit=Signal("momentum") < -0.005,
        position_size=100.0,
    )

    result = backtest(
        prices, signals, strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
    )

    return result.n_trades, result.final_value


def main():
    print("=" * 60)
    print("Signal-Only Comparison (No Trailing Stop)")
    print("=" * 60)

    for n_bars in [100, 500, 1000]:
        print(f"\n--- {n_bars} bars ---")
        data = generate_data(n_bars)

        entries = data["momentum"] > 0.01
        exits = data["momentum"] < -0.005
        print(f"Entry signals: {np.sum(entries)}")
        print(f"Exit signals: {np.sum(exits)}")

        vbt_trades, vbt_value = run_vbt_signal_only(data)
        nb_trades, nb_value = run_nb_signal_only(data)

        print(f"VBT Pro: {vbt_trades} trades, ${vbt_value:,.2f}")
        print(f"backtest-nb: {nb_trades} trades, ${nb_value:,.2f}")
        print(f"Match: {'YES' if vbt_trades == nb_trades and abs(vbt_value - nb_value) < 0.01 else 'NO'}")


if __name__ == "__main__":
    main()
