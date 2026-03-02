#!/usr/bin/env python3
"""Compare trailing stop behavior at different percentages."""

import numpy as np
import polars as pl


def generate_data(n_bars: int = 500, seed: int = 42):
    """Generate test data."""
    rng = np.random.default_rng(seed)

    returns = rng.normal(0.0002, 0.015, n_bars)
    cumret = np.clip(np.cumsum(returns), -5, 5)
    close = 100.0 * np.exp(cumret)

    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.3, n_bars)

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


def run_vbt(data, trailing_stop):
    """Run VBT Pro."""
    import vectorbtpro as vbt

    close = data["close"]
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = momentum < -0.005

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

    return pf.trades.count()


def run_nb(data, trailing_stop, hwm_source):
    """Run backtest-nb."""
    from ml4t.backtest_nb import HWM_CLOSE, HWM_HIGH, RuleEngine, Signal, backtest

    prices = pl.DataFrame({
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
    })
    signals = pl.DataFrame({"momentum": data["momentum"]})

    hwm = HWM_HIGH if hwm_source == "HIGH" else HWM_CLOSE

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -0.005,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    result = backtest(
        prices, signals, strategy,
        initial_cash=100_000.0,
        commission=0.001,
        slippage=0.0005,
        trail_hwm_source=hwm,
    )

    return result.n_trades


def main():
    print("=" * 80)
    print("Trailing Stop Comparison")
    print("=" * 80)

    data = generate_data(1000)

    print(f"\n{'TS %':>6} | {'VBT Pro':>10} | {'NB HWM_CLOSE':>12} | {'NB HWM_HIGH':>12} | {'Diff (VBT-HIGH)':>15}")
    print("-" * 80)

    for ts_pct in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]:
        vbt_trades = run_vbt(data, ts_pct)
        nb_close_trades = run_nb(data, ts_pct, "CLOSE")
        nb_high_trades = run_nb(data, ts_pct, "HIGH")

        diff = vbt_trades - nb_high_trades
        print(f"{ts_pct*100:>5.0f}% | {vbt_trades:>10} | {nb_close_trades:>12} | {nb_high_trades:>12} | {diff:>15}")


if __name__ == "__main__":
    main()
