#!/usr/bin/env python3
"""Debug trailing stop re-entry behavior."""

import numpy as np
import polars as pl


def generate_data():
    """Generate data with specific trailing stop scenario."""
    n_bars = 30
    # Price rises to 110, drops to trigger 3% trailing stop at ~106.7, then rises again
    close = np.array([
        100.0, 101.0, 102.0, 103.0, 104.0,  # Rise bars 0-4
        105.0, 106.0, 107.0, 108.0, 109.0,  # Rise bars 5-9
        110.0,  # Peak bar 10
        109.0, 108.0, 107.0, 106.0, 105.0,  # Drop bars 11-15 (3% stop triggers around 106.7)
        106.0, 107.0, 108.0, 109.0, 110.0,  # Rise again bars 16-20
        111.0, 112.0, 113.0, 114.0, 115.0,  # Continue rise 21-25
        116.0, 117.0, 118.0, 119.0,         # bars 26-29
    ])

    high = close + 0.5
    low = close - 0.5
    open_ = close

    # Entry signal at bar 2 (and stays above threshold until bar 20)
    momentum = np.zeros(n_bars)
    momentum[2:20] = 0.05  # Continuous entry signal

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
    momentum = data["momentum"]

    entries = momentum > 0.01
    exits = momentum < -0.005

    print("\nVBT Pro Debug:")
    print(f"  Entry bars: {np.where(entries)[0].tolist()}")
    print(f"  Exit bars: {np.where(exits)[0].tolist()}")

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

    print(f"  Trades: {pf.trades.count()}")
    if pf.trades.count() > 0:
        trades = pf.trades.records_readable
        for _, row in trades.iterrows():
            entry_idx = int(row.get("Entry Index", row.get("entry_idx", 0)))
            exit_idx = int(row.get("Exit Index", row.get("exit_idx", 0)))
            status = row.get("Status", row.get("status", ""))
            print(f"    bar {entry_idx}->{exit_idx} ({status})")

    return pf


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
        exit=Signal("momentum") < -0.005,
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

    print("\nbacktest-nb Debug:")
    print(f"  Trades: {result.n_trades}")
    if result.n_trades > 0:
        trades = result.trades[:result.n_trades]
        for i, t in enumerate(trades):
            exit_reasons = {0: "SIGNAL", 1: "STOP_LOSS", 2: "TAKE_PROFIT", 3: "TRAILING_STOP", 4: "TIME_STOP"}
            reason = exit_reasons.get(int(t["exit_reason"]), "?")
            print(f"    bar {t['entry_bar']}->{t['exit_bar']} ({reason})")

    return result


def main():
    print("=" * 60)
    print("Trailing Stop Re-entry Debug")
    print("=" * 60)

    data = generate_data()

    print("\nPrice data:")
    for i in range(len(data["close"])):
        mom = data["momentum"][i]
        print(f"  Bar {i:2d}: close={data['close'][i]:.1f}, mom={mom:.2f}")

    run_vbt(data, trailing_stop=0.03)
    run_nb(data, trailing_stop=0.03)


if __name__ == "__main__":
    main()
