#!/usr/bin/env python3
"""Debug basic entry behavior."""

import numpy as np
import polars as pl


def main():
    # Simple test: single entry signal
    n_bars = 20
    close = np.array([100.0 + i * 0.5 for i in range(n_bars)])
    high = close + 0.2
    low = close - 0.2
    open_ = close

    momentum = np.zeros(n_bars)
    momentum[5] = 0.05  # Entry signal at bar 5

    print("Input data:")
    for i in range(n_bars):
        print(f"  Bar {i:2d}: close={close[i]:.2f}, momentum={momentum[i]:.2f}")

    prices = pl.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1e6,
    })
    signals = pl.DataFrame({"momentum": momentum})

    from ml4t.backtest_nb import RuleEngine, Signal, backtest

    strategy = RuleEngine(
        entry=Signal("momentum") > 0.01,
        exit=Signal("momentum") < -0.01,
        position_size=10.0,
    )

    result = backtest(
        prices,
        signals,
        strategy,
        initial_cash=10_000.0,
        commission=0.0,
        slippage=0.0,
    )

    print(f"\nResults:")
    print(f"  n_trades: {result.n_trades}")
    print(f"  n_fills: {result.n_fills}")
    print(f"  final_value: ${result.final_value:.2f}")

    if result.n_fills > 0:
        print("\nFills:")
        fills = result.fills[:result.n_fills]
        for i, f in enumerate(fills):
            print(f"  Fill {i}: bar={f['bar_idx']}, side={f['side']}, qty={f['quantity']}, price={f['price']:.2f}")

    if result.n_trades > 0:
        print("\nTrades:")
        trades = result.trades[:result.n_trades]
        for i, t in enumerate(trades):
            print(f"  Trade {i}: bar {t['entry_bar']}->{t['exit_bar']}, pnl=${t['pnl']:.2f}")


if __name__ == "__main__":
    main()
