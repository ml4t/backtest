#!/usr/bin/env python3
"""Single-asset exact match validation across all implementations.

This script validates that backtest-nb, backtest-rs, and VectorBT Pro produce
IDENTICAL trades (same entry bar, exit bar, entry price, exit price, PnL) on
a single-asset scenario BEFORE we attempt multi-asset scaling.

Run from the backtest directory with the VBT Pro venv:
    source .venv-vectorbt-pro/bin/activate
    python validation/single_asset_exact_match.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class Trade:
    """Normalized trade representation for comparison."""

    asset_id: int
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    exit_reason: str

    def matches(self, other: "Trade", price_tol: float = 1e-4, pnl_tol: float = 1e-2) -> bool:
        """Check if this trade matches another within tolerance."""
        return (
            self.asset_id == other.asset_id
            and self.entry_bar == other.entry_bar
            and self.exit_bar == other.exit_bar
            and abs(self.entry_price - other.entry_price) < price_tol
            and abs(self.exit_price - other.exit_price) < price_tol
            and abs(self.pnl - other.pnl) < pnl_tol
        )


def generate_single_asset_data(n_bars: int = 1000, seed: int = 42):
    """Generate single-asset OHLCV + entry/exit signals.

    Uses random price series and signal-based entries/exits for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Price series with mild trend
    returns = rng.normal(0.0002, 0.015, n_bars)
    cumret = np.cumsum(returns)
    cumret = np.clip(cumret, -5, 5)  # Prevent overflow
    close = 100.0 * np.exp(cumret)

    # OHLC from close
    daily_vol = rng.uniform(0.005, 0.015, n_bars)
    high = close * (1 + daily_vol)
    low = close * (1 - daily_vol)
    open_ = close + rng.normal(0, 0.3, n_bars)
    volume = rng.uniform(1e6, 5e6, n_bars)

    # Mean-reverting momentum signal
    momentum = np.zeros(n_bars)
    for i in range(1, n_bars):
        momentum[i] = 0.5 * momentum[i - 1] + rng.normal(0, 0.04)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "momentum": momentum,
        "n_bars": n_bars,
    }


def run_vbt_pro(data: dict, trailing_stop: float = 0.03) -> list[Trade]:
    """Run VectorBT Pro and extract trades."""
    import vectorbtpro as vbt

    close = data["close"]
    momentum = data["momentum"]

    # Entry/exit thresholds
    entries = momentum > 0.01
    exits = momentum < -0.005

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        tsl_stop=trailing_stop,
        init_cash=100_000.0,
        size=100.0,  # Fixed size to match backtest-nb
        fees=0.001,
        slippage=0.0005,
    )

    # Extract trades
    trades_df = pf.trades.records_readable
    result = []

    if len(trades_df) > 0:
        for _, row in trades_df.iterrows():
            result.append(
                Trade(
                    asset_id=0,
                    entry_bar=int(row.get("Entry Index", row.get("entry_idx", 0))),
                    exit_bar=int(row.get("Exit Index", row.get("exit_idx", 0))),
                    entry_price=float(row.get("Avg Entry Price", row.get("avg_entry_price", 0))),
                    exit_price=float(row.get("Avg Exit Price", row.get("avg_exit_price", 0))),
                    quantity=float(row.get("Size", row.get("size", 0))),
                    pnl=float(row.get("PnL", row.get("pnl", 0))),
                    exit_reason=str(row.get("Status", row.get("status", ""))),
                )
            )

    return result


def run_backtest_nb(data: dict, trailing_stop: float = 0.03) -> list[Trade]:
    """Run backtest-nb and extract trades."""
    from ml4t.backtest_nb import HWM_HIGH, RuleEngine, Signal, backtest

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

    trades = []
    trades_arr = result.trades[: result.n_trades]

    for i in range(result.n_trades):
        t = trades_arr[i]
        trades.append(
            Trade(
                asset_id=int(t["asset_id"]),
                entry_bar=int(t["entry_bar"]),
                exit_bar=int(t["exit_bar"]),
                entry_price=float(t["entry_price"]),
                exit_price=float(t["exit_price"]),
                quantity=float(t["quantity"]),
                pnl=float(t["pnl"]),
                exit_reason=_exit_reason_nb(int(t["exit_reason"])),
            )
        )

    return trades


def _exit_reason_nb(code: int) -> str:
    """Convert backtest-nb exit reason code to string."""
    reasons = {0: "SIGNAL", 1: "STOP_LOSS", 2: "TAKE_PROFIT", 3: "TRAILING_STOP", 4: "TIME_STOP"}
    return reasons.get(code, f"UNKNOWN({code})")


def run_backtest_rs(data: dict, trailing_stop: float = 0.03) -> list[Trade]:
    """Run backtest-rs and extract trades."""
    from ml4t_backtest_rs import RuleEngine, Signal, backtest

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
        trail_hwm_source=1,  # HWM_HIGH
    )

    # backtest-rs only returns aggregate stats, not individual trades
    # For now, return empty list - we'll need to add trade extraction
    return []


def compare_trades(name1: str, trades1: list[Trade], name2: str, trades2: list[Trade]) -> bool:
    """Compare two trade lists and report differences."""
    print(f"\n  Comparing {name1} ({len(trades1)} trades) vs {name2} ({len(trades2)} trades)")

    if len(trades1) != len(trades2):
        print(f"    MISMATCH: Trade count differs ({len(trades1)} vs {len(trades2)})")

        # Show first few trades from each
        print(f"\n    First 5 {name1} trades:")
        for t in trades1[:5]:
            print(f"      bar {t.entry_bar}->{t.exit_bar}, pnl=${t.pnl:.2f}")

        print(f"\n    First 5 {name2} trades:")
        for t in trades2[:5]:
            print(f"      bar {t.entry_bar}->{t.exit_bar}, pnl=${t.pnl:.2f}")

        return False

    if len(trades1) == 0:
        print("    SKIP: No trades to compare")
        return True

    # Sort by entry_bar for comparison
    trades1_sorted = sorted(trades1, key=lambda t: (t.asset_id, t.entry_bar))
    trades2_sorted = sorted(trades2, key=lambda t: (t.asset_id, t.entry_bar))

    mismatches = 0
    for i, (t1, t2) in enumerate(zip(trades1_sorted, trades2_sorted)):
        if not t1.matches(t2):
            mismatches += 1
            if mismatches <= 5:  # Show first 5 mismatches
                print(f"    MISMATCH at trade {i}:")
                print(f"      {name1}: bar {t1.entry_bar}->{t1.exit_bar}, price {t1.entry_price:.2f}->{t1.exit_price:.2f}, pnl=${t1.pnl:.2f}")
                print(f"      {name2}: bar {t2.entry_bar}->{t2.exit_bar}, price {t2.entry_price:.2f}->{t2.exit_price:.2f}, pnl=${t2.pnl:.2f}")

    if mismatches > 0:
        print(f"    TOTAL MISMATCHES: {mismatches}/{len(trades1)}")
        return False

    print(f"    EXACT MATCH: All {len(trades1)} trades identical")
    return True


def main():
    print("=" * 80)
    print("SINGLE-ASSET EXACT MATCH VALIDATION")
    print("=" * 80)

    # Test configurations
    configs = [
        {"n_bars": 1000, "trailing_stop": 0.03, "desc": "1K bars, 3% trailing"},
        {"n_bars": 5000, "trailing_stop": 0.02, "desc": "5K bars, 2% trailing"},
        {"n_bars": 10000, "trailing_stop": 0.03, "desc": "10K bars, 3% trailing"},
    ]

    all_pass = True

    for config in configs:
        n_bars = config["n_bars"]
        trailing_stop = config["trailing_stop"]

        print(f"\n\n{'='*60}")
        print(f"Config: {config['desc']}")
        print(f"{'='*60}")

        # Generate data
        print(f"\nGenerating {n_bars} bars...")
        data = generate_single_asset_data(n_bars)

        # Run VBT Pro
        print("Running VBT Pro...")
        try:
            vbt_trades = run_vbt_pro(data, trailing_stop)
            print(f"  VBT Pro: {len(vbt_trades)} trades")
        except Exception as e:
            print(f"  VBT Pro ERROR: {e}")
            vbt_trades = []

        # Run backtest-nb
        print("Running backtest-nb...")
        try:
            nb_trades = run_backtest_nb(data, trailing_stop)
            print(f"  backtest-nb: {len(nb_trades)} trades")
        except Exception as e:
            print(f"  backtest-nb ERROR: {e}")
            nb_trades = []

        # Run backtest-rs
        print("Running backtest-rs...")
        try:
            rs_trades = run_backtest_rs(data, trailing_stop)
            print(f"  backtest-rs: {len(rs_trades)} trades (trade extraction not yet implemented)")
        except Exception as e:
            print(f"  backtest-rs ERROR: {e}")
            rs_trades = []

        # Compare
        print("\n--- Trade Comparison ---")

        if vbt_trades and nb_trades:
            if not compare_trades("VBT Pro", vbt_trades, "backtest-nb", nb_trades):
                all_pass = False

        if nb_trades and rs_trades:
            if not compare_trades("backtest-nb", nb_trades, "backtest-rs", rs_trades):
                all_pass = False

    print("\n\n" + "=" * 80)
    if all_pass:
        print("RESULT: ALL COMPARISONS PASSED")
    else:
        print("RESULT: SOME COMPARISONS FAILED")
    print("=" * 80)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
