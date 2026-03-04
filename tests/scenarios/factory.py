"""Scenario factory with analytically computed expected results.

Build scenarios where the expected result is computed BEFORE the SUT runs.
The factory uses only basic arithmetic — no shared code with ml4t.backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True)
class ExpectedResult:
    """Analytically computed expected values for a round-trip trade."""

    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    pnl_percent: float  # Gross return on notional (direction-aware)
    final_cash: float


@dataclass
class Scenario:
    """Complete test scenario with prices and expected results."""

    name: str
    prices_df: pl.DataFrame
    expected: ExpectedResult
    config_overrides: dict
    entry_bar: int = 0
    exit_bar: int = 2


def make_round_trip(
    entry_price: float,
    exit_price: float,
    quantity: float = 100.0,
    direction: str = "long",
    commission_rate: float = 0.0,
    slippage_rate: float = 0.0,
    initial_cash: float = 100_000.0,
    asset: str = "TEST",
) -> Scenario:
    """Create a round-trip scenario with analytically computed expected results.

    Args:
        entry_price: Price at entry bar.
        exit_price: Price at exit bar.
        quantity: Unsigned trade quantity.
        direction: "long" or "short".
        commission_rate: Percentage commission (0.001 = 0.1%).
        slippage_rate: Percentage slippage (0.001 = 0.1%).
        initial_cash: Starting cash.
        asset: Asset symbol.

    Returns:
        Scenario with prices DataFrame and analytically computed expected results.
    """
    # --- Compute fill prices with slippage ---
    if direction == "long":
        # Buy: slippage increases price; Sell: slippage decreases price
        actual_entry = entry_price * (1.0 + slippage_rate)
        actual_exit = exit_price * (1.0 - slippage_rate)
    else:
        # Sell (entry): slippage decreases price; Buy (exit): slippage increases price
        actual_entry = entry_price * (1.0 - slippage_rate)
        actual_exit = exit_price * (1.0 + slippage_rate)

    # --- Compute fees ---
    entry_fee = commission_rate * actual_entry * quantity
    exit_fee = commission_rate * actual_exit * quantity
    total_fees = entry_fee + exit_fee

    # --- Compute PnL ---
    if direction == "long":
        gross_pnl = (actual_exit - actual_entry) * quantity
    else:
        gross_pnl = (actual_entry - actual_exit) * quantity

    net_pnl = gross_pnl - total_fees

    # --- Compute return ---
    notional = actual_entry * quantity
    pnl_percent = gross_pnl / notional if notional > 0 else 0.0

    # --- Compute final cash ---
    if direction == "long":
        cash_after_entry = initial_cash - actual_entry * quantity - entry_fee
        final_cash = cash_after_entry + actual_exit * quantity - exit_fee
    else:
        cash_after_entry = initial_cash + actual_entry * quantity - entry_fee
        final_cash = cash_after_entry - actual_exit * quantity - exit_fee

    # --- Build prices DataFrame ---
    # 3 bars: entry, intermediate, exit
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(days=i) for i in range(3)]

    # All bars at entry_price except exit bar at exit_price
    closes = [entry_price, (entry_price + exit_price) / 2, exit_price]

    prices_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": [asset] * 3,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000.0] * 3,
        }
    )

    expected = ExpectedResult(
        direction=direction,
        entry_price=actual_entry,
        exit_price=actual_exit,
        quantity=quantity,
        gross_pnl=gross_pnl,
        fees=total_fees,
        net_pnl=net_pnl,
        pnl_percent=pnl_percent,
        final_cash=final_cash,
    )

    config_overrides = {
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "initial_cash": initial_cash,
        "allow_short_selling": True,
        "allow_leverage": True,
        "execution_mode": "SAME_BAR",
    }

    name = f"{direction}_{entry_price:.0f}_{exit_price:.0f}_q{quantity:.0f}_c{commission_rate}_s{slippage_rate}"

    return Scenario(
        name=name,
        prices_df=prices_df,
        expected=expected,
        config_overrides=config_overrides,
    )
