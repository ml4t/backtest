"""Pure-Python reference oracle engine for differential testing.

Design rules:
    1. NO imports from ml4t.backtest (zero shared code)
    2. Pure functions + simple dataclasses
    3. Deliberately simpler: market orders only, no risk rules, no limit orders
    4. Explicit about every parameter
    5. Computes: gross_pnl, fees, net_pnl, pnl_percent, final_cash

This oracle handles:
    - Market orders (long and short)
    - SAME_BAR / NEXT_BAR fill timing
    - Percentage commission and slippage
    - Average-cost accounting
    - Signed quantities (positive=long, negative=short)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FillTiming(Enum):
    """When orders fill relative to the signal bar."""
    SAME_BAR = "same_bar"    # Fill at signal bar's close
    NEXT_BAR = "next_bar"    # Fill at next bar's open


@dataclass(frozen=True)
class OracleFillRule:
    """Configuration for how fills are processed."""
    timing: FillTiming = FillTiming.SAME_BAR
    commission_rate: float = 0.0   # Fraction (0.001 = 0.1%)
    slippage_rate: float = 0.0     # Fraction (0.001 = 0.1%)


@dataclass(frozen=True)
class OracleBar:
    """Single price bar."""
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class OracleSignal:
    """Trade signal."""
    bar_index: int
    direction: str   # "long" or "short"
    action: str      # "entry" or "exit"
    quantity: float   # Unsigned


@dataclass
class OracleTrade:
    """Completed round-trip trade computed by the oracle."""
    direction: str
    entry_price: float
    exit_price: float
    quantity: float     # Unsigned
    gross_pnl: float    # Price-move PnL before costs
    fees: float         # Total fees (entry + exit commission)
    net_pnl: float      # gross_pnl - fees
    pnl_percent: float  # Direction-aware return on notional (gross, before fees)
    net_return: float   # net_pnl / notional
    entry_slippage_cost: float
    exit_slippage_cost: float


@dataclass
class OracleResult:
    """Result of running the oracle."""
    trades: list[OracleTrade]
    final_cash: float
    initial_cash: float

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.trades)


def _compute_fill_price(
    bar: OracleBar,
    timing: FillTiming,
    is_entry: bool,
    is_long: bool,
    slippage_rate: float,
    next_bar: OracleBar | None = None,
) -> float:
    """Compute the fill price including slippage.

    Slippage always works against the trader:
    - Buying: fill price is higher (close * (1 + slippage))
    - Selling: fill price is lower (close * (1 - slippage))
    """
    if timing == FillTiming.SAME_BAR:
        base_price = bar.close
    else:
        if next_bar is None:
            return -1.0  # Cannot fill
        base_price = next_bar.open

    # Determine if this is a buy or sell
    if is_long:
        is_buy = is_entry
    else:
        is_buy = not is_entry

    if is_buy:
        return base_price * (1.0 + slippage_rate)
    else:
        return base_price * (1.0 - slippage_rate)


def _compute_commission(price: float, quantity: float, rate: float) -> float:
    """Compute commission: rate * price * quantity."""
    return rate * price * quantity


def run_oracle(
    bars: list[OracleBar],
    signals: list[OracleSignal],
    fill_rule: OracleFillRule | None = None,
    initial_cash: float = 100_000.0,
) -> OracleResult:
    """Run the reference oracle on price bars and signals.

    Args:
        bars: List of OracleBar (one per time step).
        signals: List of OracleSignal (entry/exit signals).
        fill_rule: Fill configuration (timing, commission, slippage).
        initial_cash: Starting cash.

    Returns:
        OracleResult with trades and final cash.
    """
    if fill_rule is None:
        fill_rule = OracleFillRule()

    cash = initial_cash
    trades: list[OracleTrade] = []

    # Sort signals by bar_index
    sorted_signals = sorted(signals, key=lambda s: s.bar_index)

    # Track open position
    position_direction: str | None = None
    position_entry_price: float = 0.0
    position_qty: float = 0.0
    entry_commission: float = 0.0
    entry_slippage_cost: float = 0.0

    for signal in sorted_signals:
        bar_idx = signal.bar_index
        if bar_idx < 0 or bar_idx >= len(bars):
            continue

        bar = bars[bar_idx]
        next_bar = bars[bar_idx + 1] if bar_idx + 1 < len(bars) else None

        if signal.action == "entry" and position_direction is None:
            # Open new position
            is_long = signal.direction == "long"
            fill_price = _compute_fill_price(
                bar, fill_rule.timing, is_entry=True, is_long=is_long,
                slippage_rate=fill_rule.slippage_rate, next_bar=next_bar,
            )
            if fill_price < 0:
                continue  # Cannot fill (no next bar)

            comm = _compute_commission(fill_price, signal.quantity, fill_rule.commission_rate)

            # Slippage cost: difference from base price
            if fill_rule.timing == FillTiming.SAME_BAR:
                base = bar.close
            else:
                base = next_bar.open if next_bar else bar.close
            slippage_cost = abs(fill_price - base) * signal.quantity

            if is_long:
                cash -= fill_price * signal.quantity + comm
            else:
                cash += fill_price * signal.quantity - comm

            position_direction = signal.direction
            position_entry_price = fill_price
            position_qty = signal.quantity
            entry_commission = comm
            entry_slippage_cost = slippage_cost

        elif signal.action == "exit" and position_direction is not None:
            # Close position
            is_long = position_direction == "long"
            fill_price = _compute_fill_price(
                bar, fill_rule.timing, is_entry=False, is_long=is_long,
                slippage_rate=fill_rule.slippage_rate, next_bar=next_bar,
            )
            if fill_price < 0:
                continue

            exit_comm = _compute_commission(fill_price, position_qty, fill_rule.commission_rate)

            if fill_rule.timing == FillTiming.SAME_BAR:
                base = bar.close
            else:
                base = next_bar.open if next_bar else bar.close
            exit_slippage_cost = abs(fill_price - base) * position_qty

            if is_long:
                cash += fill_price * position_qty - exit_comm
            else:
                cash -= fill_price * position_qty + exit_comm

            # Compute trade PnL
            total_fees = entry_commission + exit_comm

            if is_long:
                gross_pnl = (fill_price - position_entry_price) * position_qty
            else:
                gross_pnl = (position_entry_price - fill_price) * position_qty

            net_pnl = gross_pnl - total_fees

            notional = position_entry_price * position_qty
            pnl_percent = gross_pnl / notional if notional > 0 else 0.0
            net_return = net_pnl / notional if notional > 0 else 0.0

            trades.append(OracleTrade(
                direction=position_direction,
                entry_price=position_entry_price,
                exit_price=fill_price,
                quantity=position_qty,
                gross_pnl=gross_pnl,
                fees=total_fees,
                net_pnl=net_pnl,
                pnl_percent=pnl_percent,
                net_return=net_return,
                entry_slippage_cost=entry_slippage_cost,
                exit_slippage_cost=exit_slippage_cost,
            ))

            position_direction = None
            position_entry_price = 0.0
            position_qty = 0.0
            entry_commission = 0.0
            entry_slippage_cost = 0.0

    return OracleResult(
        trades=trades,
        final_cash=cash,
        initial_cash=initial_cash,
    )
