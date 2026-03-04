"""Property-based PnL invariant tests using Hypothesis.

Replaces the existing 109 lines (2 files) with comprehensive invariant testing
over random inputs. ~800 randomized scenarios per test run, with Hypothesis
shrinking to minimal reproducer on failure.

Bug coverage:
    - Bug 1 (short PnL sign): random shorts hit sign mismatch immediately
    - Future bugs: random exploration of edge cases
"""

from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from ml4t.backtest import Broker, OrderSide
from ml4t.backtest.models import NoCommission, NoSlippage, PercentageCommission


def _set_bar(
    broker: Broker,
    price: float,
    *,
    asset: str = "TEST",
    ts: datetime = datetime(2024, 1, 1),
    high: float | None = None,
    low: float | None = None,
) -> None:
    """Set broker bar state for testing."""
    h = high if high is not None else price
    lo = low if low is not None else price
    broker._update_time(
        ts,
        {asset: price},
        {asset: price},
        {asset: h},
        {asset: lo},
        {asset: 1_000_000.0},
        {asset: {}},
    )


# ============================================================================
# Cash Conservation: initial + pnl == final (no costs)
# ============================================================================


@settings(max_examples=200, deadline=5000)
@given(
    entry=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    exit_=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    direction=st.sampled_from(["long", "short"]),
)
def test_cash_conservation(entry: float, exit_: float, qty: float, direction: str) -> None:
    """Cash is perfectly conserved in a round-trip with no costs."""
    initial_cash = 200_000.0
    broker = Broker(
        initial_cash,
        NoCommission(),
        NoSlippage(),
        allow_short_selling=True,
        allow_leverage=True,
    )

    entry_side = OrderSide.BUY if direction == "long" else OrderSide.SELL

    _set_bar(broker, entry)
    broker.submit_order("TEST", qty, entry_side)
    broker._process_orders()

    if broker.get_position("TEST") is None:
        return  # Order rejected (insufficient cash)

    _set_bar(broker, exit_)
    broker.close_position("TEST")
    broker._process_orders()

    assert broker.get_position("TEST") is None
    assert broker.trades

    trade = broker.trades[-1]
    if direction == "long":
        expected_pnl = (exit_ - entry) * qty
    else:
        expected_pnl = (entry - exit_) * qty

    assert abs(trade.pnl - expected_pnl) < 1e-6, (
        f"PnL mismatch: expected {expected_pnl}, got {trade.pnl}"
    )
    assert abs((initial_cash + expected_pnl) - broker.cash) < 1e-6, (
        f"Cash mismatch: expected {initial_cash + expected_pnl}, got {broker.cash}"
    )
    assert abs(broker.get_account_value() - broker.cash) < 1e-6


# ============================================================================
# PnL Sign == PnL Percent Sign (skip breakeven)
# ============================================================================


@settings(max_examples=200, deadline=5000)
@given(
    entry=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    exit_=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    direction=st.sampled_from(["long", "short"]),
)
def test_pnl_sign_matches_pnl_percent_sign(
    entry: float, exit_: float, qty: float, direction: str,
) -> None:
    """sign(pnl) must equal sign(pnl_percent) for non-zero PnL."""
    if abs(entry - exit_) < 0.01:
        return  # Skip near-breakeven (floating point noise)

    initial_cash = 200_000.0
    broker = Broker(
        initial_cash,
        NoCommission(),
        NoSlippage(),
        allow_short_selling=True,
        allow_leverage=True,
    )

    entry_side = OrderSide.BUY if direction == "long" else OrderSide.SELL

    _set_bar(broker, entry)
    broker.submit_order("TEST", qty, entry_side)
    broker._process_orders()

    if broker.get_position("TEST") is None:
        return  # Order rejected

    _set_bar(broker, exit_)
    broker.close_position("TEST")
    broker._process_orders()

    if not broker.trades:
        return

    trade = broker.trades[-1]
    if abs(trade.pnl) < 1e-8:
        return  # Breakeven

    pnl_sign = 1 if trade.pnl > 0 else -1
    pct_sign = 1 if trade.pnl_percent > 0 else -1

    assert pnl_sign == pct_sign, (
        f"Sign mismatch: pnl={trade.pnl} (sign={pnl_sign}), "
        f"pnl_percent={trade.pnl_percent} (sign={pct_sign}), "
        f"direction={direction}, entry={entry}, exit={exit_}, qty={qty}"
    )


# ============================================================================
# Gross - Fees == Net (with random commission rate)
# ============================================================================


@settings(max_examples=200, deadline=5000)
@given(
    entry=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    exit_=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    direction=st.sampled_from(["long", "short"]),
    comm_rate=st.floats(min_value=0.0, max_value=0.01, allow_nan=False, allow_infinity=False),
)
def test_gross_minus_fees_equals_net(
    entry: float, exit_: float, qty: float, direction: str, comm_rate: float,
) -> None:
    """gross_pnl - fees == pnl for any commission rate."""
    initial_cash = 500_000.0
    broker = Broker(
        initial_cash,
        PercentageCommission(comm_rate),
        NoSlippage(),
        allow_short_selling=True,
        allow_leverage=True,
    )

    entry_side = OrderSide.BUY if direction == "long" else OrderSide.SELL

    _set_bar(broker, entry)
    broker.submit_order("TEST", qty, entry_side)
    broker._process_orders()

    _set_bar(broker, exit_)
    broker.close_position("TEST")
    broker._process_orders()

    if not broker.trades:
        return  # Trade may have been rejected

    trade = broker.trades[-1]
    expected_net = trade.gross_pnl - trade.fees
    assert abs(expected_net - trade.pnl) < max(1e-6, abs(trade.gross_pnl) * 1e-6), (
        f"Decomposition failed: gross({trade.gross_pnl}) - fees({trade.fees}) "
        f"= {expected_net} != pnl({trade.pnl})"
    )


# ============================================================================
# Price Scale Invariance: returns unchanged when scaling prices and cash
# ============================================================================


@settings(max_examples=100, deadline=5000)
@given(
    base_entry=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    base_exit=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    direction=st.sampled_from(["long", "short"]),
)
def test_price_scale_invariance(
    base_entry: float, base_exit: float, qty: float, scale: float, direction: str,
) -> None:
    """Percentage returns should be the same regardless of price scale."""
    if abs(base_entry - base_exit) < 0.01:
        return

    def _run_trip(entry_p, exit_p, cash):
        broker = Broker(
            cash,
            NoCommission(),
            NoSlippage(),
            allow_short_selling=True,
            allow_leverage=True,
        )
        side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        _set_bar(broker, entry_p)
        broker.submit_order("TEST", qty, side)
        broker._process_orders()
        _set_bar(broker, exit_p)
        broker.close_position("TEST")
        broker._process_orders()
        return broker.trades[-1] if broker.trades else None

    t1 = _run_trip(base_entry, base_exit, 500_000.0)
    t2 = _run_trip(base_entry * scale, base_exit * scale, 500_000.0 * scale)

    if t1 is None or t2 is None:
        return

    assert abs(t1.pnl_percent - t2.pnl_percent) < 1e-6, (
        f"Returns differ with scale={scale}: {t1.pnl_percent} vs {t2.pnl_percent}"
    )


# ============================================================================
# Idempotent Close: closing an already-flat position has no effect
# ============================================================================


@settings(max_examples=100, deadline=5000)
@given(
    price=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
)
def test_idempotent_close(price: float) -> None:
    """Closing a position that doesn't exist should have no effect."""
    broker = Broker(100_000.0, NoCommission(), NoSlippage())

    _set_bar(broker, price)
    cash_before = broker.cash
    trades_before = len(broker.trades)

    broker.close_position("TEST")
    broker._process_orders()

    assert broker.cash == cash_before
    assert len(broker.trades) == trades_before
    assert broker.get_position("TEST") is None
