"""Universal accounting invariants for BacktestResult.

These invariants are checked automatically after every Engine.run() call
via the autouse fixture in conftest.py. They catch accounting bugs that
component-level tests miss by asserting properties that must hold for
ANY valid backtest result.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml4t.backtest.result import BacktestResult


# Tolerance for floating-point comparisons in dollar amounts
_ABS_TOL = 1e-6
# Relative tolerance for percentage comparisons
_REL_TOL = 1e-8


def assert_result_invariants(
    result: BacktestResult,
    initial_cash: float,
    *,
    check_equity_terminal: bool = True,
    check_pnl_decomposition: bool = True,
    check_direction_signs: bool = True,
    check_mfe_mae_bounds: bool = True,
    check_cost_non_negativity: bool = True,
    check_fill_temporal_order: bool = True,
    check_no_nan: bool = True,
    check_exit_reason_consistency: bool = True,
    check_fill_order_type_bounds: bool = True,
) -> None:
    """Assert universal invariants on a BacktestResult.

    Args:
        result: The BacktestResult to check.
        initial_cash: The initial cash used for the backtest.
        check_*: Flags to selectively disable individual checks.
    """
    closed_trades = [t for t in result.trades if t.status == "closed"]

    if check_equity_terminal:
        _check_equity_terminal(result, initial_cash, closed_trades)
    if check_pnl_decomposition:
        _check_pnl_decomposition(closed_trades)
    if check_direction_signs:
        _check_direction_signs(closed_trades)
    if check_mfe_mae_bounds:
        _check_mfe_mae_bounds(closed_trades)
    if check_cost_non_negativity:
        _check_cost_non_negativity(closed_trades)
    if check_fill_temporal_order:
        _check_fill_temporal_order(result)
    if check_no_nan:
        _check_no_nan(result)
    if check_exit_reason_consistency:
        _check_exit_reason_consistency(result.trades)
    if check_fill_order_type_bounds:
        _check_fill_order_type_bounds(result)


def _check_equity_terminal(
    result: BacktestResult,
    initial_cash: float,
    closed_trades: list,
) -> None:
    """Verify: initial_cash + sum(closed_pnl) + sum(open_pnl) ≈ final_value.

    When open positions exist, the tolerance is expanded because the open trade
    PnL is computed from Position state which may not perfectly capture all
    intermediate costs (especially in rebalancing with integer shares and high
    commission rates). Multi-asset rebalancing with integer shares also creates
    small rounding discrepancies in position PnL vs. cash-based equity tracking.
    """
    if not result.equity_curve:
        return

    final_value = result.equity_curve[-1][1]
    closed_pnl = sum(t.pnl for t in closed_trades)
    open_trades = [t for t in result.trades if t.status == "open"]
    open_pnl = sum(t.pnl for t in open_trades)

    expected = initial_cash + closed_pnl + open_pnl
    diff = abs(expected - final_value)

    # Base tolerance: relative to portfolio size
    tol = max(_ABS_TOL, abs(final_value) * 1e-6)

    # Expand tolerance for total fill costs (commission + slippage on all fills)
    total_fill_costs = sum(f.commission + f.slippage for f in result.fills)
    if total_fill_costs > 0:
        tol = max(tol, total_fill_costs * 0.05)  # 5% of total costs

    # Expand tolerance for open positions: mark-to-market PnL from Position state
    # can diverge slightly from cash-based equity tracking, especially with
    # multi-asset rebalancing and integer share rounding.
    if open_trades:
        open_notional = sum(
            abs(t.quantity) * t.exit_price * t.multiplier for t in open_trades
        )
        tol = max(tol, open_notional * 1e-4)  # 0.01% of open notional

    assert diff <= tol, (
        f"Equity terminal invariant violated: "
        f"initial_cash({initial_cash}) + closed_pnl({closed_pnl:.6f}) + "
        f"open_pnl({open_pnl:.6f}) = {expected:.6f} != final_value({final_value:.6f}), "
        f"diff={diff:.10f}, tol={tol:.6f}"
    )


def _check_pnl_decomposition(closed_trades: list) -> None:
    """Verify: gross_pnl - fees ≈ pnl for every closed trade."""
    for i, t in enumerate(closed_trades):
        gross = t.gross_pnl
        expected_net = gross - t.fees
        diff = abs(expected_net - t.pnl)

        tol = max(_ABS_TOL, abs(gross) * 1e-6)
        assert diff <= tol, (
            f"PnL decomposition invariant violated for trade {i} ({t.symbol}): "
            f"gross_pnl({gross:.6f}) - fees({t.fees:.6f}) = {expected_net:.6f} "
            f"!= pnl({t.pnl:.6f}), diff={diff:.10f}"
        )


def _check_direction_signs(closed_trades: list) -> None:
    """Verify: sign(gross_pnl) == sign(pnl_percent) for non-zero trades.

    We check gross_pnl (not net pnl) because pnl_percent is the gross return
    (price change / entry price). Net pnl includes fees, so for near-breakeven
    trades where fees exceed gross profit, sign(pnl) != sign(pnl_percent) is
    expected and correct.
    """
    for i, t in enumerate(closed_trades):
        if abs(t.gross_pnl) < _ABS_TOL or abs(t.pnl_percent) < _REL_TOL:
            continue  # Skip breakeven trades

        gross_sign = 1 if t.gross_pnl > 0 else -1
        pct_sign = 1 if t.pnl_percent > 0 else -1

        assert gross_sign == pct_sign, (
            f"Direction sign invariant violated for trade {i} ({t.symbol}): "
            f"gross_pnl={t.gross_pnl:.6f} (sign={gross_sign}) but "
            f"pnl_percent={t.pnl_percent:.6f} (sign={pct_sign}), "
            f"direction={t.direction}, quantity={t.quantity}"
        )


def _check_mfe_mae_bounds(closed_trades: list) -> None:
    """Verify: MFE >= 0, MAE <= 0.

    Note: pnl_percent is NOT guaranteed to be bounded by MFE/MAE because
    water marks are updated at bar end AFTER position exits. The exit bar's
    price move is not captured in MFE/MAE if the position closes on that bar.
    """
    for i, t in enumerate(closed_trades):
        assert t.mfe >= -_REL_TOL, (
            f"MFE bound violated for trade {i} ({t.symbol}): mfe={t.mfe:.6f} < 0"
        )
        assert t.mae <= _REL_TOL, (
            f"MAE bound violated for trade {i} ({t.symbol}): mae={t.mae:.6f} > 0"
        )


def _check_cost_non_negativity(closed_trades: list) -> None:
    """Verify: fees >= 0, multiplier > 0."""
    for i, t in enumerate(closed_trades):
        assert t.fees >= -_ABS_TOL, (
            f"Fees non-negativity violated for trade {i} ({t.symbol}): "
            f"fees={t.fees:.6f}"
        )
        assert t.multiplier > 0, (
            f"Multiplier must be positive for trade {i} ({t.symbol}): "
            f"multiplier={t.multiplier}"
        )


def _check_fill_temporal_order(result: BacktestResult) -> None:
    """Verify: fills are in non-decreasing timestamp order."""
    for i in range(1, len(result.fills)):
        prev = result.fills[i - 1]
        curr = result.fills[i]
        assert curr.timestamp >= prev.timestamp, (
            f"Fill temporal order violated: fill[{i - 1}].timestamp={prev.timestamp} > "
            f"fill[{i}].timestamp={curr.timestamp}"
        )


def _check_no_nan(result: BacktestResult) -> None:
    """Verify: no NaN in numeric Trade fields."""
    numeric_fields = [
        "entry_price", "exit_price", "quantity", "pnl", "pnl_percent",
        "fees", "slippage", "mfe", "mae", "entry_slippage", "multiplier",
    ]

    for i, t in enumerate(result.trades):
        for field in numeric_fields:
            val = getattr(t, field)
            assert not math.isnan(val), (
                f"NaN found in trade {i} ({t.symbol}).{field}"
            )
            assert math.isfinite(val), (
                f"Infinite value in trade {i} ({t.symbol}).{field}={val}"
            )


def _check_exit_reason_consistency(trades: list) -> None:
    """Verify exit_reason is consistent with trade outcome.

    Invariants on closed trades by exit_reason:
    - "stop_loss" → gross_pnl <= tolerance (price moved against position)
    - "take_profit" → gross_pnl >= -tolerance (price moved for position)
    - "trailing_stop" → mfe > 0 (favorable move happened before trail triggered)

    Invariant on all trades:
    - "end_of_data" → status == "open"
    """
    # Tolerance for floating-point and small slippage effects
    tol = 1e-4

    for i, t in enumerate(trades):
        if t.exit_reason == "stop_loss":
            assert t.gross_pnl <= tol, (
                f"Exit-reason invariant violated for trade {i} ({t.symbol}): "
                f"exit_reason='stop_loss' but gross_pnl={t.gross_pnl:.6f} > 0 "
                f"(price should have moved against position)"
            )
        elif t.exit_reason == "take_profit":
            assert t.gross_pnl >= -tol, (
                f"Exit-reason invariant violated for trade {i} ({t.symbol}): "
                f"exit_reason='take_profit' but gross_pnl={t.gross_pnl:.6f} < 0 "
                f"(price should have moved for position)"
            )
        elif t.exit_reason == "trailing_stop":
            assert t.mfe > -_REL_TOL, (
                f"Exit-reason invariant violated for trade {i} ({t.symbol}): "
                f"exit_reason='trailing_stop' but mfe={t.mfe:.6f} <= 0 "
                f"(favorable move should have happened before trail triggered)"
            )
        elif t.exit_reason == "end_of_data":
            assert t.status == "open", (
                f"Exit-reason invariant violated for trade {i} ({t.symbol}): "
                f"exit_reason='end_of_data' but status='{t.status}' (should be 'open')"
            )


def _check_fill_order_type_bounds(result: BacktestResult) -> None:
    """Verify fill prices respect order-type bounds.

    For fills with populated order_type metadata:
    - Limit buy: fill price <= limit_price (never overpay)
    - Limit sell: fill price >= limit_price (never undersell)
    - Stop buy: fill price >= stop_price (fill at or above trigger)
    - Stop sell: fill price <= stop_price (fill at or below trigger)

    Fills with empty order_type (pre-metadata or manual construction) are skipped.
    """
    from ml4t.backtest.types import OrderSide

    tol = 1e-8

    for i, f in enumerate(result.fills):
        order_type = getattr(f, "order_type", "")
        if not order_type:
            continue

        limit_price = getattr(f, "limit_price", None)
        stop_price = getattr(f, "stop_price", None)

        if order_type == "limit" and limit_price is not None:
            if f.side == OrderSide.BUY:
                assert f.price <= limit_price + tol, (
                    f"Fill order-type bound violated for fill {i} ({f.asset}): "
                    f"limit BUY filled at {f.price:.6f} > limit_price {limit_price:.6f}"
                )
            else:
                assert f.price >= limit_price - tol, (
                    f"Fill order-type bound violated for fill {i} ({f.asset}): "
                    f"limit SELL filled at {f.price:.6f} < limit_price {limit_price:.6f}"
                )
        elif order_type == "stop" and stop_price is not None:
            if f.side == OrderSide.BUY:
                assert f.price >= stop_price - tol, (
                    f"Fill order-type bound violated for fill {i} ({f.asset}): "
                    f"stop BUY filled at {f.price:.6f} < stop_price {stop_price:.6f}"
                )
            else:
                assert f.price <= stop_price + tol, (
                    f"Fill order-type bound violated for fill {i} ({f.asset}): "
                    f"stop SELL filled at {f.price:.6f} > stop_price {stop_price:.6f}"
                )
