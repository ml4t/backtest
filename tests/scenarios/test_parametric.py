"""Parametric sweep: directions x prices x quantities x commissions.

2 directions x 5 price pairs x 3 quantities x 3 commission rates = 90 scenarios,
each with analytically computed expected results.
"""

from __future__ import annotations

from itertools import product

import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine

from ..helpers.strategies import RoundTripStrategy
from .factory import Scenario, make_round_trip

# ============================================================================
# Build parametric sweep
# ============================================================================

DIRECTIONS = ["long", "short"]
PRICE_PAIRS = [(100.0, 110.0), (100.0, 90.0), (100.0, 100.0), (50.0, 75.0), (200.0, 180.0)]
QUANTITIES = [10.0, 100.0, 500.0]
COMMISSIONS = [0.0, 0.001, 0.005]

SCENARIOS: list[Scenario] = []
for direction, (ep, xp), qty, cr in product(DIRECTIONS, PRICE_PAIRS, QUANTITIES, COMMISSIONS):
    SCENARIOS.append(make_round_trip(ep, xp, qty, direction, commission_rate=cr))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_round_trip_matches_expected(scenario: Scenario):
    """Run SUT and compare each field against analytically computed expected values."""
    config = BacktestConfig(**scenario.config_overrides)
    feed = DataFeed(prices_df=scenario.prices_df)
    strategy = RoundTripStrategy(
        asset="TEST",
        qty=scenario.expected.quantity,
        entry_bar=scenario.entry_bar,
        exit_bar=scenario.exit_bar,
        direction=scenario.expected.direction,
    )
    engine = Engine(feed, strategy, config)
    result = engine.run()

    closed = [t for t in result.trades if t.status == "closed"]
    assert len(closed) == 1, f"Expected 1 closed trade, got {len(closed)}"
    trade = closed[0]

    exp = scenario.expected
    tol = max(0.02, abs(exp.gross_pnl) * 1e-4)  # Adaptive tolerance

    # Direction
    assert trade.direction == exp.direction

    # Gross PnL
    assert trade.gross_pnl == pytest.approx(exp.gross_pnl, abs=tol), (
        f"Gross PnL: SUT={trade.gross_pnl}, expected={exp.gross_pnl}"
    )

    # Fees
    assert trade.fees == pytest.approx(exp.fees, abs=tol), (
        f"Fees: SUT={trade.fees}, expected={exp.fees}"
    )

    # Net PnL (= pnl field on Trade)
    assert trade.pnl == pytest.approx(exp.net_pnl, abs=tol), (
        f"Net PnL: SUT={trade.pnl}, expected={exp.net_pnl}"
    )

    # PnL percent (gross return)
    if abs(exp.pnl_percent) > 1e-8:
        assert trade.pnl_percent == pytest.approx(exp.pnl_percent, abs=1e-4), (
            f"PnL%: SUT={trade.pnl_percent}, expected={exp.pnl_percent}"
        )

    # Final portfolio value
    if result.equity_curve:
        final_value = result.equity_curve[-1][1]
        assert final_value == pytest.approx(exp.final_cash, abs=tol), (
            f"Final cash: SUT={final_value}, expected={exp.final_cash}"
        )


# ============================================================================
# Slippage scenarios (separate sweep since they interact with fill prices)
# ============================================================================

SLIPPAGE_SCENARIOS: list[Scenario] = []
for direction, (ep, xp) in product(DIRECTIONS, [(100.0, 110.0), (100.0, 90.0)]):
    for slip in [0.001, 0.005]:
        SLIPPAGE_SCENARIOS.append(make_round_trip(ep, xp, 100.0, direction, slippage_rate=slip))


@pytest.mark.parametrize(
    "scenario",
    SLIPPAGE_SCENARIOS,
    ids=[s.name for s in SLIPPAGE_SCENARIOS],
)
def test_slippage_scenario_matches_expected(scenario: Scenario):
    """Verify slippage scenarios match expected results."""
    config = BacktestConfig(**scenario.config_overrides)
    feed = DataFeed(prices_df=scenario.prices_df)
    strategy = RoundTripStrategy(
        asset="TEST",
        qty=scenario.expected.quantity,
        entry_bar=scenario.entry_bar,
        exit_bar=scenario.exit_bar,
        direction=scenario.expected.direction,
    )
    engine = Engine(feed, strategy, config)
    result = engine.run()

    closed = [t for t in result.trades if t.status == "closed"]
    assert len(closed) == 1
    trade = closed[0]
    exp = scenario.expected
    tol = max(0.05, abs(exp.gross_pnl) * 1e-3)

    assert trade.direction == exp.direction
    assert trade.pnl == pytest.approx(exp.net_pnl, abs=tol), (
        f"Net PnL: SUT={trade.pnl}, expected={exp.net_pnl}"
    )
