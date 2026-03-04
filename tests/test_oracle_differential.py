"""Differential tests: compare ml4t-backtest (SUT) against the reference oracle.

Each test runs the same scenario through both engines and compares:
pnl, gross_pnl, pnl_percent, fees, final_cash.

This produces 2 directions x 5 price pairs x 3 commission rates = 30 test cases
from a single parametrized function.

Bug coverage:
    - Bug 1 (short PnL sign): independent pnl_percent computation
    - Bug 2 (cost decomposition): independent fee tracking
"""

from __future__ import annotations

from itertools import product

import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine

from .helpers.data import make_prices
from .helpers.strategies import RoundTripStrategy
from .oracle.engine import OracleBar, OracleFillRule, OracleSignal, run_oracle

# ============================================================================
# Parameters
# ============================================================================

DIRECTIONS = ["long", "short"]
PRICE_PAIRS = [
    (100.0, 110.0),  # Up 10%
    (100.0, 90.0),  # Down 10%
    (100.0, 100.0),  # Flat
    (50.0, 75.0),  # Up 50%
    (200.0, 180.0),  # Down 10%
]
COMMISSION_RATES = [0.0, 0.001, 0.005]
QUANTITIES = [100.0]

# Absolute tolerance for dollar comparisons
_TOL = 0.02  # 2 cents — accounts for slippage rounding differences


def _make_test_id(direction, prices, commission):
    return f"{direction}-{prices[0]:.0f}_{prices[1]:.0f}-comm{commission}"


# Build parameter list
_PARAMS = list(product(DIRECTIONS, PRICE_PAIRS, COMMISSION_RATES))
_IDS = [_make_test_id(d, p, c) for d, p, c in _PARAMS]


@pytest.mark.parametrize(
    "direction,price_pair,commission_rate",
    _PARAMS,
    ids=_IDS,
)
def test_round_trip_matches_oracle(direction, price_pair, commission_rate):
    """Compare SUT round-trip against oracle for PnL, fees, and final cash."""
    entry_price, exit_price = price_pair
    qty = 100.0
    initial_cash = 100_000.0

    # ---- Run Oracle ----
    oracle_bars = [
        OracleBar(entry_price, entry_price, entry_price, entry_price),
        OracleBar(exit_price, exit_price, exit_price, exit_price),  # Intermediate
        OracleBar(exit_price, exit_price, exit_price, exit_price),  # Exit bar
    ]
    oracle_signals = [
        OracleSignal(0, direction, "entry", qty),
        OracleSignal(2, direction, "exit", qty),
    ]
    oracle_rule = OracleFillRule(commission_rate=commission_rate)
    oracle_result = run_oracle(oracle_bars, oracle_signals, oracle_rule, initial_cash)

    assert len(oracle_result.trades) == 1
    oracle_trade = oracle_result.trades[0]

    # ---- Run SUT ----
    closes = [entry_price, exit_price, exit_price]
    prices_df = make_prices(closes)

    config = BacktestConfig(
        allow_short_selling=True,
        allow_leverage=True,
        commission_rate=commission_rate,
        slippage_rate=0.0,
        initial_cash=initial_cash,
        execution_mode="SAME_BAR",
    )
    feed = DataFeed(prices_df=prices_df)
    strategy = RoundTripStrategy(
        asset="TEST",
        qty=qty,
        entry_bar=0,
        exit_bar=2,
        direction=direction,
    )
    engine = Engine(feed, strategy, config)
    sut_result = engine.run()

    sut_closed = [t for t in sut_result.trades if t.status == "closed"]
    assert len(sut_closed) == 1, f"Expected 1 closed trade, got {len(sut_closed)}"
    sut_trade = sut_closed[0]

    # ---- Compare ----
    assert sut_trade.direction == oracle_trade.direction

    # Gross PnL (price move before costs)
    assert sut_trade.gross_pnl == pytest.approx(oracle_trade.gross_pnl, abs=_TOL), (
        f"Gross PnL mismatch: SUT={sut_trade.gross_pnl}, Oracle={oracle_trade.gross_pnl}"
    )

    # Fees
    assert sut_trade.fees == pytest.approx(oracle_trade.fees, abs=_TOL), (
        f"Fees mismatch: SUT={sut_trade.fees}, Oracle={oracle_trade.fees}"
    )

    # Net PnL
    assert sut_trade.pnl == pytest.approx(oracle_trade.net_pnl, abs=_TOL), (
        f"Net PnL mismatch: SUT={sut_trade.pnl}, Oracle={oracle_trade.net_pnl}"
    )

    # PnL percent (direction-aware gross return)
    assert sut_trade.pnl_percent == pytest.approx(oracle_trade.pnl_percent, abs=1e-6), (
        f"PnL% mismatch: SUT={sut_trade.pnl_percent}, Oracle={oracle_trade.pnl_percent}"
    )

    # Final portfolio value
    sut_final = sut_result.equity_curve[-1][1] if sut_result.equity_curve else 0
    assert sut_final == pytest.approx(oracle_result.final_cash, abs=_TOL), (
        f"Final value mismatch: SUT={sut_final}, Oracle={oracle_result.final_cash}"
    )
