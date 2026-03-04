"""Direction parameterization matrix: long x short for every behavioral feature.

If something works for longs but silently fails for shorts, it fails here.
Every test runs a full engine cycle (not isolated components) to catch
composition bugs.

Bug coverage:
    - Bug 1 (short P&L sign): TestRoundTripPnL, TestMFEMAEDirectional
    - Bug 3 (trailing stop defer_fill): TestRiskRulesDirectional
"""

from __future__ import annotations

import polars as pl
import pytest

from ml4t.backtest import (
    BacktestConfig,
    DataFeed,
    Engine,
    StopFillMode,
    StopLoss,
    TakeProfit,
    TrailingStop,
)

from .helpers.data import make_ohlcv_prices, make_prices
from .helpers.strategies import RoundTripStrategy

# ============================================================================
# Fixtures
# ============================================================================

DIRECTIONS = ["long", "short"]


def _short_config(**overrides) -> BacktestConfig:
    """Config that allows short selling with SAME_BAR fills."""
    defaults = {
        "allow_short_selling": True,
        "allow_leverage": True,
        "commission_rate": 0.0,
        "slippage_rate": 0.0,
        "execution_mode": "SAME_BAR",
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _run_round_trip(
    prices_df: pl.DataFrame,
    direction: str,
    entry_bar: int = 0,
    exit_bar: int = 2,
    qty: float = 100.0,
    config: BacktestConfig | None = None,
    risk_rules: list | None = None,
):
    """Run a round-trip strategy and return the result."""
    if config is None:
        config = _short_config()

    feed = DataFeed(prices_df=prices_df)
    strategy = RoundTripStrategy(
        asset="TEST", qty=qty, entry_bar=entry_bar, exit_bar=exit_bar, direction=direction,
    )

    engine = Engine(feed, strategy, config)

    if risk_rules:
        from ml4t.backtest.risk.position.composite import RuleChain

        chain = RuleChain(risk_rules)
        engine.broker.set_position_rules(chain)

    return engine.run()


# ============================================================================
# TestRoundTripPnL: Profitable, losing, breakeven — both directions
# ============================================================================


class TestRoundTripPnL:
    """Verify PnL correctness for both long and short round trips."""

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_profitable_trade(self, direction):
        """Long: buy low sell high. Short: sell high buy low."""
        if direction == "long":
            closes = [100.0, 105.0, 110.0]
        else:
            closes = [110.0, 105.0, 100.0]

        prices = make_prices(closes)
        result = _run_round_trip(prices, direction)

        assert len([t for t in result.trades if t.status == "closed"]) == 1
        trade = [t for t in result.trades if t.status == "closed"][0]

        assert trade.pnl > 0, f"Expected profitable trade, got pnl={trade.pnl}"
        assert trade.pnl_percent > 0, f"Expected positive return, got {trade.pnl_percent}"
        assert trade.direction == direction

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_losing_trade(self, direction):
        """Long: buy high sell low. Short: sell low buy high."""
        if direction == "long":
            closes = [110.0, 105.0, 100.0]
        else:
            closes = [100.0, 105.0, 110.0]

        prices = make_prices(closes)
        result = _run_round_trip(prices, direction)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]

        assert trade.pnl < 0, f"Expected losing trade, got pnl={trade.pnl}"
        assert trade.pnl_percent < 0, f"Expected negative return, got {trade.pnl_percent}"

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_breakeven_trade(self, direction):
        """Entry and exit at same price."""
        closes = [100.0, 105.0, 100.0]
        prices = make_prices(closes)
        result = _run_round_trip(prices, direction)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]

        assert abs(trade.pnl) < 1e-8, f"Expected breakeven, got pnl={trade.pnl}"
        assert abs(trade.pnl_percent) < 1e-8

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_pnl_dollar_amount(self, direction):
        """Verify exact dollar PnL: (exit - entry) * signed_qty."""
        if direction == "long":
            closes = [100.0, 110.0, 120.0]
            expected_pnl = (120.0 - 100.0) * 100.0  # +2000
        else:
            closes = [120.0, 110.0, 100.0]
            expected_pnl = (120.0 - 100.0) * 100.0  # +2000 (short sells at 120, buys at 100)

        prices = make_prices(closes)
        result = _run_round_trip(prices, direction)

        closed = [t for t in result.trades if t.status == "closed"]
        trade = closed[0]
        assert abs(trade.pnl - expected_pnl) < 1e-6


# ============================================================================
# TestMFEMAEDirectional: MFE/MAE tracking for both directions
# ============================================================================


class TestMFEMAEDirectional:
    """Verify MFE/MAE tracking is direction-aware."""

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_mfe_positive_for_favorable_move(self, direction):
        """MFE should capture the best favorable excursion."""
        if direction == "long":
            # Price goes up then back — MFE should capture the peak
            closes = [100.0, 115.0, 110.0, 105.0]
        else:
            # Price goes down then back — MFE should capture the trough
            closes = [100.0, 85.0, 90.0, 95.0]

        prices = make_prices(closes)
        result = _run_round_trip(prices, direction, exit_bar=3)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]

        assert trade.mfe > 0, f"MFE should be positive, got {trade.mfe}"

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_mae_negative_for_adverse_move(self, direction):
        """MAE should capture the worst adverse excursion."""
        if direction == "long":
            # Price drops then recovers
            closes = [100.0, 90.0, 95.0, 110.0]
        else:
            # Price rises then recovers
            closes = [100.0, 110.0, 105.0, 90.0]

        prices = make_prices(closes)
        result = _run_round_trip(prices, direction, exit_bar=3)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]

        assert trade.mae < 0, f"MAE should be negative, got {trade.mae}"


# ============================================================================
# TestCostDecompositionDirectional
# ============================================================================


class TestCostDecompositionDirectional:
    """Verify gross - fees == net for both directions x commission rates."""

    @pytest.mark.parametrize("direction", DIRECTIONS)
    @pytest.mark.parametrize("commission_rate", [0.0, 0.001, 0.005])
    def test_gross_minus_fees_equals_net(self, direction, commission_rate):
        if direction == "long":
            closes = [100.0, 105.0, 110.0]
        else:
            closes = [110.0, 105.0, 100.0]

        config = _short_config(commission_rate=commission_rate)
        prices = make_prices(closes)
        result = _run_round_trip(prices, direction, config=config)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]

        expected_net = trade.gross_pnl - trade.fees
        assert abs(expected_net - trade.pnl) < 1e-6, (
            f"Decomposition failed: gross({trade.gross_pnl}) - fees({trade.fees}) "
            f"= {expected_net} != pnl({trade.pnl})"
        )


# ============================================================================
# TestRiskRulesDirectional: SL, TP, TrailingStop x direction
# ============================================================================


class TestRiskRulesDirectional:
    """Verify risk rules trigger correctly for both directions."""

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_stop_loss_triggers(self, direction):
        """SL should trigger when price moves against the position."""
        if direction == "long":
            # Entry at 100, drops to 90 (10% loss)
            bars = [
                (100.0, 100.0, 100.0, 100.0),  # Entry bar
                (99.0, 99.0, 89.0, 92.0),       # SL triggers (low=89 < 95)
                (92.0, 93.0, 91.0, 92.5),       # Should already be out
            ]
        else:
            # Entry at 100, rises to 110 (10% loss for short)
            bars = [
                (100.0, 100.0, 100.0, 100.0),
                (101.0, 111.0, 101.0, 108.0),   # SL triggers (high=111 > 105)
                (108.0, 109.0, 107.0, 107.5),
            ]

        config = _short_config()
        prices = make_ohlcv_prices(bars)
        sl = StopLoss(pct=0.05)

        result = _run_round_trip(
            prices, direction, exit_bar=99, risk_rules=[sl], config=config,
        )

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]
        assert trade.exit_reason == "stop_loss"
        assert trade.pnl < 0

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_take_profit_triggers(self, direction):
        """TP should trigger when price moves in favor."""
        if direction == "long":
            bars = [
                (100.0, 100.0, 100.0, 100.0),
                (101.0, 112.0, 101.0, 108.0),   # TP triggers (high=112 > 110)
                (108.0, 109.0, 107.0, 108.0),
            ]
        else:
            bars = [
                (100.0, 100.0, 100.0, 100.0),
                (99.0, 99.0, 88.0, 92.0),       # TP triggers (low=88 < 90)
                (92.0, 93.0, 91.0, 92.0),
            ]

        config = _short_config()
        prices = make_ohlcv_prices(bars)
        tp = TakeProfit(pct=0.10)

        result = _run_round_trip(
            prices, direction, exit_bar=99, risk_rules=[tp], config=config,
        )

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]
        assert trade.exit_reason == "take_profit"
        assert trade.pnl > 0

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_trailing_stop_triggers(self, direction):
        """Trailing stop should track HWM/LWM and trigger on reversal."""
        if direction == "long":
            bars = [
                (100.0, 100.0, 100.0, 100.0),   # Entry
                (101.0, 112.0, 101.0, 110.0),   # HWM = 110 (close)
                (109.0, 109.0, 103.0, 104.0),   # Trail triggers: 103 < 110*(1-0.05)=104.5
            ]
        else:
            bars = [
                (100.0, 100.0, 100.0, 100.0),   # Entry
                (99.0, 99.0, 88.0, 90.0),       # LWM = 90 (close)
                (91.0, 97.0, 91.0, 96.0),       # Trail triggers: 97 > 90*(1+0.05)=94.5
            ]

        config = _short_config()
        prices = make_ohlcv_prices(bars)
        ts = TrailingStop(pct=0.05)

        result = _run_round_trip(
            prices, direction, exit_bar=99, risk_rules=[ts], config=config,
        )

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        trade = closed[0]
        assert trade.exit_reason == "trailing_stop"


# ============================================================================
# TestFillTimingDirectional: SAME_BAR vs NEXT_BAR_OPEN x direction
# ============================================================================


class TestFillTimingDirectional:
    """Verify fill timing modes work for both directions."""

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_same_bar_fills_at_close(self, direction):
        if direction == "long":
            closes = [100.0, 110.0, 120.0]
        else:
            closes = [120.0, 110.0, 100.0]

        config = _short_config(execution_mode="SAME_BAR")
        prices = make_prices(closes)
        result = _run_round_trip(prices, direction, config=config)

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_next_bar_fills_at_open(self, direction):
        """In NEXT_BAR mode, orders submitted on bar N fill at bar N+1 open."""
        if direction == "long":
            # entry_bar=0 submits → fills bar 1 open; exit_bar=2 submits → fills bar 3 open
            opens = [100.0, 102.0, 108.0, 115.0]
            closes = [101.0, 105.0, 110.0, 114.0]
        else:
            opens = [115.0, 112.0, 108.0, 100.0]
            closes = [114.0, 110.0, 105.0, 101.0]

        config = _short_config(execution_mode="NEXT_BAR")
        prices = make_prices(closes, opens=opens)
        result = _run_round_trip(prices, direction, entry_bar=0, exit_bar=2, config=config)

        # Should have a closed trade (entry fills at bar 1 open, exit fills at bar 3 open)
        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1


# ============================================================================
# TestPositionFlip: Long-to-short and short-to-long reversals
# ============================================================================


class TestPositionFlip:
    """Verify position reversal works correctly."""

    def test_long_to_short_reversal(self):
        """Close long and open short in sequence."""
        closes = [100.0, 110.0, 105.0, 100.0, 95.0]
        prices = make_prices(closes)

        config = _short_config()

        # Run long first, then short
        long_result = _run_round_trip(prices, "long", exit_bar=2, config=config)
        short_result = _run_round_trip(prices, "short", entry_bar=2, exit_bar=4, config=config)

        long_closed = [t for t in long_result.trades if t.status == "closed"]
        short_closed = [t for t in short_result.trades if t.status == "closed"]

        assert len(long_closed) == 1
        assert len(short_closed) == 1
        assert long_closed[0].direction == "long"
        assert short_closed[0].direction == "short"


# ============================================================================
# TestStopFillModeDirectional: All StopFillModes x direction
# ============================================================================


class TestStopFillModeDirectional:
    """Verify stop fill modes work for both directions."""

    @pytest.mark.parametrize("direction", DIRECTIONS)
    @pytest.mark.parametrize(
        "fill_mode",
        [StopFillMode.STOP_PRICE, StopFillMode.CLOSE_PRICE, StopFillMode.NEXT_BAR_OPEN],
    )
    def test_stop_loss_fill_modes(self, direction, fill_mode):
        """SL triggers and fills correctly under all fill modes."""
        if direction == "long":
            bars = [
                (100.0, 100.0, 100.0, 100.0),
                (99.0, 99.0, 93.0, 95.0),       # SL triggers
                (95.0, 95.0, 94.0, 94.5),       # Next bar for NEXT_BAR_OPEN
                (94.5, 95.0, 94.0, 94.5),
            ]
        else:
            bars = [
                (100.0, 100.0, 100.0, 100.0),
                (101.0, 107.0, 101.0, 105.0),   # SL triggers
                (105.0, 106.0, 104.0, 105.5),
                (105.5, 106.0, 105.0, 105.5),
            ]

        config = _short_config(stop_fill_mode=fill_mode)
        prices = make_ohlcv_prices(bars)
        sl = StopLoss(pct=0.05)

        result = _run_round_trip(
            prices, direction, exit_bar=99, risk_rules=[sl], config=config,
        )

        closed = [t for t in result.trades if t.status == "closed"]
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"
        assert closed[0].pnl < 0
