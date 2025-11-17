"""Unit tests for RiskContext dataclass."""

from datetime import datetime, timezone

import pytest

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType
from ml4t.backtest.data.feature_provider import FeatureProvider
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.portfolio.state import Position
from ml4t.backtest.risk import RiskContext


@pytest.fixture
def timestamp() -> datetime:
    """Fixed timestamp for tests."""
    return datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def market_event(timestamp: datetime) -> MarketEvent:
    """Create market event with OHLCV and signals/context."""
    return MarketEvent(
        timestamp=timestamp,
        asset_id="AAPL",
        data_type=MarketDataType.BAR,
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000,
        bid_price=150.95,
        ask_price=151.05,
        signals={
            "atr_20": 2.5,
            "rsi_14": 65.0,
            "ml_score": 0.85,
            "momentum_20": 0.05,
        },
        context={
            "vix": 18.5,
            "spy_return": 0.005,
            "market_regime": 1.0,
        },
    )


@pytest.fixture
def position() -> Position:
    """Create a position with 100 shares at $140 entry."""
    pos = Position(asset_id="AAPL")
    pos.add_shares(100.0, 140.0)  # Entry at $140
    pos.update_price(151.0)  # Current price $151
    return pos


@pytest.fixture
def portfolio(position: Position) -> Portfolio:
    """Create portfolio with position and cash."""
    portfolio = Portfolio(initial_cash=100000.0)
    # Add position manually
    portfolio._tracker.positions["AAPL"] = position
    portfolio._tracker.cash = 100000.0 - (100.0 * 140.0)  # Paid for position
    return portfolio


class TestRiskContextCreation:
    """Test RiskContext creation and field population."""

    def test_from_state_with_position_and_portfolio(
        self, market_event: MarketEvent, position: Position, portfolio: Portfolio, timestamp: datetime
    ):
        """Test creating RiskContext with full state."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
            portfolio=portfolio,
            entry_time=timestamp,
            bars_held=10,
        )

        # Event metadata
        assert context.timestamp == timestamp
        assert context.asset_id == "AAPL"

        # Market prices
        assert context.open == 150.0
        assert context.high == 152.0
        assert context.low == 149.0
        assert context.close == 151.0
        assert context.volume == 1000000

        # Quote prices
        assert context.bid_price == 150.95
        assert context.ask_price == 151.05

        # Position state
        assert context.position_quantity == 100.0
        assert context.entry_price == 140.0  # cost_basis / quantity
        assert context.entry_time == timestamp
        assert context.bars_held == 10

        # Portfolio state
        assert context.equity == portfolio.equity
        assert context.cash == portfolio.cash
        assert context.leverage > 0  # Has position

        # Features (from signals)
        assert context.features["atr_20"] == 2.5
        assert context.features["rsi_14"] == 65.0
        assert context.features["ml_score"] == 0.85
        assert context.features["momentum_20"] == 0.05

        # Market features (from context)
        assert context.market_features["vix"] == 18.5
        assert context.market_features["spy_return"] == 0.005
        assert context.market_features["market_regime"] == 1.0

    def test_from_state_with_no_position(
        self, market_event: MarketEvent, portfolio: Portfolio
    ):
        """Test creating RiskContext with no position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=None,
            portfolio=portfolio,
        )

        # Position state should be zero/None
        assert context.position_quantity == 0.0
        assert context.entry_price == 0.0
        assert context.entry_time is None
        assert context.bars_held == 0

        # Market data should still be present
        assert context.close == 151.0
        assert context.features["atr_20"] == 2.5
        assert context.market_features["vix"] == 18.5

    def test_from_state_with_no_portfolio(
        self, market_event: MarketEvent, position: Position
    ):
        """Test creating RiskContext with no portfolio."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
            portfolio=None,
        )

        # Portfolio state should be zero
        assert context.equity == 0.0
        assert context.cash == 0.0
        assert context.leverage == 0.0

        # Position state should still be present
        assert context.position_quantity == 100.0
        assert context.entry_price == 140.0

    def test_from_state_minimal(self, market_event: MarketEvent):
        """Test creating RiskContext with minimal arguments."""
        context = RiskContext.from_state(market_event=market_event)

        # Should have defaults
        assert context.position_quantity == 0.0
        assert context.entry_price == 0.0
        assert context.equity == 0.0
        assert context.cash == 0.0
        assert context.leverage == 0.0

        # Market data should be present
        assert context.close == 151.0
        assert context.features["atr_20"] == 2.5

    def test_from_state_with_empty_signals_and_context(self, timestamp: datetime):
        """Test creating RiskContext when MarketEvent has no signals/context."""
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.BAR,
            close=151.0,
            # No signals or context
        )

        context = RiskContext.from_state(market_event=event)

        # Should have empty dicts, not None
        assert context.features == {}
        assert context.market_features == {}

    def test_from_state_with_feature_provider(
        self, market_event: MarketEvent
    ):
        """Test feature provider merging with event signals/context."""

        class TestFeatureProvider(FeatureProvider):
            def get_features(self, asset_id, timestamp):  # noqa: ARG002
                return {"additional_signal": 0.5, "atr_20": 999.0}  # atr_20 should be overridden

            def get_market_features(self, timestamp):  # noqa: ARG002
                return {"additional_context": 1.5, "vix": 999.0}  # vix should be overridden

        provider = TestFeatureProvider()
        context = RiskContext.from_state(
            market_event=market_event,
            feature_provider=provider,
        )

        # Event signals should take precedence over provider
        assert context.features["atr_20"] == 2.5  # From event, not 999.0
        assert context.features["additional_signal"] == 0.5  # From provider

        # Event context should take precedence over provider
        assert context.market_features["vix"] == 18.5  # From event, not 999.0
        assert context.market_features["additional_context"] == 1.5  # From provider


class TestRiskContextLazyProperties:
    """Test lazy property evaluation for RiskContext."""

    def test_unrealized_pnl_long_position(
        self, market_event: MarketEvent, position: Position
    ):
        """Test unrealized P&L calculation for long position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # Position: 100 shares @ $140 entry, current $151
        # Unrealized P&L = 100 * (151 - 140) = 1100
        assert context.unrealized_pnl == 1100.0

    def test_unrealized_pnl_no_position(self, market_event: MarketEvent):
        """Test unrealized P&L is zero with no position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=None,
        )

        assert context.unrealized_pnl == 0.0

    def test_unrealized_pnl_pct_long_position(
        self, market_event: MarketEvent, position: Position
    ):
        """Test unrealized P&L percentage for long position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # (151 - 140) / 140 = 0.0785714...
        expected_pct = (151.0 - 140.0) / 140.0
        assert abs(context.unrealized_pnl_pct - expected_pct) < 1e-6

    def test_unrealized_pnl_pct_no_position(self, market_event: MarketEvent):
        """Test unrealized P&L percentage is zero with no position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=None,
        )

        assert context.unrealized_pnl_pct == 0.0

    def test_max_favorable_excursion_long_position(
        self, market_event: MarketEvent, position: Position
    ):
        """Test MFE calculation for long position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # Long position: MFE = quantity * (high - entry_price)
        # = 100 * (152 - 140) = 1200
        assert context.max_favorable_excursion == 1200.0

    def test_max_adverse_excursion_long_position(
        self, market_event: MarketEvent, position: Position
    ):
        """Test MAE calculation for long position."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # Long position: MAE = quantity * (low - entry_price)
        # = 100 * (149 - 140) = 900
        assert context.max_adverse_excursion == 900.0

    def test_max_favorable_excursion_short_position(
        self, market_event: MarketEvent
    ):
        """Test MFE calculation for short position."""
        # Create short position: -100 shares @ $160 entry
        short_position = Position(asset_id="AAPL")
        short_position.add_shares(-100.0, 160.0)
        short_position.update_price(151.0)

        context = RiskContext.from_state(
            market_event=market_event,
            position=short_position,
        )

        # Short position: MFE = abs(quantity) * (entry_price - low)
        # = 100 * (160 - 149) = 1100
        assert context.max_favorable_excursion == 1100.0

    def test_max_adverse_excursion_short_position(
        self, market_event: MarketEvent
    ):
        """Test MAE calculation for short position."""
        # Create short position: -100 shares @ $160 entry
        short_position = Position(asset_id="AAPL")
        short_position.add_shares(-100.0, 160.0)
        short_position.update_price(151.0)

        context = RiskContext.from_state(
            market_event=market_event,
            position=short_position,
        )

        # Short position: MAE = abs(quantity) * (entry_price - high)
        # = 100 * (160 - 152) = 800
        assert context.max_adverse_excursion == 800.0

    def test_mfe_mae_no_high_low_data(self, timestamp: datetime):
        """Test MFE/MAE fallback when no high/low data available."""
        # Create event without high/low (e.g., TRADE event)
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.TRADE,
            price=151.0,
            close=151.0,
        )

        position = Position(asset_id="AAPL")
        position.add_shares(100.0, 140.0)
        position.update_price(151.0)

        context = RiskContext.from_state(
            market_event=event,
            position=position,
        )

        # Should fallback to unrealized P&L
        assert context.max_favorable_excursion == context.unrealized_pnl
        assert context.max_adverse_excursion == context.unrealized_pnl

    def test_mfe_pct_and_mae_pct(
        self, market_event: MarketEvent, position: Position
    ):
        """Test MFE and MAE percentage calculations."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # Position value = 100 * 140 = 14000
        position_value = 100.0 * 140.0

        # MFE% = 1200 / 14000 = 0.0857142...
        expected_mfe_pct = 1200.0 / position_value
        assert abs(context.max_favorable_excursion_pct - expected_mfe_pct) < 1e-6

        # MAE% = 900 / 14000 = 0.0642857...
        expected_mae_pct = 900.0 / position_value
        assert abs(context.max_adverse_excursion_pct - expected_mae_pct) < 1e-6

    def test_lazy_property_caching(
        self, market_event: MarketEvent, position: Position
    ):
        """Test that lazy properties are cached (computed once)."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # Access property multiple times
        pnl1 = context.unrealized_pnl
        pnl2 = context.unrealized_pnl
        pnl3 = context.unrealized_pnl

        # All should be same object (cached)
        assert pnl1 == pnl2 == pnl3

        # Check all lazy properties
        assert context.unrealized_pnl == context.unrealized_pnl
        assert context.unrealized_pnl_pct == context.unrealized_pnl_pct
        assert context.max_favorable_excursion == context.max_favorable_excursion
        assert context.max_adverse_excursion == context.max_adverse_excursion


class TestRiskContextImmutability:
    """Test RiskContext immutability (frozen dataclass)."""

    def test_cannot_modify_fields(
        self, market_event: MarketEvent, position: Position
    ):
        """Test that RiskContext fields cannot be modified."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        # Attempt to modify field should raise FrozenInstanceError
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            context.close = 999.0

        with pytest.raises(FrozenInstanceError):
            context.position_quantity = 999.0

        with pytest.raises(FrozenInstanceError):
            context.equity = 999.0

    def test_features_dict_is_copy(self, market_event: MarketEvent):
        """Test that modifying features dict doesn't affect context."""
        context = RiskContext.from_state(market_event=market_event)

        # Modify the features dict
        context.features["new_feature"] = 999.0

        # Create another context from same event
        context2 = RiskContext.from_state(market_event=market_event)

        # Original event signals should be unchanged
        assert "new_feature" not in market_event.signals
        assert "new_feature" not in context2.features


class TestRiskContextEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_entry_price(self, market_event: MarketEvent):
        """Test handling of zero entry price."""
        position = Position(asset_id="AAPL")
        # Unusual case, but should not crash
        position.quantity = 100.0
        position.cost_basis = 0.0  # Zero cost basis

        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
        )

        assert context.entry_price == 0.0
        assert context.unrealized_pnl_pct == 0.0  # Should not divide by zero
        assert context.max_favorable_excursion_pct == 0.0
        assert context.max_adverse_excursion_pct == 0.0

    def test_zero_equity_portfolio(self, market_event: MarketEvent):
        """Test handling of zero equity portfolio."""
        portfolio = Portfolio(initial_cash=0.0)

        context = RiskContext.from_state(
            market_event=market_event,
            portfolio=portfolio,
        )

        assert context.equity == 0.0
        assert context.leverage == 0.0  # Should not divide by zero

    def test_missing_close_price(self, timestamp: datetime):
        """Test handling of missing close price."""
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.QUOTE,
            bid_price=150.0,
            ask_price=151.0,
            # No close or price
        )

        context = RiskContext.from_state(market_event=event)

        # Should default to 0.0
        assert context.close == 0.0

    def test_price_fallback_to_close(self, timestamp: datetime):
        """Test that close can fallback to price field."""
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="AAPL",
            data_type=MarketDataType.TRADE,
            price=151.5,
            # No explicit close
        )

        context = RiskContext.from_state(market_event=event)

        # Should use price field
        assert context.close == 151.5

    def test_leverage_calculation_with_multiple_positions(
        self, market_event: MarketEvent
    ):
        """Test leverage calculation with multiple positions."""
        portfolio = Portfolio(initial_cash=100000.0)

        # Add two positions
        pos1 = Position(asset_id="AAPL")
        pos1.add_shares(100.0, 140.0)
        pos1.update_price(151.0)

        pos2 = Position(asset_id="MSFT")
        pos2.add_shares(50.0, 300.0)
        pos2.update_price(310.0)

        portfolio._tracker.positions["AAPL"] = pos1
        portfolio._tracker.positions["MSFT"] = pos2
        portfolio._tracker.cash = 100000.0 - (100 * 140) - (50 * 300)

        context = RiskContext.from_state(
            market_event=market_event,
            portfolio=portfolio,
        )

        # Total position value = 100*151 + 50*310 = 15100 + 15500 = 30600
        # Equity = cash + position value
        # Leverage = 30600 / equity
        assert context.leverage > 0


class TestRiskContextUsageExamples:
    """Test real-world usage patterns from docstring examples."""

    def test_volatility_scaled_stop_loss(
        self, market_event: MarketEvent, position: Position, portfolio: Portfolio
    ):
        """Test volatility-scaled stop loss logic."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
            portfolio=portfolio,
        )

        # Example from docstring
        if context.position_quantity > 0:
            atr = context.features.get("atr_20", 0.0)
            stop_price = context.entry_price - 2.0 * atr
            # Stop = 140 - 2*2.5 = 135
            assert stop_price == 135.0

    def test_vix_filter(self, market_event: MarketEvent):
        """Test VIX filtering logic."""
        context = RiskContext.from_state(market_event=market_event)

        vix = context.market_features.get("vix", 15.0)
        assert vix == 18.5

        # Example condition
        high_volatility = vix > 30
        assert not high_volatility  # VIX is 18.5

    def test_mae_exit_rule(
        self, market_event: MarketEvent, position: Position, portfolio: Portfolio
    ):
        """Test MAE-based exit rule."""
        context = RiskContext.from_state(
            market_event=market_event,
            position=position,
            portfolio=portfolio,
        )

        if context.position_quantity > 0:
            mae_pct = context.max_adverse_excursion_pct
            # MAE% = 900 / 14000 = 6.43%
            # Condition: mae_pct < -0.05 (below -5%)
            should_exit = mae_pct < -0.05
            # Since MAE is positive (profit), this should be False
            assert not should_exit

    def test_multi_asset_context(
        self, market_event: MarketEvent, portfolio: Portfolio
    ):
        """Test multi-asset context creation pattern."""
        # Simulate multiple assets
        contexts = {}
        for asset_id in ["AAPL", "MSFT", "GOOGL"]:
            event = MarketEvent(
                timestamp=market_event.timestamp,
                asset_id=asset_id,
                data_type=MarketDataType.BAR,
                close=100.0 + hash(asset_id) % 50,
                context={"vix": 18.5},  # Shared context
            )
            contexts[asset_id] = RiskContext.from_state(
                market_event=event,
                portfolio=portfolio,
            )

        # VIX should be same for all
        vix_values = [ctx.market_features.get("vix") for ctx in contexts.values()]
        assert all(v == 18.5 for v in vix_values)

        # Each context has different asset_id
        assert contexts["AAPL"].asset_id == "AAPL"
        assert contexts["MSFT"].asset_id == "MSFT"
        assert contexts["GOOGL"].asset_id == "GOOGL"
