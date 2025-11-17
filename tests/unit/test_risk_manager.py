"""Unit tests for RiskManager orchestration and context caching.

Tests cover:
    - Rule registration and management (add_rule, remove_rule)
    - Context caching for performance
    - evaluate_all_rules with decision merging
    - validate_order method
    - record_fill for position state tracking
    - check_position_exits integration
    - PositionTradeState tracking (bars_held, MFE, MAE)
    - PositionLevels management (SL/TP)
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock

from ml4t.backtest.core.event import FillEvent, MarketEvent
from ml4t.backtest.core.types import AssetId, MarketDataType, OrderType
from ml4t.backtest.execution.order import Order
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.portfolio.state import Position
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import ExitType, RiskDecision
from ml4t.backtest.risk.manager import RiskManager, PositionTradeState, PositionLevels
from ml4t.backtest.risk.rule import RiskRule


# Fixtures


@pytest.fixture
def mock_broker():
    """Mock broker with position lookup."""
    broker = Mock()
    broker.get_positions = Mock(return_value={})
    broker.get_position = Mock(return_value=Position(
        asset_id="TEST",
        quantity=0.0,
        cost_basis=Decimal("0.0"),
        last_price=Decimal("100.0"),
    ))
    return broker


@pytest.fixture
def mock_portfolio():
    """Mock portfolio with equity and cash."""
    portfolio = Mock(spec=Portfolio)
    portfolio.equity = 10000.0
    portfolio.cash = 5000.0
    portfolio.leverage = 1.0
    return portfolio


@pytest.fixture
def market_event():
    """Create a basic market event."""
    return MarketEvent(
        timestamp=datetime(2025, 1, 1, 10, 0),
        asset_id="TEST",
        data_type=MarketDataType.BAR,
        open=Decimal("99.0"),
        high=Decimal("101.0"),
        low=Decimal("98.0"),
        close=Decimal("100.0"),
        volume=1000000,
        signals={},
        context={},
    )


@pytest.fixture
def manager():
    """Create RiskManager instance."""
    return RiskManager()


# Mock Rules for Testing


class AlwaysExitRule(RiskRule):
    """Rule that always exits."""
    def evaluate(self, context: RiskContext) -> RiskDecision:
        return RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT,
            reason="Test exit",
        )


class NeverExitRule(RiskRule):
    """Rule that never exits."""
    def evaluate(self, context: RiskContext) -> RiskDecision:
        return RiskDecision.no_action(reason="No action")


class UpdateStopsRule(RiskRule):
    """Rule that updates stops."""
    def __init__(self, sl_price: Decimal, tp_price: Decimal):
        self.sl_price = sl_price
        self.tp_price = tp_price

    def evaluate(self, context: RiskContext) -> RiskDecision:
        return RiskDecision.update_stops(
            update_stop_loss=self.sl_price,
            update_take_profit=self.tp_price,
            reason="Update stops",
        )


class RejectOrderRule(RiskRule):
    """Rule that rejects orders in validate_order."""
    def evaluate(self, context: RiskContext) -> RiskDecision:
        return RiskDecision.no_action()

    def validate_order(self, order, context):
        return None  # Reject


class ModifyOrderRule(RiskRule):
    """Rule that modifies orders in validate_order."""
    def __init__(self, scale_factor: float = 0.5):
        self.scale_factor = scale_factor

    def evaluate(self, context: RiskContext) -> RiskDecision:
        return RiskDecision.no_action()

    def validate_order(self, order, context):
        # Reduce order size
        modified = Order(
            asset_id=order.asset_id,
            order_type=OrderType.MARKET,
            quantity=order.quantity * self.scale_factor,
            timestamp=order.timestamp,
        )
        return modified


# Test Rule Registration


def test_add_rule_class_based(manager):
    """Test adding class-based RiskRule."""
    rule = NeverExitRule()
    manager.add_rule(rule)
    assert len(manager._rules) == 1
    assert manager._rules[0] == rule


def test_add_rule_callable(manager):
    """Test adding callable rule (Protocol)."""
    def simple_rule(context):
        return RiskDecision.no_action()

    manager.add_rule(simple_rule)
    assert len(manager._rules) == 1
    assert manager._rules[0] == simple_rule


def test_add_rule_invalid_type_raises(manager):
    """Test adding invalid rule type raises TypeError."""
    with pytest.raises(TypeError, match="must be RiskRule"):
        manager.add_rule("not a rule")


def test_remove_rule_success(manager):
    """Test removing registered rule."""
    rule = NeverExitRule()
    manager.add_rule(rule)
    manager.remove_rule(rule)
    assert len(manager._rules) == 0


def test_remove_rule_not_found_raises(manager):
    """Test removing non-existent rule raises ValueError."""
    rule = NeverExitRule()
    with pytest.raises(ValueError, match="not registered"):
        manager.remove_rule(rule)


# Test evaluate_all_rules


def test_evaluate_all_rules_no_rules_returns_no_action(manager):
    """Test evaluate with no rules returns no_action."""
    context = Mock(spec=RiskContext)
    decision = manager.evaluate_all_rules(context)

    assert not decision.should_exit
    assert decision.reason == "No rules registered"


def test_evaluate_all_rules_single_rule(manager):
    """Test evaluate with single rule."""
    manager.add_rule(AlwaysExitRule())
    context = Mock(spec=RiskContext)

    decision = manager.evaluate_all_rules(context)

    assert decision.should_exit
    assert decision.exit_type == ExitType.RISK_EXIT
    assert decision.reason == "Test exit"


def test_evaluate_all_rules_multiple_rules_merge(manager):
    """Test evaluate with multiple rules merges decisions."""
    manager.add_rule(NeverExitRule())
    manager.add_rule(AlwaysExitRule())  # This one should win (exit > no action)
    context = Mock(spec=RiskContext)

    decision = manager.evaluate_all_rules(context)

    # Exit decision should win over no_action
    assert decision.should_exit
    assert decision.exit_type == ExitType.RISK_EXIT


def test_evaluate_all_rules_callable_rule(manager):
    """Test evaluate with callable rule."""
    def exit_rule(context):
        return RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS,
            reason="Callable exit"
        )

    manager.add_rule(exit_rule)
    context = Mock(spec=RiskContext)

    decision = manager.evaluate_all_rules(context)

    assert decision.should_exit
    assert decision.reason == "Callable exit"


# Test validate_order


def test_validate_order_no_rules_returns_order(manager, market_event, mock_broker, mock_portfolio):
    """Test validate_order with no rules returns original order."""
    order = Order(
        asset_id="TEST",
        order_type=OrderType.MARKET,
        quantity=100.0,
        
    )

    validated = manager.validate_order(order, market_event, mock_broker, mock_portfolio)

    assert validated == order


def test_validate_order_passing_rule_returns_order(manager, market_event, mock_broker, mock_portfolio):
    """Test validate_order with passing rule returns order."""
    manager.add_rule(NeverExitRule())  # Has no validate_order method
    order = Order(
        asset_id="TEST",
        order_type=OrderType.MARKET,
        quantity=100.0,
        
    )

    validated = manager.validate_order(order, market_event, mock_broker, mock_portfolio)

    assert validated == order


def test_validate_order_reject_rule_returns_none(manager, market_event, mock_broker, mock_portfolio):
    """Test validate_order with rejecting rule returns None."""
    manager.add_rule(RejectOrderRule())
    order = Order(
        asset_id="TEST",
        order_type=OrderType.MARKET,
        quantity=100.0,
        
    )

    validated = manager.validate_order(order, market_event, mock_broker, mock_portfolio)

    assert validated is None


def test_validate_order_modify_rule_modifies_order(manager, market_event, mock_broker, mock_portfolio):
    """Test validate_order with modifying rule returns modified order."""
    manager.add_rule(ModifyOrderRule(scale_factor=0.5))
    order = Order(
        asset_id="TEST",
        order_type=OrderType.MARKET,
        quantity=100.0,
        
    )

    validated = manager.validate_order(order, market_event, mock_broker, mock_portfolio)

    assert validated is not None
    assert validated.quantity == 50.0  # Reduced by 50%


def test_validate_order_multiple_rules_chain(manager, market_event, mock_broker, mock_portfolio):
    """Test validate_order chains modifications from multiple rules."""
    manager.add_rule(ModifyOrderRule(scale_factor=0.5))  # 100 → 50
    manager.add_rule(ModifyOrderRule(scale_factor=0.8))  # 50 → 40
    order = Order(
        asset_id="TEST",
        order_type=OrderType.MARKET,
        quantity=100.0,
        
    )

    validated = manager.validate_order(order, market_event, mock_broker, mock_portfolio)

    assert validated is not None
    assert validated.quantity == 40.0  # 100 * 0.5 * 0.8


# Test record_fill and PositionTradeState


def test_record_fill_new_position_creates_state(manager, market_event):
    """Test recording fill for new position creates PositionTradeState."""
    fill_event = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )

    manager.record_fill(fill_event, market_event)

    assert "TEST" in manager._position_state
    state = manager._position_state["TEST"]
    assert state.asset_id == "TEST"
    assert state.entry_time == fill_event.timestamp
    assert state.entry_price == Decimal("100.0")
    assert state.entry_quantity == 100.0
    assert state.bars_held == 0


def test_record_fill_new_position_creates_levels(manager, market_event):
    """Test recording fill creates PositionLevels."""
    fill_event = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )

    manager.record_fill(fill_event, market_event)

    assert "TEST" in manager._position_levels
    levels = manager._position_levels["TEST"]
    assert levels.asset_id == "TEST"
    assert levels.stop_loss is None
    assert levels.take_profit is None


def test_record_fill_add_to_position_updates_avg_price(manager, market_event):
    """Test adding to position updates average entry price."""
    # Initial fill: 100 shares @ $100
    fill1 = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill1, market_event)

    # Add: 100 shares @ $110
    fill2 = FillEvent(
        asset_id="TEST",
        timestamp=market_event.timestamp + timedelta(hours=1),
        quantity=100.0,
        fill_price=Decimal("110.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill2, market_event)

    state = manager._position_state["TEST"]
    # Average: (100*100 + 100*110) / 200 = 105
    assert state.entry_price == Decimal("105.0")
    assert state.entry_quantity == 200.0


def test_record_fill_close_position_removes_state(manager, market_event):
    """Test closing position removes PositionTradeState and PositionLevels."""
    # Open position
    fill_open = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill_open, market_event)

    # Close position
    fill_close = FillEvent(
        asset_id="TEST",
        timestamp=market_event.timestamp + timedelta(hours=1),
        quantity=-100.0,  # Opposite quantity
        fill_price=Decimal("105.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill_close, market_event)

    assert "TEST" not in manager._position_state
    assert "TEST" not in manager._position_levels


def test_record_fill_reverse_position_resets_state(manager, market_event):
    """Test reversing position resets PositionTradeState."""
    # Open long position
    fill_long = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill_long, market_event)

    # Reverse to short
    fill_short = FillEvent(
        asset_id="TEST",
        timestamp=market_event.timestamp + timedelta(hours=1),
        quantity=-200.0,  # Close 100 long + open 100 short
        fill_price=Decimal("105.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill_short, market_event)

    state = manager._position_state["TEST"]
    assert state.entry_price == Decimal("105.0")  # New entry
    assert state.entry_quantity == -100.0  # Short position
    assert state.bars_held == 0  # Reset


# Test PositionTradeState.update_on_market_event


def test_position_state_update_increments_bars_held():
    """Test update_on_market_event increments bars_held."""
    state = PositionTradeState(
        asset_id="TEST",
        entry_time=datetime(2025, 1, 1, 10, 0),
        entry_price=Decimal("100.0"),
        entry_quantity=100.0,
        bars_held=0,
    )

    state.update_on_market_event(market_price=Decimal("105.0"))

    assert state.bars_held == 1


def test_position_state_update_tracks_mfe_long():
    """Test MFE tracking for long position."""
    state = PositionTradeState(
        asset_id="TEST",
        entry_time=datetime(2025, 1, 1, 10, 0),
        entry_price=Decimal("100.0"),
        entry_quantity=100.0,
    )

    # Price goes up (favorable)
    state.update_on_market_event(market_price=Decimal("105.0"))
    assert state.max_favorable_excursion == Decimal("5.0")

    # Price goes up more
    state.update_on_market_event(market_price=Decimal("110.0"))
    assert state.max_favorable_excursion == Decimal("10.0")

    # Price goes down (but MFE stays at max)
    state.update_on_market_event(market_price=Decimal("102.0"))
    assert state.max_favorable_excursion == Decimal("10.0")


def test_position_state_update_tracks_mae_long():
    """Test MAE tracking for long position."""
    state = PositionTradeState(
        asset_id="TEST",
        entry_time=datetime(2025, 1, 1, 10, 0),
        entry_price=Decimal("100.0"),
        entry_quantity=100.0,
    )

    # Price goes down (adverse)
    state.update_on_market_event(market_price=Decimal("95.0"))
    assert state.max_adverse_excursion == Decimal("5.0")

    # Price goes down more
    state.update_on_market_event(market_price=Decimal("90.0"))
    assert state.max_adverse_excursion == Decimal("10.0")

    # Price recovers (but MAE stays at max)
    state.update_on_market_event(market_price=Decimal("98.0"))
    assert state.max_adverse_excursion == Decimal("10.0")


# Test check_position_exits


def test_check_position_exits_no_positions_returns_empty(manager, market_event, mock_broker, mock_portfolio):
    """Test check_position_exits with no positions returns empty list."""
    exit_orders = manager.check_position_exits(market_event, mock_broker, mock_portfolio)
    assert exit_orders == []


def test_check_position_exits_no_rules_returns_empty(manager, market_event, mock_broker, mock_portfolio):
    """Test check_position_exits with no rules returns empty."""
    # Setup position
    position = Position(
        asset_id="TEST",
        quantity=100.0,
        cost_basis=Decimal("100.0"),
        last_price=Decimal("100.0"),
    )
    mock_broker.get_positions.return_value = {"TEST": position}
    mock_broker.get_position.return_value = position

    exit_orders = manager.check_position_exits(market_event, mock_broker, mock_portfolio)

    assert exit_orders == []


def test_check_position_exits_exit_rule_generates_order(manager, market_event, mock_broker, mock_portfolio):
    """Test check_position_exits with exit rule generates exit order."""
    # Setup position
    position = Position(
        asset_id="TEST",
        quantity=100.0,
        cost_basis=Decimal("100.0"),
        last_price=Decimal("100.0"),
    )
    mock_broker.get_positions.return_value = {"TEST": position}
    mock_broker.get_position.return_value = position

    # Add rule that exits
    manager.add_rule(AlwaysExitRule())

    exit_orders = manager.check_position_exits(market_event, mock_broker, mock_portfolio)

    assert len(exit_orders) == 1
    order = exit_orders[0]
    assert order.asset_id == "TEST"
    assert order.quantity == 100.0  # Absolute value (Order handles buy/sell separately)
    assert order.order_type == OrderType.MARKET


def test_check_position_exits_updates_levels(manager, market_event, mock_broker, mock_portfolio):
    """Test check_position_exits updates PositionLevels from rules."""
    # Setup position
    position = Position(
        asset_id="TEST",
        quantity=100.0,
        cost_basis=Decimal("100.0"),
        last_price=Decimal("100.0"),
    )
    mock_broker.get_positions.return_value = {"TEST": position}
    mock_broker.get_position.return_value = position

    # Record fill to create state
    fill_event = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill_event, market_event)

    # Add rule that updates stops
    manager.add_rule(UpdateStopsRule(
        sl_price=Decimal("95.0"),
        tp_price=Decimal("110.0"),
    ))

    manager.check_position_exits(market_event, mock_broker, mock_portfolio)

    # Check levels updated
    assert "TEST" in manager._position_levels
    levels = manager._position_levels["TEST"]
    assert levels.stop_loss == Decimal("95.0")
    assert levels.take_profit == Decimal("110.0")


def test_check_position_exits_updates_position_state(manager, market_event, mock_broker, mock_portfolio):
    """Test check_position_exits updates PositionTradeState."""
    # Setup position
    position = Position(
        asset_id="TEST",
        quantity=100.0,
        cost_basis=Decimal("100.0"),
        last_price=Decimal("100.0"),
    )
    mock_broker.get_positions.return_value = {"TEST": position}
    mock_broker.get_position.return_value = position

    # Record fill to create state
    fill_event = FillEvent(
        asset_id="TEST",
        
        quantity=100.0,
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
    )
    manager.record_fill(fill_event, market_event)

    # Add rule (doesn't matter what it does)
    manager.add_rule(NeverExitRule())

    # Initial bars_held
    assert manager._position_state["TEST"].bars_held == 0

    # Call check_position_exits
    manager.check_position_exits(market_event, mock_broker, mock_portfolio)

    # bars_held incremented
    assert manager._position_state["TEST"].bars_held == 1


# Test Context Caching


def test_build_context_caches_result(manager, market_event, mock_broker, mock_portfolio):
    """Test _build_context caches RiskContext."""
    # First call - should build
    context1 = manager._build_context("TEST", market_event, mock_broker, mock_portfolio)

    # Second call - should use cache
    context2 = manager._build_context("TEST", market_event, mock_broker, mock_portfolio)

    assert context1 is context2  # Same object (cached)


def test_build_context_different_asset_separate_cache(manager, market_event, mock_broker, mock_portfolio):
    """Test different assets have separate cache entries."""
    context1 = manager._build_context("TEST1", market_event, mock_broker, mock_portfolio)
    context2 = manager._build_context("TEST2", market_event, mock_broker, mock_portfolio)

    assert context1 is not context2


def test_build_context_different_timestamp_separate_cache(manager, mock_broker, mock_portfolio):
    """Test different timestamps have separate cache entries."""
    event1 = MarketEvent(
        timestamp=datetime(2025, 1, 1, 10, 0),
        asset_id="TEST",
        data_type=MarketDataType.BAR,
        close=Decimal("100.0"),
        signals={},
        context={},
    )
    event2 = MarketEvent(
        timestamp=datetime(2025, 1, 1, 11, 0),
        asset_id="TEST",
        data_type=MarketDataType.BAR,
        close=Decimal("105.0"),
        signals={},
        context={},
    )

    context1 = manager._build_context("TEST", event1, mock_broker, mock_portfolio)
    context2 = manager._build_context("TEST", event2, mock_broker, mock_portfolio)

    assert context1 is not context2


def test_clear_cache_empties_cache(manager, market_event, mock_broker, mock_portfolio):
    """Test clear_cache empties the cache."""
    # Build some contexts
    manager._build_context("TEST", market_event, mock_broker, mock_portfolio)
    assert len(manager._context_cache) == 1

    # Clear
    manager.clear_cache()
    assert len(manager._context_cache) == 0


def test_clear_cache_before_timestamp_partial_clear(manager, mock_broker, mock_portfolio):
    """Test clear_cache with before_timestamp does partial clear."""
    # Build contexts at different times
    event1 = MarketEvent(
        timestamp=datetime(2025, 1, 1, 10, 0),
        asset_id="TEST",
        data_type=MarketDataType.BAR,
        close=Decimal("100.0"),
        signals={},
        context={},
    )
    event2 = MarketEvent(
        timestamp=datetime(2025, 1, 1, 11, 0),
        asset_id="TEST",
        data_type=MarketDataType.BAR,
        close=Decimal("105.0"),
        signals={},
        context={},
    )

    manager._build_context("TEST", event1, mock_broker, mock_portfolio)
    manager._build_context("TEST", event2, mock_broker, mock_portfolio)
    assert len(manager._context_cache) == 2

    # Clear before 10:30
    manager.clear_cache(before_timestamp=datetime(2025, 1, 1, 10, 30))

    # Only event2 context should remain
    assert len(manager._context_cache) == 1
    assert ("TEST", event2.timestamp) in manager._context_cache


# Test Feature Provider Integration


def test_manager_with_feature_provider(market_event, mock_broker, mock_portfolio):
    """Test RiskManager uses feature_provider in context building."""
    # Create mock feature provider
    mock_fp = Mock()
    mock_fp.get_features = Mock(return_value={"atr_20": 2.5})
    mock_fp.get_market_features = Mock(return_value={"vix": 18.0})

    manager = RiskManager(feature_provider=mock_fp)

    # Build context
    context = manager._build_context("TEST", market_event, mock_broker, mock_portfolio)

    # FeatureProvider should be passed to RiskContext.from_state
    # (We can't verify directly without inspecting RiskContext internals,
    #  but this tests that the parameter is wired through)
    assert manager.feature_provider is mock_fp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
