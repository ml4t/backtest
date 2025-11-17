"""Unit tests for RiskDecision and RiskRule interfaces."""

from datetime import datetime
from decimal import Decimal

import pytest

from ml4t.backtest.core.types import AssetId, OrderSide, OrderType
from ml4t.backtest.execution.order import Order
from ml4t.backtest.portfolio.core import Position
from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import ExitType, RiskDecision
from ml4t.backtest.risk.rule import (
    CompositeRule,
    RiskRule,
    RiskRuleProtocol,
)


# ============================================================================
# Test Helpers
# ============================================================================


def create_test_context(
    close_price: float = 100.0,
    entry_price: float = 100.0,
    position_quantity: float = 100.0,
    equity: float = 10000.0,
    features: dict | None = None,
    market_context: dict | None = None,
) -> RiskContext:
    """Create a minimal RiskContext for testing."""
    return RiskContext(
        asset_id="TEST",
        timestamp=datetime.now(),
        open=None,
        high=None,
        low=None,
        close=Decimal(str(close_price)),
        volume=None,
        bid_price=None,
        ask_price=None,
        position_quantity=position_quantity,
        entry_price=Decimal(str(entry_price)),
        entry_time=datetime.now() if position_quantity > 0 else None,
        bars_held=1 if position_quantity > 0 else 0,
        equity=equity,
        cash=0.0,
        leverage=1.0,
        features=features or {},
        market_features=market_context or {},
    )


# ============================================================================
# RiskDecision Tests
# ============================================================================


class TestRiskDecisionCreation:
    """Test RiskDecision creation and validation."""

    def test_no_action_decision(self):
        """Test creating a no-action decision."""
        decision = RiskDecision.no_action(
            reason="Position within limits", metadata={"check": "passed"}
        )

        assert not decision.should_exit
        assert decision.exit_type is None
        assert decision.exit_price is None
        assert decision.update_stop_loss is None
        assert decision.update_take_profit is None
        assert decision.reason == "Position within limits"
        assert decision.priority == 0
        assert decision.metadata == {"check": "passed"}
        assert not decision.is_action_required()

    def test_exit_now_decision(self):
        """Test creating an immediate exit decision."""
        decision = RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS,
            reason="Stop-loss breach at $95.00",
            exit_price=Decimal("95.00"),
            priority=10,
            metadata={"breach_pct": 0.05},
        )

        assert decision.should_exit
        assert decision.exit_type == ExitType.STOP_LOSS
        assert decision.exit_price == Decimal("95.00")
        assert decision.reason == "Stop-loss breach at $95.00"
        assert decision.priority == 10
        assert decision.metadata == {"breach_pct": 0.05}
        assert decision.is_action_required()

    def test_update_stops_decision(self):
        """Test creating a stop update decision."""
        decision = RiskDecision.update_stops(
            update_stop_loss=Decimal("98.50"),
            update_take_profit=Decimal("105.00"),
            reason="Trailing stop update",
            priority=5,
        )

        assert not decision.should_exit
        assert decision.update_stop_loss == Decimal("98.50")
        assert decision.update_take_profit == Decimal("105.00")
        assert decision.reason == "Trailing stop update"
        assert decision.priority == 5
        assert decision.is_action_required()

    def test_exit_requires_exit_type(self):
        """Test that should_exit=True requires exit_type."""
        with pytest.raises(ValueError, match="exit_type must be specified"):
            RiskDecision(should_exit=True, exit_type=None)

    def test_exit_type_requires_should_exit(self):
        """Test that exit_type requires should_exit=True."""
        with pytest.raises(ValueError, match="should_exit must be True"):
            RiskDecision(should_exit=False, exit_type=ExitType.STOP_LOSS)

    def test_update_stops_requires_at_least_one_value(self):
        """Test that update_stops requires at least one stop value."""
        with pytest.raises(ValueError, match="At least one of"):
            RiskDecision.update_stops()


class TestRiskDecisionMerging:
    """Test merging multiple RiskDecisions."""

    def test_merge_single_decision(self):
        """Test merging a single decision returns it unchanged."""
        decision = RiskDecision.no_action()
        merged = RiskDecision.merge([decision])

        assert merged is decision

    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            RiskDecision.merge([])

    def test_merge_exit_takes_precedence_over_no_action(self):
        """Test that exit decisions take precedence over no-action."""
        no_action = RiskDecision.no_action(reason="OK")
        exit_decision = RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS, reason="Stop breach", priority=10
        )

        merged = RiskDecision.merge([no_action, exit_decision])

        assert merged.should_exit
        assert merged.exit_type == ExitType.STOP_LOSS
        assert "Stop breach" in merged.reason

    def test_merge_exit_takes_precedence_over_stop_update(self):
        """Test that exit decisions take precedence over stop updates."""
        stop_update = RiskDecision.update_stops(
            update_stop_loss=Decimal("98.00"), reason="Trailing stop", priority=5
        )
        exit_decision = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="VIX spike", priority=10
        )

        merged = RiskDecision.merge([stop_update, exit_decision])

        assert merged.should_exit
        assert merged.exit_type == ExitType.RISK_EXIT
        assert "VIX spike" in merged.reason

    def test_merge_highest_priority_exit_wins(self):
        """Test that highest priority exit decision wins."""
        exit1 = RiskDecision.exit_now(
            exit_type=ExitType.TIME_EXIT, reason="Time limit", priority=3
        )
        exit2 = RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS, reason="Stop breach", priority=10
        )
        exit3 = RiskDecision.exit_now(
            exit_type=ExitType.TAKE_PROFIT, reason="Target hit", priority=7
        )

        merged = RiskDecision.merge([exit1, exit2, exit3])

        assert merged.should_exit
        assert merged.exit_type == ExitType.STOP_LOSS
        assert merged.priority == 10
        assert "Stop breach" in merged.reason

    def test_merge_with_priority_ordering_breaks_ties(self):
        """Test that priority ordering breaks ties among equal-priority exits."""
        exit1 = RiskDecision.exit_now(
            exit_type=ExitType.TAKE_PROFIT, reason="TP", priority=10
        )
        exit2 = RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS, reason="SL", priority=10
        )

        # STOP_LOSS has higher priority in the ordering
        merged = RiskDecision.merge(
            [exit1, exit2],
            default_priority_order=[ExitType.STOP_LOSS, ExitType.TAKE_PROFIT],
        )

        assert merged.exit_type == ExitType.STOP_LOSS

    def test_merge_stop_updates_use_highest_priority(self):
        """Test that stop updates use highest priority values."""
        update1 = RiskDecision.update_stops(
            update_stop_loss=Decimal("97.00"), reason="First update", priority=3
        )
        update2 = RiskDecision.update_stops(
            update_stop_loss=Decimal("98.50"), reason="Second update", priority=7
        )
        update3 = RiskDecision.update_stops(
            update_take_profit=Decimal("105.00"), reason="TP update", priority=5
        )

        merged = RiskDecision.merge([update1, update2, update3])

        assert not merged.should_exit
        assert merged.update_stop_loss == Decimal("98.50")  # priority=7
        assert merged.update_take_profit == Decimal("105.00")  # priority=5
        assert merged.priority == 7  # max priority

    def test_merge_combines_metadata(self):
        """Test that metadata is merged from all decisions."""
        decision1 = RiskDecision.no_action(metadata={"metric_a": 1.0, "metric_b": 2.0})
        decision2 = RiskDecision.no_action(metadata={"metric_b": 3.0, "metric_c": 4.0})

        merged = RiskDecision.merge([decision1, decision2])

        assert merged.metadata == {
            "metric_a": 1.0,
            "metric_b": 3.0,  # Later value overwrites
            "metric_c": 4.0,
        }

    def test_merge_combines_reasons_for_exit(self):
        """Test that reasons are combined when multiple exits present."""
        exit1 = RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS, reason="Primary reason", priority=10
        )
        exit2 = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="Secondary reason", priority=5
        )

        merged = RiskDecision.merge([exit1, exit2])

        assert "Primary reason" in merged.reason
        assert "Secondary reason" in merged.reason


class TestRiskDecisionStringRepresentation:
    """Test string representation of RiskDecision."""

    def test_str_exit_decision(self):
        """Test string representation of exit decision."""
        decision = RiskDecision.exit_now(
            exit_type=ExitType.STOP_LOSS, reason="Stop breach", priority=10
        )

        result = str(decision)

        assert "EXIT" in result
        assert "stop_loss" in result
        assert "Stop breach" in result
        assert "priority=10" in result

    def test_str_update_decision(self):
        """Test string representation of stop update decision."""
        decision = RiskDecision.update_stops(
            update_stop_loss=Decimal("98.50"),
            update_take_profit=Decimal("105.00"),
            reason="Trailing stop",
        )

        result = str(decision)

        assert "UPDATE" in result
        assert "SL=98.50" in result
        assert "TP=105.00" in result
        assert "Trailing stop" in result

    def test_str_no_action_decision(self):
        """Test string representation of no-action decision."""
        decision = RiskDecision.no_action(reason="Position OK")

        result = str(decision)

        assert "NO_ACTION" in result
        assert "Position OK" in result


# ============================================================================
# RiskRule Tests
# ============================================================================


class SimpleStopLossRule(RiskRule):
    """Simple stop-loss rule for testing."""

    def __init__(self, stop_pct: float = 0.05):
        self.stop_pct = stop_pct

    def evaluate(self, context: RiskContext) -> RiskDecision:
        if context.unrealized_pnl_pct < -self.stop_pct:
            return RiskDecision.exit_now(
                exit_type=ExitType.STOP_LOSS,
                reason=f"Stop-loss breach: {context.unrealized_pnl_pct:.2%}",
                priority=10,
            )
        return RiskDecision.no_action()

    @property
    def priority(self) -> int:
        return 10


class VIXFilterRule(RiskRule):
    """VIX-based risk filter for testing."""

    def __init__(self, vix_threshold: float = 30.0):
        self.vix_threshold = vix_threshold

    def evaluate(self, context: RiskContext) -> RiskDecision:
        vix = context.market_features.get("vix")
        if vix and vix > self.vix_threshold:
            return RiskDecision.exit_now(
                exit_type=ExitType.RISK_EXIT,
                reason=f"VIX spike: {vix}",
                priority=8,
            )
        return RiskDecision.no_action()

    def validate_order(self, order: Order, context: RiskContext) -> Order | None:
        # Prevent new positions when VIX too high
        vix = context.market_features.get("vix")
        if vix and vix > self.vix_threshold:
            return None  # Reject order
        return order

    @property
    def priority(self) -> int:
        return 8


class TestRiskRuleInterface:
    """Test RiskRule abstract base class."""

    def test_rule_must_implement_evaluate(self):
        """Test that RiskRule subclasses must implement evaluate()."""

        class IncompleteRule(RiskRule):
            pass  # Missing evaluate() implementation

        with pytest.raises(TypeError):
            IncompleteRule()

    def test_rule_evaluate_returns_decision(self):
        """Test that rule evaluation returns RiskDecision."""
        rule = SimpleStopLossRule(stop_pct=0.05)

        # Create context with loss (6% loss from entry)
        context = create_test_context(
            close_price=94.0,  # 6% loss
            entry_price=100.0,
            position_quantity=100.0,
        )

        decision = rule.evaluate(context)

        assert isinstance(decision, RiskDecision)
        assert decision.should_exit
        assert decision.exit_type == ExitType.STOP_LOSS

    def test_rule_default_priority_is_zero(self):
        """Test that rules have default priority of 0."""

        class DefaultPriorityRule(RiskRule):
            def evaluate(self, context: RiskContext) -> RiskDecision:
                return RiskDecision.no_action()

        rule = DefaultPriorityRule()
        assert rule.priority == 0

    def test_rule_can_override_priority(self):
        """Test that rules can override priority property."""
        rule = SimpleStopLossRule()
        assert rule.priority == 10

    def test_rule_default_validate_order_passes_through(self):
        """Test that default validate_order() passes order through."""

        class BasicRule(RiskRule):
            def evaluate(self, context: RiskContext) -> RiskDecision:
                return RiskDecision.no_action()

        rule = BasicRule()
        order = Order(
            asset_id="TEST",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        context = create_test_context(close_price=100.0, position_quantity=0)

        result = rule.validate_order(order, context)

        assert result is order  # Same object, unmodified

    def test_rule_can_reject_order(self):
        """Test that rules can reject orders by returning None."""
        rule = VIXFilterRule(vix_threshold=30.0)
        order = Order(
            asset_id="TEST",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        context = create_test_context(
            close_price=100.0,
            position_quantity=0,
            market_context={"vix": 35.0},
        )

        result = rule.validate_order(order, context)

        assert result is None  # Order rejected

    def test_rule_repr(self):
        """Test string representation of rule."""
        rule = SimpleStopLossRule()
        result = repr(rule)

        assert "SimpleStopLossRule" in result
        assert "priority=10" in result


class TestRiskRuleProtocol:
    """Test RiskRuleProtocol for callable rules."""

    def test_callable_function_satisfies_protocol(self):
        """Test that callable with correct signature satisfies protocol."""

        def simple_rule(context: RiskContext) -> RiskDecision:
            return RiskDecision.no_action()

        # Check if it satisfies protocol
        assert isinstance(simple_rule, RiskRuleProtocol)

    def test_callable_can_be_invoked(self):
        """Test that protocol-satisfying callable can be invoked."""

        def stop_loss_rule(context: RiskContext) -> RiskDecision:
            if context.unrealized_pnl_pct < -0.05:
                return RiskDecision.exit_now(
                    exit_type=ExitType.STOP_LOSS, reason="Stop breach"
                )
            return RiskDecision.no_action()

        context = create_test_context(
            close_price=94.0,
            entry_price=100.0,
            position_quantity=100.0,
        )

        decision = stop_loss_rule(context)

        assert isinstance(decision, RiskDecision)
        assert decision.should_exit


class TestCompositeRule:
    """Test CompositeRule for combining multiple rules."""

    def test_composite_evaluates_all_subrules(self):
        """Test that composite rule evaluates all sub-rules."""
        rule1 = SimpleStopLossRule(stop_pct=0.05)
        rule2 = VIXFilterRule(vix_threshold=30.0)

        composite = CompositeRule([rule1, rule2])

        context = create_test_context(
            close_price=94.0,
            entry_price=100.0,
            position_quantity=100.0,
            market_context={"vix": 35.0},
        )

        decision = composite.evaluate(context)

        # Should exit due to either rule (both trigger)
        assert decision.should_exit
        # Higher priority wins (stop-loss=10 vs vix=8)
        assert decision.exit_type == ExitType.STOP_LOSS

    def test_composite_works_with_callable_rules(self):
        """Test that composite rule works with Protocol-satisfying callables."""

        def custom_rule(context: RiskContext) -> RiskDecision:
            return RiskDecision.update_stops(
                update_stop_loss=context.close * Decimal("0.95"),
                reason="Custom trailing stop",
            )

        rule1 = SimpleStopLossRule()
        composite = CompositeRule([rule1, custom_rule])

        context = create_test_context(
            close_price=100.00, position_quantity=0
        )

        decision = composite.evaluate(context)

        # Should have stop update from custom_rule
        assert decision.update_stop_loss is not None

    def test_composite_validate_order_chains_validation(self):
        """Test that composite validate_order chains through all rules."""
        rule1 = VIXFilterRule(vix_threshold=30.0)
        rule2 = SimpleStopLossRule()

        composite = CompositeRule([rule1, rule2])

        order = Order(
            asset_id="TEST",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100.0,
        )
        context = create_test_context(
            close_price=100.0,
            position_quantity=0,
            market_context={"vix": 35.0},
        )

        result = composite.validate_order(order, context)

        assert result is None  # Rejected by VIXFilterRule

    def test_composite_priority_is_max_of_subrules(self):
        """Test that composite priority is max of all sub-rules."""
        rule1 = SimpleStopLossRule()  # priority=10
        rule2 = VIXFilterRule()  # priority=8

        composite = CompositeRule([rule1, rule2])

        assert composite.priority == 10

    def test_composite_repr(self):
        """Test string representation of composite rule."""
        rule1 = SimpleStopLossRule()
        rule2 = VIXFilterRule()

        composite = CompositeRule([rule1, rule2])
        result = repr(composite)

        assert "CompositeRule" in result
        assert "2 rules" in result
        assert "priority=10" in result


# ============================================================================
# Integration Tests
# ============================================================================


class TestRiskDecisionAndRuleIntegration:
    """Integration tests for RiskDecision and RiskRule working together."""

    def test_multiple_rules_merged_correctly(self):
        """Test that multiple rule decisions are merged correctly."""
        stop_loss = SimpleStopLossRule(stop_pct=0.05)
        vix_filter = VIXFilterRule(vix_threshold=30.0)

        context = create_test_context(
            close_price=94.0,
            entry_price=100.0,
            position_quantity=100.0,
            market_context={"vix": 35.0},
        )

        # Evaluate both rules
        decision1 = stop_loss.evaluate(context)
        decision2 = vix_filter.evaluate(context)

        # Merge decisions
        merged = RiskDecision.merge([decision1, decision2])

        assert merged.should_exit
        # Higher priority wins (stop-loss=10 vs vix=8)
        assert merged.exit_type == ExitType.STOP_LOSS
        assert merged.priority == 10

    def test_callable_rule_integration(self):
        """Test integration with callable (protocol-based) rules."""

        def trailing_stop(context: RiskContext) -> RiskDecision:
            if context.position_quantity == 0:
                return RiskDecision.no_action()

            trail_price = context.close * Decimal("0.98")
            return RiskDecision.update_stops(
                update_stop_loss=trail_price, reason="Trailing stop", priority=5
            )

        context = create_test_context(
            close_price=105.0,
            entry_price=100.0,
            position_quantity=100.0,
        )

        decision = trailing_stop(context)

        assert decision.update_stop_loss == Decimal("102.90")  # 105 * 0.98
        assert decision.priority == 5
