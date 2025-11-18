"""Unit tests for RegimeDependentRule."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from ml4t.backtest.risk.context import RiskContext
from ml4t.backtest.risk.decision import ExitType, RiskDecision
from ml4t.backtest.risk.rule import RiskRule
from ml4t.backtest.risk.rules.regime_dependent import RegimeDependentRule
from ml4t.backtest.risk.rules.volatility_scaled import VolatilityScaledStopLoss
from ml4t.backtest.risk.rules.price_based import PriceBasedStopLoss


def create_mock_context(
    position_quantity: float = 100.0,
    entry_price: float = 100.0,
    current_price: float = 100.0,
    features: dict[str, float] | None = None,
    market_features: dict[str, float] | None = None,
) -> RiskContext:
    """Create a mock RiskContext for testing."""
    return RiskContext(
        timestamp=datetime(2024, 1, 1, 10, 0),
        asset_id="TEST",
        open=current_price,
        high=current_price,
        low=current_price,
        close=current_price,
        volume=1000.0,
        bid_price=None,
        ask_price=None,
        position_quantity=position_quantity,
        entry_price=entry_price,
        entry_time=datetime(2024, 1, 1, 9, 30) if position_quantity > 0 else None,
        bars_held=0,
        equity=100000.0,
        cash=50000.0,
        leverage=1.0,
        features=features or {},
        market_features=market_features or {},
    )


class TestRegimeDependentRuleInitialization:
    """Test RegimeDependentRule initialization."""

    def test_valid_initialization(self):
        """Test that valid regime_rules dict is accepted."""
        mock_rule1 = Mock(spec=RiskRule)
        mock_rule2 = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={
                "high_vol": mock_rule1,
                "low_vol": mock_rule2,
            }
        )

        assert rule.regime_rules == {"high_vol": mock_rule1, "low_vol": mock_rule2}
        assert rule.regime_key == "regime"  # default
        assert rule.default_regime is None  # default
        assert rule.priority == 100  # default

    def test_empty_regime_rules_rejected(self):
        """Test that empty regime_rules dict is rejected."""
        with pytest.raises(ValueError, match="regime_rules cannot be empty"):
            RegimeDependentRule(regime_rules={})

    def test_custom_regime_key(self):
        """Test initialization with custom regime_key."""
        mock_rule = Mock(spec=RiskRule)
        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="market_regime"
        )

        assert rule.regime_key == "market_regime"

    def test_valid_default_regime(self):
        """Test initialization with valid default_regime."""
        mock_rule1 = Mock(spec=RiskRule)
        mock_rule2 = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"high_vol": mock_rule1, "low_vol": mock_rule2},
            default_regime="low_vol",
        )

        assert rule.default_regime == "low_vol"

    def test_invalid_default_regime_rejected(self):
        """Test that default_regime not in regime_rules is rejected."""
        mock_rule = Mock(spec=RiskRule)

        with pytest.raises(
            ValueError, match="default_regime 'unknown' not found in regime_rules"
        ):
            RegimeDependentRule(
                regime_rules={"high_vol": mock_rule}, default_regime="unknown"
            )

    def test_custom_priority(self):
        """Test initialization with custom priority."""
        mock_rule = Mock(spec=RiskRule)
        rule = RegimeDependentRule(
            regime_rules={"high_vol": mock_rule}, priority=150
        )

        assert rule.priority == 150


class TestRegimeDependentRuleVixFactory:
    """Test RegimeDependentRule.from_vix_threshold() factory method."""

    def test_vix_factory_creates_correct_configuration(self):
        """Test that VIX factory method creates correct regime mapping."""
        high_vol_rule = VolatilityScaledStopLoss(1.5)
        low_vol_rule = VolatilityScaledStopLoss(2.5)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        assert rule._is_vix_based is True
        assert rule._vix_threshold == 20.0
        assert "high_vol" in rule.regime_rules
        assert "low_vol" in rule.regime_rules
        assert rule.regime_rules["high_vol"] == high_vol_rule
        assert rule.regime_rules["low_vol"] == low_vol_rule
        assert rule.default_regime == "low_vol"

    def test_vix_factory_rejects_negative_threshold(self):
        """Test that VIX factory rejects negative threshold."""
        high_vol_rule = VolatilityScaledStopLoss(1.5)
        low_vol_rule = VolatilityScaledStopLoss(2.5)

        with pytest.raises(ValueError, match="vix_threshold must be positive"):
            RegimeDependentRule.from_vix_threshold(
                vix_threshold=-10.0,
                high_vol_rule=high_vol_rule,
                low_vol_rule=low_vol_rule,
            )

    def test_vix_factory_rejects_zero_threshold(self):
        """Test that VIX factory rejects zero threshold."""
        high_vol_rule = VolatilityScaledStopLoss(1.5)
        low_vol_rule = VolatilityScaledStopLoss(2.5)

        with pytest.raises(ValueError, match="vix_threshold must be positive"):
            RegimeDependentRule.from_vix_threshold(
                vix_threshold=0.0,
                high_vol_rule=high_vol_rule,
                low_vol_rule=low_vol_rule,
            )

    def test_vix_factory_custom_priority(self):
        """Test VIX factory with custom priority."""
        high_vol_rule = VolatilityScaledStopLoss(1.5)
        low_vol_rule = VolatilityScaledStopLoss(2.5)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0,
            high_vol_rule=high_vol_rule,
            low_vol_rule=low_vol_rule,
            priority=150,
        )

        assert rule.priority == 150


class TestRegimeDependentRuleRegimeDetection:
    """Test regime detection logic."""

    def test_vix_based_high_vol_when_above_threshold(self):
        """Test that VIX-based rule detects high_vol when VIX > threshold."""
        high_vol_rule = Mock(spec=RiskRule)
        low_vol_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={"vix": 25.0})

        regime = rule._get_current_regime(context)
        assert regime == "high_vol"

    def test_vix_based_low_vol_when_below_threshold(self):
        """Test that VIX-based rule detects low_vol when VIX <= threshold."""
        high_vol_rule = Mock(spec=RiskRule)
        low_vol_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={"vix": 15.0})

        regime = rule._get_current_regime(context)
        assert regime == "low_vol"

    def test_vix_based_low_vol_when_exactly_at_threshold(self):
        """Test that VIX exactly at threshold is classified as low_vol."""
        high_vol_rule = Mock(spec=RiskRule)
        low_vol_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={"vix": 20.0})

        regime = rule._get_current_regime(context)
        assert regime == "low_vol"

    def test_direct_regime_reads_from_market_features(self):
        """Test that direct regime reads from market_features[regime_key]."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={"regime": "trending"})

        regime = rule._get_current_regime(context)
        assert regime == "trending"

    def test_direct_regime_custom_key(self):
        """Test that direct regime can use custom regime_key."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"bullish": mock_rule}, regime_key="market_state"
        )

        context = create_mock_context(market_features={"market_state": "bullish"})

        regime = rule._get_current_regime(context)
        assert regime == "bullish"

    def test_missing_vix_returns_none(self):
        """Test that missing VIX returns None."""
        high_vol_rule = Mock(spec=RiskRule)
        low_vol_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={})  # No VIX

        regime = rule._get_current_regime(context)
        assert regime is None

    def test_missing_regime_key_returns_none(self):
        """Test that missing regime key returns None."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={})  # No regime

        regime = rule._get_current_regime(context)
        assert regime is None

    def test_regime_value_converted_to_string(self):
        """Test that numeric regime values are converted to strings."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"0": mock_rule, "1": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={"regime": 0})

        regime = rule._get_current_regime(context)
        assert regime == "0"
        assert isinstance(regime, str)


class TestRegimeDependentRuleRuleDelegation:
    """Test rule delegation logic."""

    def test_delegates_to_correct_rule_for_regime(self):
        """Test that correct rule is selected and called for given regime."""
        # Create mock rules that return distinct decisions
        high_vol_rule = Mock(spec=RiskRule)
        high_vol_rule.evaluate.return_value = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="high vol exit"
        )

        low_vol_rule = Mock(spec=RiskRule)
        low_vol_rule.evaluate.return_value = RiskDecision.no_action()

        rule = RegimeDependentRule(
            regime_rules={"high_vol": high_vol_rule, "low_vol": low_vol_rule},
            regime_key="regime",
        )

        context = create_mock_context(market_features={"regime": "high_vol"})

        decision = rule.evaluate(context)

        # Verify high vol rule was called
        high_vol_rule.evaluate.assert_called_once_with(context)
        low_vol_rule.evaluate.assert_not_called()

        # Verify decision came from high vol rule
        assert decision.should_exit is True

    def test_passes_context_to_delegated_rule(self):
        """Test that context is passed unchanged to delegated rule."""
        mock_rule = Mock(spec=RiskRule)
        mock_rule.evaluate.return_value = RiskDecision.no_action()

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={"regime": "trending"})

        rule.evaluate(context)

        # Verify same context object passed
        mock_rule.evaluate.assert_called_once()
        call_args = mock_rule.evaluate.call_args
        assert call_args[0][0] is context

    def test_returns_decision_from_delegated_rule(self):
        """Test that decision from delegated rule is returned."""
        mock_rule = Mock(spec=RiskRule)
        expected_decision = RiskDecision.update_stops(
            update_stop_loss=95.0, reason="test stop"
        )
        mock_rule.evaluate.return_value = expected_decision

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={"regime": "trending"})

        decision = rule.evaluate(context)

        # Decision content should match (but metadata will be enhanced)
        assert decision.should_exit == expected_decision.should_exit
        assert decision.update_stop_loss == expected_decision.update_stop_loss
        assert decision.reason == expected_decision.reason

    def test_vix_threshold_delegates_to_high_vol_rule(self):
        """Test VIX threshold delegates to high_vol_rule when VIX high."""
        high_vol_rule = Mock(spec=RiskRule)
        high_vol_rule.evaluate.return_value = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="high vol"
        )

        low_vol_rule = Mock(spec=RiskRule)
        low_vol_rule.evaluate.return_value = RiskDecision.no_action()

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={"vix": 25.0})

        decision = rule.evaluate(context)

        high_vol_rule.evaluate.assert_called_once_with(context)
        low_vol_rule.evaluate.assert_not_called()
        assert decision.should_exit is True

    def test_vix_threshold_delegates_to_low_vol_rule(self):
        """Test VIX threshold delegates to low_vol_rule when VIX low."""
        high_vol_rule = Mock(spec=RiskRule)
        high_vol_rule.evaluate.return_value = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="high vol"
        )

        low_vol_rule = Mock(spec=RiskRule)
        low_vol_rule.evaluate.return_value = RiskDecision.no_action(
            reason="low vol, hold"
        )

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={"vix": 15.0})

        decision = rule.evaluate(context)

        low_vol_rule.evaluate.assert_called_once_with(context)
        high_vol_rule.evaluate.assert_not_called()
        assert decision.should_exit is False


class TestRegimeDependentRuleMetadataEnhancement:
    """Test metadata enhancement in decisions."""

    def test_metadata_includes_regime(self):
        """Test that decision metadata includes regime label."""
        mock_rule = Mock(spec=RiskRule)
        mock_rule.evaluate.return_value = RiskDecision.no_action()

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={"regime": "trending"})

        decision = rule.evaluate(context)

        assert "regime" in decision.metadata
        assert decision.metadata["regime"] == "trending"

    def test_metadata_includes_delegated_rule_name(self):
        """Test that metadata includes delegated rule class name."""
        stop_loss_rule = VolatilityScaledStopLoss(2.0)

        rule = RegimeDependentRule(
            regime_rules={"trending": stop_loss_rule}, regime_key="regime"
        )

        context = create_mock_context(
            market_features={"regime": "trending"}, features={"atr": 2.0}
        )

        decision = rule.evaluate(context)

        assert "delegated_to" in decision.metadata
        assert decision.metadata["delegated_to"] == "VolatilityScaledStopLoss"

    def test_metadata_includes_regime_key(self):
        """Test that metadata includes regime_key used."""
        mock_rule = Mock(spec=RiskRule)
        mock_rule.evaluate.return_value = RiskDecision.no_action()

        rule = RegimeDependentRule(
            regime_rules={"bullish": mock_rule}, regime_key="market_state"
        )

        context = create_mock_context(market_features={"market_state": "bullish"})

        decision = rule.evaluate(context)

        assert "regime_key" in decision.metadata
        assert decision.metadata["regime_key"] == "market_state"

    def test_vix_based_metadata_includes_vix_value(self):
        """Test that VIX-based decision includes VIX value in metadata."""
        high_vol_rule = Mock(spec=RiskRule)
        high_vol_rule.evaluate.return_value = RiskDecision.no_action()

        low_vol_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        context = create_mock_context(market_features={"vix": 25.0})

        decision = rule.evaluate(context)

        assert "vix" in decision.metadata
        assert decision.metadata["vix"] == 25.0
        assert "vix_threshold" in decision.metadata
        assert decision.metadata["vix_threshold"] == 20.0

    def test_preserves_existing_metadata_from_delegated_rule(self):
        """Test that existing metadata from delegated rule is preserved."""
        mock_rule = Mock(spec=RiskRule)
        mock_rule.evaluate.return_value = RiskDecision.no_action(
            metadata={"original_key": "original_value"}
        )

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={"regime": "trending"})

        decision = rule.evaluate(context)

        # Should have both original metadata and regime metadata
        assert "original_key" in decision.metadata
        assert decision.metadata["original_key"] == "original_value"
        assert "regime" in decision.metadata
        assert decision.metadata["regime"] == "trending"


class TestRegimeDependentRuleEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_regime_returns_no_action(self):
        """Test that missing regime returns no_action."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(market_features={})  # No regime

        decision = rule.evaluate(context)

        assert decision.should_exit is False
        assert decision.update_stop_loss is None
        assert "not found in market_features" in decision.reason

    def test_regime_not_in_mapping_uses_default(self):
        """Test that regime not in mapping uses default_regime."""
        default_rule = Mock(spec=RiskRule)
        default_rule.evaluate.return_value = RiskDecision.no_action(
            reason="default rule"
        )

        other_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"trending": default_rule, "mean_reverting": other_rule},
            regime_key="regime",
            default_regime="trending",
        )

        context = create_mock_context(
            market_features={"regime": "unknown_regime"}
        )

        decision = rule.evaluate(context)

        # Should use default rule
        default_rule.evaluate.assert_called_once_with(context)
        other_rule.evaluate.assert_not_called()

    def test_regime_not_in_mapping_no_default_returns_no_action(self):
        """Test that regime not in mapping with no default returns no_action."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule},
            regime_key="regime",
            default_regime=None,
        )

        context = create_mock_context(
            market_features={"regime": "unknown_regime"}
        )

        decision = rule.evaluate(context)

        mock_rule.evaluate.assert_not_called()
        assert decision.should_exit is False
        assert "not in rules mapping" in decision.reason

    def test_no_position_delegates_normally(self):
        """Test that no position still delegates to rule (rule may return no_action)."""
        mock_rule = Mock(spec=RiskRule)
        mock_rule.evaluate.return_value = RiskDecision.no_action(
            reason="no position"
        )

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule}, regime_key="regime"
        )

        context = create_mock_context(
            position_quantity=0.0,  # No position
            market_features={"regime": "trending"},
        )

        decision = rule.evaluate(context)

        # Should still delegate (delegated rule handles no position)
        mock_rule.evaluate.assert_called_once_with(context)


class TestRegimeDependentRuleRegimeTransitions:
    """Test behavior during regime transitions."""

    def test_switches_rule_when_regime_changes(self):
        """Test that rule switches when regime changes between calls."""
        high_vol_rule = Mock(spec=RiskRule)
        high_vol_rule.evaluate.return_value = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="high vol"
        )

        low_vol_rule = Mock(spec=RiskRule)
        low_vol_rule.evaluate.return_value = RiskDecision.no_action(reason="low vol")

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        # First call: low VIX
        context1 = create_mock_context(market_features={"vix": 15.0})
        decision1 = rule.evaluate(context1)

        assert decision1.should_exit is False
        low_vol_rule.evaluate.assert_called_once()
        high_vol_rule.evaluate.assert_not_called()

        # Reset mocks
        low_vol_rule.reset_mock()
        high_vol_rule.reset_mock()

        # Second call: high VIX (regime changed)
        context2 = create_mock_context(market_features={"vix": 30.0})
        decision2 = rule.evaluate(context2)

        assert decision2.should_exit is True
        high_vol_rule.evaluate.assert_called_once()
        low_vol_rule.evaluate.assert_not_called()

    def test_vix_transition_at_threshold(self):
        """Test regime transition when VIX crosses threshold."""
        high_vol_rule = Mock(spec=RiskRule)
        high_vol_rule.evaluate.return_value = RiskDecision.exit_now(
            exit_type=ExitType.RISK_EXIT, reason="high vol"
        )

        low_vol_rule = Mock(spec=RiskRule)
        low_vol_rule.evaluate.return_value = RiskDecision.no_action()

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        # Just below threshold
        context1 = create_mock_context(market_features={"vix": 19.99})
        decision1 = rule.evaluate(context1)
        assert "regime" in decision1.metadata
        assert decision1.metadata["regime"] == "low_vol"

        # Reset mocks
        low_vol_rule.reset_mock()
        high_vol_rule.reset_mock()

        # Just above threshold
        context2 = create_mock_context(market_features={"vix": 20.01})
        decision2 = rule.evaluate(context2)
        assert "regime" in decision2.metadata
        assert decision2.metadata["regime"] == "high_vol"


class TestRegimeDependentRuleIntegration:
    """Integration tests with real risk rules."""

    def test_vix_adaptation_with_real_volatility_scaled_rules(self):
        """Integration test: VIX-based adaptation with VolatilityScaledStopLoss."""
        # Create regime-dependent rule with real VolatilityScaledStopLoss
        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0,
            high_vol_rule=VolatilityScaledStopLoss(1.5),  # tight
            low_vol_rule=VolatilityScaledStopLoss(2.5),  # wide
        )

        # Scenario 1: Low VIX (calm market) - expect 2.5x ATR stop
        context_calm = create_mock_context(
            position_quantity=100.0,
            entry_price=100.0,
            current_price=100.0,
            features={"atr": 2.0},
            market_features={"vix": 15.0},  # Low VIX
        )

        decision_calm = rule.evaluate(context_calm)

        # Expected: 2.5x ATR = 100 - (2.5 × 2.0) = 95.0
        assert decision_calm.update_stop_loss == pytest.approx(95.0)
        assert decision_calm.metadata["regime"] == "low_vol"
        assert decision_calm.metadata["delegated_to"] == "VolatilityScaledStopLoss"

        # Scenario 2: High VIX (panic) - expect 1.5x ATR stop (tighter)
        context_panic = create_mock_context(
            position_quantity=100.0,
            entry_price=100.0,
            current_price=100.0,
            features={"atr": 2.0},
            market_features={"vix": 30.0},  # High VIX
        )

        decision_panic = rule.evaluate(context_panic)

        # Expected: 1.5x ATR = 100 - (1.5 × 2.0) = 97.0 (tighter)
        assert decision_panic.update_stop_loss == pytest.approx(97.0)
        assert decision_panic.metadata["regime"] == "high_vol"
        assert decision_panic.metadata["vix"] == 30.0

        # Verify stops are tighter in high VIX
        assert decision_panic.update_stop_loss > decision_calm.update_stop_loss

    def test_custom_regime_with_different_rule_types(self):
        """Integration test: Custom regime with different rule types."""
        # Mix different rule types for different regimes
        rule = RegimeDependentRule(
            regime_rules={
                "trending": VolatilityScaledStopLoss(2.0),
                "mean_reverting": PriceBasedStopLoss(98.0),  # Fixed price stop
            },
            regime_key="market_regime",
            default_regime="trending",
        )

        # Trending regime - volatility-scaled stop
        context_trending = create_mock_context(
            position_quantity=100.0,
            entry_price=100.0,
            features={"atr": 1.5},
            market_features={"market_regime": "trending"},
        )

        decision_trending = rule.evaluate(context_trending)

        # Expected: 2.0x ATR = 100 - (2.0 × 1.5) = 97.0
        assert decision_trending.update_stop_loss == pytest.approx(97.0)

        # Mean-reverting regime - fixed price stop at 98
        # PriceBasedStopLoss only exits when stop is hit, doesn't update levels
        # It returns exit_now() when price <= stop_loss_price
        # Since current_price=100 > stop_price=98, no action
        context_mr = create_mock_context(
            position_quantity=100.0,
            entry_price=100.0,
            current_price=100.0,
            market_features={"market_regime": "mean_reverting"},
        )

        decision_mr = rule.evaluate(context_mr)

        # PriceBasedStopLoss returns no_action when stop not hit
        # It doesn't update stop levels, only exits when breached
        assert decision_mr.should_exit is False
        assert decision_mr.update_stop_loss is None

        # Test when price drops below stop
        context_mr_hit = create_mock_context(
            position_quantity=100.0,
            entry_price=100.0,
            current_price=97.0,  # Below stop
            market_features={"market_regime": "mean_reverting"},
        )

        decision_mr_hit = rule.evaluate(context_mr_hit)

        # Should trigger exit
        assert decision_mr_hit.should_exit is True
        assert decision_mr_hit.exit_type == ExitType.STOP_LOSS


class TestRegimeDependentRuleRepr:
    """Test string representation."""

    def test_repr_vix_based(self):
        """Test __repr__ for VIX-based rule."""
        high_vol_rule = Mock(spec=RiskRule)
        low_vol_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=20.0, high_vol_rule=high_vol_rule, low_vol_rule=low_vol_rule
        )

        repr_str = repr(rule)

        assert "RegimeDependentRule" in repr_str
        assert "VIX-based" in repr_str
        assert "threshold=20.0" in repr_str
        assert "high_vol" in repr_str
        assert "low_vol" in repr_str

    def test_repr_custom_regime(self):
        """Test __repr__ for custom regime rule."""
        mock_rule = Mock(spec=RiskRule)

        rule = RegimeDependentRule(
            regime_rules={"trending": mock_rule, "mean_reverting": mock_rule},
            regime_key="market_regime",
            default_regime="trending",
        )

        repr_str = repr(rule)

        assert "RegimeDependentRule" in repr_str
        assert "regime_key='market_regime'" in repr_str
        assert "trending" in repr_str
        assert "mean_reverting" in repr_str
        assert "default='trending'" in repr_str
