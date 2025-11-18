"""Unit tests for volatility-scaled risk management rules."""

import pytest
from unittest.mock import Mock

from ml4t.backtest.risk.rules.volatility_scaled import (
    VolatilityScaledStopLoss,
    VolatilityScaledTakeProfit,
)
from ml4t.backtest.risk.context import RiskContext


class TestVolatilityScaledStopLoss:
    """Tests for VolatilityScaledStopLoss rule."""

    def test_init_validates_atr_multiplier(self):
        """Test that atr_multiplier must be positive."""
        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            VolatilityScaledStopLoss(atr_multiplier=0.0)

        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            VolatilityScaledStopLoss(atr_multiplier=-1.5)

        # Valid values should work
        VolatilityScaledStopLoss(atr_multiplier=0.5)
        VolatilityScaledStopLoss(atr_multiplier=2.0)
        VolatilityScaledStopLoss(atr_multiplier=5.0)

    def test_init_accepts_custom_volatility_key(self):
        """Test that custom volatility_key is accepted."""
        rule = VolatilityScaledStopLoss(
            atr_multiplier=2.0, volatility_key="realized_volatility"
        )
        assert rule.volatility_key == "realized_volatility"

    def test_init_accepts_custom_priority(self):
        """Test that custom priority is accepted."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0, priority=50)
        assert rule.priority == 50

    def test_no_position_returns_no_action(self):
        """Test that rule returns NO_ACTION when there's no position."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 0.0
        context.features = {"atr": 2.5}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_stop_loss is None
        assert "No position" in decision.reason

    def test_missing_atr_returns_no_action(self):
        """Test that rule returns NO_ACTION when ATR is missing from features."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.features = {"rsi": 50.0}  # No ATR
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_stop_loss is None
        assert "Missing atr" in decision.reason
        assert "available_features" in decision.metadata

    def test_zero_atr_returns_no_action(self):
        """Test that rule returns NO_ACTION when ATR is zero."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.features = {"atr": 0.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_stop_loss is None
        assert "Invalid atr" in decision.reason
        assert decision.metadata["atr"] == 0.0

    def test_negative_atr_returns_no_action(self):
        """Test that rule returns NO_ACTION when ATR is negative."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.features = {"atr": -1.5}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_stop_loss is None
        assert "Invalid atr" in decision.reason

    def test_long_position_stop_loss_calculation(self):
        """Test stop loss calculation for long positions."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Expected: 100.0 - (2.0 × 2.0) = 96.0
        assert not decision.should_exit
        assert decision.update_stop_loss is not None
        assert decision.update_stop_loss == pytest.approx(96.0)
        assert decision.update_take_profit is None
        assert "long" in decision.metadata["position_direction"]

    def test_short_position_stop_loss_calculation(self):
        """Test stop loss calculation for short positions."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Expected: 100.0 + (2.0 × 2.0) = 104.0
        assert not decision.should_exit
        assert decision.update_stop_loss is not None
        assert decision.update_stop_loss == pytest.approx(104.0)
        assert decision.update_take_profit is None
        assert "short" in decision.metadata["position_direction"]

    def test_different_atr_multipliers(self):
        """Test that different multipliers produce different stop levels."""
        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        # 1.5x ATR stop
        rule_1_5x = VolatilityScaledStopLoss(atr_multiplier=1.5)
        decision_1_5x = rule_1_5x.evaluate(context)
        # Expected: 100.0 - (1.5 × 2.0) = 97.0

        # 2.5x ATR stop
        rule_2_5x = VolatilityScaledStopLoss(atr_multiplier=2.5)
        decision_2_5x = rule_2_5x.evaluate(context)
        # Expected: 100.0 - (2.5 × 2.0) = 95.0

        assert decision_1_5x.update_stop_loss == pytest.approx(97.0)
        assert decision_2_5x.update_stop_loss == pytest.approx(95.0)
        assert decision_2_5x.update_stop_loss < decision_1_5x.update_stop_loss

    def test_different_atr_values(self):
        """Test that different ATR values produce different stop levels."""
        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.asset_id = "SPY"

        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        # Low volatility: ATR = 1.0
        context.features = {"atr": 1.0}
        decision_low_vol = rule.evaluate(context)
        # Expected: 100.0 - (2.0 × 1.0) = 98.0

        # High volatility: ATR = 4.0
        context.features = {"atr": 4.0}
        decision_high_vol = rule.evaluate(context)
        # Expected: 100.0 - (2.0 × 4.0) = 92.0

        assert decision_low_vol.update_stop_loss == pytest.approx(98.0)
        assert decision_high_vol.update_stop_loss == pytest.approx(92.0)
        assert decision_high_vol.update_stop_loss < decision_low_vol.update_stop_loss

    def test_realized_volatility_key(self):
        """Test using realized_volatility instead of atr."""
        rule = VolatilityScaledStopLoss(
            atr_multiplier=2.0, volatility_key="realized_volatility"
        )

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"realized_volatility": 3.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Expected: 100.0 - (2.0 × 3.0) = 94.0
        assert decision.update_stop_loss == pytest.approx(94.0)
        assert decision.metadata["volatility_key"] == "realized_volatility"
        assert decision.metadata["volatility_value"] == 3.0

    def test_metadata_includes_all_relevant_info(self):
        """Test that decision metadata includes all relevant information."""
        rule = VolatilityScaledStopLoss(atr_multiplier=2.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert "volatility_key" in decision.metadata
        assert "volatility_value" in decision.metadata
        assert "atr_multiplier" in decision.metadata
        assert "stop_loss_price" in decision.metadata
        assert "entry_price" in decision.metadata
        assert "stop_distance" in decision.metadata
        assert "stop_distance_pct" in decision.metadata
        assert "position_direction" in decision.metadata

        assert decision.metadata["volatility_key"] == "atr"
        assert decision.metadata["volatility_value"] == 2.0
        assert decision.metadata["atr_multiplier"] == 2.0
        assert decision.metadata["stop_loss_price"] == pytest.approx(96.0)
        assert decision.metadata["entry_price"] == 100.0
        assert decision.metadata["stop_distance"] == pytest.approx(4.0)
        assert decision.metadata["stop_distance_pct"] == pytest.approx(0.04)


class TestVolatilityScaledTakeProfit:
    """Tests for VolatilityScaledTakeProfit rule."""

    def test_init_validates_atr_multiplier(self):
        """Test that atr_multiplier must be positive."""
        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            VolatilityScaledTakeProfit(atr_multiplier=0.0)

        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            VolatilityScaledTakeProfit(atr_multiplier=-1.5)

        # Valid values should work
        VolatilityScaledTakeProfit(atr_multiplier=0.5)
        VolatilityScaledTakeProfit(atr_multiplier=3.0)
        VolatilityScaledTakeProfit(atr_multiplier=5.0)

    def test_init_accepts_custom_volatility_key(self):
        """Test that custom volatility_key is accepted."""
        rule = VolatilityScaledTakeProfit(
            atr_multiplier=3.0, volatility_key="realized_volatility"
        )
        assert rule.volatility_key == "realized_volatility"

    def test_init_accepts_custom_priority(self):
        """Test that custom priority is accepted."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0, priority=50)
        assert rule.priority == 50

    def test_no_position_returns_no_action(self):
        """Test that rule returns NO_ACTION when there's no position."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 0.0
        context.features = {"atr": 2.5}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_take_profit is None
        assert "No position" in decision.reason

    def test_missing_atr_returns_no_action(self):
        """Test that rule returns NO_ACTION when ATR is missing from features."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.features = {"rsi": 50.0}  # No ATR
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_take_profit is None
        assert "Missing atr" in decision.reason

    def test_zero_atr_returns_no_action(self):
        """Test that rule returns NO_ACTION when ATR is zero."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.features = {"atr": 0.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_take_profit is None
        assert "Invalid atr" in decision.reason

    def test_negative_atr_returns_no_action(self):
        """Test that rule returns NO_ACTION when ATR is negative."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.features = {"atr": -1.5}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_take_profit is None
        assert "Invalid atr" in decision.reason

    def test_long_position_take_profit_calculation(self):
        """Test take profit calculation for long positions."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Expected: 100.0 + (3.0 × 2.0) = 106.0
        assert not decision.should_exit
        assert decision.update_take_profit is not None
        assert decision.update_take_profit == pytest.approx(106.0)
        assert decision.update_stop_loss is None
        assert "long" in decision.metadata["position_direction"]

    def test_short_position_take_profit_calculation(self):
        """Test take profit calculation for short positions."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Expected: 100.0 - (3.0 × 2.0) = 94.0
        assert not decision.should_exit
        assert decision.update_take_profit is not None
        assert decision.update_take_profit == pytest.approx(94.0)
        assert decision.update_stop_loss is None
        assert "short" in decision.metadata["position_direction"]

    def test_different_atr_multipliers(self):
        """Test that different multipliers produce different target levels."""
        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        # 2.5x ATR target
        rule_2_5x = VolatilityScaledTakeProfit(atr_multiplier=2.5)
        decision_2_5x = rule_2_5x.evaluate(context)
        # Expected: 100.0 + (2.5 × 2.0) = 105.0

        # 4.0x ATR target
        rule_4_0x = VolatilityScaledTakeProfit(atr_multiplier=4.0)
        decision_4_0x = rule_4_0x.evaluate(context)
        # Expected: 100.0 + (4.0 × 2.0) = 108.0

        assert decision_2_5x.update_take_profit == pytest.approx(105.0)
        assert decision_4_0x.update_take_profit == pytest.approx(108.0)
        assert decision_4_0x.update_take_profit > decision_2_5x.update_take_profit

    def test_different_atr_values(self):
        """Test that different ATR values produce different target levels."""
        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.asset_id = "SPY"

        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        # Low volatility: ATR = 1.0
        context.features = {"atr": 1.0}
        decision_low_vol = rule.evaluate(context)
        # Expected: 100.0 + (3.0 × 1.0) = 103.0

        # High volatility: ATR = 4.0
        context.features = {"atr": 4.0}
        decision_high_vol = rule.evaluate(context)
        # Expected: 100.0 + (3.0 × 4.0) = 112.0

        assert decision_low_vol.update_take_profit == pytest.approx(103.0)
        assert decision_high_vol.update_take_profit == pytest.approx(112.0)
        assert decision_high_vol.update_take_profit > decision_low_vol.update_take_profit

    def test_realized_volatility_key(self):
        """Test using realized_volatility instead of atr."""
        rule = VolatilityScaledTakeProfit(
            atr_multiplier=3.0, volatility_key="realized_volatility"
        )

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"realized_volatility": 3.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Expected: 100.0 + (3.0 × 3.0) = 109.0
        assert decision.update_take_profit == pytest.approx(109.0)
        assert decision.metadata["volatility_key"] == "realized_volatility"
        assert decision.metadata["volatility_value"] == 3.0

    def test_metadata_includes_all_relevant_info(self):
        """Test that decision metadata includes all relevant information."""
        rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert "volatility_key" in decision.metadata
        assert "volatility_value" in decision.metadata
        assert "atr_multiplier" in decision.metadata
        assert "take_profit_price" in decision.metadata
        assert "entry_price" in decision.metadata
        assert "target_distance" in decision.metadata
        assert "target_distance_pct" in decision.metadata
        assert "position_direction" in decision.metadata

        assert decision.metadata["volatility_key"] == "atr"
        assert decision.metadata["volatility_value"] == 2.0
        assert decision.metadata["atr_multiplier"] == 3.0
        assert decision.metadata["take_profit_price"] == pytest.approx(106.0)
        assert decision.metadata["entry_price"] == 100.0
        assert decision.metadata["target_distance"] == pytest.approx(6.0)
        assert decision.metadata["target_distance_pct"] == pytest.approx(0.06)


class TestVolatilityScaledRiskRewardRatio:
    """Test risk/reward ratios when combining stop loss and take profit rules."""

    def test_2_to_1_risk_reward_ratio(self):
        """Test that 2x stop and 4x target gives 2:1 reward/risk ratio."""
        stop_rule = VolatilityScaledStopLoss(atr_multiplier=2.0)
        profit_rule = VolatilityScaledTakeProfit(atr_multiplier=4.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        stop_decision = stop_rule.evaluate(context)
        profit_decision = profit_rule.evaluate(context)

        # Stop: 100 - (2 × 2) = 96 (risk = 4.0)
        # Target: 100 + (4 × 2) = 108 (reward = 8.0)
        # Ratio: 8.0 / 4.0 = 2:1

        stop_distance = abs(stop_decision.update_stop_loss - context.entry_price)
        target_distance = abs(profit_decision.update_take_profit - context.entry_price)

        assert stop_distance == pytest.approx(4.0)
        assert target_distance == pytest.approx(8.0)
        assert target_distance / stop_distance == pytest.approx(2.0)

    def test_1_5_to_1_risk_reward_ratio(self):
        """Test that 2x stop and 3x target gives 1.5:1 reward/risk ratio."""
        stop_rule = VolatilityScaledStopLoss(atr_multiplier=2.0)
        profit_rule = VolatilityScaledTakeProfit(atr_multiplier=3.0)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.features = {"atr": 2.0}
        context.asset_id = "SPY"

        stop_decision = stop_rule.evaluate(context)
        profit_decision = profit_rule.evaluate(context)

        # Stop: 100 - (2 × 2) = 96 (risk = 4.0)
        # Target: 100 + (3 × 2) = 106 (reward = 6.0)
        # Ratio: 6.0 / 4.0 = 1.5:1

        stop_distance = abs(stop_decision.update_stop_loss - context.entry_price)
        target_distance = abs(profit_decision.update_take_profit - context.entry_price)

        assert target_distance / stop_distance == pytest.approx(1.5)
