"""Unit tests for risk decision conflict resolution.

Tests the priority + conservative conflict resolution logic in RiskDecision.merge():
- Priority override: highest priority wins
- Same priority: most conservative wins (tightest stop, nearest target)
"""

import pytest
from decimal import Decimal

from ml4t.backtest.risk.decision import RiskDecision, ExitType


class TestStopLossConflictResolution:
    """Test stop-loss conflict resolution logic."""

    def test_different_priorities_highest_wins(self):
        """Test that highest priority stop-loss wins regardless of value."""
        # Lower priority but tighter stop (99.0)
        d1 = RiskDecision(
            should_exit=False,
            update_stop_loss=99.0,
            priority=5,
            reason="Rule 1"
        )

        # Higher priority but looser stop (98.0)
        d2 = RiskDecision(
            should_exit=False,
            update_stop_loss=98.0,
            priority=10,
            reason="Rule 2"
        )

        merged = RiskDecision.merge([d1, d2])

        # Priority 10 wins even though 98.0 is looser
        assert merged.update_stop_loss == 98.0
        assert merged.priority == 10

    def test_same_priority_most_conservative_wins(self):
        """Test that most conservative (tightest/max) stop wins at same priority."""
        # Same priority, different stop-loss values
        d1 = RiskDecision(
            should_exit=False,
            update_stop_loss=98.0,
            priority=5,
            reason="Rule 1"
        )

        d2 = RiskDecision(
            should_exit=False,
            update_stop_loss=99.0,  # Tighter stop
            priority=5,
            reason="Rule 2"
        )

        merged = RiskDecision.merge([d1, d2])

        # max(98.0, 99.0) = 99.0 (tightest stop for long)
        assert merged.update_stop_loss == 99.0
        assert merged.priority == 5

    def test_three_rules_same_priority(self):
        """Test conflict resolution with three rules at same priority."""
        d1 = RiskDecision(update_stop_loss=97.0, priority=5)
        d2 = RiskDecision(update_stop_loss=99.0, priority=5)  # Tightest
        d3 = RiskDecision(update_stop_loss=98.0, priority=5)

        merged = RiskDecision.merge([d1, d2, d3])

        # max(97, 99, 98) = 99 (most conservative)
        assert merged.update_stop_loss == 99.0

    def test_mixed_priorities(self):
        """Test with mixed priorities - highest priority group uses conservative logic."""
        # Priority 5: 97.0, 98.0 → conservative would be 98.0
        # Priority 10: 95.0, 96.0 → conservative would be 96.0
        # Result: Priority 10 wins, so conservative within priority 10 = 96.0

        d1 = RiskDecision(update_stop_loss=97.0, priority=5)
        d2 = RiskDecision(update_stop_loss=98.0, priority=5)
        d3 = RiskDecision(update_stop_loss=95.0, priority=10)
        d4 = RiskDecision(update_stop_loss=96.0, priority=10)

        merged = RiskDecision.merge([d1, d2, d3, d4])

        # Priority 10 group wins, then max(95, 96) = 96
        assert merged.update_stop_loss == 96.0
        assert merged.priority == 10

    def test_single_stop_loss(self):
        """Test with only one stop-loss decision."""
        d1 = RiskDecision(update_stop_loss=98.5, priority=7)

        merged = RiskDecision.merge([d1])

        assert merged.update_stop_loss == 98.5
        assert merged.priority == 7


class TestTakeProfitConflictResolution:
    """Test take-profit conflict resolution logic."""

    def test_different_priorities_highest_wins(self):
        """Test that highest priority take-profit wins."""
        d1 = RiskDecision(
            update_take_profit=105.0,  # Nearer target
            priority=5
        )

        d2 = RiskDecision(
            update_take_profit=110.0,  # Further target
            priority=10
        )

        merged = RiskDecision.merge([d1, d2])

        # Priority 10 wins even though 110 is further
        assert merged.update_take_profit == 110.0
        assert merged.priority == 10

    def test_same_priority_most_conservative_wins(self):
        """Test that most conservative (nearest/min) target wins at same priority."""
        d1 = RiskDecision(
            update_take_profit=110.0,  # Further
            priority=5
        )

        d2 = RiskDecision(
            update_take_profit=105.0,  # Nearer (more conservative)
            priority=5
        )

        merged = RiskDecision.merge([d1, d2])

        # min(110, 105) = 105 (nearest/most conservative)
        assert merged.update_take_profit == 105.0
        assert merged.priority == 5

    def test_three_rules_same_priority(self):
        """Test with three rules at same priority."""
        d1 = RiskDecision(update_take_profit=108.0, priority=5)
        d2 = RiskDecision(update_take_profit=105.0, priority=5)  # Nearest
        d3 = RiskDecision(update_take_profit=112.0, priority=5)

        merged = RiskDecision.merge([d1, d2, d3])

        # min(108, 105, 112) = 105 (most conservative)
        assert merged.update_take_profit == 105.0


class TestCombinedStopAndTakeProfit:
    """Test conflict resolution when both stop-loss and take-profit are set."""

    def test_both_same_priority(self):
        """Test merging decisions with both SL and TP at same priority."""
        # Rule 1: SL=98, TP=105
        d1 = RiskDecision(
            update_stop_loss=98.0,
            update_take_profit=105.0,
            priority=5
        )

        # Rule 2: SL=99, TP=110
        d2 = RiskDecision(
            update_stop_loss=99.0,
            update_take_profit=110.0,
            priority=5
        )

        merged = RiskDecision.merge([d1, d2])

        # SL: max(98, 99) = 99 (tightest)
        # TP: min(105, 110) = 105 (nearest)
        assert merged.update_stop_loss == 99.0
        assert merged.update_take_profit == 105.0

    def test_both_different_priorities(self):
        """Test that each resolves independently based on priority."""
        # Rule 1 (priority 10): SL=98, TP=105
        d1 = RiskDecision(
            update_stop_loss=98.0,
            update_take_profit=105.0,
            priority=10
        )

        # Rule 2 (priority 5): SL=99, TP=110
        d2 = RiskDecision(
            update_stop_loss=99.0,
            update_take_profit=110.0,
            priority=5
        )

        merged = RiskDecision.merge([d1, d2])

        # Both SL and TP: priority 10 wins for both
        assert merged.update_stop_loss == 98.0  # From d1 (priority 10)
        assert merged.update_take_profit == 105.0  # From d1 (priority 10)

    def test_sl_and_tp_from_different_rules(self):
        """Test when SL and TP come from different rules."""
        # Rule 1: only SL
        d1 = RiskDecision(
            update_stop_loss=98.0,
            priority=10
        )

        # Rule 2: only TP
        d2 = RiskDecision(
            update_take_profit=105.0,
            priority=10
        )

        merged = RiskDecision.merge([d1, d2])

        # Both should be present
        assert merged.update_stop_loss == 98.0
        assert merged.update_take_profit == 105.0


class TestExitDecisionPriority:
    """Test that exit decisions take precedence over stop updates."""

    def test_exit_overrides_stop_update(self):
        """Test exit decision takes precedence over stop update."""
        # Stop update
        d1 = RiskDecision(
            update_stop_loss=99.0,
            priority=10
        )

        # Exit decision
        d2 = RiskDecision(
            should_exit=True,
            exit_type=ExitType.STOP_LOSS,
            priority=5,  # Lower priority
            reason="Stop-loss breach"
        )

        merged = RiskDecision.merge([d1, d2])

        # Exit takes precedence regardless of priority
        assert merged.should_exit
        assert merged.exit_type == ExitType.STOP_LOSS

    def test_multiple_exits_highest_priority_wins(self):
        """Test that highest priority exit wins among multiple exits."""
        d1 = RiskDecision(
            should_exit=True,
            exit_type=ExitType.TAKE_PROFIT,
            priority=5,
            reason="Target reached"
        )

        d2 = RiskDecision(
            should_exit=True,
            exit_type=ExitType.STOP_LOSS,
            priority=10,
            reason="Stop breach"
        )

        merged = RiskDecision.merge([d1, d2])

        # Priority 10 wins
        assert merged.should_exit
        assert merged.exit_type == ExitType.STOP_LOSS
        assert merged.priority == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_stop_or_tp_updates(self):
        """Test merge with no stop or take-profit updates."""
        d1 = RiskDecision(reason="No action 1", priority=5)
        d2 = RiskDecision(reason="No action 2", priority=10)

        merged = RiskDecision.merge([d1, d2])

        assert merged.update_stop_loss is None
        assert merged.update_take_profit is None
        assert merged.priority == 10  # Max priority

    def test_decimal_prices(self):
        """Test with Decimal prices for precision."""
        d1 = RiskDecision(
            update_stop_loss=Decimal("98.50"),
            priority=5
        )

        d2 = RiskDecision(
            update_stop_loss=Decimal("98.75"),
            priority=5
        )

        merged = RiskDecision.merge([d1, d2])

        # max(Decimal("98.50"), Decimal("98.75")) = Decimal("98.75")
        assert merged.update_stop_loss == Decimal("98.75")

    def test_metadata_merged(self):
        """Test that metadata is properly merged."""
        d1 = RiskDecision(
            update_stop_loss=98.0,
            priority=5,
            metadata={"rule": "VolatilityScaled", "atr": 2.5}
        )

        d2 = RiskDecision(
            update_stop_loss=99.0,
            priority=5,
            metadata={"rule": "DynamicTrailing", "trail_pct": 0.02}
        )

        merged = RiskDecision.merge([d1, d2])

        # Metadata should be merged
        assert "atr" in merged.metadata
        assert "trail_pct" in merged.metadata

    def test_reasons_concatenated(self):
        """Test that reasons from all decisions are captured."""
        d1 = RiskDecision(update_stop_loss=98.0, priority=5, reason="ATR-based stop")
        d2 = RiskDecision(update_stop_loss=99.0, priority=5, reason="Trailing stop")

        merged = RiskDecision.merge([d1, d2])

        # Reasons should be merged (implementation detail may vary)
        assert merged.reason is not None
        assert len(merged.reason) > 0
