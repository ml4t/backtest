"""Unit tests for dynamic trailing stop risk management rule."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from ml4t.backtest.risk.rules.dynamic_trailing import DynamicTrailingStop
from ml4t.backtest.risk.context import RiskContext


class TestDynamicTrailingStop:
    """Tests for DynamicTrailingStop rule."""

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_init_validates_initial_trail_pct(self):
        """Test that initial_trail_pct must be positive."""
        with pytest.raises(ValueError, match="initial_trail_pct must be positive"):
            DynamicTrailingStop(initial_trail_pct=0.0, tighten_rate=0.001)

        with pytest.raises(ValueError, match="initial_trail_pct must be positive"):
            DynamicTrailingStop(initial_trail_pct=-0.05, tighten_rate=0.001)

        # Valid values should work
        DynamicTrailingStop(initial_trail_pct=0.01, tighten_rate=0.001)
        DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)
        DynamicTrailingStop(initial_trail_pct=0.10, tighten_rate=0.001)

    def test_init_validates_tighten_rate(self):
        """Test that tighten_rate must be non-negative."""
        with pytest.raises(ValueError, match="tighten_rate must be non-negative"):
            DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=-0.001)

        # Zero is allowed (no tightening)
        DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.0)

        # Positive values work
        DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

    def test_init_validates_minimum_trail_pct(self):
        """Test that minimum_trail_pct must be positive and less than initial."""
        with pytest.raises(ValueError, match="minimum_trail_pct must be positive"):
            DynamicTrailingStop(
                initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=0.0
            )

        with pytest.raises(ValueError, match="minimum_trail_pct must be positive"):
            DynamicTrailingStop(
                initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=-0.001
            )

        with pytest.raises(
            ValueError,
            match="minimum_trail_pct .* must be less than initial_trail_pct",
        ):
            DynamicTrailingStop(
                initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=0.05
            )

        with pytest.raises(
            ValueError,
            match="minimum_trail_pct .* must be less than initial_trail_pct",
        ):
            DynamicTrailingStop(
                initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=0.06
            )

        # Valid minimum
        DynamicTrailingStop(
            initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=0.005
        )

    def test_init_accepts_custom_priority(self):
        """Test that custom priority is accepted."""
        rule = DynamicTrailingStop(
            initial_trail_pct=0.05, tighten_rate=0.001, priority=50
        )
        assert rule.priority == 50

    # -------------------------------------------------------------------------
    # No Position Tests
    # -------------------------------------------------------------------------

    def test_no_position_returns_no_action(self):
        """Test that rule returns NO_ACTION when there's no position."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 0.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        assert not decision.should_exit
        assert decision.update_stop_loss is None
        assert "No position" in decision.reason

    # -------------------------------------------------------------------------
    # Trail Tightening Tests
    # -------------------------------------------------------------------------

    def test_trail_uses_initial_pct_at_bar_zero(self):
        """Test that trail uses initial_trail_pct at bars_held=0."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 0
        context.max_favorable_excursion = 0.0  # No profit yet
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail should be 5% at bar 0
        # Peak = 100 + 0 = 100
        # Stop = 100 × (1 - 0.05) = 95.0
        assert decision.update_stop_loss == pytest.approx(95.0)
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.05)

    def test_trail_tightens_after_n_bars(self):
        """Test that trail tightens correctly after N bars."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 20
        context.max_favorable_excursion = 500.0  # $5 per share × 100 shares
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 20: 5% - (20 × 0.1%) = 3%
        expected_trail = 0.05 - (20 * 0.001)
        assert decision.metadata["current_trail_pct"] == pytest.approx(expected_trail)
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.03)

        # Peak = 100 + (500 / 100) = 105
        # Stop = 105 × (1 - 0.03) = 101.85
        expected_peak = 100.0 + (500.0 / 100.0)
        expected_stop = expected_peak * (1 - expected_trail)
        assert decision.update_stop_loss == pytest.approx(expected_stop)
        assert decision.update_stop_loss == pytest.approx(101.85)

    def test_trail_reaches_minimum_floor(self):
        """Test that trail never goes below minimum_trail_pct."""
        rule = DynamicTrailingStop(
            initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=0.005
        )

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 100  # Very long hold
        context.max_favorable_excursion = 1000.0  # $10 per share
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 100: 5% - (100 × 0.1%) = -5% → clamped to 0.5%
        # Should be clamped to minimum_trail_pct = 0.005
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.005)

        # Peak = 100 + 10 = 110
        # Stop = 110 × (1 - 0.005) = 109.45
        expected_peak = 100.0 + (1000.0 / 100.0)
        expected_stop = expected_peak * (1 - 0.005)
        assert decision.update_stop_loss == pytest.approx(expected_stop)
        assert decision.update_stop_loss == pytest.approx(109.45)

    def test_zero_tighten_rate_keeps_constant_trail(self):
        """Test that tighten_rate=0 means trail never tightens."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.0)

        for bars_held in [0, 10, 50, 100]:
            context = Mock(spec=RiskContext)
            context.position_quantity = 100.0  # Long
            context.entry_price = 100.0
            context.bars_held = bars_held
            context.max_favorable_excursion = 500.0
            context.asset_id = "SPY"

            decision = rule.evaluate(context)

            # Trail should always be 5%
            assert decision.metadata["current_trail_pct"] == pytest.approx(0.05)

    # -------------------------------------------------------------------------
    # Stop Level Calculation Tests (Long Positions)
    # -------------------------------------------------------------------------

    def test_long_stop_trails_peak_price(self):
        """Test that long position stop trails the peak price (entry + MFE)."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 10
        context.max_favorable_excursion = 1000.0  # $10 per share × 100 shares
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 10: 5% - (10 × 0.1%) = 4%
        # Peak = 100 + (1000 / 100) = 110
        # Stop = 110 × (1 - 0.04) = 105.6
        assert decision.metadata["peak_price"] == pytest.approx(110.0)
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.04)
        assert decision.update_stop_loss == pytest.approx(105.6)

    def test_long_stop_with_zero_mfe(self):
        """Test long position with zero MFE (no profit yet)."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 5
        context.max_favorable_excursion = 0.0  # No profit yet
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 5: 5% - (5 × 0.1%) = 4.5%
        # Peak = 100 + 0 = 100
        # Stop = 100 × (1 - 0.045) = 95.5
        assert decision.metadata["peak_price"] == pytest.approx(100.0)
        assert decision.update_stop_loss == pytest.approx(95.5)

    def test_long_stop_never_moves_backward(self):
        """Test that long position stop only moves up (tightens), never down."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        # Scenario: Price moved up to 110, then pulled back to 105
        # MFE tracks the peak (110), not current price
        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 20
        context.max_favorable_excursion = 1000.0  # Peak was $10 per share
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 20: 3%
        # Peak = 100 + 10 = 110 (MFE tracks peak, not current)
        # Stop = 110 × 0.97 = 106.7
        assert decision.metadata["peak_price"] == pytest.approx(110.0)
        assert decision.update_stop_loss == pytest.approx(106.7)

    # -------------------------------------------------------------------------
    # Stop Level Calculation Tests (Short Positions)
    # -------------------------------------------------------------------------

    def test_short_stop_trails_peak_price(self):
        """Test that short position stop trails the peak price (entry - MFE)."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.entry_price = 100.0
        context.bars_held = 10
        context.max_favorable_excursion = 1000.0  # $10 per share profit (short)
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 10: 5% - (10 × 0.1%) = 4%
        # Peak (lowest) = 100 - (1000 / 100) = 90
        # Stop = 90 × (1 + 0.04) = 93.6
        assert decision.metadata["peak_price"] == pytest.approx(90.0)
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.04)
        assert decision.update_stop_loss == pytest.approx(93.6)

    def test_short_stop_with_zero_mfe(self):
        """Test short position with zero MFE (no profit yet)."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.entry_price = 100.0
        context.bars_held = 5
        context.max_favorable_excursion = 0.0  # No profit yet
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 5: 5% - (5 × 0.1%) = 4.5%
        # Peak = 100 - 0 = 100
        # Stop = 100 × (1 + 0.045) = 104.5
        assert decision.metadata["peak_price"] == pytest.approx(100.0)
        assert decision.update_stop_loss == pytest.approx(104.5)

    def test_short_stop_never_moves_backward(self):
        """Test that short position stop only moves down (tightens), never up."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        # Scenario: Price moved down to 90, then bounced to 95
        # MFE tracks the peak (90 for short), not current price
        context = Mock(spec=RiskContext)
        context.position_quantity = -100.0  # Short
        context.entry_price = 100.0
        context.bars_held = 20
        context.max_favorable_excursion = 1000.0  # Peak was $10 profit
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 20: 3%
        # Peak (lowest) = 100 - 10 = 90 (MFE tracks peak, not current)
        # Stop = 90 × 1.03 = 92.7
        assert decision.metadata["peak_price"] == pytest.approx(90.0)
        assert decision.update_stop_loss == pytest.approx(92.7)

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_very_long_holding_period(self):
        """Test behavior with very long holding period (trail at minimum)."""
        rule = DynamicTrailingStop(
            initial_trail_pct=0.05, tighten_rate=0.001, minimum_trail_pct=0.005
        )

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0  # Long
        context.entry_price = 100.0
        context.bars_held = 200  # Very long hold
        context.max_favorable_excursion = 2000.0  # $20 profit
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail should be at minimum (0.5%)
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.005)

        # Peak = 100 + 20 = 120
        # Stop = 120 × (1 - 0.005) = 119.4
        assert decision.metadata["peak_price"] == pytest.approx(120.0)
        assert decision.update_stop_loss == pytest.approx(119.4)

    def test_metadata_completeness(self):
        """Test that decision includes comprehensive metadata."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.bars_held = 15
        context.max_favorable_excursion = 750.0
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Check all expected metadata keys
        assert "bars_held" in decision.metadata
        assert "initial_trail_pct" in decision.metadata
        assert "current_trail_pct" in decision.metadata
        assert "tighten_rate" in decision.metadata
        assert "minimum_trail_pct" in decision.metadata
        assert "peak_price" in decision.metadata
        assert "stop_level" in decision.metadata
        assert "trail_distance" in decision.metadata
        assert "trail_distance_pct" in decision.metadata
        assert "mfe" in decision.metadata
        assert "mfe_per_share" in decision.metadata
        assert "entry_price" in decision.metadata
        assert "position_direction" in decision.metadata

        # Verify some values
        assert decision.metadata["bars_held"] == 15
        assert decision.metadata["entry_price"] == 100.0
        assert decision.metadata["position_direction"] == "long"

    # -------------------------------------------------------------------------
    # Scenario Tests
    # -------------------------------------------------------------------------

    def test_trend_capture_scenario(self):
        """Test capturing profit in a strong trend that reverses."""
        rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        # Scenario: Entry at 100, trends up to 110 over 20 bars, then reverses
        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.bars_held = 20
        context.max_favorable_excursion = 1000.0  # Peaked at +$10/share
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # After 20 bars: trail = 5% - 2% = 3%
        # Peak = 110, Stop = 110 × 0.97 = 106.7
        # Captured most of the 10% move, locked in ~6.7% profit
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.03)
        assert decision.metadata["peak_price"] == pytest.approx(110.0)
        assert decision.update_stop_loss == pytest.approx(106.7)

        # If price drops to 106.5, stop would trigger
        # Realized profit: (106.5 - 100) / 100 = 6.5% (vs 10% peak)
        # Dynamic trail captured 65% of the peak move

    def test_aggressive_settings(self):
        """Test aggressive settings (tight initial trail, fast tightening)."""
        rule = DynamicTrailingStop(
            initial_trail_pct=0.03, tighten_rate=0.002, minimum_trail_pct=0.005
        )

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.bars_held = 10
        context.max_favorable_excursion = 500.0  # +$5/share
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 10: 3% - (10 × 0.2%) = 1%
        # Peak = 105, Stop = 105 × 0.99 = 103.95
        # Very tight trail locks in profit quickly
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.01)
        assert decision.update_stop_loss == pytest.approx(103.95)

    def test_patient_settings(self):
        """Test patient settings (wide initial trail, slow tightening)."""
        rule = DynamicTrailingStop(
            initial_trail_pct=0.08, tighten_rate=0.0005, minimum_trail_pct=0.01
        )

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.bars_held = 40
        context.max_favorable_excursion = 1000.0  # +$10/share
        context.asset_id = "SPY"

        decision = rule.evaluate(context)

        # Trail at bar 40: 8% - (40 × 0.05%) = 6%
        # Peak = 110, Stop = 110 × 0.94 = 103.4
        # Wide trail gives trend room to breathe
        assert decision.metadata["current_trail_pct"] == pytest.approx(0.06)
        assert decision.update_stop_loss == pytest.approx(103.4)


class TestDynamicVsFixedTrailingComparison:
    """Integration tests comparing dynamic vs fixed trailing stops."""

    def test_dynamic_captures_more_of_trend_than_fixed(self):
        """Test that dynamic trail captures more profit than fixed trail."""
        # Setup: Price goes from 100 → 110 over 20 bars, then drops to 107

        # Dynamic trail (5% initial, 0.1% tightening)
        dynamic_rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.bars_held = 20
        context.max_favorable_excursion = 1000.0  # Peaked at 110
        context.asset_id = "SPY"

        dynamic_decision = dynamic_rule.evaluate(context)

        # Dynamic: trail = 3%, stop = 110 × 0.97 = 106.7
        # Would exit at ~107 (locked in 7% vs 10% peak)
        dynamic_stop = dynamic_decision.update_stop_loss
        assert dynamic_stop == pytest.approx(106.7)

        # Fixed 5% trail comparison:
        # Stop = 110 × 0.95 = 104.5
        # Would exit at ~104.5 (locked in 4.5% vs 10% peak)

        # Dynamic trail locks in more profit (7% vs 4.5%)
        fixed_stop_approx = 104.5
        assert dynamic_stop > fixed_stop_approx
        assert (dynamic_stop - 100.0) / (110.0 - 100.0) > 0.67  # Captured >67% of move

    def test_dynamic_exits_faster_on_reversal(self):
        """Test that tightened dynamic trail exits faster than fixed on reversal."""
        # Setup: Position held for 40 bars with good profit, then price reverses

        dynamic_rule = DynamicTrailingStop(initial_trail_pct=0.05, tighten_rate=0.001)

        context = Mock(spec=RiskContext)
        context.position_quantity = 100.0
        context.entry_price = 100.0
        context.bars_held = 40  # Long hold
        context.max_favorable_excursion = 1000.0  # Peak at 110
        context.asset_id = "SPY"

        dynamic_decision = dynamic_rule.evaluate(context)

        # Dynamic: trail = 5% - 4% = 1%, stop = 110 × 0.99 = 108.9
        # Very tight trail after 40 bars
        dynamic_stop = dynamic_decision.update_stop_loss
        assert dynamic_stop == pytest.approx(108.9)

        # Fixed 5% trail: stop = 110 × 0.95 = 104.5
        # Dynamic exits much faster (at 108.9 vs 104.5)
        fixed_stop_approx = 104.5
        assert dynamic_stop > fixed_stop_approx
