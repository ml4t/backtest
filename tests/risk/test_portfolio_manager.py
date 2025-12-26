"""Tests for RiskManager portfolio-level risk management."""

from datetime import datetime

from ml4t.backtest.risk.portfolio.limits import (
    GrossExposureLimit,
    MaxDrawdownLimit,
    MaxExposureLimit,
    MaxPositionsLimit,
    PortfolioState,
)
from ml4t.backtest.risk.portfolio.manager import RiskManager


class TestRiskManagerInitialization:
    """Test RiskManager initialization."""

    def test_default_initialization(self):
        """Test default RiskManager creation."""
        manager = RiskManager()
        assert manager.limits == []
        assert not manager.is_halted
        assert manager.halt_reason == ""
        assert manager.warnings == []

    def test_initialize_with_limits(self):
        """Test RiskManager with limits."""
        limits = [MaxDrawdownLimit(max_drawdown=0.20), MaxPositionsLimit(max_positions=5)]
        manager = RiskManager(limits=limits)
        assert len(manager.limits) == 2

    def test_initialize_method(self):
        """Test initialize() sets tracking state."""
        manager = RiskManager()
        ts = datetime(2024, 1, 1, 9, 30)
        manager.initialize(initial_equity=100000.0, timestamp=ts)

        assert manager._initial_equity == 100000.0
        assert manager._high_water_mark == 100000.0
        assert manager._daily_start_equity == 100000.0
        assert manager._last_date == ts.date()
        assert not manager._halted
        assert manager._halt_reason == ""

    def test_initialize_without_timestamp(self):
        """Test initialize() without timestamp."""
        manager = RiskManager()
        manager.initialize(initial_equity=50000.0)

        assert manager._initial_equity == 50000.0
        assert manager._last_date is None


class TestRiskManagerUpdate:
    """Test RiskManager update method."""

    def test_update_high_water_mark(self):
        """Test high water mark updates on equity increase."""
        manager = RiskManager()
        manager.initialize(initial_equity=100000.0)

        # Equity increases
        manager.update(equity=110000.0, positions={"AAPL": 50000.0})
        assert manager._high_water_mark == 110000.0

        # Equity decreases (no update)
        manager.update(equity=105000.0, positions={"AAPL": 45000.0})
        assert manager._high_water_mark == 110000.0

        # Equity increases again
        manager.update(equity=115000.0, positions={"AAPL": 55000.0})
        assert manager._high_water_mark == 115000.0

    def test_update_new_trading_day(self):
        """Test daily P&L reset on new day."""
        manager = RiskManager()
        ts1 = datetime(2024, 1, 1, 9, 30)
        ts2 = datetime(2024, 1, 2, 9, 30)

        manager.initialize(initial_equity=100000.0, timestamp=ts1)
        manager.update(equity=105000.0, positions={}, timestamp=ts1)

        # New day - daily start should reset
        manager.update(equity=105000.0, positions={}, timestamp=ts2)
        assert manager._daily_start_equity == 105000.0
        assert manager._last_date == ts2.date()

    def test_update_checks_all_limits(self):
        """Test that update checks all configured limits."""
        limits = [
            MaxDrawdownLimit(max_drawdown=0.10),
            MaxPositionsLimit(max_positions=2),
        ]
        manager = RiskManager(limits=limits)
        manager.initialize(initial_equity=100000.0)

        # Trigger drawdown limit
        results = manager.update(equity=85000.0, positions={"A": 40000.0, "B": 45000.0})

        # Should detect drawdown breach
        assert len(results) >= 1
        assert any("drawdown" in r.reason for r in results)

    def test_update_halt_action(self):
        """Test that halt action sets halted state."""
        limits = [MaxDrawdownLimit(max_drawdown=0.10, action="halt")]
        manager = RiskManager(limits=limits)
        manager.initialize(initial_equity=100000.0)

        manager.update(equity=85000.0, positions={})

        assert manager.is_halted
        assert "drawdown" in manager.halt_reason

    def test_update_warn_action(self):
        """Test that warn action adds to warnings."""
        limits = [MaxExposureLimit(max_exposure_pct=0.50, action="warn")]
        manager = RiskManager(limits=limits)
        manager.initialize(initial_equity=100000.0)

        manager.update(equity=100000.0, positions={"AAPL": 60000.0})

        assert not manager.is_halted
        assert len(manager.warnings) >= 1
        assert any("exposure" in w for w in manager.warnings)


class TestRiskManagerBuildState:
    """Test _build_state method."""

    def test_build_state_basic(self):
        """Test basic state building."""
        manager = RiskManager()
        manager.initialize(initial_equity=100000.0)
        manager._high_water_mark = 110000.0
        manager._daily_start_equity = 95000.0

        state = manager._build_state(
            equity=100000.0,
            positions={"AAPL": 30000.0, "GOOG": -20000.0},
            timestamp=datetime(2024, 1, 1),
        )

        assert state.equity == 100000.0
        assert state.initial_equity == 100000.0
        assert state.high_water_mark == 110000.0
        assert abs(state.current_drawdown - 0.0909) < 0.001  # ~9.09%
        assert state.num_positions == 2
        assert state.daily_pnl == 5000.0  # 100000 - 95000
        assert state.gross_exposure == 50000.0  # |30000| + |-20000|
        assert state.net_exposure == 10000.0  # 30000 - 20000

    def test_build_state_zero_high_water_mark(self):
        """Test state building with zero high water mark."""
        manager = RiskManager()
        manager.initialize(initial_equity=0.0)

        state = manager._build_state(equity=0.0, positions={}, timestamp=None)

        assert state.current_drawdown == 0.0


class TestRiskManagerPermissions:
    """Test permission checking methods."""

    def test_can_open_position_not_halted(self):
        """Test can_open_position returns True when not halted."""
        manager = RiskManager()
        assert manager.can_open_position()

    def test_can_open_position_halted(self):
        """Test can_open_position returns False when halted."""
        manager = RiskManager()
        manager._halted = True
        assert not manager.can_open_position()

    def test_can_increase_position_not_halted(self):
        """Test can_increase_position when not halted."""
        manager = RiskManager()
        allowed, reason = manager.can_increase_position("AAPL", 10000.0)
        assert allowed
        assert reason == ""

    def test_can_increase_position_halted(self):
        """Test can_increase_position when halted."""
        manager = RiskManager()
        manager._halted = True
        manager._halt_reason = "drawdown breach"

        allowed, reason = manager.can_increase_position("AAPL", 10000.0)
        assert not allowed
        assert reason == "drawdown breach"


class TestRiskManagerProperties:
    """Test property accessors."""

    def test_is_halted_property(self):
        """Test is_halted property."""
        manager = RiskManager()
        assert not manager.is_halted

        manager._halted = True
        assert manager.is_halted

    def test_halt_reason_property(self):
        """Test halt_reason property."""
        manager = RiskManager()
        assert manager.halt_reason == ""

        manager._halt_reason = "test reason"
        assert manager.halt_reason == "test reason"

    def test_warnings_property(self):
        """Test warnings property."""
        manager = RiskManager()
        assert manager.warnings == []

        manager._warnings = ["warning1", "warning2"]
        assert manager.warnings == ["warning1", "warning2"]

    def test_current_drawdown_property(self):
        """Test current_drawdown property."""
        manager = RiskManager()
        # Default returns 0.0
        assert manager.current_drawdown == 0.0

        manager._high_water_mark = 100000.0
        # Still returns 0.0 since it's a snapshot (needs equity)
        assert manager.current_drawdown == 0.0


class TestRiskManagerReset:
    """Test reset functionality."""

    def test_reset_halt(self):
        """Test reset_halt clears halt state."""
        manager = RiskManager()
        manager._halted = True
        manager._halt_reason = "some reason"

        manager.reset_halt()

        assert not manager._halted
        assert manager._halt_reason == ""


class TestRiskManagerGetState:
    """Test get_state method."""

    def test_get_state(self):
        """Test get_state returns correct PortfolioState."""
        manager = RiskManager()
        manager.initialize(initial_equity=100000.0)
        manager._last_date = datetime(2024, 1, 1).date()

        state = manager.get_state(equity=95000.0, positions={"AAPL": 45000.0})

        assert isinstance(state, PortfolioState)
        assert state.equity == 95000.0
        assert state.initial_equity == 100000.0
        assert state.num_positions == 1


class TestRiskManagerIntegration:
    """Integration tests for RiskManager."""

    def test_full_workflow(self):
        """Test complete RiskManager workflow."""
        limits = [
            MaxDrawdownLimit(max_drawdown=0.20, warn_threshold=0.10),
            MaxPositionsLimit(max_positions=3),
        ]
        manager = RiskManager(limits=limits)

        # Initialize
        ts = datetime(2024, 1, 1, 9, 30)
        manager.initialize(initial_equity=100000.0, timestamp=ts)

        # Normal update - no breaches
        results = manager.update(equity=102000.0, positions={"AAPL": 50000.0}, timestamp=ts)
        assert len(results) == 0
        assert manager.can_open_position()

        # Drawdown warning (12.7% from 102000 to 89000)
        results = manager.update(equity=89000.0, positions={"AAPL": 45000.0}, timestamp=ts)
        assert any(r.action == "warn" for r in results)
        assert manager.can_open_position()  # Warn doesn't halt

        # Drawdown breach - halt (21.6% from 102000 to 80000)
        results = manager.update(equity=80000.0, positions={"AAPL": 40000.0}, timestamp=ts)
        assert manager.is_halted
        assert not manager.can_open_position()

        # Reset and continue
        manager.reset_halt()
        assert manager.can_open_position()

    def test_multiple_limit_breaches(self):
        """Test that all breaches are reported."""
        limits = [
            MaxDrawdownLimit(max_drawdown=0.10),
            MaxPositionsLimit(max_positions=2),
            GrossExposureLimit(max_gross_exposure=1.0),
        ]
        manager = RiskManager(limits=limits)
        manager.initialize(initial_equity=100000.0)

        # Trigger multiple breaches
        results = manager.update(
            equity=80000.0,  # 20% drawdown
            positions={"A": 50000.0, "B": 40000.0, "C": 30000.0},  # 3 positions, 150% gross
        )

        # Should detect multiple breaches
        assert len(results) >= 2
