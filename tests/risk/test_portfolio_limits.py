"""Tests for portfolio-level risk limits."""

from ml4t.backtest.risk.portfolio.limits import (
    DailyLossLimit,
    GrossExposureLimit,
    LimitResult,
    MaxDrawdownLimit,
    MaxExposureLimit,
    MaxPositionsLimit,
    NetExposureLimit,
    PortfolioState,
)


class TestLimitResult:
    """Test LimitResult factory methods."""

    def test_ok(self):
        """Test ok() factory method."""
        result = LimitResult.ok()
        assert not result.breached
        assert result.action == "none"
        assert result.reason == ""
        assert result.reduction_pct == 0.0

    def test_warn(self):
        """Test warn() factory method."""
        result = LimitResult.warn("test warning")
        assert result.breached
        assert result.action == "warn"
        assert result.reason == "test warning"
        assert result.reduction_pct == 0.0

    def test_reduce(self):
        """Test reduce() factory method."""
        result = LimitResult.reduce("reduce position", 0.25)
        assert result.breached
        assert result.action == "reduce"
        assert result.reason == "reduce position"
        assert result.reduction_pct == 0.25

    def test_halt(self):
        """Test halt() factory method."""
        result = LimitResult.halt("halt trading")
        assert result.breached
        assert result.action == "halt"
        assert result.reason == "halt trading"


class TestPortfolioState:
    """Test PortfolioState dataclass."""

    def test_creation(self):
        """Test creating PortfolioState."""
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=105000.0,
            current_drawdown=0.05,
            num_positions=3,
            positions={"A": 30000.0, "B": 20000.0, "C": -10000.0},
            daily_pnl=-500.0,
            gross_exposure=60000.0,
            net_exposure=40000.0,
        )
        assert state.equity == 100000.0
        assert state.num_positions == 3
        assert state.gross_exposure == 60000.0


class TestMaxDrawdownLimit:
    """Test MaxDrawdownLimit."""

    def test_no_breach(self):
        """Test no breach when drawdown is below limit."""
        limit = MaxDrawdownLimit(max_drawdown=0.20)
        state = PortfolioState(
            equity=90000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.10,
            num_positions=1,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_halt(self):
        """Test halt action on breach."""
        limit = MaxDrawdownLimit(max_drawdown=0.10, action="halt")
        state = PortfolioState(
            equity=85000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.15,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"
        assert "15.0%" in result.reason

    def test_warn_threshold(self):
        """Test warning at warn_threshold."""
        limit = MaxDrawdownLimit(max_drawdown=0.20, warn_threshold=0.10)
        state = PortfolioState(
            equity=88000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.12,  # Above warn, below max
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "warn"


class TestMaxPositionsLimit:
    """Test MaxPositionsLimit."""

    def test_no_breach(self):
        """Test no breach when under limit."""
        limit = MaxPositionsLimit(max_positions=5)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=3,
            positions={"A": 10000, "B": 10000, "C": 10000},
            daily_pnl=0.0,
            gross_exposure=30000.0,
            net_exposure=30000.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_at_limit(self):
        """Test breach when at limit."""
        limit = MaxPositionsLimit(max_positions=3, action="halt")
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=3,
            positions={"A": 10000, "B": 10000, "C": 10000},
            daily_pnl=0.0,
            gross_exposure=30000.0,
            net_exposure=30000.0,
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"


class TestMaxExposureLimit:
    """Test MaxExposureLimit (single asset concentration)."""

    def test_no_breach(self):
        """Test no breach when exposure is acceptable."""
        limit = MaxExposureLimit(max_exposure_pct=0.25)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 20000.0, "B": 15000.0},  # 20% and 15%
            daily_pnl=0.0,
            gross_exposure=35000.0,
            net_exposure=35000.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_single_asset(self):
        """Test breach when single asset exceeds limit."""
        limit = MaxExposureLimit(max_exposure_pct=0.20, action="warn")
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 30000.0, "B": 15000.0},  # 30% exceeds 20%
            daily_pnl=0.0,
            gross_exposure=45000.0,
            net_exposure=45000.0,
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "warn"
        assert "A" in result.reason  # Should mention the asset

    def test_zero_equity(self):
        """Test with zero equity (avoid division by zero)."""
        limit = MaxExposureLimit(max_exposure_pct=0.20)
        state = PortfolioState(
            equity=0.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert not result.breached  # No error


class TestDailyLossLimit:
    """Test DailyLossLimit."""

    def test_no_breach_profit(self):
        """Test no breach when profitable."""
        limit = DailyLossLimit(max_daily_loss_pct=0.02)
        state = PortfolioState(
            equity=102000.0,
            initial_equity=100000.0,
            high_water_mark=102000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"A": 50000.0},
            daily_pnl=2000.0,  # Profitable
            gross_exposure=50000.0,
            net_exposure=50000.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_no_breach_small_loss(self):
        """Test no breach with small loss."""
        limit = DailyLossLimit(max_daily_loss_pct=0.05)
        state = PortfolioState(
            equity=98000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"A": 48000.0},
            daily_pnl=-2000.0,  # 2% loss
            gross_exposure=48000.0,
            net_exposure=48000.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_large_loss(self):
        """Test breach with large loss."""
        limit = DailyLossLimit(max_daily_loss_pct=0.02, action="halt")
        state = PortfolioState(
            equity=95000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"A": 45000.0},
            daily_pnl=-5000.0,  # 5.26% loss of current equity
            gross_exposure=45000.0,
            net_exposure=45000.0,
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"

    def test_zero_equity(self):
        """Test with zero equity."""
        limit = DailyLossLimit(max_daily_loss_pct=0.02)
        state = PortfolioState(
            equity=0.0,
            initial_equity=0.0,
            high_water_mark=0.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=-1000.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert not result.breached  # No error


class TestGrossExposureLimit:
    """Test GrossExposureLimit (leverage limit)."""

    def test_no_breach(self):
        """Test no breach when under limit."""
        limit = GrossExposureLimit(max_gross_exposure=2.0)  # 2x leverage allowed
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 80000.0, "B": 70000.0},  # 150% gross
            daily_pnl=0.0,
            gross_exposure=150000.0,
            net_exposure=150000.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_over_leverage(self):
        """Test breach when over leverage limit."""
        limit = GrossExposureLimit(max_gross_exposure=1.5, action="halt")
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 100000.0, "B": 80000.0},  # 180% gross
            daily_pnl=0.0,
            gross_exposure=180000.0,
            net_exposure=180000.0,
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"

    def test_zero_equity(self):
        """Test with zero equity."""
        limit = GrossExposureLimit(max_gross_exposure=1.0)
        state = PortfolioState(
            equity=0.0,
            initial_equity=0.0,
            high_water_mark=0.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert not result.breached  # No error


class TestNetExposureLimit:
    """Test NetExposureLimit (market neutral enforcement)."""

    def test_no_breach(self):
        """Test no breach when within bounds."""
        limit = NetExposureLimit(max_net_exposure=0.20, min_net_exposure=-0.20)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 55000.0, "B": -45000.0},  # 10% net long
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=10000.0,
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_too_long(self):
        """Test breach when too net long."""
        limit = NetExposureLimit(max_net_exposure=0.10, min_net_exposure=-0.10, action="warn")
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 80000.0, "B": -50000.0},  # 30% net long
            daily_pnl=0.0,
            gross_exposure=130000.0,
            net_exposure=30000.0,
        )
        result = limit.check(state)
        assert result.breached
        assert "30.0%" in result.reason or "net exposure" in result.reason.lower()

    def test_breach_too_short(self):
        """Test breach when too net short."""
        limit = NetExposureLimit(max_net_exposure=0.10, min_net_exposure=-0.10, action="warn")
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"A": 30000.0, "B": -60000.0},  # 30% net short
            daily_pnl=0.0,
            gross_exposure=90000.0,
            net_exposure=-30000.0,
        )
        result = limit.check(state)
        assert result.breached
        assert "min" in result.reason.lower()

    def test_zero_equity(self):
        """Test with zero equity."""
        limit = NetExposureLimit(max_net_exposure=0.10, min_net_exposure=-0.10)
        state = PortfolioState(
            equity=0.0,
            initial_equity=0.0,
            high_water_mark=0.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        result = limit.check(state)
        assert not result.breached  # No error
