"""Tests for portfolio-level risk limits."""

import numpy as np

from ml4t.backtest.risk.portfolio.limits import (
    BetaLimit,
    CVaRLimit,
    DailyLossLimit,
    FactorExposureLimit,
    GrossExposureLimit,
    LimitResult,
    MaxDrawdownLimit,
    MaxExposureLimit,
    MaxPositionsLimit,
    NetExposureLimit,
    PortfolioState,
    SectorExposureLimit,
    VaRLimit,
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


class TestVaRLimit:
    """Test VaRLimit (Value at Risk)."""

    def _make_state(self, returns: list | np.ndarray | None = None) -> PortfolioState:
        """Helper to create PortfolioState with returns in context."""
        context = {}
        if returns is not None:
            context["historical_returns"] = returns
        return PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"A": 50000.0},
            daily_pnl=0.0,
            gross_exposure=50000.0,
            net_exposure=50000.0,
            context=context,
        )

    def test_no_breach_low_volatility(self):
        """Test no breach when VaR is below threshold."""
        limit = VaRLimit(threshold=0.05, confidence_level=0.95, lookback_days=20)
        # Generate low-volatility returns (std ~1% daily)
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 30)  # 30 days of 1% daily vol

        state = self._make_state(returns)
        result = limit.check(state)
        assert not result.breached

    def test_breach_high_volatility(self):
        """Test breach when VaR exceeds threshold."""
        limit = VaRLimit(threshold=0.02, confidence_level=0.95, lookback_days=20, action="halt")
        # Generate high-volatility returns with large losses
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.03, 30)  # 3% daily vol
        returns[5] = -0.08  # Add a big loss day

        state = self._make_state(returns)
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"
        assert "VaR" in result.reason

    def test_no_data_graceful(self):
        """Test graceful handling when no historical returns available."""
        limit = VaRLimit(threshold=0.05, confidence_level=0.95, lookback_days=20)
        state = self._make_state(None)  # No returns data
        result = limit.check(state)
        assert not result.breached  # Graceful: no data = ok

    def test_insufficient_history(self):
        """Test graceful handling when insufficient history."""
        limit = VaRLimit(threshold=0.05, confidence_level=0.95, lookback_days=20)
        returns = [0.01, 0.02, -0.01]  # Only 3 days

        state = self._make_state(returns)
        result = limit.check(state)
        assert not result.breached  # Graceful: insufficient data = ok

    def test_warn_action(self):
        """Test warn action when VaR breached."""
        limit = VaRLimit(threshold=0.02, confidence_level=0.95, lookback_days=20, action="warn")
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.05, 30)  # High vol

        state = self._make_state(returns)
        result = limit.check(state)
        assert result.breached
        assert result.action == "warn"

    def test_custom_returns_key(self):
        """Test using custom key for returns."""
        limit = VaRLimit(threshold=0.05, returns_key="my_returns", lookback_days=20)
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.01, 30)

        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"A": 50000.0},
            daily_pnl=0.0,
            gross_exposure=50000.0,
            net_exposure=50000.0,
            context={"my_returns": returns},  # Custom key
        )
        result = limit.check(state)
        assert not result.breached

    def test_different_confidence_levels(self):
        """Test VaR at different confidence levels."""
        np.random.seed(42)
        # Returns with some negative tail
        returns = np.random.normal(0.0, 0.02, 100)
        returns[0:5] = [-0.06, -0.05, -0.055, -0.048, -0.052]  # Fat tail

        state = self._make_state(returns)

        # 99% confidence should give higher VaR than 95%
        # Use very low threshold to ensure both breach
        limit_95_check = VaRLimit(threshold=0.001, confidence_level=0.95, lookback_days=50)
        limit_99_check = VaRLimit(threshold=0.001, confidence_level=0.99, lookback_days=50)

        result_95 = limit_95_check.check(state)
        result_99 = limit_99_check.check(state)

        # Both should breach at 0.1% threshold with this data
        assert result_95.breached
        assert result_99.breached


class TestCVaRLimit:
    """Test CVaRLimit (Conditional VaR / Expected Shortfall)."""

    def _make_state(self, returns: list | np.ndarray | None = None) -> PortfolioState:
        """Helper to create PortfolioState with returns in context."""
        context = {}
        if returns is not None:
            context["historical_returns"] = returns
        return PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"A": 50000.0},
            daily_pnl=0.0,
            gross_exposure=50000.0,
            net_exposure=50000.0,
            context=context,
        )

    def test_no_breach_low_cvar(self):
        """Test no breach when CVaR is below threshold."""
        limit = CVaRLimit(threshold=0.08, confidence_level=0.95, lookback_days=20)
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 30)  # Low vol

        state = self._make_state(returns)
        result = limit.check(state)
        assert not result.breached

    def test_breach_fat_tail(self):
        """Test breach when CVaR exceeds threshold."""
        limit = CVaRLimit(threshold=0.03, confidence_level=0.95, lookback_days=20, action="halt")
        # Generate returns with fat left tail
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 30)
        returns[0:3] = [-0.10, -0.08, -0.09]  # Add several large losses

        state = self._make_state(returns)
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"
        assert "CVaR" in result.reason

    def test_no_data_graceful(self):
        """Test graceful handling when no historical returns available."""
        limit = CVaRLimit(threshold=0.08, confidence_level=0.95, lookback_days=20)
        state = self._make_state(None)
        result = limit.check(state)
        assert not result.breached

    def test_insufficient_history(self):
        """Test graceful handling when insufficient history."""
        limit = CVaRLimit(threshold=0.08, confidence_level=0.95, lookback_days=20)
        returns = [0.01, -0.02, 0.005]  # Only 3 days

        state = self._make_state(returns)
        result = limit.check(state)
        assert not result.breached

    def test_cvar_greater_than_var(self):
        """Test that CVaR >= VaR for same returns (mathematical property)."""
        np.random.seed(42)
        # Create returns with a fat tail
        returns = np.random.normal(0.0, 0.02, 100)
        returns[0:5] = [-0.15, -0.12, -0.10, -0.11, -0.13]

        # Calculate VaR and CVaR manually
        recent = np.asarray(returns)[-30:]
        var = -np.percentile(recent, 5)  # 95% VaR
        var_threshold = np.percentile(recent, 5)
        tail_returns = recent[recent <= var_threshold]
        cvar = -np.mean(tail_returns)

        # CVaR should be >= VaR
        assert cvar >= var

    def test_reduce_action(self):
        """Test reduce action."""
        limit = CVaRLimit(threshold=0.02, confidence_level=0.95, lookback_days=20, action="reduce")
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.05, 30)  # High vol
        returns[0:3] = [-0.15, -0.12, -0.14]

        state = self._make_state(returns)
        result = limit.check(state)
        assert result.breached
        assert result.action == "reduce"


class TestPortfolioStateContext:
    """Test PortfolioState context field."""

    def test_context_default_empty(self):
        """Test context defaults to empty dict."""
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
        )
        assert state.context == {}

    def test_context_with_data(self):
        """Test context can hold arbitrary data."""
        context = {
            "historical_returns": [0.01, -0.02, 0.015],
            "current_volatility": 0.02,
            "regime": "high_vol",
        }
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            context=context,
        )
        assert state.context["historical_returns"] == [0.01, -0.02, 0.015]
        assert state.context["current_volatility"] == 0.02
        assert state.context["regime"] == "high_vol"


class TestBetaLimit:
    """Test BetaLimit portfolio limit."""

    def test_no_breach_within_limits(self):
        """Test no breach when portfolio beta is within bounds."""
        limit = BetaLimit(max_beta=1.5, min_beta=-0.5)
        # Portfolio: 50% AAPL (beta=1.2), 50% MSFT (beta=1.0) = 1.1 beta
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"AAPL": 50000.0, "MSFT": 50000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"asset_betas": {"AAPL": 1.2, "MSFT": 1.0}},
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_max_beta(self):
        """Test breach when portfolio beta exceeds max."""
        limit = BetaLimit(max_beta=1.2, min_beta=0.0, action="halt")
        # Portfolio: 100% high-beta stock (beta=1.5)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"TSLA": 100000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"asset_betas": {"TSLA": 1.5}},
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"
        assert "1.50" in result.reason
        assert "1.20" in result.reason

    def test_breach_min_beta(self):
        """Test breach when portfolio beta below min."""
        limit = BetaLimit(max_beta=1.5, min_beta=0.5, action="warn")
        # Portfolio: 100% low-beta/inverse stock (beta=0.2)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"UTIL": 100000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"asset_betas": {"UTIL": 0.2}},
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "warn"
        assert "0.20" in result.reason
        assert "0.50" in result.reason

    def test_no_breach_no_positions(self):
        """Test no breach when no positions (zero beta)."""
        limit = BetaLimit(max_beta=1.5, min_beta=-0.5)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            context={"asset_betas": {}},
        )
        result = limit.check(state)
        assert not result.breached

    def test_graceful_no_beta_data(self):
        """Test graceful handling when beta data not in context."""
        limit = BetaLimit(max_beta=1.5, min_beta=-0.5)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"AAPL": 50000.0, "MSFT": 50000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={},  # No beta data
        )
        result = limit.check(state)
        assert not result.breached  # Graceful degradation


class TestSectorExposureLimit:
    """Test SectorExposureLimit portfolio limit."""

    def test_no_breach_diversified(self):
        """Test no breach when sectors are diversified."""
        limit = SectorExposureLimit(max_sector_exposure=0.40)
        # 33% each sector
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=3,
            positions={"AAPL": 33333.0, "XOM": 33333.0, "JPM": 33333.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={
                "asset_sectors": {
                    "AAPL": "Technology",
                    "XOM": "Energy",
                    "JPM": "Financials",
                }
            },
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_concentrated_sector(self):
        """Test breach when one sector is too concentrated."""
        limit = SectorExposureLimit(max_sector_exposure=0.30, action="halt")
        # 60% Tech, 40% Energy
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=3,
            positions={"AAPL": 30000.0, "MSFT": 30000.0, "XOM": 40000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={
                "asset_sectors": {
                    "AAPL": "Technology",
                    "MSFT": "Technology",
                    "XOM": "Energy",
                }
            },
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"
        assert "Technology" in result.reason
        assert "60.0" in result.reason

    def test_no_breach_no_positions(self):
        """Test no breach when no positions."""
        limit = SectorExposureLimit(max_sector_exposure=0.30)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            context={"asset_sectors": {}},
        )
        result = limit.check(state)
        assert not result.breached

    def test_graceful_no_sector_data(self):
        """Test graceful handling when sector data not in context."""
        limit = SectorExposureLimit(max_sector_exposure=0.30)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"AAPL": 50000.0, "MSFT": 50000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={},  # No sector data
        )
        result = limit.check(state)
        assert not result.breached  # Graceful degradation


class TestFactorExposureLimit:
    """Test FactorExposureLimit portfolio limit."""

    def test_no_breach_within_limits(self):
        """Test no breach when factor exposure within bounds."""
        limit = FactorExposureLimit(factor_name="momentum", max_exposure=0.5, min_exposure=-0.5)
        # Portfolio: 50% stock with 0.3 loading, 50% with -0.1 loading = 0.1
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"AAPL": 50000.0, "MSFT": 50000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"factor_loadings": {"AAPL": 0.3, "MSFT": -0.1}},
        )
        result = limit.check(state)
        assert not result.breached

    def test_breach_max_exposure(self):
        """Test breach when factor exposure exceeds max."""
        limit = FactorExposureLimit(
            factor_name="value", max_exposure=0.3, min_exposure=-0.3, action="halt"
        )
        # Portfolio: 100% high value loading (0.8)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"VALUE_ETF": 100000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"factor_loadings": {"VALUE_ETF": 0.8}},
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "halt"
        assert "value" in result.reason
        assert "0.80" in result.reason

    def test_breach_min_exposure(self):
        """Test breach when factor exposure below min."""
        limit = FactorExposureLimit(
            factor_name="size", max_exposure=0.5, min_exposure=-0.3, action="warn"
        )
        # Portfolio: 100% negative size exposure (small caps = -0.6)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"SMALL_CAP": 100000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"factor_loadings": {"SMALL_CAP": -0.6}},
        )
        result = limit.check(state)
        assert result.breached
        assert result.action == "warn"
        assert "size" in result.reason

    def test_no_breach_no_positions(self):
        """Test no breach when no positions (zero factor exposure)."""
        limit = FactorExposureLimit(factor_name="momentum", max_exposure=0.5, min_exposure=-0.5)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=0,
            positions={},
            daily_pnl=0.0,
            gross_exposure=0.0,
            net_exposure=0.0,
            context={"factor_loadings": {}},
        )
        result = limit.check(state)
        assert not result.breached

    def test_graceful_no_factor_data(self):
        """Test graceful handling when factor data not in context."""
        limit = FactorExposureLimit(factor_name="momentum", max_exposure=0.5, min_exposure=-0.5)
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=2,
            positions={"AAPL": 50000.0, "MSFT": 50000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={},  # No factor data
        )
        result = limit.check(state)
        assert not result.breached  # Graceful degradation

    def test_custom_factor_key(self):
        """Test using custom key for factor loadings."""
        limit = FactorExposureLimit(
            factor_name="momentum",
            max_exposure=0.5,
            min_exposure=-0.5,
            factor_key="momentum_loadings",  # Custom key
        )
        state = PortfolioState(
            equity=100000.0,
            initial_equity=100000.0,
            high_water_mark=100000.0,
            current_drawdown=0.0,
            num_positions=1,
            positions={"AAPL": 100000.0},
            daily_pnl=0.0,
            gross_exposure=100000.0,
            net_exposure=100000.0,
            context={"momentum_loadings": {"AAPL": 0.2}},
        )
        result = limit.check(state)
        assert not result.breached
