"""Portfolio-level risk limits."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np


class PortfolioLimit(ABC):
    """Base class for portfolio-level risk limits."""

    @abstractmethod
    def check(self, state: "PortfolioState") -> "LimitResult":
        """Check if limit is breached.

        Args:
            state: Current portfolio state

        Returns:
            LimitResult indicating if breached and any actions to take
        """
        pass


@dataclass
class LimitResult:
    """Result of a portfolio limit check.

    Attributes:
        breached: True if limit was breached
        action: Action to take ("none", "warn", "reduce", "halt")
        reason: Human-readable explanation
        reduction_pct: If action=="reduce", percentage to reduce by
    """

    breached: bool
    action: str = "none"  # "none", "warn", "reduce", "halt"
    reason: str = ""
    reduction_pct: float = 0.0

    @classmethod
    def ok(cls) -> "LimitResult":
        return cls(breached=False)

    @classmethod
    def warn(cls, reason: str) -> "LimitResult":
        return cls(breached=True, action="warn", reason=reason)

    @classmethod
    def reduce(cls, reason: str, pct: float) -> "LimitResult":
        return cls(breached=True, action="reduce", reason=reason, reduction_pct=pct)

    @classmethod
    def halt(cls, reason: str) -> "LimitResult":
        return cls(breached=True, action="halt", reason=reason)


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk checks.

    Attributes:
        equity: Current portfolio equity value
        initial_equity: Starting equity value
        high_water_mark: Highest equity reached
        current_drawdown: Current drawdown from high water mark (0 to 1)
        num_positions: Number of open positions
        positions: Dict of asset -> position value
        daily_pnl: P&L since start of trading day
        gross_exposure: Sum of absolute position values
        net_exposure: Sum of signed position values
        timestamp: Current time
        context: Optional strategy-provided context (e.g., historical returns for VaR)
    """

    equity: float
    initial_equity: float
    high_water_mark: float
    current_drawdown: float  # 0.0 to 1.0
    num_positions: int
    positions: dict[str, float]  # asset -> market value
    daily_pnl: float
    gross_exposure: float
    net_exposure: float
    timestamp: date | datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaxDrawdownLimit(PortfolioLimit):
    """Halt trading when drawdown exceeds threshold.

    Args:
        max_drawdown: Maximum allowed drawdown (0.0-1.0)
                     Default 0.20 = 20% max drawdown
        action: Action when breached ("warn", "reduce", "halt")
                Default "halt" - stops all new trades
        warn_threshold: Optional earlier threshold for warnings

    Example:
        limit = MaxDrawdownLimit(max_drawdown=0.20, warn_threshold=0.15)
        # Warns at 15% drawdown, halts at 20%
    """

    max_drawdown: float = 0.20
    action: str = "halt"
    warn_threshold: float | None = None

    def check(self, state: PortfolioState) -> LimitResult:
        if state.current_drawdown >= self.max_drawdown:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"drawdown {state.current_drawdown:.1%} >= {self.max_drawdown:.1%}",
            )

        if self.warn_threshold and state.current_drawdown >= self.warn_threshold:
            return LimitResult.warn(
                f"drawdown {state.current_drawdown:.1%} >= warn threshold {self.warn_threshold:.1%}"
            )

        return LimitResult.ok()


@dataclass
class MaxPositionsLimit(PortfolioLimit):
    """Limit maximum number of open positions.

    Args:
        max_positions: Maximum number of simultaneous positions
        action: Action when breached ("warn", "halt")

    Example:
        limit = MaxPositionsLimit(max_positions=10)
        # Prevents opening more than 10 positions
    """

    max_positions: int = 10
    action: str = "halt"

    def check(self, state: PortfolioState) -> LimitResult:
        if state.num_positions >= self.max_positions:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"positions {state.num_positions} >= max {self.max_positions}",
            )
        return LimitResult.ok()


@dataclass
class MaxExposureLimit(PortfolioLimit):
    """Limit maximum exposure to a single asset.

    Args:
        max_exposure_pct: Maximum position size as % of equity (0.0-1.0)
                         Default 0.10 = 10% max per asset
        action: Action when breached

    Example:
        limit = MaxExposureLimit(max_exposure_pct=0.10)
        # No single position can be > 10% of portfolio
    """

    max_exposure_pct: float = 0.10
    action: str = "warn"

    def check(self, state: PortfolioState) -> LimitResult:
        for asset, value in state.positions.items():
            exposure_pct = abs(value) / state.equity if state.equity > 0 else 0
            if exposure_pct > self.max_exposure_pct:
                return LimitResult(
                    breached=True,
                    action=self.action,
                    reason=f"{asset} exposure {exposure_pct:.1%} > max {self.max_exposure_pct:.1%}",
                )
        return LimitResult.ok()


@dataclass
class DailyLossLimit(PortfolioLimit):
    """Halt trading when daily loss exceeds threshold.

    Args:
        max_daily_loss_pct: Maximum daily loss as % of equity (0.0-1.0)
                           Default 0.02 = 2% max daily loss
        action: Action when breached

    Example:
        limit = DailyLossLimit(max_daily_loss_pct=0.02)
        # Halt if down more than 2% today
    """

    max_daily_loss_pct: float = 0.02
    action: str = "halt"

    def check(self, state: PortfolioState) -> LimitResult:
        if state.equity > 0:
            daily_loss_pct = -state.daily_pnl / state.equity if state.daily_pnl < 0 else 0
            if daily_loss_pct > self.max_daily_loss_pct:
                return LimitResult(
                    breached=True,
                    action=self.action,
                    reason=f"daily loss {daily_loss_pct:.1%} > max {self.max_daily_loss_pct:.1%}",
                )
        return LimitResult.ok()


@dataclass
class GrossExposureLimit(PortfolioLimit):
    """Limit total gross exposure (sum of absolute positions).

    Args:
        max_gross_exposure: Maximum gross exposure as multiple of equity
                           Default 1.0 = 100% gross exposure (no leverage)
        action: Action when breached

    Example:
        limit = GrossExposureLimit(max_gross_exposure=2.0)
        # Allow up to 2x leverage
    """

    max_gross_exposure: float = 1.0
    action: str = "halt"

    def check(self, state: PortfolioState) -> LimitResult:
        if state.equity > 0:
            gross_ratio = state.gross_exposure / state.equity
            if gross_ratio > self.max_gross_exposure:
                return LimitResult(
                    breached=True,
                    action=self.action,
                    reason=f"gross exposure {gross_ratio:.1%} > max {self.max_gross_exposure:.1%}",
                )
        return LimitResult.ok()


@dataclass
class NetExposureLimit(PortfolioLimit):
    """Limit net exposure (for market-neutral strategies).

    Args:
        max_net_exposure: Maximum net exposure as % of equity (-1.0 to 1.0)
        min_net_exposure: Minimum net exposure (for enforcing hedging)
        action: Action when breached

    Example:
        limit = NetExposureLimit(max_net_exposure=0.10, min_net_exposure=-0.10)
        # Stay within +/- 10% net exposure (near market-neutral)
    """

    max_net_exposure: float = 1.0
    min_net_exposure: float = -1.0
    action: str = "warn"

    def check(self, state: PortfolioState) -> LimitResult:
        if state.equity > 0:
            net_ratio = state.net_exposure / state.equity
            if net_ratio > self.max_net_exposure:
                return LimitResult(
                    breached=True,
                    action=self.action,
                    reason=f"net exposure {net_ratio:.1%} > max {self.max_net_exposure:.1%}",
                )
            if net_ratio < self.min_net_exposure:
                return LimitResult(
                    breached=True,
                    action=self.action,
                    reason=f"net exposure {net_ratio:.1%} < min {self.min_net_exposure:.1%}",
                )
        return LimitResult.ok()


@dataclass
class VaRLimit(PortfolioLimit):
    """Limit portfolio exposure based on Value at Risk.

    VaR estimates the maximum loss at a given confidence level over a period.
    This limit triggers when the portfolio's VaR exceeds a threshold.

    Uses historical returns from state.context["historical_returns"] to compute
    VaR using the historical simulation method (percentile-based).

    Args:
        threshold: Maximum acceptable VaR as decimal (0.05 = 5% max loss)
        confidence_level: Confidence level for VaR (0.95 = 95% confidence)
        lookback_days: Minimum days of history required (default: 20)
        action: Action when breached ("warn", "reduce", "halt")
        returns_key: Key in context for historical returns array

    Example:
        limit = VaRLimit(threshold=0.05, confidence_level=0.95, lookback_days=20)
        # Strategy must provide historical returns in context:
        state.context["historical_returns"] = daily_returns_array

    Note:
        If historical_returns not found or insufficient data, limit does not
        trigger (returns LimitResult.ok()). This allows graceful degradation
        during backtest warmup period.
    """

    threshold: float = 0.05  # 5% default VaR threshold
    confidence_level: float = 0.95  # 95% confidence
    lookback_days: int = 20
    action: str = "warn"
    returns_key: str = "historical_returns"

    def check(self, state: PortfolioState) -> LimitResult:
        """Check if portfolio VaR exceeds threshold."""
        # Get historical returns from context
        returns = state.context.get(self.returns_key)

        if returns is None:
            return LimitResult.ok()  # No data, cannot evaluate

        returns_arr = np.asarray(returns)

        if len(returns_arr) < self.lookback_days:
            return LimitResult.ok()  # Insufficient history

        # Use most recent lookback_days for VaR calculation
        recent_returns = returns_arr[-self.lookback_days :]

        # Historical VaR: percentile of returns at (1 - confidence) level
        # For 95% confidence, we want the 5th percentile (worst 5% of returns)
        var = -np.percentile(recent_returns, (1 - self.confidence_level) * 100)

        if var > self.threshold:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"VaR {var:.2%} > threshold {self.threshold:.2%} "
                f"at {self.confidence_level:.0%} confidence",
            )

        return LimitResult.ok()


@dataclass
class CVaRLimit(PortfolioLimit):
    """Limit portfolio exposure based on Conditional Value at Risk (CVaR).

    CVaR (Expected Shortfall) is the expected loss given that VaR is exceeded.
    It captures tail risk better than VaR by considering the average of all
    losses beyond the VaR threshold.

    Uses historical returns from state.context["historical_returns"] to compute
    CVaR using the historical simulation method.

    Args:
        threshold: Maximum acceptable CVaR as decimal (0.08 = 8% expected shortfall)
        confidence_level: Confidence level for CVaR (0.95 = 95% confidence)
        lookback_days: Minimum days of history required (default: 20)
        action: Action when breached ("warn", "reduce", "halt")
        returns_key: Key in context for historical returns array

    Example:
        limit = CVaRLimit(threshold=0.08, confidence_level=0.95, action="halt")
        # CVaR is always >= VaR at the same confidence level

    Note:
        CVaR is generally preferred over VaR for tail risk management
        as it considers the severity of losses beyond the VaR threshold.
    """

    threshold: float = 0.08  # 8% default CVaR threshold
    confidence_level: float = 0.95
    lookback_days: int = 20
    action: str = "warn"
    returns_key: str = "historical_returns"

    def check(self, state: PortfolioState) -> LimitResult:
        """Check if portfolio CVaR exceeds threshold."""
        # Get historical returns from context
        returns = state.context.get(self.returns_key)

        if returns is None:
            return LimitResult.ok()  # No data, cannot evaluate

        returns_arr = np.asarray(returns)

        if len(returns_arr) < self.lookback_days:
            return LimitResult.ok()  # Insufficient history

        # Use most recent lookback_days
        recent_returns = returns_arr[-self.lookback_days :]

        # CVaR: mean of returns below VaR threshold
        var_threshold = np.percentile(recent_returns, (1 - self.confidence_level) * 100)
        tail_returns = recent_returns[recent_returns <= var_threshold]

        if len(tail_returns) == 0:
            return LimitResult.ok()  # Edge case: no tail returns

        cvar = -np.mean(tail_returns)

        if cvar > self.threshold:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"CVaR {cvar:.2%} > threshold {self.threshold:.2%} "
                f"at {self.confidence_level:.0%} confidence",
            )

        return LimitResult.ok()


@dataclass
class BetaLimit(PortfolioLimit):
    """Limit portfolio beta exposure to market.

    Controls directional market exposure by limiting the portfolio's
    beta to a benchmark (typically the market portfolio).

    Requires beta values in state.context["asset_betas"] as a dict mapping
    asset -> beta value.

    Args:
        max_beta: Maximum allowed portfolio beta (default 1.5)
        min_beta: Minimum allowed portfolio beta (default -0.5)
        action: Action when breached ("warn", "reduce", "halt")
        betas_key: Key in context for asset beta dict

    Example:
        limit = BetaLimit(max_beta=1.2, min_beta=0.5)
        # Strategy must provide betas in context:
        state.context["asset_betas"] = {"AAPL": 1.2, "JNJ": 0.6, "SPY": 1.0}

    Note:
        Portfolio beta = sum(weight_i * beta_i) where weights are
        position values / equity.
    """

    max_beta: float = 1.5
    min_beta: float = -0.5
    action: str = "warn"
    betas_key: str = "asset_betas"

    def check(self, state: PortfolioState) -> LimitResult:
        """Check if portfolio beta is within limits."""
        asset_betas = state.context.get(self.betas_key)

        if asset_betas is None or not state.positions:
            return LimitResult.ok()  # No data or no positions

        if state.equity <= 0:
            return LimitResult.ok()

        # Calculate portfolio beta
        portfolio_beta = 0.0
        for asset, value in state.positions.items():
            beta = asset_betas.get(asset, 1.0)  # Default to market beta
            weight = value / state.equity
            portfolio_beta += weight * beta

        if portfolio_beta > self.max_beta:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"portfolio beta {portfolio_beta:.2f} > max {self.max_beta:.2f}",
            )

        if portfolio_beta < self.min_beta:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"portfolio beta {portfolio_beta:.2f} < min {self.min_beta:.2f}",
            )

        return LimitResult.ok()


@dataclass
class SectorExposureLimit(PortfolioLimit):
    """Limit exposure to any single sector.

    Ensures diversification by limiting concentration in any one sector.

    Requires sector mappings in state.context["asset_sectors"] as a dict
    mapping asset -> sector name.

    Args:
        max_sector_exposure: Maximum exposure to any sector as % of equity (0.0-1.0)
                            Default 0.30 = 30% max per sector
        action: Action when breached
        sectors_key: Key in context for asset sector dict

    Example:
        limit = SectorExposureLimit(max_sector_exposure=0.25)
        # Strategy must provide sectors in context:
        state.context["asset_sectors"] = {
            "AAPL": "Technology", "MSFT": "Technology",
            "JNJ": "Healthcare", "XOM": "Energy"
        }
    """

    max_sector_exposure: float = 0.30
    action: str = "warn"
    sectors_key: str = "asset_sectors"

    def check(self, state: PortfolioState) -> LimitResult:
        """Check if any sector exceeds exposure limit."""
        asset_sectors = state.context.get(self.sectors_key)

        if asset_sectors is None or not state.positions:
            return LimitResult.ok()

        if state.equity <= 0:
            return LimitResult.ok()

        # Calculate sector exposures
        sector_exposure: dict[str, float] = {}
        for asset, value in state.positions.items():
            sector = asset_sectors.get(asset, "Unknown")
            weight = abs(value) / state.equity
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + weight

        # Check for breaches
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                return LimitResult(
                    breached=True,
                    action=self.action,
                    reason=f"{sector} exposure {exposure:.1%} > max {self.max_sector_exposure:.1%}",
                )

        return LimitResult.ok()


@dataclass
class FactorExposureLimit(PortfolioLimit):
    """Limit portfolio exposure to a specific risk factor.

    Generic factor exposure limit that can be used for any factor
    (momentum, value, size, volatility, etc.).

    Requires factor loadings in state.context[factor_key] as a dict
    mapping asset -> factor loading.

    Args:
        factor_name: Name of the factor (for error messages)
        max_exposure: Maximum allowed factor exposure
        min_exposure: Minimum allowed factor exposure
        action: Action when breached
        factor_key: Key in context for factor loadings dict

    Example:
        # Momentum factor limit
        limit = FactorExposureLimit(
            factor_name="momentum",
            max_exposure=0.5,
            min_exposure=-0.5,
            factor_key="momentum_loadings"
        )
        state.context["momentum_loadings"] = {"AAPL": 0.8, "JNJ": -0.2}

    Note:
        Factor exposure = sum(weight_i * loading_i) where weights are
        position values / equity (signed, so shorts contribute negatively).
    """

    factor_name: str = "factor"
    max_exposure: float = 1.0
    min_exposure: float = -1.0
    action: str = "warn"
    factor_key: str = "factor_loadings"

    def check(self, state: PortfolioState) -> LimitResult:
        """Check if factor exposure is within limits."""
        factor_loadings = state.context.get(self.factor_key)

        if factor_loadings is None or not state.positions:
            return LimitResult.ok()

        if state.equity <= 0:
            return LimitResult.ok()

        # Calculate factor exposure
        factor_exposure = 0.0
        for asset, value in state.positions.items():
            loading = factor_loadings.get(asset, 0.0)  # Default to neutral
            weight = value / state.equity  # Signed weight
            factor_exposure += weight * loading

        if factor_exposure > self.max_exposure:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"{self.factor_name} exposure {factor_exposure:.2f} > max {self.max_exposure:.2f}",
            )

        if factor_exposure < self.min_exposure:
            return LimitResult(
                breached=True,
                action=self.action,
                reason=f"{self.factor_name} exposure {factor_exposure:.2f} < min {self.min_exposure:.2f}",
            )

        return LimitResult.ok()
