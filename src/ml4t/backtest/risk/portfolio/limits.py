"""Portfolio-level risk limits."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime


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
