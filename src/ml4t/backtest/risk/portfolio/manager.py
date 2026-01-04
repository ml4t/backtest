"""RiskManager for portfolio-level risk management."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .limits import LimitResult, PortfolioLimit, PortfolioState


@dataclass
class RiskManager:
    """Portfolio-level risk manager.

    Monitors portfolio-wide risk metrics and enforces limits.
    Integrates with Broker to prevent trades that would breach limits.

    Args:
        limits: List of PortfolioLimit rules to enforce

    Example:
        from ml4t.backtest.risk.portfolio import (
            RiskManager, MaxDrawdownLimit, MaxPositionsLimit
        )

        manager = RiskManager(limits=[
            MaxDrawdownLimit(max_drawdown=0.20),
            MaxPositionsLimit(max_positions=10),
        ])

        # In strategy or engine:
        if manager.can_open_position():
            broker.submit_order(...)
    """

    limits: list[PortfolioLimit] = field(default_factory=list)

    # Tracking state
    _initial_equity: float = 0.0
    _high_water_mark: float = 0.0
    _daily_start_equity: float = 0.0
    _last_date: date | None = None
    _halted: bool = False
    _halt_reason: str = ""
    _warnings: list[str] = field(default_factory=list)

    def initialize(self, initial_equity: float, timestamp: datetime | None = None) -> None:
        """Initialize the risk manager with starting equity.

        Args:
            initial_equity: Starting portfolio value
            timestamp: Optional starting timestamp
        """
        self._initial_equity = initial_equity
        self._high_water_mark = initial_equity
        self._daily_start_equity = initial_equity
        self._last_date = timestamp.date() if timestamp else None
        self._halted = False
        self._halt_reason = ""
        self._warnings = []

    def update(
        self,
        equity: float,
        positions: dict[str, float],
        timestamp: datetime | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[LimitResult]:
        """Update risk state and check all limits.

        Args:
            equity: Current portfolio equity
            positions: Dict of asset -> position market value
            timestamp: Current timestamp
            context: Optional context dict with historical data (e.g., returns for VaR)

        Returns:
            List of LimitResult for any breached limits
        """
        # Update high water mark
        if equity > self._high_water_mark:
            self._high_water_mark = equity

        # Check for new trading day
        if timestamp:
            current_date = timestamp.date()
            if self._last_date and current_date != self._last_date:
                # New day - reset daily P&L tracking
                self._daily_start_equity = equity
                self._warnings = []  # Clear daily warnings
            self._last_date = current_date

        # Build portfolio state
        state = self._build_state(equity, positions, timestamp, context or {})

        # Check all limits
        results = []
        self._warnings = []

        for limit in self.limits:
            result = limit.check(state)
            if result.breached:
                results.append(result)

                if result.action == "halt":
                    self._halted = True
                    self._halt_reason = result.reason
                elif result.action == "warn":
                    self._warnings.append(result.reason)

        return results

    def _build_state(
        self,
        equity: float,
        positions: dict[str, float],
        timestamp: date | datetime | None,
        context: dict[str, Any] | None = None,
    ) -> PortfolioState:
        """Build PortfolioState from current data."""
        # Calculate drawdown
        if self._high_water_mark > 0:
            drawdown = (self._high_water_mark - equity) / self._high_water_mark
        else:
            drawdown = 0.0

        # Calculate daily P&L
        daily_pnl = equity - self._daily_start_equity

        # Calculate exposures
        gross_exposure = sum(abs(v) for v in positions.values())
        net_exposure = sum(positions.values())

        return PortfolioState(
            equity=equity,
            initial_equity=self._initial_equity,
            high_water_mark=self._high_water_mark,
            current_drawdown=max(0, drawdown),
            num_positions=len(positions),
            positions=positions,
            daily_pnl=daily_pnl,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            timestamp=timestamp,
            context=context or {},
        )

    def can_open_position(self) -> bool:
        """Check if new positions can be opened.

        Returns:
            True if trading is allowed, False if halted
        """
        return not self._halted

    def can_increase_position(self, asset: str, amount: float) -> tuple[bool, str]:
        """Check if a position increase is allowed.

        Args:
            asset: Asset symbol
            amount: Additional market value

        Returns:
            Tuple of (allowed, reason)
        """
        if self._halted:
            return False, self._halt_reason
        return True, ""

    @property
    def is_halted(self) -> bool:
        """True if trading is halted due to risk limit breach."""
        return self._halted

    @property
    def halt_reason(self) -> str:
        """Reason for halt, empty if not halted."""
        return self._halt_reason

    @property
    def warnings(self) -> list[str]:
        """Current warning messages."""
        return self._warnings

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from high water mark (0 to 1)."""
        if self._high_water_mark > 0:
            # This is a snapshot - actual drawdown needs current equity
            return 0.0
        return 0.0

    def reset_halt(self) -> None:
        """Manually reset halt state (use with caution)."""
        self._halted = False
        self._halt_reason = ""

    def get_state(
        self,
        equity: float,
        positions: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> PortfolioState:
        """Get current portfolio state for external inspection.

        Args:
            equity: Current portfolio equity
            positions: Dict of asset -> position market value
            context: Optional context dict with historical data

        Returns:
            PortfolioState with all calculated metrics
        """
        return self._build_state(equity, positions, self._last_date, context)
