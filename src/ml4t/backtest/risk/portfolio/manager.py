"""RiskManager for portfolio-level risk management."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import TYPE_CHECKING, Any

from .limits import LimitResult, PortfolioLimit, PortfolioState

if TYPE_CHECKING:
    from ...broker import Broker


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
        manager.initialize(initial_equity=broker.get_account_value())

        # In strategy or engine:
        manager.update(
            equity=broker.get_account_value(),
            positions={asset: pos.market_value for asset, pos in broker.positions.items()},
            timestamp=timestamp,
            broker=broker,
        )
        if manager.can_open_position():
            broker.submit_order(...)
    """

    limits: list[PortfolioLimit] = field(default_factory=list)

    # Tracking state
    _initial_equity: float = 0.0
    _high_water_mark: float = 0.0
    _daily_start_equity: float = 0.0
    _last_equity: float = 0.0  # Track for current_drawdown property
    _last_date: date | None = None
    _halted: bool = False
    _halt_reason: str = ""
    _warnings: list[str] = field(default_factory=list)
    _liquidation_applied: bool = False
    _reduction_applied: set[int] = field(default_factory=set)

    def initialize(self, initial_equity: float, timestamp: datetime | None = None) -> None:
        """Initialize the risk manager with starting equity.

        Args:
            initial_equity: Starting portfolio value
            timestamp: Optional starting timestamp
        """
        self._initial_equity = initial_equity
        self._high_water_mark = initial_equity
        self._daily_start_equity = initial_equity
        self._last_equity = initial_equity  # Track for current_drawdown property
        self._last_date = timestamp.date() if timestamp else None
        self._halted = False
        self._halt_reason = ""
        self._warnings = []
        self._liquidation_applied = False
        self._reduction_applied.clear()

    def update(
        self,
        equity: float,
        positions: dict[str, float],
        timestamp: datetime | None = None,
        context: dict[str, Any] | None = None,
        broker: Broker | None = None,
    ) -> list[LimitResult]:
        """Check limits, then apply supported actions through the broker.

        A reduction requires a broker and runs once per continuous breach of
        each limit. A new breach after recovery may reduce exposure again.
        """
        high_water_mark = max(self._high_water_mark, equity)
        daily_start_equity = self._daily_start_equity
        last_date = self._last_date
        if timestamp is not None:
            current_date = timestamp.date()
            if last_date is not None and current_date != last_date:
                daily_start_equity = equity
            last_date = current_date

        state = self._build_state(
            equity,
            positions,
            timestamp,
            context or {},
            high_water_mark=high_water_mark,
            daily_start_equity=daily_start_equity,
        )
        results: list[LimitResult] = []
        reduction_results: dict[int, LimitResult] = {}
        liquidation_reasons: list[str] = []
        warning_reasons: list[str] = []
        halted = self._halted
        halt_reason = self._halt_reason

        for index, limit in enumerate(self.limits):
            result = limit.check(state)
            if not result.breached:
                continue
            results.append(result)
            if result.action in {"halt", "liquidate"}:
                halted = True
                halt_reason = result.reason
                if result.action == "liquidate":
                    liquidation_reasons.append(result.reason)
            elif result.action == "warn":
                warning_reasons.append(result.reason)
            elif result.action == "reduce":
                pct = result.reduction_pct
                if not isinstance(pct, (int, float)) or not isfinite(pct) or not 0 < pct <= 1:
                    raise ValueError("reduce action requires a finite reduction_pct in (0, 1]")
                reduction_results[index] = result
            elif result.action != "none":
                raise ValueError(f"unsupported portfolio-limit action: {result.action!r}")

        new_reductions = set(reduction_results) - self._reduction_applied
        has_exposure = bool(positions) or (broker is not None and bool(broker.positions))
        apply_reduction = bool(new_reductions and has_exposure and not liquidation_reasons)
        if apply_reduction and broker is None:
            raise ValueError("broker is required to apply action='reduce'")
        if apply_reduction and broker is not None and not broker.positions:
            raise ValueError("broker has no positions to reduce")

        liquidation_applied = self._liquidation_applied
        if liquidation_reasons and not liquidation_applied:
            reason = "; ".join(dict.fromkeys(liquidation_reasons))
            if broker is not None:
                broker.flatten_all_positions(reason=reason)
                liquidation_applied = True
            else:
                warnings.warn(
                    "RiskManager.update() produced action='liquidate' but no broker was "
                    "provided. Pass broker=... or explicitly call "
                    "broker.flatten_all_positions(...).",
                    UserWarning,
                    stacklevel=2,
                )
        elif apply_reduction:
            assert broker is not None
            reason = "; ".join(
                dict.fromkeys(reduction_results[index].reason for index in sorted(new_reductions))
            )
            fraction = max(reduction_results[index].reduction_pct for index in new_reductions)
            broker.reduce_all_positions(fraction=fraction, reason=reason)

        self._last_equity = equity
        self._high_water_mark = high_water_mark
        self._daily_start_equity = daily_start_equity
        self._last_date = last_date
        self._warnings = warning_reasons
        self._halted = halted
        self._halt_reason = halt_reason
        self._liquidation_applied = liquidation_applied
        self._reduction_applied.intersection_update(reduction_results)
        if apply_reduction:
            self._reduction_applied.update(new_reductions)
        return results

    def _build_state(
        self,
        equity: float,
        positions: dict[str, float],
        timestamp: date | datetime | None,
        context: dict[str, Any] | None = None,
        *,
        high_water_mark: float | None = None,
        daily_start_equity: float | None = None,
    ) -> PortfolioState:
        """Build PortfolioState from current data."""
        high_water_mark = self._high_water_mark if high_water_mark is None else high_water_mark
        daily_start_equity = (
            self._daily_start_equity if daily_start_equity is None else daily_start_equity
        )
        # Calculate drawdown
        drawdown = (high_water_mark - equity) / high_water_mark if high_water_mark > 0 else 0.0

        # Calculate daily P&L
        daily_pnl = equity - daily_start_equity

        # Calculate exposures
        gross_exposure = sum(abs(v) for v in positions.values())
        net_exposure = sum(positions.values())

        return PortfolioState(
            equity=equity,
            initial_equity=self._initial_equity,
            high_water_mark=high_water_mark,
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
            return max(0.0, (self._high_water_mark - self._last_equity) / self._high_water_mark)
        return 0.0

    def reset_halt(self) -> None:
        """Manually reset halt state (use with caution)."""
        self._halted = False
        self._halt_reason = ""
        self._liquidation_applied = False

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
