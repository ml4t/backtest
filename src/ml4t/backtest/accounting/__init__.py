"""Accounting module for backtesting engine.

Provides proper accounting constraints for different account types:
- Cash accounts: No leverage, no shorts (allow_short_selling=False, allow_leverage=False)
- Crypto accounts: Shorts allowed, no leverage (allow_short_selling=True, allow_leverage=False)
- Margin accounts: Leverage and shorts enabled (allow_short_selling=True, allow_leverage=True)

Key Components:
- Position: Unified position tracking (from types module)
- AccountPolicy: Interface for account type constraints
- UnifiedAccountPolicy: Parameter-driven policy implementation
- AccountState: Account state management and position tracking
- Gatekeeper: Order validation before execution

Usage:
    from ml4t.backtest.accounting import UnifiedAccountPolicy

    # Cash account (no shorts, no leverage)
    policy = UnifiedAccountPolicy()

    # Crypto account (shorts allowed, no leverage)
    policy = UnifiedAccountPolicy(allow_short_selling=True)

    # Margin account (shorts and leverage)
    policy = UnifiedAccountPolicy(allow_short_selling=True, allow_leverage=True)

    # Or create from config
    from ml4t.backtest import BacktestConfig
    config = BacktestConfig.from_preset("backtrader")
    policy = UnifiedAccountPolicy.from_config(config)
"""

from ..types import Position
from .account import AccountState
from .gatekeeper import Gatekeeper
from .policy import (
    AccountPolicy,
    UnifiedAccountPolicy,
)

__all__ = [
    "Position",
    "AccountPolicy",
    "UnifiedAccountPolicy",
    "AccountState",
    "Gatekeeper",
]
