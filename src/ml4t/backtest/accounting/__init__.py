"""Accounting module for backtesting engine.

Provides proper accounting constraints for both cash accounts (no leverage, no shorts)
and margin accounts (leverage enabled, shorts allowed).

Key Components:
- Position: Unified position tracking with cost basis
- AccountPolicy: Interface for account type constraints
- CashAccountPolicy: Cash account constraints (cash >= 0, no shorts)
- MarginAccountPolicy: Margin account constraints (NLV/BP/MM calculations)
- AccountState: Account state management and position tracking
- Gatekeeper: Order validation before execution
"""

from .account import AccountState
from .gatekeeper import Gatekeeper
from .models import Position
from .policy import AccountPolicy, CashAccountPolicy, MarginAccountPolicy

__all__ = [
    "Position",
    "AccountPolicy",
    "CashAccountPolicy",
    "MarginAccountPolicy",
    "AccountState",
    "Gatekeeper",
]
