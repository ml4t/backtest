"""Risk management module for ml4t.backtest.

This module provides risk context and rule evaluation infrastructure for
position and portfolio risk management.

Main Components:
    - RiskContext: Immutable snapshot of position/portfolio state for risk evaluation
"""

from ml4t.backtest.risk.context import RiskContext

__all__ = ["RiskContext"]
