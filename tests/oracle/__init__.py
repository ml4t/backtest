"""Independent reference oracle for differential testing.

This package contains a pure-Python backtesting engine that shares ZERO code
with ml4t.backtest. It is deliberately simpler (market orders only, no risk
rules) and computes all values independently.

Any difference between the oracle and the SUT indicates a bug in one or the other.
"""

from .engine import OracleFillRule, OracleResult, OracleTrade, run_oracle

__all__ = ["run_oracle", "OracleFillRule", "OracleResult", "OracleTrade"]
