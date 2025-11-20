"""ml4t.backtest adapter for cross-platform validation.

NOTE: This adapter is currently stubbed out. The ml4t.backtest framework
has been migrated from an event-driven architecture to a modular callback-based API.
This adapter needs to be rewritten to use the new API.

For working examples, see:
- tests/validation/common/engine_wrappers.py (BacktestWrapper)
- Integration tests that use the new modular API

TODO: Implement adapter using new ml4t.backtest API:
  - from ml4t.backtest import Engine, Strategy, Broker, DataFeed
  - Callback-based strategy (on_data, on_fill)
  - Polars DataFrame-based data feeds
"""
import time
from typing import Any

import polars as pl

from .base import BacktestResult, PlatformAdapter, Trade


class BacktestAdapter(PlatformAdapter):
    """Adapter for ml4t.backtest platform (currently stubbed)."""

    def __init__(self):
        super().__init__("ml4t.backtest")

    def run_backtest(
        self,
        signals: list,
        data: pl.DataFrame,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0,
        **kwargs
    ) -> BacktestResult:
        """Run backtest using ml4t.backtest.

        Currently raises NotImplementedError because the adapter needs to be
        rewritten for the new modular API.
        """
        raise NotImplementedError(
            "ml4t.backtest adapter not yet migrated to new modular API. "
            "The old event-driven architecture (core.event, data.feed, strategy.base) "
            "has been replaced with a callback-based API (Engine, Strategy, Broker). "
            "\n\n"
            "This requires a complete rewrite of the adapter. See:\n"
            "- tests/validation/common/engine_wrappers.py for working example\n"
            "- src/ml4t/backtest/ for new modular API structure\n"
            "\n"
            "To run validation tests, use other platforms:\n"
            "  --platforms vectorbt,backtrader,zipline"
        )

    def supports_stop_loss(self) -> bool:
        """Whether platform supports stop loss orders."""
        return True

    def supports_take_profit(self) -> bool:
        """Whether platform supports take profit orders."""
        return True

    def supports_trailing_stop(self) -> bool:
        """Whether platform supports trailing stop orders."""
        return True
