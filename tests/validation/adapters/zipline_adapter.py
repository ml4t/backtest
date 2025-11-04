"""Zipline-reloaded adapter for cross-platform validation."""
import time
from datetime import datetime

import pandas as pd
import polars as pl

from zipline import run_algorithm
from zipline.api import order, set_commission, set_slippage, symbol
from zipline.finance.commission import PerShare
from zipline.finance.slippage import FixedSlippage

from .base import BacktestResult, PlatformAdapter, Trade


class ZiplineAdapter(PlatformAdapter):
    """Adapter for zipline-reloaded backtesting platform."""

    def __init__(self):
        super().__init__("zipline_reloaded")
        self.signals = []
        self.signal_index = 0

    def run_backtest(
        self,
        signals: list,
        data: pd.DataFrame,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0,
        **kwargs
    ) -> BacktestResult:
        """Run backtest using zipline-reloaded."""
        start_time = time.time()

        # Store signals for strategy access
        self.signals = sorted(signals, key=lambda s: s.timestamp)
        self.signal_index = 0

        # Determine date range
        start_date = data.index.min()
        end_date = data.index.max()

        def initialize(context):
            """Initialize strategy."""
            # Set commission
            set_commission(PerShare(cost=commission * 100))  # Convert to per-share

            # Set slippage
            if slippage > 0:
                set_slippage(FixedSlippage(spread=slippage))

            # Store signals in context
            context.signals = self.signals
            context.signal_index = 0
            context.executed_signals = set()

        def handle_data(context, data):
            """Process each bar."""
            current_time = data.current_dt

            # Check for signals at this timestamp
            while context.signal_index < len(context.signals):
                signal = context.signals[context.signal_index]

                if signal.timestamp > current_time:
                    break  # Future signal

                if signal.timestamp == current_time:
                    # Execute signal
                    if signal.signal_id not in context.executed_signals:
                        self._execute_signal(signal, data)
                        context.executed_signals.add(signal.signal_id)

                context.signal_index += 1

        # Run zipline algorithm
        try:
            results = run_algorithm(
                start=start_date,
                end=end_date,
                initialize=initialize,
                handle_data=handle_data,
                capital_base=initial_capital,
                data_frequency='daily',
                bundle='quandl',  # Might need custom bundle
            )
        except Exception as e:
            # Zipline might need custom bundle setup
            print(f"Zipline error: {e}")
            # Return empty results
            execution_time = time.time() - start_time
            return BacktestResult(
                platform=self.name,
                trades=[],
                equity_curve=pl.DataFrame({'timestamp': [], 'equity': []}),
                metrics={},
                execution_time=execution_time,
                config={'error': str(e)},
            )

        execution_time = time.time() - start_time

        # Convert results
        trades = self._convert_trades(results)
        equity_curve = self._convert_equity_curve(results)
        metrics = self._extract_metrics(results)

        return BacktestResult(
            platform=self.name,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            execution_time=execution_time,
            config={
                'initial_capital': initial_capital,
                'commission': commission,
                'slippage': slippage,
            }
        )

    def _execute_signal(self, signal, data):
        """Execute a signal in zipline."""
        try:
            sym = symbol(signal.symbol)

            if signal.action == 'BUY':
                order(sym, signal.quantity or 100)
            elif signal.action == 'SELL':
                order(sym, -(signal.quantity or 100))
            elif signal.action == 'CLOSE':
                # Close position
                current_position = data.current(sym, 'close')  # Get position
                # order to close...
                pass
        except Exception as e:
            print(f"Zipline order error: {e}")

    def _convert_trades(self, results: pd.DataFrame) -> list[Trade]:
        """Convert zipline results to trades."""
        # Zipline doesn't directly provide trades, need to extract from transactions
        trades = []
        # TODO: Parse zipline transactions into trades
        return trades

    def _convert_equity_curve(self, results: pd.DataFrame) -> pl.DataFrame:
        """Extract equity curve from zipline results."""
        if 'portfolio_value' in results.columns:
            df = pd.DataFrame({
                'timestamp': results.index,
                'equity': results['portfolio_value'].values
            })
            return pl.from_pandas(df)
        return pl.DataFrame({'timestamp': [], 'equity': []})

    def _extract_metrics(self, results: pd.DataFrame) -> dict:
        """Extract metrics from zipline results."""
        metrics = {}
        if 'returns' in results.columns:
            returns = results['returns']
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
        return metrics

    def supports_stop_loss(self) -> bool:
        """Zipline supports stop loss via order types."""
        return True

    def supports_take_profit(self) -> bool:
        """Zipline supports take profit via limit orders."""
        return True

    def supports_trailing_stop(self) -> bool:
        """Zipline supports trailing stop via custom logic."""
        return False  # Not natively
