"""
ml4t.backtest Framework Adapter for Cross-Framework Validation

Now uses the REAL ml4t.backtest library via BacktestWrapper for accurate validation.
"""

import time
import tracemalloc
from typing import Any
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseFrameworkAdapter, Signal, TradeRecord, ValidationResult, MomentumStrategy

# Add common module to path
common_path = Path(__file__).parent.parent / "common"
if str(common_path) not in sys.path:
    sys.path.insert(0, str(common_path))

from engine_wrappers import BacktestWrapper, BacktestConfig


class BacktestAdapter(BaseFrameworkAdapter):
    """Adapter for ml4t.backtest backtesting framework using real ml4t.backtest library."""

    def __init__(self):
        super().__init__("ml4t.backtest")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest using real ml4t.backtest library via BacktestWrapper."""

        result = ValidationResult(
            framework=self.framework_name,
            strategy=strategy_params.get("name", "Unknown"),
            initial_capital=initial_capital,
        )

        try:
            tracemalloc.start()
            start_time = time.time()

            # Extract strategy parameters
            short_window = strategy_params.get("short_window", 20)
            long_window = strategy_params.get("long_window", 50)

            # Generate signals using MomentumStrategy
            strategy = MomentumStrategy(short_window=short_window, long_window=long_window)
            signals_df = strategy.generate_signals(data)

            entries = signals_df['entries']
            exits = signals_df['exits']

            # Prepare OHLCV data for BacktestWrapper
            # BacktestWrapper expects DatetimeIndex named 'timestamp', normalize column names
            ohlcv = data.copy()

            # Ensure we have the required OHLCV columns (lowercase)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in ohlcv.columns:
                    # Try uppercase version
                    col_upper = col.upper()
                    if col_upper in ohlcv.columns:
                        ohlcv[col] = ohlcv[col_upper]

            # Select only OHLCV columns
            ohlcv = ohlcv[required_cols]

            # Rename index to 'timestamp' (BacktestWrapper expects this)
            ohlcv.index.name = 'timestamp'

            # Create BacktestConfig (no fees/slippage for baseline test)
            config = BacktestConfig(
                initial_cash=initial_capital,
                fees=0.0,
                slippage=0.0,
                order_type='market',
            )

            # Run backtest via BacktestWrapper
            wrapper = BacktestWrapper()
            backtest_result = wrapper.run_backtest(
                ohlcv=ohlcv,
                entries=entries,
                exits=exits,
                config=config,
            )

            # Convert BacktestResult to ValidationResult
            result.final_value = backtest_result.final_value
            result.total_return = (backtest_result.final_value / initial_capital - 1) * 100
            result.num_trades = backtest_result.num_trades

            # Convert trades DataFrame to list of TradeRecord
            trades = []
            if len(backtest_result.trades) > 0:
                for _, trade_row in backtest_result.trades.iterrows():
                    # Entry trade
                    entry_trade = TradeRecord(
                        timestamp=trade_row['entry_time'],
                        action="BUY",
                        quantity=trade_row['size'],
                        price=trade_row['entry_price'],
                        value=trade_row['size'] * trade_row['entry_price'],
                        commission=0.0,
                    )
                    trades.append(entry_trade)

                    # Exit trade
                    exit_trade = TradeRecord(
                        timestamp=trade_row['exit_time'],
                        action="SELL",
                        quantity=trade_row['size'],
                        price=trade_row['exit_price'],
                        value=trade_row['size'] * trade_row['exit_price'],
                        commission=0.0,
                    )
                    trades.append(exit_trade)

            result.trades = trades

            # Calculate equity curve from portfolio values
            # For simplicity, reconstruct from trades (real implementation would track bar-by-bar)
            # Start with initial capital and update on each trade
            equity = [initial_capital]
            cash = initial_capital
            shares = 0.0

            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda t: t.timestamp)

            for trade in sorted_trades:
                if trade.action == "BUY":
                    shares = trade.quantity
                    cash = 0.0
                elif trade.action == "SELL":
                    cash = trade.value
                    shares = 0.0

                # Portfolio value after trade
                # If holding shares, value them at last known price (approximation)
                if shares > 0:
                    equity.append(shares * trade.price)
                else:
                    equity.append(cash)

            result.equity_curve = pd.Series(equity)

            # Calculate performance metrics
            if len(equity) > 1:
                returns = result.equity_curve.pct_change().dropna()
                result.daily_returns = returns

                if len(returns) > 0 and returns.std() > 0:
                    result.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

                    # Calculate max drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    result.max_drawdown = drawdown.min() * 100

            # Calculate win rate from paired trades
            if len(backtest_result.trades) > 0 and 'pnl' in backtest_result.trades.columns:
                winning_trades = (backtest_result.trades['pnl'] > 0).sum()
                result.win_rate = winning_trades / len(backtest_result.trades)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ ml4t.backtest backtest completed (using real ml4t.backtest library)")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except Exception as e:
            error_msg = f"ml4t.backtest backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result

    def run_with_signals(
        self,
        data: pd.DataFrame,
        signals: list[Signal],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """
        Run backtest with pre-computed signals (pure execution validation).

        This eliminates variance from indicator calculations by using
        pre-computed entry/exit signals.

        Args:
            data: OHLCV data with DatetimeIndex
            signals: Pre-computed trading signals
            initial_capital: Starting capital

        Returns:
            ValidationResult with performance metrics
        """
        result = ValidationResult(
            framework=self.framework_name,
            strategy="SignalBased",
            initial_capital=initial_capital,
        )

        try:
            tracemalloc.start()
            start_time = time.time()

            # Convert signal list to boolean entry/exit series
            # Initialize with all False
            entries = pd.Series(False, index=data.index)
            exits = pd.Series(False, index=data.index)

            # Process signals and set True at signal timestamps
            for signal in signals:
                timestamp = signal["timestamp"]
                action = signal["action"]

                # Find closest timestamp in data index (handles minor timing differences)
                if timestamp in data.index:
                    idx = timestamp
                else:
                    # Find nearest timestamp
                    idx = data.index[data.index.get_indexer([timestamp], method='nearest')[0]]

                if action == "BUY":
                    entries.loc[idx] = True
                elif action == "SELL":
                    exits.loc[idx] = True

            # Prepare OHLCV data
            ohlcv = data.copy()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in ohlcv.columns:
                    col_upper = col.upper()
                    if col_upper in ohlcv.columns:
                        ohlcv[col] = ohlcv[col_upper]

            ohlcv = ohlcv[required_cols]
            ohlcv.index.name = 'timestamp'

            # Create BacktestConfig (no fees/slippage for signal validation)
            config = BacktestConfig(
                initial_cash=initial_capital,
                fees=0.0,
                slippage=0.0,
                order_type='market',
            )

            # Run backtest via BacktestWrapper
            wrapper = BacktestWrapper()
            backtest_result = wrapper.run_backtest(
                ohlcv=ohlcv,
                entries=entries,
                exits=exits,
                config=config,
            )

            # Convert to ValidationResult
            result.final_value = backtest_result.final_value
            result.total_return = (backtest_result.final_value / initial_capital - 1) * 100
            result.num_trades = backtest_result.num_trades

            # Convert trades to TradeRecord list
            trades = []
            if len(backtest_result.trades) > 0:
                for _, trade_row in backtest_result.trades.iterrows():
                    # Entry
                    trades.append(TradeRecord(
                        timestamp=trade_row['entry_time'],
                        action="BUY",
                        quantity=trade_row['size'],
                        price=trade_row['entry_price'],
                        value=trade_row['size'] * trade_row['entry_price'],
                        commission=0.0,
                    ))
                    # Exit
                    trades.append(TradeRecord(
                        timestamp=trade_row['exit_time'],
                        action="SELL",
                        quantity=trade_row['size'],
                        price=trade_row['exit_price'],
                        value=trade_row['size'] * trade_row['exit_price'],
                        commission=0.0,
                    ))

            result.trades = trades

            # Calculate metrics
            if len(backtest_result.trades) > 0 and 'pnl' in backtest_result.trades.columns:
                winning_trades = (backtest_result.trades['pnl'] > 0).sum()
                result.win_rate = winning_trades / len(backtest_result.trades)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ ml4t.backtest signal-based backtest completed")
            print(f"  Signals Processed: {len(signals)}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")

        except Exception as e:
            error_msg = f"ml4t.backtest signal-based backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
