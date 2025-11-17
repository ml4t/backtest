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

from .base import BaseFrameworkAdapter, FrameworkConfig, Signal, TradeRecord, ValidationResult, MomentumStrategy

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

            # Use daily returns and equity curve from backtest engine if available
            if hasattr(backtest_result, 'returns') and backtest_result.returns is not None and len(backtest_result.returns) > 0:
                result.daily_returns = backtest_result.returns
                result.equity_curve = (1 + result.daily_returns).cumprod() * initial_capital
            else:
                # Reconstruct DAILY portfolio values from trades and price data
                # This gives proper daily granularity for comparison
                daily_equity = {}

                # Build trade lookup by date
                trade_dict = {}
                for trade in trades:
                    trade_date = trade.timestamp.date()
                    if trade_date not in trade_dict:
                        trade_dict[trade_date] = []
                    trade_dict[trade_date].append(trade)

                # Track portfolio state day by day
                cash = initial_capital
                shares = 0.0

                for date, row in data.iterrows():
                    date_key = date.date() if hasattr(date, 'date') else date
                    close_price = row['close']

                    # Apply any trades that occurred on this date
                    if date_key in trade_dict:
                        for trade in sorted(trade_dict[date_key], key=lambda t: t.timestamp):
                            if trade.action == "BUY":
                                shares = trade.quantity
                                cash = 0.0
                            elif trade.action == "SELL":
                                cash = trade.value
                                shares = 0.0

                    # Value portfolio at end of day
                    if shares > 0:
                        portfolio_value = shares * close_price
                    else:
                        portfolio_value = cash

                    daily_equity[date] = portfolio_value

                # Convert to Series
                result.equity_curve = pd.Series(daily_equity)
                result.daily_returns = result.equity_curve.pct_change().fillna(0.0)

            # Calculate performance metrics from daily returns
            if result.daily_returns is not None and len(result.daily_returns) > 0 and result.daily_returns.std() > 0:
                result.sharpe_ratio = np.sqrt(252) * result.daily_returns.mean() / result.daily_returns.std()

                # Calculate max drawdown
                cumulative = (1 + result.daily_returns).cumprod()
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
        signals: pd.DataFrame,
        config: FrameworkConfig | None = None,
    ) -> ValidationResult:
        """
        Run backtest with pre-computed signals (pure execution validation).

        This eliminates variance from indicator calculations by using
        pre-computed entry/exit signals.

        Args:
            data: OHLCV data with DatetimeIndex
            signals: DataFrame with 'entry' and 'exit' boolean columns
            config: FrameworkConfig for execution parameters (costs, timing, etc.).
                   If None, uses FrameworkConfig.realistic() defaults.

        Returns:
            ValidationResult with performance metrics
        """
        # Use default config if not provided
        if config is None:
            config = FrameworkConfig.realistic()

        # Validate inputs
        self.validate_data(data)
        self.validate_signals(signals, data)

        result = ValidationResult(
            framework=self.framework_name,
            strategy="SignalBased",
            initial_capital=config.initial_capital,
        )

        try:
            tracemalloc.start()
            start_time = time.time()

            # Use signals directly (already boolean DataFrames)
            entries = signals["entry"]
            exits = signals["exit"]

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

            # Create BacktestConfig from FrameworkConfig
            # Note: BacktestWrapper expects BacktestConfig, not FrameworkConfig
            # NO LONGER shift signals here - broker handles execution delay internally
            # with execution_delay=True (fills at next bar's open)
            backtest_config = BacktestConfig(
                initial_cash=config.initial_capital,
                fees=config.commission_pct,
                slippage=config.slippage_pct,
                order_type='market',
                close_final_position=config.close_final_position,
                execution_delay_days=0,  # Always 0 - broker handles delay internally
            )

            # Run backtest via BacktestWrapper
            wrapper = BacktestWrapper()
            backtest_result = wrapper.run_backtest(
                ohlcv=ohlcv,
                entries=entries,
                exits=exits,
                config=backtest_config,
            )

            # Convert to ValidationResult
            result.final_value = backtest_result.final_value
            result.total_return = (backtest_result.final_value / config.initial_capital - 1) * 100
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
                        commission=trade_row.get('entry_fees', 0.0),
                    ))
                    # Exit
                    trades.append(TradeRecord(
                        timestamp=trade_row['exit_time'],
                        action="SELL",
                        quantity=trade_row['size'],
                        price=trade_row['exit_price'],
                        value=trade_row['size'] * trade_row['exit_price'],
                        commission=trade_row.get('exit_fees', 0.0),
                    ))

            # Add open position as a final trade (if exists and not already closed)
            # This ensures we count all individual orders, matching Backtrader/VectorBT behavior
            if hasattr(backtest_result, 'final_position') and backtest_result.final_position:
                final_qty = backtest_result.final_position.quantity if hasattr(backtest_result.final_position, 'quantity') else backtest_result.final_position

                if final_qty and abs(float(final_qty)) > 0.001:  # Non-zero position
                    # We have an open position - find the unclosed entry from signals
                    # Count completed trades to find which entry is still open
                    num_completed_trades = len(backtest_result.trades) if len(backtest_result.trades) > 0 else 0

                    # The (num_completed_trades + 1)th entry signal is the unclosed one
                    entry_count = 0
                    for idx in range(len(entries)):
                        if entries.iloc[idx]:
                            entry_count += 1
                            if entry_count == num_completed_trades + 1:  # This is the unclosed entry
                                # Add it as a trade
                                entry_date = entries.index[idx]
                                entry_price = ohlcv.loc[entry_date, 'close']
                                trades.append(TradeRecord(
                                    timestamp=entry_date,
                                    action="BUY",
                                    quantity=abs(float(final_qty)),
                                    price=float(entry_price),
                                    value=abs(float(final_qty)) * float(entry_price),
                                    commission=0.0,
                                ))
                                break

            result.trades = trades

            # Reconstruct DAILY portfolio values from trades and price data
            if hasattr(backtest_result, 'returns') and backtest_result.returns is not None and len(backtest_result.returns) > 0:
                result.daily_returns = backtest_result.returns
                result.equity_curve = (1 + result.daily_returns).cumprod() * config.initial_capital
            else:
                # Reconstruct daily equity curve from trades and price data
                daily_equity = {}
                trade_dict = {}

                # Build trade lookup by date
                for trade in trades:
                    trade_date = trade.timestamp.date()
                    if trade_date not in trade_dict:
                        trade_dict[trade_date] = []
                    trade_dict[trade_date].append(trade)

                # Track portfolio state day by day
                cash = config.initial_capital
                shares = 0.0

                for date, row in data.iterrows():
                    date_key = date.date() if hasattr(date, 'date') else date
                    close_price = row['close']

                    # Apply any trades that occurred on this date
                    if date_key in trade_dict:
                        for trade in sorted(trade_dict[date_key], key=lambda t: t.timestamp):
                            if trade.action == "BUY":
                                shares = trade.quantity
                                cash = 0.0
                            elif trade.action == "SELL":
                                cash = trade.value
                                shares = 0.0

                    # Value portfolio at end of day
                    if shares > 0:
                        portfolio_value = shares * close_price
                    else:
                        portfolio_value = cash

                    daily_equity[date] = portfolio_value

                # Convert to Series
                result.equity_curve = pd.Series(daily_equity)
                result.daily_returns = result.equity_curve.pct_change().fillna(0.0)

            # Calculate performance metrics from daily returns
            if result.daily_returns is not None and len(result.daily_returns) > 0 and result.daily_returns.std() > 0:
                result.sharpe_ratio = np.sqrt(252) * result.daily_returns.mean() / result.daily_returns.std()

                # Calculate max drawdown
                cumulative = (1 + result.daily_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                result.max_drawdown = drawdown.min() * 100

            # Calculate metrics
            if len(backtest_result.trades) > 0 and 'pnl' in backtest_result.trades.columns:
                winning_trades = (backtest_result.trades['pnl'] > 0).sum()
                result.win_rate = winning_trades / len(backtest_result.trades)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            num_entry_signals = signals["entry"].sum()
            num_exit_signals = signals["exit"].sum()
            print("✓ ml4t.backtest signal-based backtest completed")
            print(f"  Entry Signals: {num_entry_signals}, Exit Signals: {num_exit_signals}")
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
