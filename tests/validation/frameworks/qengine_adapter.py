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
        data: pd.DataFrame | dict[str, pd.DataFrame],
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

        # Check if multi-asset (dict) or single-asset (DataFrame)
        if isinstance(data, dict):
            return self._run_multi_asset_with_signals(data, signals, config)

        # Single-asset path
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
    def _run_multi_asset_with_signals(
        self,
        data: dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        config: FrameworkConfig,
    ) -> ValidationResult:
        """Run backtest with multi-asset rotation signals.

        Strategy logic MATCHES Backtrader:
        - BUY signal: Only buy if NO existing position (no accumulation)
        - SELL signal: Only sell if HAVE a position to close
        """
        from datetime import datetime
        from ml4t.backtest import (
            Engine,
            DataFeed,
            Strategy,
            PercentageCommission,
            PercentageSlippage,
            NoCommission,
            NoSlippage,
            OrderType,
            OrderSide,
            ExecutionMode,
        )
        import polars as pl

        result = ValidationResult(
            framework=self.framework_name,
            strategy="TopNMomentum",
            initial_capital=config.initial_capital,
        )

        try:
            import tracemalloc
            import time

            tracemalloc.start()
            start_time = time.time()

            # Convert multi-asset data dict to Polars DataFrame for DataFeed
            # Format: timestamp, asset, open, high, low, close, volume
            price_records = []
            for symbol, ohlcv in data.items():
                for idx in range(len(ohlcv)):
                    row = ohlcv.iloc[idx]
                    ts = row.name
                    # Convert to Python datetime if needed
                    if hasattr(ts, 'to_pydatetime'):
                        ts = ts.to_pydatetime()
                    price_records.append({
                        'timestamp': ts,
                        'asset': symbol,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                    })

            prices_df = pl.DataFrame(price_records)

            # Convert signals DataFrame to Polars signals DataFrame
            # Input format: timestamp, symbol, signal (1=BUY, -1=SELL)
            # Output format: timestamp, asset, entry (bool), exit (bool)
            signal_records = []
            for _, row in signals.iterrows():
                ts = row['timestamp']
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                signal_records.append({
                    'timestamp': ts,
                    'asset': row['symbol'],
                    'entry': row['signal'] == 1,
                    'exit': row['signal'] == -1,
                })

            signals_df = pl.DataFrame(signal_records)

            print(f"\nml4t.backtest multi-asset setup:")
            print(f"  Prices shape: ({len(prices_df)}, {len(prices_df.columns)})")
            print(f"  Entry signals: {signals_df['entry'].sum()}")
            print(f"  Exit signals: {signals_df['exit'].sum()}")

            # Strategy class matching Backtrader logic
            class RotationStrategy(Strategy):
                def __init__(self, target_pct=0.20):
                    super().__init__()
                    self.target_pct = target_pct

                def on_data(self, timestamp, data, context, broker):
                    """Process signals for all assets at this timestamp."""
                    for symbol, asset_data in data.items():
                        signals = asset_data.get('signals', {})
                        entry_signal = signals.get('entry', False)
                        exit_signal = signals.get('exit', False)

                        # Get current position
                        position = broker.get_position(symbol)
                        current_qty = position.quantity if position else 0.0

                        # MATCHES Backtrader: if action == "BUY" and current_size == 0
                        if entry_signal and abs(current_qty) < 0.001:
                            # Calculate portfolio value for position sizing
                            portfolio_value = broker.cash
                            for s, s_data in data.items():
                                pos = broker.get_position(s)
                                if pos and pos.quantity > 0:
                                    portfolio_value += pos.quantity * s_data.get('close', 0)

                            # MATCH BACKTRADER: Use integer shares and account for commission
                            # Backtrader rounds down to whole shares, which leaves small buffers
                            # that accumulate to allow more positions to fit
                            target_value = portfolio_value * self.target_pct
                            price = asset_data.get('close')

                            # Calculate quantity and round to integer shares (like Backtrader)
                            raw_quantity = target_value / price if price > 0 else 0
                            quantity = int(raw_quantity)  # Round DOWN to integer shares

                            if quantity > 0:
                                broker.submit_order(
                                    asset=symbol,
                                    side=OrderSide.BUY,
                                    quantity=quantity,
                                    order_type=OrderType.MARKET,
                                )

                        # MATCHES Backtrader: elif action == "SELL" and current_size > 0
                        elif exit_signal and abs(current_qty) > 0.001:
                            broker.submit_order(
                                asset=symbol,
                                side=OrderSide.SELL,
                                quantity=abs(current_qty),
                                order_type=OrderType.MARKET,
                            )

            # Create commission and slippage models
            commission = PercentageCommission(rate=config.commission_pct) if config.commission_pct > 0 else NoCommission()
            slippage = PercentageSlippage(rate=config.slippage_pct) if config.slippage_pct > 0 else NoSlippage()

            # Create DataFeed with prices and signals
            feed = DataFeed(prices_df=prices_df, signals_df=signals_df)

            # Create strategy
            strategy = RotationStrategy(target_pct=0.20)

            # Determine execution mode from config
            # - "next_open" -> NEXT_BAR (matches Backtrader default)
            # - "same_close" -> SAME_BAR (look-ahead bias, but matches some frameworks)
            if config.fill_timing == "next_open":
                exec_mode = ExecutionMode.NEXT_BAR
            else:
                exec_mode = ExecutionMode.SAME_BAR

            # Run backtest
            engine = Engine(
                feed=feed,
                strategy=strategy,
                initial_cash=config.initial_capital,
                commission_model=commission,
                slippage_model=slippage,
                execution_mode=exec_mode,
                account_type='cash',
            )

            results = engine.run()

            result.final_value = results['final_value']
            result.total_return = results['total_return'] * 100

            # Extract trades
            trades = []
            raw_trades = engine.broker.trades
            result.num_trades = len(raw_trades)

            for trade in raw_trades:
                if trade.exit_time is not None:
                    trades.append(TradeRecord(
                        timestamp=trade.entry_time,
                        action="BUY",
                        quantity=trade.quantity,
                        price=trade.entry_price,
                        value=trade.quantity * trade.entry_price,
                        commission=trade.commission / 2,
                    ))
                    trades.append(TradeRecord(
                        timestamp=trade.exit_time,
                        action="SELL",
                        quantity=trade.quantity,
                        price=trade.exit_price,
                        value=trade.quantity * trade.exit_price,
                        commission=trade.commission / 2,
                    ))

            result.trades = trades

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print(f"✓ ml4t.backtest multi-asset backtest completed")
            print(f"  Entry Signals: {signals_df['entry'].sum()}, Exit Signals: {signals_df['exit'].sum()}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except Exception as e:
            error_msg = f"ml4t.backtest multi-asset failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
