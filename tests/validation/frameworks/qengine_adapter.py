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
            Order,
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

            # Convert signals to dict for fast lookup
            signal_dict = {}
            for _, row in signals.iterrows():
                key = (pd.Timestamp(row['timestamp']), row['symbol'])
                signal_dict[key] = row['signal']

            # Strategy class matching Backtrader logic EXACTLY
            class RotationStrategy(Strategy):
                def __init__(self, signals_dict, target_pct=0.20):
                    super().__init__()
                    self.signals_dict = signals_dict
                    self.target_pct = target_pct
                    self.pending_orders = {}  # Track pending orders by symbol (CRITICAL)

                def on_data(self, timestamp, data, context, broker):
                    """New callback-based API."""
                    # Process each asset's data
                    for symbol, asset_data in data.items():
                        # Check if we have a signal for this timestamp+symbol
                        key = (pd.Timestamp(timestamp), symbol)
                        if key not in self.signals_dict:
                            continue

                        signal_value = self.signals_dict[key]

                        # Get current position
                        position = broker.get_position(symbol)
                        current_qty = position.quantity if position else 0.0

                        # CRITICAL: Skip if we already have a pending order for this symbol
                        # (Matches Backtrader logic at backtrader_adapter.py:608-610)
                        if symbol in self.pending_orders and self.pending_orders[symbol]:
                            continue

                        # MATCHES Backtrader: if action == "BUY" and current_size == 0
                        if signal_value == 1 and abs(current_qty) < 0.001:
                            portfolio_value = broker.cash + sum(
                                broker.get_position(s).quantity * data[s].get('close', 0)
                                for s in data.keys()
                                if broker.get_position(s) is not None
                            )
                            target_value = portfolio_value * self.target_pct
                            price = asset_data.get('close')
                            quantity = target_value / price if price > 0 else 0

                            if quantity > 0.01:
                                broker.submit_order(
                                    asset=symbol,
                                    side=OrderSide.BUY,
                                    quantity=quantity,
                                    order_type=OrderType.MARKET,
                                )
                                self.pending_orders[symbol] = True  # Track pending order

                        # MATCHES Backtrader: elif action == "SELL" and current_size > 0
                        elif signal_value == -1 and abs(current_qty) > 0.001:
                            broker.submit_order(
                                asset=symbol,
                                side=OrderSide.SELL,
                                quantity=abs(current_qty),
                                order_type=OrderType.MARKET,
                            )
                            self.pending_orders[symbol] = True  # Track pending order

            # TODO: Multi-asset support needs complete rewrite for new modular API
            # The old event-driven approach with custom MultiAssetDataFeed doesn't work
            # Need to create Polars DataFrame with multi-asset data and use new DataFeed
            raise NotImplementedError(
                "Multi-asset backtesting with new modular API not yet implemented. "
                "The old event-driven architecture has been replaced. This requires a "
                "complete rewrite to use Polars DataFrames and the new callback-based Strategy API."
            )

            # OLD CODE BELOW - needs to be rewritten for new API
            # Create data feed
            class MultiAssetDataFeed_OLD(DataFeed):
                def __init__(self, ohlcv_dict):
                    self.data_list = []
                    for asset_id, ohlcv in ohlcv_dict.items():
                        for idx in range(len(ohlcv)):
                            row = ohlcv.iloc[idx]
                            self.data_list.append({
                                'timestamp': row.name,
                                'asset_id': asset_id,
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row['volume'],
                            })
                    self.data_list.sort(key=lambda x: x['timestamp'])
                    self.index = 0

                def get_next_event(self) -> MarketEvent | None:
                    """Get next event from data feed."""
                    if self.is_exhausted:
                        return None
                    data = self.data_list[self.index]
                    self.index += 1
                    return MarketEvent(
                        timestamp=data['timestamp'],
                        asset_id=data['asset_id'],
                        data_type=MarketDataType.BAR,
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        volume=data['volume'],
                    )

                @property
                def is_exhausted(self) -> bool:
                    """Check if feed exhausted."""
                    return self.index >= len(self.data_list)

                def peek_next_timestamp(self) -> datetime | None:
                    """Peek at next timestamp without consuming."""
                    if self.is_exhausted:
                        return None
                    return self.data_list[self.index]['timestamp']

                def reset(self) -> None:
                    """Reset to beginning."""
                    self.index = 0

                def seek(self, timestamp: datetime) -> None:
                    """Seek to timestamp."""
                    for i, data in enumerate(self.data_list):
                        if data['timestamp'] >= timestamp:
                            self.index = i
                            return
                    self.index = len(self.data_list)  # Exhausted if not found

            feed = MultiAssetDataFeed(data)

            commission = PercentageCommission(rate=config.commission_pct)
            slippage = PercentageSlippage(slippage_pct=config.slippage_pct)

            broker = SimulationBroker(
                initial_cash=config.initial_capital,
                commission_model=commission,
                slippage_model=slippage,
            )

            strategy = RotationStrategy(signal_dict, target_pct=0.20)

            engine = BacktestEngine(
                data_feed=feed,
                broker=broker,
                strategy=strategy,
            )

            engine.run()

            result.final_value = broker.portfolio.equity
            result.total_return = ((result.final_value / config.initial_capital) - 1) * 100

            # Extract trades
            trades = []
            if hasattr(broker, 'trade_tracker') and broker.trade_tracker:
                raw_trades = broker.trade_tracker._trades
                result.num_trades = len(raw_trades) * 2

                for trade in raw_trades:
                    trades.append(TradeRecord(
                        timestamp=trade.entry_dt,
                        action="BUY" if trade.direction == "long" else "SELL",
                        quantity=trade.entry_quantity,
                        price=trade.entry_price,
                        value=trade.entry_quantity * trade.entry_price,
                        commission=trade.entry_commission,
                    ))
                    trades.append(TradeRecord(
                        timestamp=trade.exit_dt,
                        action="SELL" if trade.direction == "long" else "BUY",
                        quantity=trade.exit_quantity,
                        price=trade.exit_price,
                        value=trade.exit_quantity * trade.exit_price,
                        commission=trade.exit_commission,
                    ))
            else:
                result.num_trades = 0

            result.trades = trades

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

        except Exception as e:
            error_msg = f"ml4t.backtest multi-asset failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
