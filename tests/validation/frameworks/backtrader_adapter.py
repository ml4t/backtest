"""
Backtrader Framework Adapter for Cross-Framework Validation

Clean implementation that avoids the position tracking bugs found in earlier attempts.
"""

import time
import tracemalloc
from typing import Any

import pandas as pd

from .base import BaseFrameworkAdapter, Signal, TradeRecord, ValidationResult


class BacktraderAdapter(BaseFrameworkAdapter):
    """Adapter for Backtrader backtesting framework."""

    def __init__(self):
        super().__init__("Backtrader")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest using Backtrader."""

        result = ValidationResult(
            framework=self.framework_name,
            strategy=strategy_params.get("name", "Unknown"),
            initial_capital=initial_capital,
        )

        try:
            import backtrader as bt

            tracemalloc.start()
            start_time = time.time()

            # Extract strategy parameters
            short_window = strategy_params.get("short_window", 20)
            long_window = strategy_params.get("long_window", 50)

            class FixedMomentumStrategy(bt.Strategy):
                """Fixed momentum strategy that properly tracks positions."""

                params = (
                    ("short_window", short_window),
                    ("long_window", long_window),
                )

                def __init__(self):
                    # Create moving averages
                    self.ma_short = bt.indicators.SimpleMovingAverage(
                        self.data.close,
                        period=self.params.short_window,
                    )
                    self.ma_long = bt.indicators.SimpleMovingAverage(
                        self.data.close,
                        period=self.params.long_window,
                    )

                    # Track trades manually for debugging
                    self.trade_log = []
                    self.order = None  # Keep track of pending orders

                def next(self):
                    # Skip if we have a pending order
                    if self.order:
                        return

                    current_date = self.data.datetime.date(0)
                    current_price = self.data.close[0]

                    # Manual crossover detection to match ml4t.backtest/VectorBT logic
                    # This uses asymmetric operators to prevent whipsaw
                    current_short = self.ma_short[0]
                    current_long = self.ma_long[0]
                    prev_short = self.ma_short[-1]
                    prev_long = self.ma_long[-1]

                    # Golden cross: short MA crosses above long MA
                    golden_cross = (prev_short <= prev_long) and (current_short > current_long)

                    # Death cross: short MA crosses below long MA
                    death_cross = (prev_short > prev_long) and (current_short <= current_long)

                    # Check for golden cross (go long)
                    if not self.position:
                        if golden_cross:
                            # Calculate position size (use all available cash)
                            cash = self.broker.getcash()
                            size = cash / current_price

                            # Place buy order
                            self.order = self.buy(size=size)
                            self.trade_log.append(
                                {
                                    "date": current_date,
                                    "action": "BUY_SIGNAL",
                                    "price": current_price,
                                    "size": size,
                                    "cash_before": cash,
                                    "ma_short": current_short,
                                    "ma_long": current_long,
                                },
                            )

                    # Check for death cross (go flat)
                    else:
                        if death_cross:
                            # Close position
                            self.order = self.close()
                            self.trade_log.append(
                                {
                                    "date": current_date,
                                    "action": "SELL_SIGNAL",
                                    "price": current_price,
                                    "position_size": self.position.size,
                                    "ma_short": current_short,
                                    "ma_long": current_long,
                                },
                            )

                def notify_order(self, order):
                    """Called when order status changes."""
                    if order.status in [order.Completed]:
                        action = "BUY" if order.isbuy() else "SELL"

                        self.trade_log.append(
                            {
                                "date": bt.num2date(order.executed.dt).date(),
                                "action": f"{action}_EXECUTED",
                                "price": order.executed.price,
                                "size": order.executed.size,
                                "value": order.executed.value,
                                "commission": order.executed.comm,
                            },
                        )

                    if not order.alive():
                        self.order = None  # Clear pending order

            # Create Cerebro engine
            cerebro = bt.Cerebro()
            cerebro.addstrategy(FixedMomentumStrategy)

            # Prepare data for Backtrader
            bt_data = data.copy().reset_index()

            # Create Backtrader data feed
            data_feed = bt.feeds.PandasData(
                dataname=bt_data,
                datetime="date",  # Use the date column name from reset_index()
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                openinterest=-1,
            )
            cerebro.adddata(data_feed)

            # Set broker parameters
            cerebro.broker.setcash(initial_capital)
            cerebro.broker.setcommission(commission=0.0)  # No commission for comparison

            # CRITICAL: Enable cheat-on-close to execute at close of current bar (not next bar open)
            # This matches ml4t.backtest/VectorBT behavior which execute on same bar as signal
            cerebro.broker.set_coc(True)

            # Run backtest
            strategies = cerebro.run()
            strategy_instance = strategies[0]

            # Extract results
            result.final_value = cerebro.broker.getvalue()
            result.total_return = (result.final_value / initial_capital - 1) * 100

            # Process trade log to extract clean trade records
            executed_trades = [
                log for log in strategy_instance.trade_log if log["action"].endswith("_EXECUTED")
            ]

            result.trades = []
            for trade_info in executed_trades:
                trade_record = TradeRecord(
                    timestamp=pd.to_datetime(trade_info["date"]),
                    action=trade_info["action"].replace("_EXECUTED", ""),
                    quantity=abs(trade_info["size"]),
                    price=trade_info["price"],
                    value=abs(trade_info["value"]),
                    commission=trade_info.get("commission", 0.0),
                )
                result.trades.append(trade_record)

            # Count round-trip trades (pairs of buy/sell), not individual orders
            result.num_trades = len(result.trades) // 2

            # Calculate win rate from paired trades
            if len(result.trades) >= 2:
                trade_returns = []
                for i in range(0, len(result.trades) - 1, 2):
                    if i + 1 < len(result.trades):
                        entry = result.trades[i]
                        exit_trade = result.trades[i + 1]
                        if entry.action == "BUY" and exit_trade.action == "SELL":
                            ret = exit_trade.price / entry.price - 1
                            trade_returns.append(ret)

                if trade_returns:
                    result.win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ Backtrader backtest completed")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

            # Debug information
            signal_trades = [
                log for log in strategy_instance.trade_log if log["action"].endswith("_SIGNAL")
            ]
            print(
                f"  Debug: {len(signal_trades)} signals generated, {len(executed_trades)} trades executed",
            )

        except ImportError as e:
            error_msg = f"Backtrader not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Backtrader backtest failed: {e}"
            print(f"✗ {error_msg}")
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
            import backtrader as bt

            tracemalloc.start()
            start_time = time.time()

            # Convert signals to date-indexed dict for quick lookup
            signal_dict = {}
            for signal in signals:
                date = signal["timestamp"].date()
                if date not in signal_dict:
                    signal_dict[date] = []
                signal_dict[date].append(signal)

            class SignalStrategy(bt.Strategy):
                """Strategy that trades on pre-computed signals."""

                def __init__(self):
                    self.signal_dict = signal_dict
                    self.trade_log = []
                    self.order = None

                def next(self):
                    # Skip if we have a pending order
                    if self.order:
                        return

                    current_date = self.data.datetime.date(0)
                    current_price = self.data.close[0]

                    # Check if we have signals for today
                    if current_date in self.signal_dict:
                        for signal in self.signal_dict[current_date]:
                            action = signal["action"]

                            if action == "BUY" and not self.position:
                                # Enter long position
                                cash = self.broker.getcash()
                                size = cash / current_price

                                self.order = self.buy(size=size)
                                self.trade_log.append({
                                    "date": current_date,
                                    "action": "BUY_SIGNAL",
                                    "price": current_price,
                                    "size": size,
                                })

                            elif action == "SELL" and self.position:
                                # Exit position
                                self.order = self.close()
                                self.trade_log.append({
                                    "date": current_date,
                                    "action": "SELL_SIGNAL",
                                    "price": current_price,
                                    "position_size": self.position.size,
                                })

                def notify_order(self, order):
                    """Called when order status changes."""
                    if order.status in [order.Completed]:
                        action = "BUY" if order.isbuy() else "SELL"

                        self.trade_log.append({
                            "date": bt.num2date(order.executed.dt).date(),
                            "action": f"{action}_EXECUTED",
                            "price": order.executed.price,
                            "size": order.executed.size,
                            "value": order.executed.value,
                            "commission": order.executed.comm,
                        })

                    if not order.alive():
                        self.order = None

            # Create Cerebro engine
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SignalStrategy)

            # Prepare data for Backtrader
            bt_data = data.copy().reset_index()

            # Backtrader expects the datetime column to be named appropriately
            # After reset_index(), the column name depends on the original index name
            datetime_col = bt_data.columns[0]  # First column is the datetime after reset_index

            # Create Backtrader data feed
            data_feed = bt.feeds.PandasData(
                dataname=bt_data,
                datetime=datetime_col,  # Use the actual column name
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                openinterest=-1,
            )
            cerebro.adddata(data_feed)

            # Set broker parameters (no commission for signal validation)
            cerebro.broker.setcash(initial_capital)
            cerebro.broker.setcommission(commission=0.0)

            # Enable cheat-on-close to match other frameworks
            cerebro.broker.set_coc(True)

            # Run backtest
            strategies = cerebro.run()
            strategy_instance = strategies[0]

            # Extract results
            result.final_value = cerebro.broker.getvalue()
            result.total_return = (result.final_value / initial_capital - 1) * 100

            # Process trade log
            executed_trades = [
                log for log in strategy_instance.trade_log if log["action"].endswith("_EXECUTED")
            ]

            result.num_trades = len(executed_trades) // 2

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ Backtrader signal-based backtest completed")
            print(f"  Signals Processed: {len(signals)}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")

        except ImportError as e:
            error_msg = f"Backtrader not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Backtrader signal-based backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
