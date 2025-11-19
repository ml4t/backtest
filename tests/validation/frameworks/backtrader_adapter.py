"""
Backtrader Framework Adapter for Cross-Framework Validation

Clean implementation that avoids the position tracking bugs found in earlier attempts.
"""

import time
import tracemalloc
from typing import Any

import pandas as pd

from .base import BaseFrameworkAdapter, FrameworkConfig, Signal, TradeRecord, ValidationResult


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

            # CRITICAL: Disable cheat-on-close to execute at NEXT bar's OPEN (not signal bar's close)
            # COC=True would fill at signal bar's close (look-ahead bias)
            # COC=False fills at next bar's open (industry standard for daily data)
            # Source: backtrader/brokers/bbroker.py:895-903
            cerebro.broker.set_coc(False)

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
        data: pd.DataFrame | dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        config: FrameworkConfig | None = None,
    ) -> ValidationResult:
        """
        Run backtest with pre-computed signals (pure execution validation).

        Args:
            data: OHLCV data - either:
                  - Single DataFrame with DatetimeIndex (single-asset)
                  - Dict mapping symbol -> DataFrame (multi-asset)
            signals: DataFrame with either:
                  - 'entry' and 'exit' boolean columns (single-asset)
                  - 'timestamp', 'symbol', 'signal' columns (multi-asset)
            config: FrameworkConfig for execution parameters (costs, timing, etc.).
                   If None, uses FrameworkConfig.realistic() defaults.

        Returns:
            ValidationResult with performance metrics
        """
        # Use default config if not provided
        if config is None:
            config = FrameworkConfig.realistic()

        # Check if this is multi-asset (dict) or single-asset (DataFrame)
        is_multi_asset = isinstance(data, dict)

        if is_multi_asset:
            return self._run_multi_asset_with_signals(data, signals, config)

        # Single-asset path (original logic)
        # Validate inputs
        self.validate_data(data)
        self.validate_signals(signals, data)

        result = ValidationResult(
            framework=self.framework_name,
            strategy="SignalBased",
            initial_capital=config.initial_capital,
        )

        try:
            import backtrader as bt

            tracemalloc.start()
            start_time = time.time()

            # Apply execution delay if configured (e.g., to match Zipline's next-day fills)
            if config.fill_timing in ["next_open", "next_close"]:
                # Shift signals forward by 1 day to execute later
                signals = signals.shift(1, fill_value=False)

            # Convert boolean signals to date-indexed dict for quick lookup
            signal_dict = {}
            for timestamp, row in signals.iterrows():
                date = timestamp.date()
                if row["entry"]:
                    if date not in signal_dict:
                        signal_dict[date] = []
                    signal_dict[date].append({"action": "BUY"})
                if row["exit"]:
                    if date not in signal_dict:
                        signal_dict[date] = []
                    signal_dict[date].append({"action": "SELL"})


            class SignalStrategy(bt.Strategy):
                """Strategy that trades on pre-computed signals."""

                def __init__(self):
                    self.signal_dict = signal_dict
                    self.trade_log = []
                    self.order = None
                    self.portfolio_values = []  # Track portfolio value over time

                def next(self):
                    # Track portfolio value at each bar for daily returns calculation
                    current_date = self.data.datetime.date(0)
                    portfolio_value = self.broker.getvalue()
                    self.portfolio_values.append({
                        "date": current_date,
                        "value": portfolio_value,
                    })

                    # Skip if we have a pending order
                    if self.order:
                        return

                    current_price = self.data.close[0]

                    # Check if we have signals for today
                    if current_date in self.signal_dict:
                        for signal in self.signal_dict[current_date]:
                            action = signal["action"]

                            if action == "BUY" and not self.position:
                                # Enter long position using target value (not fixed size)
                                # With COC=True, order fills at SAME bar's close (same price used for sizing)
                                # Minimal buffer needed for commission/slippage
                                cash = self.broker.getcash()
                                target_value = cash * 0.998  # 0.2% buffer for commission + slippage

                                self.order = self.order_target_value(target=target_value)
                                self.trade_log.append({
                                    "date": current_date,
                                    "action": "BUY_SIGNAL",
                                    "price": current_price,
                                    "target_value": target_value,
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
                    elif order.status in [order.Margin, order.Rejected]:
                        # Log rejected orders to understand why Backtrader skips signals
                        self.trade_log.append({
                            "date": self.data.datetime.date(0),
                            "action": f"ORDER_REJECTED_{order.Margin if order.status == order.Margin else 'OTHER'}",
                            "reason": "Insufficient cash/margin",
                            "cash": self.broker.getcash(),
                            "value": self.broker.getvalue(),
                        })

                    if not order.alive():
                        self.order = None

            # Create Cerebro engine
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SignalStrategy)

            # Prepare data for Backtrader
            bt_data = data.copy()
            # Ensure index is named 'date' for Backtrader compatibility
            bt_data.index.name = 'date'
            bt_data = bt_data.reset_index()

            # Backtrader expects the datetime column
            datetime_col = 'date'

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

            # Set broker parameters from config
            cerebro.broker.setcash(config.initial_capital)
            cerebro.broker.setcommission(commission=config.commission_pct)

            # Configure slippage
            # Note: Backtrader slip_perc expects percentage as decimal (0.01 = 1%)
            # Our config.slippage_pct is also decimal (0.0005 = 0.05%)
            if config.slippage_pct > 0 or config.slippage_fixed > 0:
                cerebro.broker.set_slippage_perc(
                    perc=config.slippage_pct,
                    slip_open=True,   # Apply slippage to opening fills
                    slip_limit=True,  # Limit slippage to bar high/low
                )

            # Configure execution timing based on config
            # COO (Cheat-On-Open): Execute at bar open
            # COC (Cheat-On-Close): Execute at bar close
            if config.backtrader_coo:
                cerebro.broker.set_coo(True)
            if config.backtrader_coc:
                cerebro.broker.set_coc(True)

            # Default behavior (coo=False, coc=False): Execute at next bar open
            # This is realistic and avoids look-ahead bias

            # Run backtest
            strategies = cerebro.run()
            strategy_instance = strategies[0]

            # Extract results
            result.final_value = cerebro.broker.getvalue()
            result.total_return = (result.final_value / config.initial_capital - 1) * 100

            # Process trade log to create trade records
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

            result.num_trades = len(executed_trades) // 2

            # Extract equity curve and daily returns
            if strategy_instance.portfolio_values:
                # Convert portfolio values to DataFrame
                equity_df = pd.DataFrame(strategy_instance.portfolio_values)
                equity_df["date"] = pd.to_datetime(equity_df["date"])
                equity_df = equity_df.set_index("date")

                # Create equity curve (portfolio value over time)
                result.equity_curve = equity_df["value"]

                # Calculate daily returns from portfolio values
                result.daily_returns = result.equity_curve.pct_change().fillna(0.0)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            num_entry_signals = signals["entry"].sum()
            num_exit_signals = signals["exit"].sum()

            # Count rejections
            rejected_orders = [log for log in strategy_instance.trade_log if "REJECTED" in log.get("action", "")]
            num_signals_generated = len([log for log in strategy_instance.trade_log if log["action"].endswith("_SIGNAL")])

            print("✓ Backtrader signal-based backtest completed")
            print(f"  Entry Signals: {num_entry_signals}, Exit Signals: {num_exit_signals}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            if rejected_orders:
                print(f"  ⚠️  Rejected Orders: {len(rejected_orders)} (insufficient cash/margin)")
                print(f"  Signal Generation: {num_signals_generated} signals → {len(executed_trades)} executions")

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

    def _run_multi_asset_with_signals(
        self,
        data: dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        config: FrameworkConfig,
    ) -> ValidationResult:
        """
        Run backtest with multi-asset rotation signals using Backtrader.

        Args:
            data: Dict mapping symbol -> OHLCV DataFrame
            signals: DataFrame with columns: timestamp, symbol, signal (1=buy, -1=sell)
            config: FrameworkConfig for execution parameters

        Returns:
            ValidationResult with performance metrics
        """
        result = ValidationResult(
            framework=self.framework_name,
            strategy="TopNMomentum",
            initial_capital=config.initial_capital,
        )

        try:
            import backtrader as bt

            tracemalloc.start()
            start_time = time.time()

            # Convert signals from long format to dict for fast lookup
            # signal_dict[date][symbol] = action ("BUY" or "SELL")
            signal_dict = {}
            for _, row in signals.iterrows():
                date = pd.Timestamp(row['timestamp']).date()
                symbol = row['symbol']
                signal_value = row['signal']

                if date not in signal_dict:
                    signal_dict[date] = {}

                if signal_value == 1:  # BUY
                    signal_dict[date][symbol] = "BUY"
                elif signal_value == -1:  # SELL
                    signal_dict[date][symbol] = "SELL"

            num_entry_signals = (signals['signal'] == 1).sum()
            num_exit_signals = (signals['signal'] == -1).sum()

            print(f"\nBacktrader multi-asset setup:")
            print(f"  Assets: {len(data)}")
            print(f"  Entry signals: {num_entry_signals}")
            print(f"  Exit signals: {num_exit_signals}")

            # Create the signal-based rotation strategy
            class MultiAssetSignalStrategy(bt.Strategy):
                """Multi-asset rotation strategy using pre-computed signals."""

                def __init__(self):
                    self.signal_dict = signal_dict
                    self.target_pct = 0.20  # 20% per position (5 positions max)
                    self.trade_log = []
                    self.orders = {}  # Track pending orders by symbol

                    # Create a mapping from data feed to symbol name
                    self.data_to_symbol = {}
                    for i, d in enumerate(self.datas):
                        # Symbol name is stored in data feed's _name attribute
                        symbol = d._name
                        self.data_to_symbol[d] = symbol

                def next(self):
                    current_date = self.data.datetime.date(0)

                    # Check if we have signals for today
                    if current_date not in self.signal_dict:
                        return

                    day_signals = self.signal_dict[current_date]

                    # Process each data feed (symbol)
                    for data_feed in self.datas:
                        symbol = self.data_to_symbol[data_feed]

                        # Skip if we have a pending order for this symbol
                        if symbol in self.orders and self.orders[symbol]:
                            continue

                        # Check if we have a signal for this symbol today
                        if symbol not in day_signals:
                            continue

                        action = day_signals[symbol]
                        current_price = data_feed.close[0]

                        # Get current position size for this symbol
                        position = self.getposition(data_feed)
                        current_size = position.size if position else 0

                        if action == "BUY" and current_size == 0:
                            # Calculate target value (20% of portfolio)
                            portfolio_value = self.broker.getvalue()
                            target_value = portfolio_value * self.target_pct

                            # Place order using order_target_value for exact percentage
                            # With COC=True, this fills at same bar's close
                            order = self.order_target_value(
                                data=data_feed,
                                target=target_value,
                            )

                            if order:
                                self.orders[symbol] = order
                                self.trade_log.append({
                                    "date": current_date,
                                    "symbol": symbol,
                                    "action": "BUY_SIGNAL",
                                    "price": current_price,
                                    "target_value": target_value,
                                })

                        elif action == "SELL" and current_size > 0:
                            # Close position (target = 0)
                            order = self.order_target_value(
                                data=data_feed,
                                target=0.0,
                            )

                            if order:
                                self.orders[symbol] = order
                                self.trade_log.append({
                                    "date": current_date,
                                    "symbol": symbol,
                                    "action": "SELL_SIGNAL",
                                    "price": current_price,
                                    "position_size": current_size,
                                })

                def notify_order(self, order):
                    """Called when order status changes."""
                    if order.status in [order.Completed]:
                        # Find which symbol this order belongs to
                        symbol = None
                        for data_feed, sym in self.data_to_symbol.items():
                            if order.data == data_feed:
                                symbol = sym
                                break

                        action = "BUY" if order.isbuy() else "SELL"

                        self.trade_log.append({
                            "date": bt.num2date(order.executed.dt).date(),
                            "symbol": symbol,
                            "action": f"{action}_EXECUTED",
                            "price": order.executed.price,
                            "size": abs(order.executed.size),
                            "value": abs(order.executed.value),
                            "commission": order.executed.comm,
                        })

                    # Clear pending order when completed/cancelled/rejected
                    if not order.alive():
                        for sym, ord in list(self.orders.items()):
                            if ord == order:
                                self.orders[sym] = None
                                break

            # Create Cerebro engine
            cerebro = bt.Cerebro()
            cerebro.addstrategy(MultiAssetSignalStrategy)

            # Add data feed for each symbol
            for symbol, ohlcv in sorted(data.items()):
                bt_data = ohlcv.copy()
                bt_data.index.name = 'date'
                bt_data = bt_data.reset_index()

                data_feed = bt.feeds.PandasData(
                    dataname=bt_data,
                    datetime='date',
                    open='open',
                    high='high',
                    low='low',
                    close='close',
                    volume='volume',
                    openinterest=-1,
                )

                # Set the name of the data feed to the symbol
                data_feed._name = symbol
                cerebro.adddata(data_feed, name=symbol)

            # Set broker parameters from config
            cerebro.broker.setcash(config.initial_capital)
            cerebro.broker.setcommission(commission=config.commission_pct)

            # Configure slippage
            if config.slippage_pct > 0 or config.slippage_fixed > 0:
                cerebro.broker.set_slippage_perc(
                    perc=config.slippage_pct,
                    slip_open=True,
                    slip_limit=True,
                )

            # Configure execution timing
            # Use COC (Cheat-On-Close) for same-bar fills like VectorBT
            # This matches the behavior of from_signals where signals execute immediately
            if config.backtrader_coc:
                cerebro.broker.set_coc(True)

            # Run backtest
            strategies = cerebro.run()
            strategy_instance = strategies[0]

            # Extract results
            result.final_value = cerebro.broker.getvalue()
            result.total_return = (result.final_value / config.initial_capital - 1) * 100

            # Process trade log to create trade records
            executed_trades = [
                log for log in strategy_instance.trade_log if log["action"].endswith("_EXECUTED")
            ]

            result.trades = []
            for trade_info in executed_trades:
                trade_record = TradeRecord(
                    timestamp=pd.to_datetime(trade_info["date"]),
                    action=trade_info["action"].replace("_EXECUTED", ""),
                    quantity=trade_info["size"],
                    price=trade_info["price"],
                    value=trade_info["value"],
                    commission=trade_info.get("commission", 0.0),
                )
                result.trades.append(trade_record)

            # Count trades (individual orders, not round trips)
            result.num_trades = len(executed_trades)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ Backtrader multi-asset backtest completed")
            print(f"  Entry Signals: {num_entry_signals}, Exit Signals: {num_exit_signals}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except ImportError as e:
            error_msg = f"Backtrader not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Backtrader multi-asset backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
