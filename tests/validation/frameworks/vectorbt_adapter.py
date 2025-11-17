"""
VectorBT Framework Adapter for Cross-Framework Validation

Fixed implementation for vectorbt 0.28.0 (open source version)
"""

import time
import tracemalloc
from typing import Any

import pandas as pd

from .base import BaseFrameworkAdapter, FrameworkConfig, Signal, TradeRecord, ValidationResult


class VectorBTAdapter(BaseFrameworkAdapter):
    """Adapter for VectorBT backtesting framework."""

    def __init__(self):
        super().__init__("VectorBT")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest using VectorBT."""

        result = ValidationResult(
            framework=self.framework_name,
            strategy=strategy_params.get("name", "Unknown"),
            initial_capital=initial_capital,
        )

        try:
            # Use open-source vectorbt for correctness validation
            # (vectorbtpro should only be used for performance benchmarking)
            import vectorbt as vbt

            tracemalloc.start()
            start_time = time.time()

            # Extract strategy parameters
            short_window = strategy_params.get("short_window", 20)
            long_window = strategy_params.get("long_window", 50)

            # Calculate moving averages
            close_prices = data["close"]

            # Use pandas rolling for consistency with other frameworks
            ma_short_values = close_prices.rolling(window=short_window, min_periods=short_window).mean()
            ma_long_values = close_prices.rolling(window=long_window, min_periods=long_window).mean()

            # Generate entry and exit signals
            # Entry: short MA crosses above long MA
            entries = (ma_short_values > ma_long_values) & (
                ma_short_values.shift(1) <= ma_long_values.shift(1)
            )

            # Exit: short MA crosses below long MA
            exits = (ma_short_values <= ma_long_values) & (
                ma_short_values.shift(1) > ma_long_values.shift(1)
            )

            # Remove NaN values (from MA calculation)
            valid_mask = ~(ma_short_values.isna() | ma_long_values.isna())
            entries = entries & valid_mask
            exits = exits & valid_mask

            print(f"VectorBT signals: {entries.sum()} entries, {exits.sum()} exits")

            # Create portfolio using from_signals
            pf = vbt.Portfolio.from_signals(
                close_prices,
                entries=entries,
                exits=exits,
                init_cash=initial_capital,
                fees=0.0,  # No fees for comparison
                slippage=0.0,  # No slippage for comparison
                freq="D",  # Daily frequency
            )

            # Extract results - compatible with both vectorbt and vectorbtpro
            # Handle both Series (open-source) and callable (Pro) APIs
            try:
                value = pf.value
                if callable(value):
                    # Pro API - value is a method
                    value_result = value()
                    if hasattr(value_result, 'iloc'):
                        result.final_value = float(value_result.iloc[-1])
                    else:
                        result.final_value = float(value_result)
                elif hasattr(value, 'iloc'):
                    # Series - get last value
                    last_val = value.iloc[-1]
                    # Handle case where iloc[-1] still returns a Series (multi-column)
                    if hasattr(last_val, 'iloc'):
                        result.final_value = float(last_val.iloc[0])
                    else:
                        result.final_value = float(last_val)
                else:
                    # Scalar
                    result.final_value = float(value)
            except Exception as e:
                print(f"Warning: Could not extract final_value: {e}")
                print(f"  Type: {type(pf.value)}")
                if hasattr(pf.value, 'iloc'):
                    print(f"  Last value type: {type(pf.value.iloc[-1])}")
                result.final_value = 0.0

            try:
                total_return = pf.total_return
                if callable(total_return):
                    # Pro API - total_return is a method
                    tr_result = total_return()
                    if hasattr(tr_result, 'iloc'):
                        last_tr = tr_result.iloc[-1]
                        result.total_return = float(last_tr.iloc[0] if hasattr(last_tr, 'iloc') else last_tr) * 100
                    else:
                        result.total_return = float(tr_result) * 100
                elif hasattr(total_return, 'iloc'):
                    # Series - get last value
                    last_tr = total_return.iloc[-1]
                    # Handle case where iloc[-1] still returns a Series (multi-column)
                    if hasattr(last_tr, 'iloc'):
                        result.total_return = float(last_tr.iloc[0]) * 100
                    else:
                        result.total_return = float(last_tr) * 100
                else:
                    # Scalar
                    result.total_return = float(total_return) * 100
            except Exception as e:
                print(f"Warning: Could not extract total_return: {e}")
                result.total_return = 0.0

            # Get number of trades (round trips, not individual orders)
            if hasattr(pf, "trades") and hasattr(pf.trades, "records"):
                result.num_trades = len(pf.trades.records)
            elif hasattr(pf, "orders") and hasattr(pf.orders, "records"):
                # Fallback: count orders and divide by 2 for round trips
                result.num_trades = len(pf.orders.records) // 2
            else:
                result.num_trades = 0

            # Get performance metrics - compatible with both APIs
            try:
                sr = pf.sharpe_ratio
                if hasattr(sr, 'iloc'):
                    # Series - get last/scalar value
                    result.sharpe_ratio = float(sr.iloc[-1] if len(sr) > 1 else sr.iloc[0])
                elif callable(sr):
                    result.sharpe_ratio = float(sr())
                else:
                    result.sharpe_ratio = float(sr)
            except Exception as e:
                print(f"Warning: Could not extract sharpe_ratio: {e}")
                result.sharpe_ratio = 0.0

            try:
                md = pf.max_drawdown
                if hasattr(md, 'iloc'):
                    # Series - get last/max value
                    result.max_drawdown = float(md.iloc[-1] if len(md) > 1 else md.iloc[0]) * 100
                elif callable(md):
                    result.max_drawdown = float(md()) * 100
                else:
                    result.max_drawdown = float(md) * 100
            except Exception as e:
                print(f"Warning: Could not extract max_drawdown: {e}")
                result.max_drawdown = 0.0

            # Get equity curve and returns
            try:
                result.equity_curve = pf.value if hasattr(pf.value, 'index') else pf.value()
                result.daily_returns = pf.returns if hasattr(pf.returns, 'index') else pf.returns()

                print(f"  DEBUG: equity_curve type={type(result.equity_curve)}, len={len(result.equity_curve) if hasattr(result.equity_curve, '__len__') else 'N/A'}")
                print(f"  DEBUG: daily_returns type={type(result.daily_returns)}, len={len(result.daily_returns) if hasattr(result.daily_returns, '__len__') else 'N/A'}")

                # Ensure we have actual data, not empty Series
                if result.daily_returns is not None and len(result.daily_returns) == 0:
                    print("  DEBUG: daily_returns is empty, setting to None")
                    result.daily_returns = None
                if result.equity_curve is not None and len(result.equity_curve) == 0:
                    print("  DEBUG: equity_curve is empty, setting to None")
                    result.equity_curve = None
            except Exception as e:
                print(f"  Warning: Failed to extract daily returns/equity curve from VectorBT: {e}")
                import traceback
                traceback.print_exc()
                result.equity_curve = None
                result.daily_returns = None

            # Extract individual orders (not complete round-trip trades)
            # This matches ml4t.backtest and Backtrader behavior
            result.trades = []
            try:
                if hasattr(pf, "orders") and hasattr(pf.orders, "records_readable"):
                    orders_df = pf.orders.records_readable

                    for _, order in orders_df.iterrows():
                        trade_record = TradeRecord(
                            timestamp=order.get("Timestamp", order.get("timestamp")),
                            action=str(order.get("Side", "")).upper(),  # 'BUY' or 'SELL'
                            quantity=abs(float(order.get("Size", order.get("size", 0)))),
                            price=float(order.get("Price", order.get("price", 0))),
                            value=abs(
                                float(order.get("Size", order.get("size", 0)))
                                * float(order.get("Price", order.get("price", 0)))
                            ),
                        )
                        result.trades.append(trade_record)
            except Exception as e:
                print(f"Warning: Could not extract order details: {e}")

            # Calculate win rate from trades
            if len(result.trades) >= 2:
                trade_returns = []
                for i in range(0, len(result.trades) - 1, 2):
                    if i + 1 < len(result.trades):
                        entry = result.trades[i]
                        exit_trade = result.trades[i + 1]
                        if entry.action == "BUY" and exit_trade.action == "SELL":
                            ret = exit_trade.price / entry.price - 1
                            trade_returns.append(ret)
                        elif entry.action == "SELL" and exit_trade.action == "BUY":
                            ret = entry.price / exit_trade.price - 1
                            trade_returns.append(ret)

                if trade_returns:
                    result.win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ VectorBT backtest completed")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except ImportError as e:
            error_msg = f"VectorBT not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"VectorBT backtest failed: {e}"
            print(f"✗ {error_msg}")
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
            import vectorbt as vbt

            tracemalloc.start()
            start_time = time.time()

            # Determine execution price based on fill_timing config
            close_prices = data["close"]
            open_prices = data["open"]

            # Apply execution delay if configured (e.g., to match Zipline's next-day fills)
            if config.fill_timing in ["next_open", "next_close"]:
                # Shift signals forward by 1 day to execute later
                signals = signals.shift(1, fill_value=False)
                # When signals are shifted, we DON'T shift prices
                # Signal on T → shifted to T+1 → fill at T+1's open (no price shift needed)
                if config.fill_timing == "next_open":
                    fill_price = open_prices  # Use current bar's open (signal already shifted)
                else:  # next_close
                    fill_price = close_prices  # Use current bar's close (signal already shifted)
            elif config.fill_timing == "same_close":
                # Default VectorBT behavior - fill at same bar close (look-ahead bias!)
                fill_price = close_prices
                # No signal shift needed - execute on same bar
            else:
                raise ValueError(f"Unknown fill_timing: {config.fill_timing}")

            # Use signals directly (already boolean Series)
            entries = signals["entry"]
            exits = signals["exit"]

            num_entry_signals = entries.sum()
            num_exit_signals = exits.sum()

            # Create portfolio using from_signals with configuration
            pf = vbt.Portfolio.from_signals(
                close_prices,
                entries=entries,
                exits=exits,
                price=fill_price,  # Configure execution price
                init_cash=config.initial_capital,
                fees=config.commission_pct,  # Commission rate
                slippage=config.slippage_pct,  # Slippage percentage
                accumulate=config.vectorbt_accumulate,  # Allow/prevent same-bar re-entry
                freq="D",
            )

            # Extract results using same robust API handling as run_backtest
            try:
                value = pf.value
                if callable(value):
                    value_result = value()
                    if hasattr(value_result, 'iloc'):
                        result.final_value = float(value_result.iloc[-1])
                    else:
                        result.final_value = float(value_result)
                elif hasattr(value, 'iloc'):
                    last_val = value.iloc[-1]
                    if hasattr(last_val, 'iloc'):
                        result.final_value = float(last_val.iloc[0])
                    else:
                        result.final_value = float(last_val)
                else:
                    result.final_value = float(value)
            except Exception as e:
                print(f"Warning: Could not extract final_value: {e}")
                result.final_value = 0.0

            try:
                total_return = pf.total_return
                if callable(total_return):
                    tr_result = total_return()
                    if hasattr(tr_result, 'iloc'):
                        last_tr = tr_result.iloc[-1]
                        result.total_return = float(last_tr.iloc[0] if hasattr(last_tr, 'iloc') else last_tr) * 100
                    else:
                        result.total_return = float(tr_result) * 100
                elif hasattr(total_return, 'iloc'):
                    last_tr = total_return.iloc[-1]
                    if hasattr(last_tr, 'iloc'):
                        result.total_return = float(last_tr.iloc[0]) * 100
                    else:
                        result.total_return = float(last_tr) * 100
                else:
                    result.total_return = float(total_return) * 100
            except Exception as e:
                print(f"Warning: Could not extract total_return: {e}")
                result.total_return = 0.0

            # Get number of trades
            if hasattr(pf, "trades") and hasattr(pf.trades, "records"):
                result.num_trades = len(pf.trades.records)
            elif hasattr(pf, "orders") and hasattr(pf.orders, "records"):
                result.num_trades = len(pf.orders.records) // 2
            else:
                result.num_trades = 0

            # Extract individual orders (not complete round-trip trades)
            # This matches ml4t.backtest and Backtrader behavior where each
            # BUY and SELL is counted as a separate trade
            result.trades = []
            try:
                if hasattr(pf, "orders") and hasattr(pf.orders, "records_readable"):
                    orders_df = pf.orders.records_readable

                    for _, order in orders_df.iterrows():
                        trade_record = TradeRecord(
                            timestamp=order.get("Timestamp", order.get("timestamp")),
                            action=str(order.get("Side", "")).upper(),  # 'BUY' or 'SELL'
                            quantity=abs(float(order.get("Size", order.get("size", 0)))),
                            price=float(order.get("Price", order.get("price", 0))),
                            value=abs(
                                float(order.get("Size", order.get("size", 0)))
                                * float(order.get("Price", order.get("price", 0)))
                            ),
                        )
                        result.trades.append(trade_record)
            except Exception as e:
                print(f"Warning: Could not extract order details: {e}")

            # Get equity curve and daily returns
            try:
                # Extract equity curve
                value = pf.value
                if callable(value):
                    result.equity_curve = value()
                else:
                    result.equity_curve = value

                # Extract daily returns
                returns = pf.returns
                if callable(returns):
                    result.daily_returns = returns()
                else:
                    result.daily_returns = returns

                # Ensure we have valid Series, not empty
                if result.daily_returns is not None and hasattr(result.daily_returns, '__len__') and len(result.daily_returns) == 0:
                    result.daily_returns = None
                if result.equity_curve is not None and hasattr(result.equity_curve, '__len__') and len(result.equity_curve) == 0:
                    result.equity_curve = None
            except Exception as e:
                print(f"  Warning: Failed to extract daily returns/equity curve: {e}")
                result.equity_curve = None
                result.daily_returns = None

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ VectorBT signal-based backtest completed")
            print(f"  Entry Signals: {num_entry_signals}, Exit Signals: {num_exit_signals}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")

        except ImportError as e:
            error_msg = f"VectorBT not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"VectorBT signal-based backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
