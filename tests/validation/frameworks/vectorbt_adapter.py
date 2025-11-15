"""
VectorBT Framework Adapter for Cross-Framework Validation

Fixed implementation for vectorbt 0.28.0 (open source version)
"""

import time
import tracemalloc
from typing import Any

import pandas as pd

from .base import BaseFrameworkAdapter, Signal, TradeRecord, ValidationResult


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
            except:
                result.equity_curve = pd.Series()
                result.daily_returns = pd.Series()

            # Extract individual trades
            result.trades = []
            try:
                if hasattr(pf, "trades") and hasattr(pf.trades, "records_readable"):
                    trades_df = pf.trades.records_readable

                    for _, trade in trades_df.iterrows():
                        trade_record = TradeRecord(
                            timestamp=trade.get("Entry Timestamp", trade.get("entry_timestamp")),
                            action="BUY" if trade.get("Size", trade.get("size", 0)) > 0 else "SELL",
                            quantity=abs(float(trade.get("Size", trade.get("size", 0)))),
                            price=float(
                                trade.get("Avg Entry Price", trade.get("avg_entry_price", 0)),
                            ),
                            value=abs(
                                float(trade.get("Size", trade.get("size", 0)))
                                * float(
                                    trade.get("Avg Entry Price", trade.get("avg_entry_price", 0)),
                                ),
                            ),
                        )
                        result.trades.append(trade_record)
            except Exception as e:
                print(f"Warning: Could not extract trade details: {e}")

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
            import vectorbt as vbt

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

            print(f"VectorBT signal-based: {entries.sum()} entries, {exits.sum()} exits")

            # Create portfolio using from_signals
            close_prices = data["close"]
            pf = vbt.Portfolio.from_signals(
                close_prices,
                entries=entries,
                exits=exits,
                init_cash=initial_capital,
                fees=0.0,  # No fees for signal validation
                slippage=0.0,  # No slippage for signal validation
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

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ VectorBT signal-based backtest completed")
            print(f"  Signals Processed: {len(signals)}")
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
