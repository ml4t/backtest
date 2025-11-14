"""
VectorBT Framework Adapter for Cross-Framework Validation

Fixed implementation for vectorbt 0.28.0 (open source version)
"""

import time
import tracemalloc
from typing import Any

import pandas as pd

from .base import BaseFrameworkAdapter, TradeRecord, ValidationResult


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
            # Try vectorbtpro first, fallback to vectorbt (open source)
            try:
                import vectorbtpro as vbt
            except ImportError:
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
            result.final_value = float(pf.value.iloc[-1] if hasattr(pf.value, 'iloc') else pf.value())
            result.total_return = float(pf.total_return * 100 if isinstance(pf.total_return, (int, float)) else pf.total_return() * 100)

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
                result.sharpe_ratio = float(sr if isinstance(sr, (int, float)) else sr())
            except:
                result.sharpe_ratio = 0.0

            try:
                md = pf.max_drawdown
                result.max_drawdown = float(md if isinstance(md, (int, float)) else md()) * 100
            except:
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
