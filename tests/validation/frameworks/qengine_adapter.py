"""
QEngine Framework Adapter for Cross-Framework Validation

Implements QEngine backtesting using the same manual approach as other frameworks
for fair comparison.
"""

import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseFrameworkAdapter, TradeRecord, ValidationResult


class QEngineAdapter(BaseFrameworkAdapter):
    """Adapter for QEngine backtesting framework."""

    def __init__(self):
        super().__init__("QEngine")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest using QEngine (manual implementation for comparison)."""

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

            # Calculate moving averages
            data_copy = data.copy()
            data_copy["ma_short"] = (
                data_copy["close"].rolling(window=short_window, min_periods=short_window).mean()
            )
            data_copy["ma_long"] = (
                data_copy["close"].rolling(window=long_window, min_periods=long_window).mean()
            )

            # Remove rows with NaN MAs
            data_copy = data_copy.dropna()

            print(f"QEngine processing {len(data_copy)} rows after MA calculation")

            # Initialize portfolio
            cash = initial_capital
            shares = 0.0
            position = 0  # 0 = flat, 1 = long
            trades = []
            equity_curve = []

            prev_ma_short = None
            prev_ma_long = None

            for date, row in data_copy.iterrows():
                current_ma_short = row["ma_short"]
                current_ma_long = row["ma_long"]
                current_price = row["close"]

                # Track portfolio value
                portfolio_value = cash + shares * current_price
                equity_curve.append(portfolio_value)

                # Generate signals based on MA crossover
                signal = 0  # 0 = no change, 1 = go long, -1 = go flat

                if prev_ma_short is not None and prev_ma_long is not None:
                    # Check for golden cross (MA short crosses above MA long)
                    if prev_ma_short <= prev_ma_long and current_ma_short > current_ma_long:
                        signal = 1  # Go long
                    # Check for death cross (MA short crosses below MA long)
                    elif prev_ma_short > prev_ma_long and current_ma_short <= current_ma_long:
                        signal = -1  # Go flat

                # Execute trades
                if signal == 1 and position == 0 and cash > 0:
                    # Buy: use all cash
                    shares = cash / current_price
                    cash = 0.0
                    position = 1
                    trade = TradeRecord(
                        timestamp=date,
                        action="BUY",
                        quantity=shares,
                        price=current_price,
                        value=shares * current_price,
                    )
                    trades.append(trade)

                elif signal == -1 and position == 1 and shares > 0:
                    # Sell: liquidate all shares
                    cash = shares * current_price
                    shares = 0.0
                    position = 0
                    trade = TradeRecord(
                        timestamp=date,
                        action="SELL",
                        quantity=shares,
                        price=current_price,
                        value=cash,
                    )
                    trades.append(trade)

                prev_ma_short = current_ma_short
                prev_ma_long = current_ma_long

            # Final portfolio value
            final_value = cash + shares * data_copy["close"].iloc[-1]

            # Set results
            result.final_value = final_value
            result.total_return = (final_value / initial_capital - 1) * 100
            result.num_trades = len(trades)
            result.trades = trades

            # Create equity curve series
            result.equity_curve = pd.Series(equity_curve, index=data_copy.index)

            # Calculate performance metrics
            if len(equity_curve) > 1:
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
            if len(trades) >= 2:
                trade_returns = []
                for i in range(0, len(trades) - 1, 2):
                    if i + 1 < len(trades):
                        entry = trades[i]
                        exit_trade = trades[i + 1]
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

            print("✓ QEngine backtest completed")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except Exception as e:
            error_msg = f"QEngine backtest failed: {e}"
            print(f"✗ {error_msg}")
            result.errors.append(error_msg)

        return result
