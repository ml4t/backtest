"""
Zipline-Reloaded Framework Adapter for Cross-Framework Validation

Uses direct data.history() in handle_data() - no Pipeline complexity needed.
"""

import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseFrameworkAdapter, TradeRecord, ValidationResult


class ZiplineAdapter(BaseFrameworkAdapter):
    """Adapter for Zipline-Reloaded backtesting framework.

    Uses direct data access in handle_data() instead of Pipeline API
    for simplicity and reliability.
    """

    def __init__(self):
        super().__init__("Zipline-Reloaded")
        self._check_zipline_available()

    def _check_zipline_available(self):
        """Check if Zipline is importable."""
        try:
            import zipline
            self.zipline_version = zipline.__version__
            print(f"✓ Zipline-Reloaded {self.zipline_version} available")
        except ImportError:
            self.zipline_version = None
            print("⚠ Zipline-Reloaded not available")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest using Zipline-Reloaded."""

        result = ValidationResult(
            framework=self.framework_name,
            strategy=strategy_params.get("name", "Unknown"),
            initial_capital=initial_capital,
        )

        try:
            from zipline import run_algorithm
            from zipline.api import order_target_percent, record, set_commission, set_slippage, symbol
            from zipline.finance import commission, slippage

            tracemalloc.start()
            start_time = time.time()

            strategy_name = strategy_params.get("name", "Unknown")

            # Only MA Crossover supported for now
            if strategy_name != "MovingAverageCrossover":
                result.errors.append(f"Strategy {strategy_name} not implemented for Zipline")
                return result

            # Accept both 'long_window' and 'slow_window' for compatibility
            short_window = strategy_params.get("short_window", 20)
            long_window = strategy_params.get("long_window") or strategy_params.get("slow_window", 50)

            # Track trades and signals
            trades_list = []
            signal_log = []

            def initialize(context):
                context.asset = symbol('AAPL')
                context.short_window = short_window
                context.long_window = long_window
                context.trades = []

                set_commission(commission.PerShare(cost=0.0))
                set_slippage(slippage.FixedSlippage(spread=0.0))

            def handle_data(context, data):
                # Get historical data using data.history()
                history = data.history(
                    context.asset,
                    'close',
                    context.long_window + 1,
                    '1d'
                )

                if len(history) < context.long_window:
                    return

                # Calculate MAs (current values)
                ma_short = history[-context.short_window:].mean()
                ma_long = history[-context.long_window:].mean()  # FIX: Only use last long_window days

                # Previous MAs for crossover detection
                prev_history = history[:-1]  # Exclude today
                prev_ma_short = prev_history[-context.short_window:].mean()
                prev_ma_long = prev_history[-context.long_window:].mean()  # FIX: Only use last long_window days

                current_price = data.current(context.asset, 'close')
                positions = context.portfolio.positions[context.asset].amount

                # Detect crossovers with explicit boolean logic
                golden_cross = (prev_ma_short <= prev_ma_long) and (ma_short > ma_long)
                death_cross = (prev_ma_short > prev_ma_long) and (ma_short <= ma_long)

                # Log signals for debugging
                if golden_cross or death_cross:
                    signal_log.append({
                        'date': data.current_dt,
                        'type': 'GOLDEN' if golden_cross else 'DEATH',
                        'price': current_price,
                        'positions': positions,
                        'executed': False
                    })

                # Execute trades
                if golden_cross:
                    # Golden cross
                    if positions == 0:
                        order_target_percent(context.asset, 1.0)
                        context.trades.append({
                            'date': data.current_dt,
                            'action': 'BUY',
                            'price': current_price
                        })
                        if signal_log:
                            signal_log[-1]['executed'] = True
                elif death_cross:
                    # Death cross
                    if positions > 0:
                        order_target_percent(context.asset, 0.0)
                        context.trades.append({
                            'date': data.current_dt,
                            'action': 'SELL',
                            'price': current_price
                        })
                        if signal_log:
                            signal_log[-1]['executed'] = True

                record(price=current_price, ma_short=ma_short, ma_long=ma_long)

            def analyze(context, perf):
                nonlocal trades_list, signal_log
                trades_list = context.trades

                # Debug output
                print(f"  Signals detected: {len(signal_log)}")
                print(f"  Trades executed: {len(trades_list)}")
                for sig in signal_log:
                    print(f"    {sig['date'].date()}: {sig['type']} @ ${sig['price']:.2f}, "
                          f"pos={sig['positions']:.0f}, executed={sig['executed']}")

            # Determine date range - use naive dates to avoid timezone issues
            start_date = data.index[0].tz_localize(None) if data.index.tz else data.index[0]
            end_date = data.index[-1].tz_localize(None) if data.index.tz else data.index[-1]

            print(f"Running Zipline from {start_date} to {end_date}")

            # Run algorithm
            perf = run_algorithm(
                start=start_date,
                end=end_date,
                initialize=initialize,
                handle_data=handle_data,
                analyze=analyze,
                capital_base=initial_capital,
                bundle='quandl'
            )

            # Extract results
            result.final_value = perf['portfolio_value'].iloc[-1]
            result.total_return = (result.final_value / initial_capital - 1) * 100
            result.num_trades = len(trades_list)

            # Convert trades
            for trade_info in trades_list:
                trade = TradeRecord(
                    timestamp=pd.to_datetime(trade_info['date']),
                    action=trade_info['action'],
                    quantity=0,
                    price=trade_info['price'],
                    value=0
                )
                result.trades.append(trade)

            # Performance metrics
            if 'returns' in perf.columns:
                returns = perf['returns']
                result.daily_returns = returns

                if len(returns) > 0 and returns.std() > 0:
                    result.sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

                # Max drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                result.max_drawdown = drawdown.min() * 100

            result.equity_curve = perf['portfolio_value']

            print("✓ Zipline backtest completed")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Num Trades: {result.num_trades}")

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

        except ImportError as e:
            error_msg = f"Zipline not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Zipline backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            result.errors.append(error_msg)
            result.errors.append(traceback.format_exc())

        return result
