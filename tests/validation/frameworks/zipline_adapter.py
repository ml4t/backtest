"""
Zipline-Reloaded Framework Adapter for Cross-Framework Validation

Uses direct data.history() in handle_data() - no Pipeline complexity needed.
"""

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseFrameworkAdapter, FrameworkConfig, Signal, TradeRecord, ValidationResult

# Register custom test_data bundle
sys.path.insert(0, str(Path(__file__).parent.parent))
from bundles.test_data_bundle import *  # noqa: E402, F403


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
                bundle='test_data'
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
            result.errors.append(error_msg)
            # Note: traceback.format_exc() can crash on Python 3.12 with
            # "RuntimeError: generator raised StopIteration" when formatting
            # certain Zipline exception chains. Use str(e) instead.
            try:
                import traceback
                result.errors.append(traceback.format_exc())
            except RuntimeError:
                # Python 3.12 traceback bug - fall back to simple error
                result.errors.append(f"Exception type: {type(e).__name__}")

        return result

    def run_with_signals(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        config: FrameworkConfig | None = None,
    ) -> ValidationResult:
        """
        Run backtest with pre-computed signals using test_data bundle.

        Args:
            data: OHLCV data - either:
                  - Single DataFrame with DatetimeIndex (single-asset)
                  - Dict mapping symbol -> DataFrame (multi-asset)
            signals: DataFrame with either:
                  - 'entry' and 'exit' boolean columns (single-asset)
                  - 'timestamp', 'symbol', 'signal' columns (multi-asset)
            config: FrameworkConfig for execution parameters

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

        result = ValidationResult(
            framework=self.framework_name,
            strategy="SignalBased",
            initial_capital=config.initial_capital,
        )

        try:
            from zipline import run_algorithm
            from zipline.api import order_target_percent, record, set_commission, set_slippage, symbol
            from zipline.finance import commission, slippage

            tracemalloc.start()
            start_time = time.time()

            # CRITICAL: Shift signals BACKWARD by 1 day for Zipline
            # Zipline's handle_data() is called at END of bar, orders fill NEXT bar
            # To match same-bar fills (like Backtrader COC), we place orders 1 day EARLIER
            # Example: Signal on 2020-04-07 → Shift to 2020-04-06 → Zipline fills 2020-04-07
            signals_shifted = signals.shift(-1)  # Shift backward (earlier dates)

            # Convert DataFrame signals to date-indexed dict
            signal_dict = {}
            for idx, row in signals_shifted.iterrows():
                date = idx.date() if hasattr(idx, 'date') else idx
                if pd.notna(row.get('entry')) and row['entry']:
                    if date not in signal_dict:
                        signal_dict[date] = []
                    signal_dict[date].append({'action': 'BUY'})
                if pd.notna(row.get('exit')) and row['exit']:
                    if date not in signal_dict:
                        signal_dict[date] = []
                    signal_dict[date].append({'action': 'SELL'})

            num_entry_signals = signals['entry'].sum()
            num_exit_signals = signals['exit'].sum()

            def initialize(context):
                context.asset = symbol('AAPL')
                context.signal_dict = signal_dict
                context.target_position = 0.0

                # NOTE: Zipline's leverage control is incompatible with signal-based trading
                # Using set_max_leverage(1.0) causes violations even with 99% target
                # This is a known Zipline limitation - it allows slight margin usage

                # Match config settings (use PerDollar for percentage-based commissions)
                # config.commission_pct is in percent (e.g., 0.1 for 0.1%), convert to fraction
                if config.commission_pct > 0:
                    set_commission(commission.PerDollar(cost=config.commission_pct / 100.0))
                else:
                    set_commission(commission.PerShare(cost=0.0))

                # config.slippage_pct is in percent (e.g., 0.05 for 0.05%), convert to fraction
                if config.slippage_pct > 0:
                    set_slippage(slippage.FixedSlippage(spread=config.slippage_pct / 100.0))
                else:
                    set_slippage(slippage.FixedSlippage(spread=0.0))

            def handle_data(context, data):
                current_date = data.current_dt.date()

                # Check if we have signals for today
                if current_date in context.signal_dict:
                    # Check if asset exists on this date
                    if not data.can_trade(context.asset):
                        print(f"  [Zipline DEBUG] {current_date}: Signal exists but asset not tradable")
                        return

                    current_price = data.current(context.asset, 'close')

                    for signal in context.signal_dict[current_date]:
                        action = signal["action"]

                        if action == "BUY" and context.target_position == 0.0:
                            # Use 99% to avoid leverage constraint violations
                            # (100% can trigger violation due to rounding/slippage)
                            order_target_percent(context.asset, 0.99)
                            context.target_position = 0.99
                        elif action == "SELL" and context.target_position > 0.0:
                            order_target_percent(context.asset, 0.0)
                            context.target_position = 0.0

                # Always record price if available
                if data.can_trade(context.asset):
                    current_price = data.current(context.asset, 'close')
                    record(price=current_price)

            # Determine date range
            start_date = data.index[0].tz_localize(None) if data.index.tz else data.index[0]
            end_date = data.index[-1].tz_localize(None) if data.index.tz else data.index[-1]

            print(f"Zipline signal-based: {num_entry_signals} entries, {num_exit_signals} exits from {start_date} to {end_date}")
            print(f"  Note: Using custom test_data bundle (same OHLCV data as other frameworks)")

            # Run algorithm with custom test_data bundle
            perf = run_algorithm(
                start=start_date,
                end=end_date,
                initialize=initialize,
                handle_data=handle_data,
                capital_base=config.initial_capital,
                bundle='test_data'
            )

            # Extract results
            result.final_value = perf['portfolio_value'].iloc[-1]
            result.total_return = (result.final_value / config.initial_capital - 1) * 100

            # Extract actual transactions using pyfolio
            try:
                import pyfolio as pf
                returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)

                # Convert transactions DataFrame to TradeRecord list
                result.trades = []
                for idx, txn in transactions.iterrows():
                    # Determine action based on amount sign
                    action = 'BUY' if txn['amount'] > 0 else 'SELL'
                    quantity = abs(txn['amount'])

                    result.trades.append(TradeRecord(
                        timestamp=pd.Timestamp(idx),
                        action=action,
                        quantity=quantity,
                        price=txn['price'],
                        value=abs(txn['amount'] * txn['price']),
                        commission=txn.get('commission', 0.0),
                    ))

                result.num_trades = len(result.trades)

            except Exception as e:
                print(f"Warning: Could not extract transactions from Zipline: {e}")
                # Fallback: no trades extracted
                result.num_trades = 0
                result.trades = []

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ Zipline signal-based backtest completed")
            print(f"  Entry Signals: {num_entry_signals}, Exit Signals: {num_exit_signals}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")

        except ImportError as e:
            error_msg = f"Zipline not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Zipline signal-based backtest failed: {e}"
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
        Run backtest with multi-asset rotation signals using Zipline.

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
            from zipline import run_algorithm
            from zipline.api import order_target_percent, record, set_commission, set_slippage, symbol
            from zipline.finance import commission, slippage
            from zipline.data.bundles import ingest, load

            tracemalloc.start()
            start_time = time.time()

            # Set multi-asset data in bundle and re-ingest
            from bundles.test_data_bundle import set_multi_asset_data
            from zipline.utils.calendar_utils import get_calendar

            # Get NYSE trading calendar
            nyse_calendar = get_calendar('NYSE')

            # Prepare data with proper column names and filter to trading days
            prepared_data = {}
            for sym, df in data.items():
                df_copy = df.copy()
                df_copy.columns = df_copy.columns.str.lower()
                # Ensure timezone
                if df_copy.index.tz is None:
                    df_copy.index = df_copy.index.tz_localize('UTC')
                else:
                    df_copy.index = df_copy.index.tz_convert('UTC')

                # Filter to NYSE trading days only
                trading_days = nyse_calendar.sessions_in_range(
                    df_copy.index[0],
                    df_copy.index[-1]
                )
                df_filtered = df_copy[df_copy.index.isin(trading_days)]

                if len(df_filtered) == 0:
                    raise ValueError(f"No trading days found for {sym} in date range")

                prepared_data[sym] = df_filtered

            set_multi_asset_data(prepared_data)

            # Re-ingest bundle with multi-asset data
            print("  Re-ingesting bundle with multi-asset data...")
            ingest('test_data', show_progress=False)

            # Convert signals from long format to dict for fast lookup
            # Filter signals to trading days only
            # signal_dict[date][symbol] = action ("BUY" or "SELL")
            signal_dict = {}
            for _, row in signals.iterrows():
                # Shift signals backward by 1 day for Zipline
                # Zipline fills next bar, so we need to place orders 1 day early
                timestamp = pd.Timestamp(row['timestamp'])

                # Skip if not a trading day
                if timestamp not in trading_days:
                    continue

                date = timestamp.date()
                symbol_name = row['symbol']
                signal_value = row['signal']

                if date not in signal_dict:
                    signal_dict[date] = {}

                if signal_value == 1:  # BUY
                    signal_dict[date][symbol_name] = "BUY"
                elif signal_value == -1:  # SELL
                    signal_dict[date][symbol_name] = "SELL"

            # Shift signals backward by 1 day to account for Zipline's next-bar fill
            shifted_signal_dict = {}
            for date_obj, symbol_actions in signal_dict.items():
                # Convert to pandas Timestamp, shift back 1 day
                shifted_date = (pd.Timestamp(date_obj) - pd.Timedelta(days=1)).date()
                shifted_signal_dict[shifted_date] = symbol_actions

            signal_dict = shifted_signal_dict

            num_entry_signals = (signals['signal'] == 1).sum()
            num_exit_signals = (signals['signal'] == -1).sum()

            print(f"\nZipline multi-asset setup:")
            print(f"  Assets: {len(data)}")
            print(f"  Entry signals: {num_entry_signals}")
            print(f"  Exit signals: {num_exit_signals}")

            # Get list of all symbols
            symbols_list = sorted(data.keys())

            def initialize(context):
                # Create symbol objects for all assets
                context.assets = {sym: symbol(sym) for sym in symbols_list}
                context.signal_dict = signal_dict
                context.target_pct = 0.20  # 20% per position (5 positions max)
                context.positions = {sym: 0.0 for sym in symbols_list}  # Track target positions

                # Configure commission and slippage
                if config.commission_pct > 0:
                    set_commission(commission.PerDollar(cost=config.commission_pct / 100.0))
                else:
                    set_commission(commission.PerShare(cost=0.0))

                if config.slippage_pct > 0:
                    set_slippage(slippage.FixedSlippage(spread=config.slippage_pct / 100.0))
                else:
                    set_slippage(slippage.FixedSlippage(spread=0.0))

            def handle_data(context, data):
                current_date = data.current_dt.date()

                # Check if we have signals for today
                if current_date not in context.signal_dict:
                    return

                day_signals = context.signal_dict[current_date]

                # Process signals for each symbol
                for symbol_name, action in day_signals.items():
                    if symbol_name not in context.assets:
                        continue

                    asset = context.assets[symbol_name]

                    # Check if asset is tradable
                    if not data.can_trade(asset):
                        continue

                    if action == "BUY" and context.positions[symbol_name] == 0.0:
                        # Enter position: 20% of portfolio
                        # Use 19.8% to avoid leverage violations
                        order_target_percent(asset, 0.198)
                        context.positions[symbol_name] = 0.198

                    elif action == "SELL" and context.positions[symbol_name] > 0.0:
                        # Exit position
                        order_target_percent(asset, 0.0)
                        context.positions[symbol_name] = 0.0

            # Determine date range from data
            # Get min/max timestamps across all symbols
            all_timestamps = []
            for df in data.values():
                all_timestamps.extend(df.index.tolist())

            start_date = min(all_timestamps)
            end_date = max(all_timestamps)

            # Remove timezone if present
            if hasattr(start_date, 'tz_localize'):
                start_date = start_date.tz_localize(None) if start_date.tz else start_date
            if hasattr(end_date, 'tz_localize'):
                end_date = end_date.tz_localize(None) if end_date.tz else end_date

            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Using custom test_data bundle with {len(symbols_list)} symbols")

            # Run algorithm with custom test_data bundle
            perf = run_algorithm(
                start=start_date,
                end=end_date,
                initialize=initialize,
                handle_data=handle_data,
                capital_base=config.initial_capital,
                bundle='test_data'
            )

            # Extract results
            result.final_value = perf['portfolio_value'].iloc[-1]
            result.total_return = (result.final_value / config.initial_capital - 1) * 100

            # Extract actual transactions using pyfolio
            try:
                import pyfolio as pf
                returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)

                # Convert transactions DataFrame to TradeRecord list
                result.trades = []
                for idx, txn in transactions.iterrows():
                    # Get symbol from transaction
                    # Note: pyfolio transactions may have symbol in different format
                    sym = str(txn.get('symbol', 'UNKNOWN'))

                    # Determine action based on amount sign
                    action = 'BUY' if txn['amount'] > 0 else 'SELL'
                    quantity = abs(txn['amount'])

                    result.trades.append(TradeRecord(
                        timestamp=pd.Timestamp(idx),
                        action=action,
                        quantity=quantity,
                        price=txn['price'],
                        value=abs(txn['amount'] * txn['price']),
                        commission=txn.get('commission', 0.0),
                    ))

                result.num_trades = len(result.trades)

            except Exception as e:
                print(f"  Warning: Could not extract transactions from Zipline: {e}")
                result.num_trades = 0
                result.trades = []

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print("✓ Zipline multi-asset backtest completed")
            print(f"  Entry Signals: {num_entry_signals}, Exit Signals: {num_exit_signals}")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except ImportError as e:
            error_msg = f"Zipline not available: {e}"
            print(f"⚠ {error_msg}")
            result.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Zipline multi-asset backtest failed: {e}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            result.errors.append(error_msg)

        return result
