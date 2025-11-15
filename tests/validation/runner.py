#!/usr/bin/env python3
"""
Validation Test Runner

Executes scenarios across multiple backtesting platforms and compares results
using trade-by-trade comparison.

Usage:
    python runner.py --scenario 001                    # Run specific scenario
    python runner.py --all                             # Run all scenarios
    python runner.py --platforms ml4t.backtest,vectorbt      # Specific platforms only
    python runner.py --scenario 001 --report detailed  # Generate detailed report
"""

import argparse
import importlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import polars as pl

# Import extractors, matcher, and reporter
from extractors import (
    extract_backtest_trades,
    extract_vectorbt_trades,
    extract_backtrader_trades,
    extract_zipline_trades,
)
from comparison import match_trades, generate_trade_report, generate_summary_report


@dataclass
class PlatformResult:
    """Results from one platform for one scenario."""

    platform: str
    scenario: str
    data: pl.DataFrame  # Market data used
    raw_results: Any  # Platform-specific results
    execution_time: float
    errors: list[str]


class ScenarioRunner:
    """Runs a scenario across multiple platforms."""

    def __init__(self, scenario_module):
        """
        Initialize with a scenario module.

        Args:
            scenario_module: Imported scenario module (e.g., scenario_001)
        """
        self.scenario = scenario_module
        # Find the Scenario class in the module
        scenario_classes = [obj for name, obj in vars(scenario_module).items()
                           if isinstance(obj, type) and name.startswith('Scenario')]
        if not scenario_classes:
            raise ValueError(f"No Scenario class found in module {scenario_module}")
        self.scenario_class = scenario_classes[0]

    def run_ml4t.backtest(self) -> PlatformResult:
        """Run scenario on ml4t.backtest."""
        print(f"  🔧 Running ml4t.backtest...")
        start = datetime.now()

        try:
            # Import ml4t.backtest components
            from ml4t.backtest.engine import BacktestEngine
            from ml4t.backtest.strategy.base import Strategy
            from ml4t.backtest.execution.broker import SimulationBroker
            from ml4t.backtest.execution.commission import PercentageCommission
            from ml4t.backtest.execution.order import Order
            from ml4t.backtest.core.types import OrderType, OrderSide, EventType, MarketDataType
            from ml4t.backtest.core.event import MarketEvent
            from ml4t.backtest.data.feed import DataFeed

            # Get data and signals
            data = self.scenario_class.get_data()
            signals = self.scenario_class.signals
            config = self.scenario_class.config

            # Create simple data feed from DataFrame
            class SimpleDataFeed(DataFeed):
                """In-memory data feed from Polars DataFrame."""

                def __init__(self, df: pl.DataFrame):
                    self.df = df.sort('timestamp')
                    self.index = 0
                    self._exhausted = False

                def get_next_event(self) -> MarketEvent | None:
                    if self.index >= len(self.df):
                        self._exhausted = True
                        return None

                    row = self.df[self.index]
                    self.index += 1

                    return MarketEvent(
                        timestamp=row['timestamp'][0],
                        asset_id=row['symbol'][0],
                        data_type=MarketDataType.BAR,
                        open=row['open'][0],
                        high=row['high'][0],
                        low=row['low'][0],
                        close=row['close'][0],
                        volume=row['volume'][0],
                    )

                def peek_next_timestamp(self) -> datetime | None:
                    """Peek at next timestamp without consuming it."""
                    if self.index >= len(self.df):
                        return None
                    return self.df[self.index]['timestamp'][0]

                def reset(self) -> None:
                    """Reset to beginning."""
                    self.index = 0
                    self._exhausted = False

                def seek(self, timestamp: datetime) -> None:
                    """Seek to specific timestamp."""
                    for i in range(len(self.df)):
                        if self.df[i]['timestamp'][0] >= timestamp:
                            self.index = i
                            self._exhausted = False
                            return
                    self.index = len(self.df)
                    self._exhausted = True

                @property
                def is_exhausted(self) -> bool:
                    return self._exhausted

            # Create signal-driven strategy
            class SignalStrategy(Strategy):
                """Execute pre-defined signals."""

                def __init__(self, signals: list):
                    super().__init__("SignalStrategy")
                    self.signals = {sig.timestamp: sig for sig in signals}
                    self.positions = {}

                def on_start(self, portfolio, event_bus):
                    self.portfolio = portfolio
                    self.event_bus = event_bus

                def on_event(self, event):
                    if event.event_type != EventType.MARKET:
                        return

                    if event.timestamp not in self.signals:
                        return

                    signal = self.signals[event.timestamp]

                    if signal.action == 'BUY':
                        side = OrderSide.BUY
                    elif signal.action == 'SELL':
                        side = OrderSide.SELL
                    else:
                        return

                    order = Order(
                        asset_id=signal.asset,
                        order_type=OrderType.MARKET,
                        side=side,
                        quantity=signal.quantity,
                    )

                    if hasattr(self, 'broker') and self.broker:
                        self.broker.submit_order(order)

                        if signal.action == 'BUY':
                            self.positions[signal.asset] = self.positions.get(signal.asset, 0) + signal.quantity
                        elif signal.action == 'SELL':
                            self.positions[signal.asset] = self.positions.get(signal.asset, 0) - signal.quantity

            # Set up backtest
            data_feed = SimpleDataFeed(data)
            strategy = SignalStrategy(signals)
            commission_model = PercentageCommission(rate=config['commission'])
            broker = SimulationBroker(
                initial_cash=config['initial_capital'],
                commission_model=commission_model,
            )
            engine = BacktestEngine(
                data_feed=data_feed,
                strategy=strategy,
                broker=broker,
                initial_capital=config['initial_capital'],
            )
            strategy.broker = broker

            # Run
            results = engine.run()
            execution_time = (datetime.now() - start).total_seconds()

            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__

            return PlatformResult(
                platform='ml4t.backtest',
                scenario=scenario_name,
                data=data,
                raw_results=results,
                execution_time=execution_time,
                errors=[],
            )

        except Exception as e:
            import traceback
            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__
            return PlatformResult(
                platform='ml4t.backtest',
                scenario=scenario_name,
                data=self.scenario_class.get_data(),
                raw_results=None,
                execution_time=(datetime.now() - start).total_seconds(),
                errors=[f"{type(e).__name__}: {e}", traceback.format_exc()],
            )

    def run_vectorbt(self) -> PlatformResult:
        """Run scenario on VectorBT."""
        print(f"  📊 Running VectorBT...")
        start = datetime.now()

        try:
            import vectorbtpro as vbt
            import pandas as pd

            # Get data and signals
            data_polars = self.scenario_class.get_data()
            data = data_polars.to_pandas().set_index('timestamp')
            signals_list = self.scenario_class.signals
            config = self.scenario_class.config

            # Create signal series
            entries = pd.Series(False, index=data.index)
            exits = pd.Series(False, index=data.index)

            for signal in signals_list:
                if signal.timestamp in data.index:
                    if signal.action == 'BUY':
                        entries.loc[signal.timestamp] = True
                    elif signal.action == 'SELL':
                        exits.loc[signal.timestamp] = True

            # Run portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                init_cash=config['initial_capital'],
                fees=config['commission'],
                slippage=config.get('slippage', 0.0),
            )

            execution_time = (datetime.now() - start).total_seconds()

            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__

            return PlatformResult(
                platform='vectorbt',
                scenario=scenario_name,
                data=data_polars,
                raw_results={'portfolio': portfolio, 'data': data},
                execution_time=execution_time,
                errors=[],
            )

        except Exception as e:
            import traceback
            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__
            return PlatformResult(
                platform='vectorbt',
                scenario=scenario_name,
                data=self.scenario_class.get_data(),
                raw_results=None,
                execution_time=(datetime.now() - start).total_seconds(),
                errors=[f"{type(e).__name__}: {e}", traceback.format_exc()],
            )

    def run_backtrader(self) -> PlatformResult:
        """Run scenario on Backtrader."""
        print(f"  📈 Running Backtrader...")
        start = datetime.now()

        try:
            import backtrader as bt

            # Get data and signals
            data_polars = self.scenario_class.get_data()
            data = data_polars.to_pandas().set_index('timestamp')
            signals_list = self.scenario_class.signals
            config = self.scenario_class.config

            # Create Backtrader data feed
            class PandasData(bt.feeds.PandasData):
                """Custom pandas data feed."""
                params = (
                    ('datetime', None),
                    ('open', 'open'),
                    ('high', 'high'),
                    ('low', 'low'),
                    ('close', 'close'),
                    ('volume', 'volume'),
                    ('openinterest', None),
                )

            # Create strategy that follows signals
            class SignalStrategy(bt.Strategy):
                """Execute pre-defined signals."""

                def __init__(self):
                    # Normalize signals dict to handle both timezone-aware and naive timestamps
                    # Store both versions to support Backtrader's timezone-naive datetimes
                    self.signals_tz = {sig.timestamp: sig for sig in signals_list}
                    self.signals_naive = {
                        sig.timestamp.replace(tzinfo=None) if sig.timestamp.tzinfo else sig.timestamp: sig
                        for sig in signals_list
                    }
                    self.trades_list = []

                def next(self):
                    # Check for signal at current bar
                    # Backtrader returns timezone-naive datetime
                    current_dt = self.datas[0].datetime.datetime(0)

                    # Try both timezone-naive and timezone-aware lookup
                    signal = self.signals_naive.get(current_dt) or self.signals_tz.get(current_dt)

                    if signal is None:
                        return

                    if signal.action == 'BUY':
                        self.buy(size=signal.quantity)
                    elif signal.action == 'SELL':
                        self.sell(size=signal.quantity)

                def notify_trade(self, trade):
                    """Collect trade information."""
                    if trade.isclosed:
                        # Calculate exit price from P&L
                        exit_price = trade.pnlcomm / trade.size + trade.price if trade.size != 0 else 0

                        self.trades_list.append({
                            'entry_time': bt.num2date(trade.dtopen),
                            'exit_time': bt.num2date(trade.dtclose),
                            'entry_price': trade.price,
                            'exit_price': exit_price,
                            'size': trade.size,
                            'pnl': trade.pnlcomm,  # Includes commission
                            'commission': trade.commission,
                        })

            # Set up cerebro
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SignalStrategy)

            # Add data
            bt_data = PandasData(dataname=data)
            cerebro.adddata(bt_data)

            # Set initial cash
            cerebro.broker.setcash(config['initial_capital'])

            # Set commission
            cerebro.broker.setcommission(commission=config['commission'])

            # Run
            strategies = cerebro.run()
            strategy = strategies[0]

            execution_time = (datetime.now() - start).total_seconds()

            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__

            return PlatformResult(
                platform='backtrader',
                scenario=scenario_name,
                data=data_polars,
                raw_results={'trades': strategy.trades_list, 'data': data},
                execution_time=execution_time,
                errors=[],
            )

        except Exception as e:
            import traceback
            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__
            return PlatformResult(
                platform='backtrader',
                scenario=scenario_name,
                data=self.scenario_class.get_data(),
                raw_results=None,
                execution_time=(datetime.now() - start).total_seconds(),
                errors=[f"{type(e).__name__}: {e}", traceback.format_exc()],
            )

    def run_zipline(self) -> PlatformResult:
        """Run scenario on Zipline-reloaded."""
        print(f"  🚀 Running Zipline...")
        start = datetime.now()

        try:
            from zipline import run_algorithm
            from zipline.api import order, symbol, set_commission
            from zipline.finance.commission import PerShare
            import pandas as pd
            from zoneinfo import ZoneInfo

            # Get data and signals
            data_polars = self.scenario_class.get_data()
            data = data_polars.to_pandas().set_index('timestamp')

            # CRITICAL: Zipline/exchange_calendars expects ZoneInfo, not datetime.timezone
            # Convert timezone to ZoneInfo('UTC') which has .key attribute
            data.index = data.index.tz_convert(ZoneInfo('UTC'))
            signals_list = self.scenario_class.signals
            config = self.scenario_class.config

            # Create signal index
            # CRITICAL: Index by date only (not datetime) because Zipline's current_dt
            # uses market close time (21:00 UTC for NYSE) but signals use midnight
            signals_by_date = {}
            for sig in signals_list:
                # Normalize to date-only for comparison
                sig_date = sig.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                if sig_date not in signals_by_date:
                    signals_by_date[sig_date] = []
                signals_by_date[sig_date].append(sig)

            # Define algorithm
            def initialize(context):
                """Initialize algorithm."""
                # Set commission (convert percentage to per-share)
                # This is approximate - Zipline uses per-share commission
                set_commission(PerShare(cost=config['commission'] * 100))  # Rough approximation

                context.signals = signals_by_date
                context.asset = symbol('AAPL')  # TODO: Make this dynamic

            def handle_data(context, data):
                """Handle daily bar."""
                current_dt = data.current_dt

                # DEBUG: Print current date and check signals
                # print(f"DEBUG handle_data: current_dt={current_dt}, type={type(current_dt)}, signals={list(context.signals.keys())[:3]}")

                # Check for signals (need to compare dates, not exact timestamps)
                current_date = current_dt.normalize()  # Strip time component

                # Check both with and without time
                signal_list = context.signals.get(current_dt) or context.signals.get(current_date)

                if not signal_list:
                    return

                # print(f"🎯 FOUND SIGNALS at {current_dt}: {[(s.action, s.quantity) for s in signal_list]}")

                for signal in signal_list:
                    if signal.action == 'BUY':
                        order(context.asset, signal.quantity)
                        # print(f"  📈 Placed BUY order: {signal.quantity} shares")
                    elif signal.action == 'SELL':
                        order(context.asset, -signal.quantity)
                        # print(f"  📉 Placed SELL order: {signal.quantity} shares")

            # Set up validation bundle
            import os
            import sys
            from pathlib import Path

            # Set ZIPLINE_ROOT to use validation bundle
            bundle_root = Path(__file__).parent / 'bundles' / '.zipline_root'
            os.environ['ZIPLINE_ROOT'] = str(bundle_root)

            # Register validation bundle
            bundle_dir = Path(__file__).parent / 'bundles'
            sys.path.insert(0, str(bundle_dir))

            from zipline.data.bundles import register as register_bundle
            from validation_ingest import validation_to_bundle
            register_bundle('validation', validation_to_bundle(), calendar_name='NYSE')

            # Run algorithm
            # CRITICAL: Zipline expects timezone-naive dates for start/end
            # but the bundle data itself must be timezone-aware
            start_date = data.index[0].tz_localize(None)
            end_date = data.index[-1].tz_localize(None)

            perf = run_algorithm(
                start=start_date,
                end=end_date,
                initialize=initialize,
                handle_data=handle_data,
                capital_base=config['initial_capital'],
                data_frequency='daily',
                bundle='validation',  # Use our custom validation bundle
            )

            execution_time = (datetime.now() - start).total_seconds()

            # DEBUG: Check if transactions exist
            # print(f"\n🔍 DEBUG: Checking Zipline performance data...")
            # print(f"  Perf columns: {list(perf.columns)[:10]}")
            # print(f"  Perf shape: {perf.shape}")
            # has_txn_col = 'transactions' in perf.columns
            # print(f"  Has 'transactions' column: {has_txn_col}")
            # if has_txn_col:
            #     txn_rows = perf[perf['transactions'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
            #     print(f"  Rows with transactions: {len(txn_rows)}")
            #     if len(txn_rows) > 0:
            #         print(f"  Sample transactions: {txn_rows['transactions'].iloc[0]}")

            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__

            return PlatformResult(
                platform='zipline',
                scenario=scenario_name,
                data=data_polars,
                raw_results={'perf': perf, 'data': data},
                execution_time=execution_time,
                errors=[],
            )

        except Exception as e:
            import traceback
            scenario_name = self.scenario_class.name if hasattr(self.scenario_class, 'name') else self.scenario.__name__
            return PlatformResult(
                platform='zipline',
                scenario=scenario_name,
                data=self.scenario_class.get_data(),
                raw_results=None,
                execution_time=(datetime.now() - start).total_seconds(),
                errors=[f"{type(e).__name__}: {e}", traceback.format_exc()],
            )

    def run(self, platforms: list[str]) -> dict[str, PlatformResult]:
        """
        Run scenario on specified platforms.

        Args:
            platforms: List of platform names ('ml4t.backtest', 'vectorbt', etc.)

        Returns:
            Dictionary mapping platform name to result
        """
        results = {}

        for platform in platforms:
            if platform == 'ml4t.backtest':
                results[platform] = self.run_ml4t.backtest()
            elif platform == 'vectorbt':
                results[platform] = self.run_vectorbt()
            elif platform == 'backtrader':
                results[platform] = self.run_backtrader()
            elif platform == 'zipline':
                results[platform] = self.run_zipline()
            else:
                print(f"  ⚠️  Platform '{platform}' not implemented yet")

        return results


def print_summary(results: dict[str, PlatformResult]):
    """Print high-level execution summary."""
    print(f"\n{'=' * 80}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"{'Platform':<15} {'Time':<10} {'Status'}")
    print("-" * 80)

    for platform, result in results.items():
        status = "✅ OK" if not result.errors else "❌ ERROR"
        print(f"{platform:<15} {result.execution_time:<9.3f}s {status}")

    # Show errors if any
    for platform, result in results.items():
        if result.errors:
            print(f"\n{platform} errors:")
            for error in result.errors[:3]:
                print(f"  {error}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run validation scenarios')
    parser.add_argument('--scenario', type=str, help='Scenario number (e.g., 001)')
    parser.add_argument('--all', action='store_true', help='Run all scenarios')
    parser.add_argument(
        '--platforms',
        type=str,
        default='ml4t.backtest,vectorbt',
        help='Comma-separated list of platforms (default: ml4t.backtest,vectorbt)'
    )
    parser.add_argument(
        '--report',
        type=str,
        choices=['summary', 'detailed', 'both'],
        default='both',
        help='Report type: summary, detailed, or both (default: both)'
    )

    args = parser.parse_args()

    platforms = args.platforms.split(',')

    if args.scenario:
        # Run single scenario
        scenario_name = f"scenario_{args.scenario}_*"
        scenario_files = list(Path('scenarios').glob(scenario_name + '.py'))

        if not scenario_files:
            print(f"❌ Scenario {args.scenario} not found")
            return

        scenario_file = scenario_files[0]
        scenario_module_name = scenario_file.stem

        print(f"\n{'=' * 80}")
        print(f"Running {scenario_module_name}")
        print(f"Platforms: {', '.join(platforms)}")
        print(f"{'=' * 80}")

        # Import scenario
        scenario_module = importlib.import_module(f"scenarios.{scenario_module_name}")

        # Run
        runner = ScenarioRunner(scenario_module)
        results = runner.run(platforms)

        # Print execution summary
        print_summary(results)

        # Extract trades using platform-specific extractors
        print(f"\n{'=' * 80}")
        print("EXTRACTING TRADES")
        print(f"{'=' * 80}\n")

        trades_by_platform = {}

        for platform, result in results.items():
            if result.errors or not result.raw_results:
                print(f"  ⚠️  Skipping {platform} (had errors)")
                continue

            print(f"  🔍 Extracting {platform} trades...")

            if platform == 'ml4t.backtest':
                trades = extract_backtest_trades(result.raw_results, result.data)
                trades_by_platform[platform] = trades
                print(f"     Found {len(trades)} trades")

            elif platform == 'vectorbt':
                portfolio = result.raw_results['portfolio']
                data_pandas = result.raw_results['data']
                trades = extract_vectorbt_trades(portfolio, data_pandas)
                trades_by_platform[platform] = trades
                print(f"     Found {len(trades)} trades")

            elif platform == 'backtrader':
                trades_list = result.raw_results['trades']
                data_pandas = result.raw_results['data']
                trades = extract_backtrader_trades(trades_list, data_pandas)
                trades_by_platform[platform] = trades
                print(f"     Found {len(trades)} trades")

            elif platform == 'zipline':
                perf = result.raw_results['perf']
                data_pandas = result.raw_results['data']
                trades = extract_zipline_trades(perf, data_pandas)
                trades_by_platform[platform] = trades
                print(f"     Found {len(trades)} trades")

        # Match trades across platforms
        print(f"\n{'=' * 80}")
        print("MATCHING TRADES")
        print(f"{'=' * 80}\n")

        if len(trades_by_platform) < 2:
            print("⚠️  Need at least 2 platforms to compare")
        else:
            matches = match_trades(trades_by_platform, timestamp_tolerance_seconds=60)
            print(f"  ✅ Matched {len(matches)} trade groups\n")

            # Generate reports
            if args.report in ['summary', 'both']:
                summary_report = generate_summary_report(matches)
                print(summary_report)

            if args.report in ['detailed', 'both']:
                print(f"\n{'=' * 80}")
                print("DETAILED TRADE-BY-TRADE COMPARISON")
                print(f"{'=' * 80}\n")

                for match in matches:
                    trade_report = generate_trade_report(match)
                    print(trade_report)

    elif args.all:
        print("TODO: Implement --all")

    else:
        print("Please specify --scenario or --all")


if __name__ == "__main__":
    main()
