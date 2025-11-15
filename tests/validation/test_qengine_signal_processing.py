#!/usr/bin/env python3
"""
Diagnostic test for ml4t.backtest signal processing (TASK-001 RED phase).

This test verifies that ml4t.backtest correctly:
1. Receives signals from the signal-driven strategy
2. Converts signals to orders
3. Submits orders to the broker
4. Fills orders at the next bar
5. Extracts trades successfully

Currently EXPECTED TO FAIL with 0 orders placed.
"""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import pytz

# Add fixtures to path
sys.path.insert(0, str(Path(__file__).parent / 'fixtures'))
sys.path.insert(0, str(Path(__file__).parent / 'scenarios'))

from market_data import get_ticker_data
from scenario_001_simple_market_orders import Signal


def test_ml4t.backtest_executes_market_orders():
    """
    RED: Test that ml4t.backtest executes BUY/SELL market orders and generates trades.

    EXPECTED RESULT (currently failing):
    - 4 signals → 4 orders submitted → 4 fills → 2 complete trades

    CURRENT RESULT:
    - 0 trades extracted (investigation needed)

    This is the TDD RED phase for TASK-001.
    """
    # Import ml4t.backtest components
    from ml4t.backtest.engine import BacktestEngine
    from ml4t.backtest.strategy.base import Strategy
    from ml4t.backtest.execution.broker import SimulationBroker
    from ml4t.backtest.execution.commission import PercentageCommission
    from ml4t.backtest.execution.order import Order
    from ml4t.backtest.core.types import OrderType, OrderSide, EventType, MarketDataType
    from ml4t.backtest.core.event import MarketEvent
    from ml4t.backtest.data.feed import DataFeed

    # Get real AAPL 2017 data
    data = get_ticker_data(
        ticker='AAPL',
        start_date='2017-01-01',
        end_date='2017-12-31',
        use_adjusted=False
    )

    # Define 4 signals with UTC timezone (2 BUY + 2 SELL = 2 complete trades expected)
    # CRITICAL: Signals must be timezone-aware to match data timestamps
    utc = pytz.UTC
    signals = [
        Signal(timestamp=datetime(2017, 2, 6, tzinfo=utc), asset='AAPL', action='BUY', quantity=100),
        Signal(timestamp=datetime(2017, 4, 17, tzinfo=utc), asset='AAPL', action='SELL', quantity=100),
        Signal(timestamp=datetime(2017, 7, 17, tzinfo=utc), asset='AAPL', action='BUY', quantity=100),
        Signal(timestamp=datetime(2017, 12, 18, tzinfo=utc), asset='AAPL', action='SELL', quantity=100),
    ]

    # Verify signals are in dataset
    signal_dates = [sig.timestamp for sig in signals]
    data_dates = data['timestamp'].to_list()
    for sig_date in signal_dates:
        assert sig_date in data_dates, f"Signal date {sig_date} not in data"

    # Create simple data feed from DataFrame
    class SimpleDataFeed(DataFeed):
        """In-memory data feed from Polars DataFrame."""

        def __init__(self, df: pl.DataFrame):
            self.df = df.sort('timestamp')
            self.index = 0
            self._exhausted = False
            self.events_generated = 0

        def get_next_event(self) -> MarketEvent | None:
            if self.index >= len(self.df):
                self._exhausted = True
                return None

            row = self.df[self.index]
            self.index += 1
            self.events_generated += 1

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
            if self.index >= len(self.df):
                return None
            return self.df[self.index]['timestamp'][0]

        def reset(self) -> None:
            self.index = 0
            self._exhausted = False
            self.events_generated = 0

        def seek(self, timestamp: datetime) -> None:
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

    # Create signal-driven strategy with VERBOSE LOGGING
    class DiagnosticSignalStrategy(Strategy):
        """Execute pre-defined signals with detailed logging."""

        def __init__(self, signals: list):
            super().__init__("DiagnosticSignalStrategy")
            self.signals = {sig.timestamp: sig for sig in signals}
            self.positions = {}
            self.orders_submitted = []
            self.events_received = []
            print(f"[DIAGNOSTIC] Strategy initialized with {len(signals)} signals")
            print(f"[DIAGNOSTIC] Signal timestamps: {sorted(self.signals.keys())}")

        def on_start(self, portfolio, event_bus):
            self.portfolio = portfolio
            self.event_bus = event_bus
            print(f"[DIAGNOSTIC] Strategy on_start called")

        def on_event(self, event):
            self.events_received.append(event)

            if event.event_type != EventType.MARKET:
                return

            print(f"[DIAGNOSTIC] Market event at {event.timestamp}")

            # Check if we have a signal for this timestamp
            if event.timestamp in self.signals:
                signal = self.signals[event.timestamp]
                print(f"[DIAGNOSTIC] ✅ FOUND SIGNAL at {event.timestamp}: {signal.action} {signal.quantity} {signal.asset}")

                # Determine order side
                if signal.action == 'BUY':
                    side = OrderSide.BUY
                elif signal.action == 'SELL':
                    side = OrderSide.SELL
                else:
                    print(f"[DIAGNOSTIC] ❌ Unknown action: {signal.action}")
                    return

                # Create order
                order = Order(
                    asset_id=signal.asset,
                    order_type=OrderType.MARKET,
                    side=side,
                    quantity=signal.quantity,
                )

                print(f"[DIAGNOSTIC] Created order: {order}")

                # Submit to broker
                if hasattr(self, 'broker') and self.broker:
                    print(f"[DIAGNOSTIC] Submitting order to broker...")
                    self.broker.submit_order(order)
                    self.orders_submitted.append(order)
                    print(f"[DIAGNOSTIC] ✅ Order submitted (total: {len(self.orders_submitted)})")

                    # Track position
                    if signal.action == 'BUY':
                        self.positions[signal.asset] = self.positions.get(signal.asset, 0) + signal.quantity
                    elif signal.action == 'SELL':
                        self.positions[signal.asset] = self.positions.get(signal.asset, 0) - signal.quantity
                    print(f"[DIAGNOSTIC] Position tracking: {self.positions}")
                else:
                    print(f"[DIAGNOSTIC] ❌ No broker available! hasattr: {hasattr(self, 'broker')}, broker: {getattr(self, 'broker', None)}")

    # Set up backtest with diagnostic strategy
    data_feed = SimpleDataFeed(data)
    strategy = DiagnosticSignalStrategy(signals)
    commission_model = PercentageCommission(rate=0.001)  # 0.1% commission
    broker = SimulationBroker(
        initial_cash=100000.0,
        commission_model=commission_model,
    )
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        broker=broker,
        initial_capital=100000.0,
    )

    # CRITICAL: Link strategy to broker (this may be missing!)
    strategy.broker = broker
    print(f"[DIAGNOSTIC] Strategy.broker linked: {strategy.broker}")

    # Run backtest
    print(f"\n{'=' * 80}")
    print("RUNNING DIAGNOSTIC BACKTEST")
    print(f"{'=' * 80}\n")

    results = engine.run()

    print(f"\n{'=' * 80}")
    print("BACKTEST COMPLETE - DIAGNOSTICS")
    print(f"{'=' * 80}\n")

    print(f"Events processed: {results['events_processed']}")
    print(f"Market events generated: {data_feed.events_generated}")
    print(f"Strategy events received: {len(strategy.events_received)}")
    print(f"Orders submitted: {len(strategy.orders_submitted)}")

    # Check trades
    trades_df = results.get('trades')
    if trades_df is not None:
        print(f"Trades DataFrame shape: {trades_df.shape}")
        print(f"Trades count: {len(trades_df)}")
        if not trades_df.is_empty():
            print("\nTrades:")
            print(trades_df)
    else:
        print("❌ No trades DataFrame in results")

    # Get broker trades directly
    broker_trades = broker.get_trades()
    print(f"\nBroker trades directly: {len(broker_trades) if broker_trades is not None else 0}")
    if broker_trades is not None and not broker_trades.is_empty():
        print(broker_trades)

    # === ASSERTIONS (RED PHASE - EXPECTED TO FAIL) ===

    # Assert 1: Strategy received all market events
    market_events = [e for e in strategy.events_received if e.event_type == EventType.MARKET]
    assert len(market_events) == len(data), \
        f"Expected {len(data)} market events, got {len(market_events)}"

    # Assert 2: Strategy submitted 4 orders (THIS WILL FAIL IF BUG EXISTS)
    assert len(strategy.orders_submitted) >= 4, \
        f"Expected 4 orders submitted, got {len(strategy.orders_submitted)}"

    # Assert 3: Broker has filled orders
    assert broker_trades is not None, "Broker trades DataFrame is None"
    assert not broker_trades.is_empty(), "Broker trades DataFrame is empty (0 fills)"

    # Assert 4: At least 4 fills (BUY + SELL for 2 trades)
    assert len(broker_trades) >= 4, \
        f"Expected at least 4 fills, got {len(broker_trades)}"

    # Assert 5: Results contain trades
    assert trades_df is not None and not trades_df.is_empty(), \
        "Results trades DataFrame is empty"

    print("\n✅ ALL ASSERTIONS PASSED (GREEN PHASE ACHIEVED)")


if __name__ == "__main__":
    # Run test directly for debugging
    test_ml4t.backtest_executes_market_orders()
