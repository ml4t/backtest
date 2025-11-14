"""
Diagnostic Test with Detailed Logging

This version adds extensive logging to trace the phantom trade bug.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import load_real_crypto_data, BacktestConfig
import pandas as pd
import polars as pl


def test_diagnostic_with_logging():
    """Diagnostic: Trace exact execution flow"""

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST: Detailed Logging")
    print("=" * 80)

    # 1. Load minimal data
    print("\n1️⃣  Loading 200 bars of BTC data...")
    ohlcv = load_real_crypto_data(symbol="BTC", data_type="spot", n_bars=200)
    print(f"   ✅ Loaded {len(ohlcv)} bars")

    # 2. Create EXPLICIT signals at specific bars
    print("\n2️⃣  Creating explicit signals...")
    entries = pd.Series([False] * 200, index=ohlcv.index)
    exits = pd.Series([False] * 200, index=ohlcv.index)

    # Explicit entry/exit pairs:
    # Trade 1: Enter at bar 10, exit at bar 20
    # Trade 2: Enter at bar 60, exit at bar 70
    # Trade 3: Enter at bar 110, exit at bar 120
    entries.iloc[10] = True
    exits.iloc[20] = True
    entries.iloc[60] = True
    exits.iloc[70] = True
    entries.iloc[110] = True
    exits.iloc[120] = True

    print(f"   ✅ Entry signals at bars: 10, 60, 110")
    print(f"   ✅ Exit signals at bars: 20, 70, 120")
    print(f"   Expected: 3 trades")

    # Verify signals at problem bars
    print(f"\n   Signal verification at key bars:")
    for bar in [20, 21, 70, 71, 120, 121]:
        if bar < len(entries):
            print(f"     Bar {bar}: entry={entries.iloc[bar]}, exit={exits.iloc[bar]}")

    # 3. Run qengine WITH FEES and logging
    print("\n3️⃣  Running qengine with detailed logging...")
    config = BacktestConfig(
        initial_cash=100000.0,
        fees=0.001,  # 0.1%
        slippage=0.0,
        order_type='market',
    )

    # Import qengine components
    from qengine.engine import BacktestEngine
    from qengine.core.event import MarketEvent
    from qengine.execution.broker import SimulationBroker
    from qengine.execution.commission import PercentageCommission
    from qengine.data.feed import DataFeed
    from qengine.strategy.base import Strategy
    from qengine.core.types import MarketDataType

    # Create custom strategy with logging
    class LoggingSignalStrategy(Strategy):
        def __init__(self, entries_ser, exits_ser):
            super().__init__()
            self.entries = entries_ser
            self.exits = exits_ser
            self.bar_idx = 0
            self._order_counter = 0
            self.trade_count = 0
            print("\n   📊 Strategy initialized")

        def on_start(self, portfolio, event_bus):
            super().on_start(portfolio, event_bus)
            self.portfolio = portfolio
            self.event_bus = event_bus

        def on_event(self, event):
            from qengine.core.event import FillEvent

            if isinstance(event, MarketEvent):
                self.on_market_event(event)
            elif isinstance(event, FillEvent):
                super().on_fill_event(event)

        def on_market_event(self, event: MarketEvent):
            from qengine.core.event import OrderEvent
            from qengine.core.types import OrderSide, OrderType

            # Get signal for this bar
            entry_signal = self.entries.iloc[self.bar_idx] if self.bar_idx < len(self.entries) else False
            exit_signal = self.exits.iloc[self.bar_idx] if self.bar_idx < len(self.exits) else False

            # Check current position
            current_qty = self._positions.get(event.asset_id, 0.0)

            # Only log bars with signals or positions
            if entry_signal or exit_signal or current_qty != 0:
                print(f"\n   Bar {self.bar_idx:3d} @ {event.timestamp.strftime('%H:%M')}: "
                      f"entry={entry_signal}, exit={exit_signal}, qty={current_qty:.4f}, "
                      f"price=${event.close:.2f}, cash=${self.portfolio.cash:.2f}")

            # Exit logic
            if exit_signal and current_qty != 0:
                print(f"     ➡️  EXIT SIGNAL: Selling {current_qty:.4f} @ ${event.close:.2f}")
                self._order_counter += 1
                order_event = OrderEvent(
                    timestamp=event.timestamp,
                    order_id=f"EXIT_{self._order_counter:04d}",
                    asset_id=event.asset_id,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                    quantity=abs(current_qty),
                )
                self.event_bus.publish(order_event)
                print(f"     📤 Published: {order_event.order_id}")

            # Entry logic
            if entry_signal and current_qty == 0:
                cash = self.portfolio.cash
                size = cash / event.close
                print(f"     ➡️  ENTRY SIGNAL: Buying ${cash:.2f} / ${event.close:.2f} = {size:.4f}")
                self._order_counter += 1
                order_event = OrderEvent(
                    timestamp=event.timestamp,
                    order_id=f"ENTRY_{self._order_counter:04d}",
                    asset_id=event.asset_id,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=size,
                )
                self.event_bus.publish(order_event)
                print(f"     📤 Published: {order_event.order_id}")
                self.trade_count += 1

            self.bar_idx += 1

    # Create data feed
    class SimpleDataFeed(DataFeed):
        def __init__(self, ohlcv_df):
            self.ohlcv = ohlcv_df.reset_index()
            self.idx = 0

        @property
        def is_exhausted(self) -> bool:
            return self.idx >= len(self.ohlcv)

        def get_next_event(self) -> MarketEvent:
            if self.is_exhausted:
                return None

            row = self.ohlcv.iloc[self.idx]
            event = MarketEvent(
                timestamp=row['timestamp'],
                asset_id="BTC",
                data_type=MarketDataType.BAR,
                price=row['close'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )
            self.idx += 1
            return event

        def peek_next_timestamp(self):
            if self.is_exhausted:
                return None
            return self.ohlcv.iloc[self.idx]['timestamp']

        def reset(self):
            self.idx = 0

        def seek(self, timestamp):
            mask = self.ohlcv['timestamp'] >= timestamp
            indices = mask[mask].index
            if len(indices) > 0:
                self.idx = indices[0]

    # Run backtest
    strategy = LoggingSignalStrategy(entries, exits)
    data_feed = SimpleDataFeed(ohlcv)

    commission = PercentageCommission(rate=config.fees) if config.fees > 0 else None
    broker = SimulationBroker(
        initial_cash=config.initial_cash,
        commission_model=commission,
        execution_delay=False,  # Same-bar execution
    )

    engine = BacktestEngine(
        strategy=strategy,
        broker=broker,
        data_feed=data_feed,
    )

    print("\n   Running backtest...")
    results = engine.run()

    print(f"\n4️⃣  Results:")
    print(f"   Final portfolio value: ${results['final_value']:,.2f}")
    print(f"   Entries published: {strategy.trade_count}")
    print(f"   Total orders: {strategy._order_counter}")

    # Get trades from broker
    trade_tracker = broker.trade_tracker
    if trade_tracker:
        trades_df = trade_tracker.get_trades_df()
        print(f"\n   Completed trades: {len(trades_df)}")
        for i, row in enumerate(trades_df.head(5).iter_rows(named=True), 1):
            print(f"     Trade {i}: entry ${row['entry_price']:,.2f} @ {row['entry_dt'].strftime('%H:%M')}, "
                  f"exit ${row['exit_price']:,.2f} @ {row['exit_dt'].strftime('%H:%M')}, "
                  f"pnl ${row['pnl']:,.2f}")

    print("\n" + "=" * 80)
    print("Expected: 3 entries at bars 10, 60, 110")
    print("Expected: 3 exits at bars 20, 70, 120")
    print("Expected: 3 completed trades")
    if strategy.trade_count > 3:
        print(f"\n🔴 PHANTOM TRADES DETECTED: {strategy.trade_count} entries instead of 3")
    print("=" * 80)


if __name__ == "__main__":
    test_diagnostic_with_logging()
