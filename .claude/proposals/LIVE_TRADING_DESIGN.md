# Proposal: Live Trading Engine Design

**Date**: 2025-11-23
**Author**: Claude Code
**Status**: Draft - Design Planning
**Library**: ml4t.backtest → ml4t.live

---

## Executive Summary

Enable users to **copy-paste their Strategy class** from backtesting to live trading with zero code changes. Only the infrastructure changes:

| Component | Backtest | Live Trading |
|-----------|----------|--------------|
| Data Source | Historical DataFrame iteration | Real-time streaming (WebSocket/callbacks) |
| Broker | Simulated `Broker` class | `IBBroker` or `AlpacaBroker` |
| Loop Model | Synchronous for-loop | Async event loop (ib_async) |
| Account State | Simulated cash/positions | Real account from broker API |

**Key Principle**: `Strategy.on_data(timestamp, data, context, broker)` signature is **immutable**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User's Strategy                                  │
│                                                                          │
│   class MyStrategy(Strategy):                                           │
│       def on_data(self, timestamp, data, context, broker):              │
│           # SAME CODE for backtest AND live                             │
│           target = {'AAPL': 0.5}                                        │
│           self.executor.execute(target, data, broker)                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BrokerProtocol Interface                            │
│                                                                          │
│   - submit_order(asset, quantity, side, ...)  → Order                   │
│   - cancel_order(order_id) → bool                                       │
│   - get_position(asset) → Position | None                               │
│   - get_account_value() → float                                         │
│   - get_cash() → float                                                  │
│   - close_position(asset) → Order | None                                │
│   - positions: dict[str, Position]                                      │
│   - pending_orders: list[Order]                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
              │                                       │
              ▼                                       ▼
┌──────────────────────────┐          ┌──────────────────────────────────┐
│   Backtest Broker        │          │      Live Brokers                │
│   (existing)             │          │                                  │
│                          │          │  ┌────────────┐  ┌────────────┐  │
│   - Simulated fills      │          │  │ IBBroker   │  │AlpacaBroker│  │
│   - Commission/slippage  │          │  │ (ib_async) │  │ (REST/WS)  │  │
│   - Historical data      │          │  └────────────┘  └────────────┘  │
│                          │          │                                  │
└──────────────────────────┘          └──────────────────────────────────┘
```

---

## Core Design: Protocol-Based Abstraction

### BrokerProtocol (New)

```python
# src/ml4t/backtest/protocols.py (or ml4t/live/protocols.py)

from typing import Protocol, runtime_checkable
from datetime import datetime

from .types import Order, OrderSide, OrderType, Position


@runtime_checkable
class BrokerProtocol(Protocol):
    """Interface that all brokers (backtest and live) must implement.

    Strategy code interacts with this interface only, enabling seamless
    transition from backtesting to live trading.
    """

    # Properties
    @property
    def positions(self) -> dict[str, Position]:
        """Current positions keyed by asset."""
        ...

    @property
    def pending_orders(self) -> list[Order]:
        """Orders submitted but not yet filled."""
        ...

    # Account queries
    def get_position(self, asset: str) -> Position | None:
        """Get position for specific asset."""
        ...

    def get_account_value(self) -> float:
        """Total account value (cash + positions)."""
        ...

    def get_cash(self) -> float:
        """Available cash."""
        ...

    # Order management
    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Submit an order. Returns Order with order_id."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order. Returns True if cancelled."""
        ...

    def close_position(self, asset: str) -> Order | None:
        """Close entire position. Returns Order or None if no position."""
        ...
```

### Existing Broker Already Implements Protocol

The current `Broker` class already has all these methods - we just need to formalize the protocol.

---

## Live Trading Components

### 1. LiveEngine

```python
# src/ml4t/live/engine.py

import asyncio
from datetime import datetime
from typing import Any

from ..backtest import Strategy
from .protocols import BrokerProtocol, DataFeedProtocol


class LiveEngine:
    """Live trading engine - async event-driven.

    Receives real-time data and dispatches to Strategy.on_data()
    using the same interface as backtesting.

    Example:
        strategy = MyStrategy()  # Same class used in backtest
        broker = IBBroker(...)
        feed = IBDataFeed(...)

        engine = LiveEngine(strategy, broker, feed)
        await engine.run()
    """

    def __init__(
        self,
        strategy: Strategy,
        broker: BrokerProtocol,
        feed: "DataFeedProtocol",
        on_error: callable | None = None,
    ):
        self.strategy = strategy
        self.broker = broker
        self.feed = feed
        self.on_error = on_error or self._default_error_handler
        self._running = False

    async def run(self):
        """Main async loop - receives bars and dispatches to strategy."""
        self._running = True
        self.strategy.on_start(self.broker)

        try:
            async for timestamp, data, context in self.feed:
                if not self._running:
                    break

                try:
                    # Call strategy with same signature as backtest
                    self.strategy.on_data(timestamp, data, context, self.broker)
                except Exception as e:
                    self.on_error(e, timestamp, data)
        finally:
            self.strategy.on_end(self.broker)
            self._running = False

    def stop(self):
        """Signal the engine to stop."""
        self._running = False

    def _default_error_handler(self, error, timestamp, data):
        """Default error handler - log and continue."""
        import logging
        logging.error(f"Strategy error at {timestamp}: {error}")
```

### 2. IBBroker (Interactive Brokers)

```python
# src/ml4t/live/brokers/ib.py

import asyncio
from datetime import datetime
from typing import Any

# ib_async is the maintained fork of ib_insync
from ib_async import IB, Stock, MarketOrder, LimitOrder, Contract, Trade as IBTrade

from ...backtest.types import Order, OrderSide, OrderType, OrderStatus, Position
from ..protocols import BrokerProtocol


class IBBroker(BrokerProtocol):
    """Interactive Brokers implementation using ib_async.

    Wraps the ib_async library to provide the BrokerProtocol interface.
    Handles async connection, position sync, and order management.

    Example:
        broker = IBBroker()
        await broker.connect(host='127.0.0.1', port=7497, client_id=1)

        # Now use like backtest broker
        broker.submit_order('AAPL', 100, OrderSide.BUY)
    """

    def __init__(self, ib: IB | None = None):
        self.ib = ib or IB()
        self._positions: dict[str, Position] = {}
        self._pending_orders: list[Order] = {}
        self._order_map: dict[str, IBTrade] = {}  # our order_id -> IB trade
        self._order_counter = 0
        self._connected = False

    async def connect(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # 7497=TWS paper, 7496=TWS live, 4001=Gateway paper, 4002=Gateway live
        client_id: int = 1,
        readonly: bool = False,
    ):
        """Connect to IB TWS or Gateway."""
        await self.ib.connectAsync(host, port, clientId=client_id, readonly=readonly)
        self._connected = True

        # Subscribe to events
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.positionEvent += self._on_position
        self.ib.errorEvent += self._on_error

        # Initial sync
        await self._sync_positions()
        await self._sync_orders()

    async def disconnect(self):
        """Disconnect from IB."""
        self.ib.disconnect()
        self._connected = False

    # === BrokerProtocol Implementation ===

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders.values())

    def get_position(self, asset: str) -> Position | None:
        return self._positions.get(asset)

    def get_account_value(self) -> float:
        """Get total account value from IB."""
        account_values = self.ib.accountValues()
        for av in account_values:
            if av.tag == 'NetLiquidation' and av.currency == 'USD':
                return float(av.value)
        return 0.0

    def get_cash(self) -> float:
        """Get available cash from IB."""
        account_values = self.ib.accountValues()
        for av in account_values:
            if av.tag == 'AvailableFunds' and av.currency == 'USD':
                return float(av.value)
        return 0.0

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Submit order to IB."""
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)

        # Create IB contract
        contract = Stock(asset, 'SMART', 'USD')

        # Create IB order
        action = 'BUY' if side == OrderSide.BUY else 'SELL'
        if order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, quantity)
        elif order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, quantity, limit_price)
        else:
            # TODO: Handle STOP, STOP_LIMIT, TRAILING_STOP
            ib_order = MarketOrder(action, quantity)

        # Place order
        trade = self.ib.placeOrder(contract, ib_order)

        # Create our Order object
        self._order_counter += 1
        order_id = f"IB-{self._order_counter}"

        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            order_id=order_id,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
        )

        self._order_map[order_id] = trade
        self._pending_orders[order_id] = order

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order in IB."""
        if order_id not in self._order_map:
            return False

        trade = self._order_map[order_id]
        self.ib.cancelOrder(trade.order)

        if order_id in self._pending_orders:
            self._pending_orders[order_id].status = OrderStatus.CANCELLED
            del self._pending_orders[order_id]

        return True

    def close_position(self, asset: str) -> Order | None:
        """Close position via IB."""
        pos = self._positions.get(asset)
        if not pos or pos.quantity == 0:
            return None

        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        return self.submit_order(asset, abs(pos.quantity), side)

    # === Event Handlers ===

    def _on_order_status(self, trade: IBTrade):
        """Handle order status updates from IB."""
        # Find our order
        for order_id, ib_trade in self._order_map.items():
            if ib_trade is trade:
                order = self._pending_orders.get(order_id)
                if order:
                    if trade.orderStatus.status == 'Filled':
                        order.status = OrderStatus.FILLED
                        order.filled_price = trade.orderStatus.avgFillPrice
                        order.filled_quantity = trade.orderStatus.filled
                        order.filled_at = datetime.now()
                        del self._pending_orders[order_id]
                    elif trade.orderStatus.status == 'Cancelled':
                        order.status = OrderStatus.CANCELLED
                        del self._pending_orders[order_id]
                break

    def _on_position(self, position):
        """Handle position updates from IB."""
        asset = position.contract.symbol
        if position.position != 0:
            self._positions[asset] = Position(
                asset=asset,
                quantity=float(position.position),
                entry_price=float(position.avgCost),
                entry_time=datetime.now(),  # IB doesn't provide entry time
            )
        elif asset in self._positions:
            del self._positions[asset]

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle errors from IB."""
        import logging
        logging.warning(f"IB Error {errorCode}: {errorString}")

    async def _sync_positions(self):
        """Sync positions from IB on connect."""
        positions = self.ib.positions()
        for pos in positions:
            self._on_position(pos)

    async def _sync_orders(self):
        """Sync open orders from IB on connect."""
        trades = self.ib.trades()
        for trade in trades:
            if trade.orderStatus.status in ('PreSubmitted', 'Submitted'):
                # Add to our tracking
                self._order_counter += 1
                order_id = f"IB-{self._order_counter}"
                self._order_map[order_id] = trade
                # ... create Order object
```

### 3. AlpacaBroker

```python
# src/ml4t/live/brokers/alpaca.py

from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

from ...backtest.types import Order, OrderSide, OrderType, OrderStatus, Position
from ..protocols import BrokerProtocol


class AlpacaBroker(BrokerProtocol):
    """Alpaca implementation using alpaca-py.

    Simpler than IB, REST-based with WebSocket for streaming.

    Example:
        broker = AlpacaBroker(
            api_key='your-key',
            secret_key='your-secret',
            paper=True,  # Paper trading
        )

        broker.submit_order('AAPL', 100, OrderSide.BUY)
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self._positions: dict[str, Position] = {}
        self._pending_orders: dict[str, Order] = {}
        self._order_counter = 0

        # Initial sync
        self._sync_positions()
        self._sync_orders()

    @property
    def positions(self) -> dict[str, Position]:
        self._sync_positions()  # Refresh
        return self._positions

    @property
    def pending_orders(self) -> list[Order]:
        self._sync_orders()  # Refresh
        return list(self._pending_orders.values())

    def get_position(self, asset: str) -> Position | None:
        return self._positions.get(asset)

    def get_account_value(self) -> float:
        account = self.client.get_account()
        return float(account.equity)

    def get_cash(self) -> float:
        account = self.client.get_account()
        return float(account.cash)

    def submit_order(
        self,
        asset: str,
        quantity: float,
        side: OrderSide | None = None,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        if side is None:
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            quantity = abs(quantity)

        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

        if order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=asset,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            request = LimitOrderRequest(
                symbol=asset,
                qty=quantity,
                side=alpaca_side,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )

        alpaca_order = self.client.submit_order(request)

        self._order_counter += 1
        order = Order(
            asset=asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            order_id=str(alpaca_order.id),
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
        )

        self._pending_orders[order.order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel_order_by_id(order_id)
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
            return True
        except Exception:
            return False

    def close_position(self, asset: str) -> Order | None:
        pos = self.get_position(asset)
        if not pos or pos.quantity == 0:
            return None

        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        return self.submit_order(asset, abs(pos.quantity), side)

    def _sync_positions(self):
        """Sync positions from Alpaca."""
        positions = self.client.get_all_positions()
        self._positions = {
            p.symbol: Position(
                asset=p.symbol,
                quantity=float(p.qty),
                entry_price=float(p.avg_entry_price),
                entry_time=datetime.now(),
            )
            for p in positions
        }

    def _sync_orders(self):
        """Sync open orders from Alpaca."""
        orders = self.client.get_orders()
        # ... convert to our Order format
```

### 4. Data Feeds

```python
# src/ml4t/live/feeds/ib.py

import asyncio
from datetime import datetime
from typing import AsyncIterator

from ib_async import IB, Stock, BarData


class IBDataFeed:
    """Real-time data feed from Interactive Brokers.

    Provides async iterator interface compatible with LiveEngine.

    Example:
        feed = IBDataFeed(ib, assets=['AAPL', 'GOOG'], bar_size='1 min')
        await feed.start()

        async for timestamp, data, context in feed:
            print(f"{timestamp}: {data}")
    """

    def __init__(
        self,
        ib: IB,
        assets: list[str],
        bar_size: str = '1 min',  # '5 secs', '1 min', '5 mins', etc.
    ):
        self.ib = ib
        self.assets = assets
        self.bar_size = bar_size
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._bars_by_asset: dict[str, BarData] = {}

    async def start(self):
        """Subscribe to real-time bars for all assets."""
        self._running = True

        for asset in self.assets:
            contract = Stock(asset, 'SMART', 'USD')

            # Request real-time bars (5-second bars aggregated to bar_size)
            bars = self.ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=True,
            )
            bars.updateEvent += lambda bars, has_new, a=asset: self._on_bar(a, bars, has_new)

    def stop(self):
        """Stop receiving data."""
        self._running = False

    def _on_bar(self, asset: str, bars: list[BarData], has_new_bar: bool):
        """Handle incoming bar data."""
        if not has_new_bar or not self._running:
            return

        bar = bars[-1]
        self._bars_by_asset[asset] = bar

        # Package data in format Strategy expects
        timestamp = bar.time
        data = {
            asset: {
                'open': bar.open_,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }
            for asset, bar in self._bars_by_asset.items()
        }
        context = {}

        # Put in queue for async iteration
        self._queue.put_nowait((timestamp, data, context))

    async def __aiter__(self) -> AsyncIterator[tuple[datetime, dict, dict]]:
        """Async iterator interface for LiveEngine."""
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield item
            except asyncio.TimeoutError:
                continue  # Check _running flag and continue
```

---

## Usage Example

```python
import asyncio
from ib_async import IB

from ml4t.backtest import Strategy, TargetWeightExecutor, RebalanceConfig
from ml4t.live import LiveEngine, IBBroker, IBDataFeed


class MyStrategy(Strategy):
    """Same strategy for backtest AND live trading."""

    def __init__(self):
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(min_trade_value=500)
        )
        self.bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1

        # Rebalance every 20 bars
        if self.bar_count % 20 != 0:
            return

        # Simple equal-weight portfolio
        assets = list(data.keys())
        weight = 0.95 / len(assets)  # 5% cash
        target_weights = {a: weight for a in assets}

        self.executor.execute(target_weights, data, broker)


async def main():
    # === BACKTEST ===
    # from ml4t.backtest import Engine, Broker, DataFeed
    # broker = Broker(initial_cash=100000)
    # feed = DataFeed(df)
    # engine = Engine(MyStrategy(), broker, feed)
    # results = engine.run()

    # === LIVE TRADING ===
    ib = IB()

    # Connect to IB
    broker = IBBroker(ib)
    await broker.connect(host='127.0.0.1', port=7497, client_id=1)

    # Set up data feed
    feed = IBDataFeed(ib, assets=['AAPL', 'GOOG', 'MSFT'])
    await feed.start()

    # Run with SAME strategy class
    strategy = MyStrategy()
    engine = LiveEngine(strategy, broker, feed)

    try:
        await engine.run()
    except KeyboardInterrupt:
        engine.stop()
        await broker.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## Package Structure

```
ml4t/
├── backtest/                    # Existing
│   ├── __init__.py
│   ├── broker.py               # Add BrokerProtocol compliance
│   ├── engine.py
│   ├── strategy.py
│   ├── types.py
│   └── ...
│
└── live/                        # NEW
    ├── __init__.py
    ├── protocols.py             # BrokerProtocol, DataFeedProtocol
    ├── engine.py                # LiveEngine (async)
    │
    ├── brokers/
    │   ├── __init__.py
    │   ├── ib.py               # IBBroker (ib_async)
    │   └── alpaca.py           # AlpacaBroker (alpaca-py)
    │
    └── feeds/
        ├── __init__.py
        ├── ib.py               # IBDataFeed
        └── alpaca.py           # AlpacaDataFeed
```

---

## Key Design Decisions

### 1. Async vs Sync Strategy

**Decision**: Strategy.on_data() stays **synchronous**.

**Rationale**:
- Users shouldn't need to learn async for simple strategies
- Heavy computation can be offloaded to thread pool if needed
- LiveEngine handles async internally

```python
# LiveEngine internally can use to_thread for blocking strategies
await asyncio.to_thread(self.strategy.on_data, timestamp, data, context, broker)
```

### 2. Position/Order Type Reuse

**Decision**: Reuse `Order`, `Position`, `Fill` types from backtest.

**Rationale**:
- Strategy code that inspects positions works identically
- TargetWeightExecutor works unchanged
- Single source of truth for types

### 3. Real-Time vs Bar-Based

**Decision**: Support both real-time ticks and aggregated bars.

**Rationale**:
- Most strategies work on bars (1min, 5min, etc.)
- Some HFT strategies need tick data
- Feed abstraction handles aggregation

### 4. Error Handling

**Decision**: Strategy errors don't crash the engine.

**Rationale**:
- Live trading must be resilient
- Log errors and continue
- User can provide custom error handler

---

## Implementation Phases

### Phase 1: Protocol & Core (Week 1)
- [ ] Define `BrokerProtocol` in protocols.py
- [ ] Verify existing `Broker` implements protocol
- [ ] Create `LiveEngine` class
- [ ] Basic async infrastructure

### Phase 2: IB Integration (Week 2)
- [ ] `IBBroker` implementation
- [ ] `IBDataFeed` implementation
- [ ] Connection handling
- [ ] Position/order sync

### Phase 3: Alpaca Integration (Week 3)
- [ ] `AlpacaBroker` implementation
- [ ] `AlpacaDataFeed` implementation
- [ ] WebSocket streaming

### Phase 4: Testing & Documentation (Week 4)
- [ ] Paper trading tests
- [ ] Error handling hardening
- [ ] Documentation and examples
- [ ] Migration guide

---

## Dependencies

```toml
# pyproject.toml additions for ml4t-live

[project.optional-dependencies]
live = [
    "ib_async>=1.0.0",      # IB API (maintained fork)
    "alpaca-py>=0.10.0",    # Alpaca API
]

ib = ["ib_async>=1.0.0"]
alpaca = ["alpaca-py>=0.10.0"]
```

---

## Open Questions

1. **Separate package or same?**
   - Option A: `ml4t-live` separate package
   - Option B: `ml4t-backtest[live]` optional dependency

2. **Order reconciliation?**
   - How to handle orders placed outside our system?
   - Sync on connect? Ignore?

3. **Risk limits for live?**
   - Maximum position size?
   - Kill switch?
   - Should be configurable per-broker

4. **Paper vs Live switching?**
   - Same code, different port (IB)?
   - Config flag?

---

## Summary

The design enables **identical Strategy code** for backtest and live:

```python
# Works in BOTH backtest and live
strategy = MyStrategy()

# Backtest
engine = Engine(strategy, Broker(), DataFeed(df))
results = engine.run()

# Live
engine = LiveEngine(strategy, IBBroker(), IBDataFeed())
await engine.run()
```

The key is the **BrokerProtocol** abstraction that both `Broker` (backtest) and `IBBroker`/`AlpacaBroker` implement.
