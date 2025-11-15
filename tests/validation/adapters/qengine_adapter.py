"""ml4t.backtest adapter for cross-platform validation."""
import time
from datetime import datetime
from typing import Any

import polars as pl

from ml4t.backtest import BacktestEngine
from ml4t.backtest.core.event import EventType, MarketEvent, OrderEvent
from ml4t.backtest.core.types import AssetId, MarketDataType, OrderSide, OrderType
from ml4t.backtest.data.feed import DataFeed
from ml4t.backtest.execution.order import Order
from ml4t.backtest.strategy.base import Strategy

from .base import BacktestResult, PlatformAdapter, Trade


class PolarsDataFeed(DataFeed):
    """In-memory data feed from Polars DataFrame."""

    def __init__(
        self,
        df: pl.DataFrame,
        asset_id: AssetId | None = None,
        data_type: MarketDataType = MarketDataType.BAR,
        timestamp_column: str = "timestamp",
    ):
        """Initialize from Polars DataFrame.

        Args:
            df: Polars DataFrame with OHLCV data
            asset_id: Asset identifier (extracted from 'symbol' column if None)
            data_type: Type of market data
            timestamp_column: Name of timestamp column
        """
        self.df = df.sort(timestamp_column)
        self.timestamp_column = timestamp_column
        self.data_type = data_type

        # Extract asset_id from first row if not provided
        if asset_id is None and 'symbol' in df.columns:
            self.asset_id = df[0, 'symbol']
        else:
            self.asset_id = asset_id or 'UNKNOWN'

        self.current_index = 0
        self.max_index = len(self.df) - 1

    def get_next_event(self) -> MarketEvent | None:
        """Get the next market event."""
        if self.is_exhausted:
            return None

        row = self.df.row(self.current_index, named=True)
        self.current_index += 1

        return self._create_market_event(row)

    def _create_market_event(self, row: dict[str, Any]) -> MarketEvent:
        """Create a MarketEvent from a data row."""
        timestamp = row[self.timestamp_column]

        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))

        return MarketEvent(
            timestamp=timestamp,
            asset_id=self.asset_id,
            data_type=self.data_type,
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            price=row.get("price", row.get("close")),
        )

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the next timestamp."""
        if self.is_exhausted:
            return None

        timestamp = self.df[self.timestamp_column][self.current_index]
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp))

        return timestamp

    def reset(self) -> None:
        """Reset to the beginning."""
        self.current_index = 0

    def seek(self, timestamp: datetime) -> None:
        """Seek to a specific timestamp."""
        mask = self.df[self.timestamp_column] >= timestamp
        indices = mask.arg_true()

        if len(indices) > 0:
            self.current_index = indices[0]
        else:
            self.current_index = self.max_index + 1

    @property
    def is_exhausted(self) -> bool:
        """Check if all data has been consumed."""
        return self.current_index > self.max_index


class SignalDrivenStrategy(Strategy):
    """Strategy that executes based on pre-generated signals."""

    def __init__(self, signals: list, name: str = "SignalDriven"):
        super().__init__(name)
        self.signals = {(s.timestamp, s.symbol): s for s in signals}
        self.signal_index = 0
        self.executed_signals = set()

    def on_event(self, event) -> None:
        """Route events to appropriate handlers."""
        if event.event_type == EventType.MARKET:
            self.on_market_event(event)
        elif event.event_type == EventType.FILL:
            self.on_fill_event(event)

    def on_market_event(self, event) -> None:
        """Check for signals matching this market event."""
        key = (event.timestamp, event.asset_id)

        if key in self.signals and key not in self.executed_signals:
            signal = self.signals[key]
            self._execute_signal(signal, event)
            self.executed_signals.add(key)

    def _execute_signal(self, signal, market_event):
        """Translate signal to ml4t.backtest order."""
        # Determine order side
        if signal.action == 'BUY':
            side = OrderSide.BUY
        elif signal.action == 'SELL':
            side = OrderSide.SELL
        elif signal.action == 'CLOSE':
            # Close position: determine side based on current position
            current_qty = self._positions.get(signal.symbol, 0)
            if current_qty > 0:
                side = OrderSide.SELL
            elif current_qty < 0:
                side = OrderSide.BUY
            else:
                return  # No position to close

        # Determine quantity
        if signal.action == 'CLOSE':
            quantity = abs(self._positions.get(signal.symbol, 0))
            if quantity == 0:
                return
        else:
            quantity = signal.quantity or 100

        # Create order with stop loss / take profit if specified
        if signal.stop_loss or signal.take_profit:
            # Use bracket order for stop loss + take profit
            order = Order(
                asset_id=signal.symbol,
                order_type=OrderType.BRACKET if signal.stop_loss and signal.take_profit else OrderType.MARKET,
                side=side,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                profit_target=signal.take_profit,
                metadata={'signal_id': signal.signal_id}
            )
        elif signal.trailing_stop_pct:
            # Trailing stop order
            order = Order(
                asset_id=signal.symbol,
                order_type=OrderType.TRAILING_STOP,
                side=side,
                quantity=quantity,
                trail_percent=signal.trailing_stop_pct * 100,  # Convert to percentage
                metadata={'signal_id': signal.signal_id}
            )
        else:
            # Simple market order
            order = Order(
                asset_id=signal.symbol,
                order_type=OrderType.MARKET,
                side=side,
                quantity=quantity,
                metadata={'signal_id': signal.signal_id}
            )

        # Submit order via broker
        if hasattr(self, 'broker') and self.broker:
            self.broker.submit_order(order)


class ml4t.backtestAdapter(PlatformAdapter):
    """Adapter for ml4t.backtest backtesting platform."""

    def __init__(self):
        super().__init__("ml4t.backtest")

    def run_backtest(
        self,
        signals: list,
        data: pl.DataFrame,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0,
        **kwargs
    ) -> BacktestResult:
        """Run backtest using ml4t.backtest."""
        start_time = time.time()

        # Create data feed
        data_feed = PolarsDataFeed(data)

        # Create signal-driven strategy
        strategy = SignalDrivenStrategy(signals)

        # Create broker with commission/slippage
        from ml4t.backtest.execution.broker import SimulationBroker
        from ml4t.backtest.execution.commission import PercentageCommission
        from ml4t.backtest.execution.slippage import FixedSlippage

        broker = SimulationBroker(
            commission_model=PercentageCommission(commission),
            slippage_model=FixedSlippage(slippage) if slippage > 0 else None
        )

        # Inject broker into strategy
        strategy.broker = broker

        # Create engine
        engine = BacktestEngine(
            data_feed=data_feed,
            strategy=strategy,
            broker=broker,
            initial_capital=initial_capital
        )

        # Run backtest
        results = engine.run()
        execution_time = time.time() - start_time

        # Convert to standardized format
        trades = self._convert_trades(results['trades'])
        equity_curve = self._convert_equity_curve(results)
        metrics = results['metrics']

        return BacktestResult(
            platform=self.name,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            execution_time=execution_time,
            config={
                'initial_capital': initial_capital,
                'commission': commission,
                'slippage': slippage,
            },
            metadata={
                'events_processed': results['events_processed'],
                'events_per_second': results['events_per_second'],
            }
        )

    def _convert_trades(self, trades_df: pl.DataFrame) -> list[Trade]:
        """Convert ml4t.backtest orders to standardized trades format.

        ml4t.backtest returns individual fills, not matched entry/exit pairs.
        We need to match BUY and SELL orders to create complete trades.
        """
        if trades_df.is_empty():
            return []

        trades = []
        open_positions = {}  # Track open positions by asset_id

        # Sort by filled_time to process in order
        sorted_df = trades_df.sort('filled_time')

        for row in sorted_df.iter_rows(named=True):
            asset_id = row['asset_id']
            side = row['side'].upper()  # Normalize to uppercase
            quantity = row['quantity']
            price = row['price']
            commission = row['commission']
            filled_time = row['filled_time']

            if side == 'BUY':
                # Open or add to position
                if asset_id not in open_positions:
                    open_positions[asset_id] = []

                open_positions[asset_id].append({
                    'entry_time': filled_time,
                    'entry_price': price,
                    'quantity': quantity,
                    'entry_commission': commission,
                })

            elif side == 'SELL':
                # Close position(s)
                if asset_id not in open_positions or not open_positions[asset_id]:
                    # Short sale or position mismatch - skip for now
                    continue

                remaining_to_close = quantity

                while remaining_to_close > 0 and open_positions[asset_id]:
                    position = open_positions[asset_id][0]
                    position_qty = position['quantity']

                    if position_qty <= remaining_to_close:
                        # Close entire position
                        pnl = (price - position['entry_price']) * position_qty
                        pnl -= (position['entry_commission'] + commission * position_qty / quantity)

                        trades.append(Trade(
                            entry_time=position['entry_time'],
                            entry_price=position['entry_price'],
                            exit_time=filled_time,
                            exit_price=price,
                            symbol=asset_id,
                            quantity=position_qty,
                            pnl=pnl,
                            commission=position['entry_commission'] + commission * position_qty / quantity,
                            signal_id='',
                        ))

                        open_positions[asset_id].pop(0)
                        remaining_to_close -= position_qty
                    else:
                        # Partial close
                        pnl = (price - position['entry_price']) * remaining_to_close
                        pnl -= (position['entry_commission'] * remaining_to_close / position_qty +
                                commission * remaining_to_close / quantity)

                        trades.append(Trade(
                            entry_time=position['entry_time'],
                            entry_price=position['entry_price'],
                            exit_time=filled_time,
                            exit_price=price,
                            symbol=asset_id,
                            quantity=remaining_to_close,
                            pnl=pnl,
                            commission=(position['entry_commission'] * remaining_to_close / position_qty +
                                       commission * remaining_to_close / quantity),
                            signal_id='',
                        ))

                        position['quantity'] -= remaining_to_close
                        position['entry_commission'] -= position['entry_commission'] * remaining_to_close / position_qty
                        remaining_to_close = 0

        return trades

    def _convert_equity_curve(self, results: dict) -> pl.DataFrame:
        """Extract equity curve from results."""
        # Try to get from returns
        if 'returns' in results and not results['returns'].is_empty():
            # Reconstruct equity from returns
            returns = results['returns']
            # Implementation depends on returns format
            # For now, return simple equity curve
            pass

        # Fallback: create simple equity curve
        initial = results['initial_capital']
        final = results['final_value']

        return pl.DataFrame({
            'timestamp': [datetime.now()],  # Placeholder
            'equity': [final]
        })

    def supports_stop_loss(self) -> bool:
        """ml4t.backtest supports stop loss via bracket orders."""
        return True

    def supports_take_profit(self) -> bool:
        """ml4t.backtest supports take profit via bracket orders."""
        return True

    def supports_trailing_stop(self) -> bool:
        """ml4t.backtest supports trailing stop orders."""
        return True
