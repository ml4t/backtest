"""Unified wrappers for all 4 backtesting engines."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional


@dataclass
class BacktestConfig:
    """Configuration for backtest runs (same across all engines)."""
    initial_cash: float = 100000.0
    fees: float | dict = 0.0  # Float for percentage-only, or dict{"percentage": float, "fixed": float}
    slippage: float = 0.0
    order_type: str = 'market'  # 'market', 'limit', 'stop'
    size: Optional[float] = None  # None = use all cash (size=np.inf in VBT)


@dataclass
class BacktestResult:
    """Standardized results from any engine."""
    trades: pd.DataFrame  # Columns: entry_time, entry_price, exit_time, exit_price, pnl, size
    final_value: float
    final_cash: float
    final_position: float
    total_pnl: float
    num_trades: int
    engine_name: str


class EngineWrapper(ABC):
    """Base class for engine wrappers."""

    @abstractmethod
    def run_backtest(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: Optional[pd.Series] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """
        Run backtest with given OHLCV and signals.

        Args:
            ohlcv: DataFrame with [open, high, low, close, volume] columns and timestamp index
            entries: Boolean series with True at entry signals
            exits: Optional boolean series with True at exit signals (if None, hold until next entry)
            config: Backtest configuration

        Returns:
            BacktestResult with standardized trade data
        """
        pass


class QEngineWrapper(EngineWrapper):
    """Wrapper for qengine."""

    def run_backtest(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: Optional[pd.Series] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Run qengine backtest."""
        from qengine.engine import BacktestEngine
        from qengine.core.assets import AssetSpec, AssetRegistry
        from qengine.data.feed import DataFeed
        from qengine.strategy.base import Strategy
        from qengine.core.event import MarketEvent
        from qengine.execution.commission import PercentageCommission
        from qengine.execution.slippage import PercentageSlippage

        if config is None:
            config = BacktestConfig()

        # Create asset
        from qengine.core.assets import AssetClass
        asset_spec = AssetSpec(
            asset_id="BTC",
            asset_class=AssetClass.CRYPTO,
        )
        registry = AssetRegistry()
        registry.register(asset_spec)

        # Create PrecisionManager for consistent rounding
        precision_mgr = asset_spec.get_precision_manager()

        # Create strategy that trades on signals
        class SignalStrategy(Strategy):
            def __init__(self, entries_ser, exits_ser, precision_mgr):
                super().__init__()
                self.entries = entries_ser
                self.exits = exits_ser if exits_ser is not None else pd.Series([False] * len(entries_ser))
                self.bar_idx = 0
                self._order_counter = 0
                self.precision_mgr = precision_mgr  # Store for use in calculations

            def on_start(self, portfolio, event_bus):
                super().on_start(portfolio, event_bus)
                # Store references
                self.portfolio = portfolio
                self.event_bus = event_bus

            def on_event(self, event):
                """Route events to appropriate handlers."""
                from qengine.core.event import FillEvent

                if isinstance(event, MarketEvent):
                    self.on_market_event(event)
                elif isinstance(event, FillEvent):
                    # Let base class update internal position tracking
                    super().on_fill_event(event)

            def on_market_event(self, event: MarketEvent):
                from qengine.core.event import OrderEvent
                from qengine.core.types import OrderSide, OrderType

                # Get signal for this bar
                entry_signal = self.entries.iloc[self.bar_idx] if self.bar_idx < len(self.entries) else False
                exit_signal = self.exits.iloc[self.bar_idx] if self.bar_idx < len(self.exits) else False

                # Check current position from portfolio (updated by FillEvents)
                # CRITICAL: Use portfolio.get_position() instead of self._positions
                # because portfolio is updated directly by FillEvents with actual filled quantities
                position = self.portfolio.get_position(event.asset_id)
                current_qty = position.quantity if position else 0.0

                # Exit logic: exit if signal and have position
                # Use PrecisionManager to determine if position is effectively zero
                if exit_signal and not self.precision_mgr.is_position_zero(current_qty):
                    # Exit EXACTLY the quantity we actually hold in portfolio
                    # This prevents position tracking mismatches with fixed fees/rounding
                    self._order_counter += 1
                    exit_qty = abs(current_qty)
                    order_event = OrderEvent(
                        timestamp=event.timestamp,
                        order_id=f"EXIT_{self._order_counter:04d}",
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                        quantity=exit_qty,
                    )
                    self.event_bus.publish(order_event)

                # Entry logic: enter only if flat (use portfolio position)
                # Use PrecisionManager to determine if position is effectively zero
                if entry_signal and self.precision_mgr.is_position_zero(current_qty):
                    # Calculate position size (use all cash if size not specified)
                    if config.size is None:
                        # Use all available cash, accounting for commission
                        cash = self.portfolio.cash
                        # Calculate size that fits within cash budget INCLUDING commission
                        # For combined fees: cash >= size * price * (1 + pct_rate) + fixed_fee
                        # For percentage only: cash >= size * price * (1 + pct_rate)
                        if isinstance(config.fees, dict):
                            # Combined fees: solve for size in: cash = size * price * (1 + pct) + fixed
                            pct_rate = config.fees.get('percentage', 0.0)
                            fixed_fee = config.fees.get('fixed', 0.0)
                            # size = (cash - fixed) / (price * (1 + pct))
                            # TASK-019: Buffer (0.9999) still needed despite PrecisionManager integration
                            # Root cause: Order sizing happens at signal generation (before fill),
                            # but actual costs include additional rounding at execution time.
                            # Buffer prevents "insufficient funds" errors from accumulated micro-differences.
                            size_raw = (cash * 0.9999 - fixed_fee) / (event.close * (1 + pct_rate))
                            # Round DOWN to valid precision for this asset
                            size = self.precision_mgr.round_quantity(size_raw)
                        elif config.fees > 0:
                            # Percentage only
                            # TASK-019: Buffer still needed (see combined fees comment above)
                            size_raw = (cash * 0.9999) / (event.close * (1 + config.fees))
                            # Round DOWN to valid precision for this asset
                            size = self.precision_mgr.round_quantity(size_raw)
                        else:
                            # No fees
                            # TASK-019: Buffer still needed (see combined fees comment above)
                            size_raw = (cash * 0.9999) / event.close
                            # Round DOWN to valid precision for this asset
                            size = self.precision_mgr.round_quantity(size_raw)
                    else:
                        size = config.size

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

                self.bar_idx += 1

        # Create data feed
        class SimpleDataFeed(DataFeed):
            def __init__(self, ohlcv_df):
                self.ohlcv = ohlcv_df.reset_index()
                self.idx = 0

            @property
            def is_exhausted(self) -> bool:
                """Check if all data has been consumed."""
                return self.idx >= len(self.ohlcv)

            def get_next_event(self) -> MarketEvent:
                """Get next market event."""
                from qengine.core.types import MarketDataType

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

            def peek_next_timestamp(self) -> datetime:
                """Peek at timestamp of next event without consuming it."""
                if self.is_exhausted:
                    return None
                return self.ohlcv.iloc[self.idx]['timestamp']

            def reset(self):
                """Reset to beginning."""
                self.idx = 0

            def seek(self, timestamp: datetime):
                """Seek to specific timestamp."""
                # Find first row with timestamp >= target
                mask = self.ohlcv['timestamp'] >= timestamp
                indices = mask[mask].index
                if len(indices) > 0:
                    self.idx = indices[0]
                else:
                    self.idx = len(self.ohlcv)

        # Create commission and slippage models
        # IMPORTANT: Must pass NoCommission/NoSlippage instead of None to avoid FillSimulator defaults
        from qengine.execution.commission import NoCommission
        from qengine.execution.slippage import NoSlippage
        # Import validation-specific VectorBT models
        import sys
        from pathlib import Path
        tests_path = Path(__file__).parent.parent.parent
        if str(tests_path) not in sys.path:
            sys.path.insert(0, str(tests_path))
        from validation.models import VectorBTCommission

        # Handle combined fees (percentage + fixed) or simple percentage
        if isinstance(config.fees, dict):
            # Combined fee model (percentage + fixed)
            commission_model = VectorBTCommission(
                fee_rate=config.fees.get('percentage', 0.0),
                fixed_fee=config.fees.get('fixed', 0.0)
            )
        elif config.fees > 0:
            # Simple percentage fee
            commission_model = PercentageCommission(rate=config.fees)
        else:
            # No fees
            commission_model = NoCommission()

        slippage_model = PercentageSlippage(slippage_pct=config.slippage) if config.slippage > 0 else NoSlippage()

        # Create broker with commission/slippage models
        from qengine.execution.broker import SimulationBroker
        broker = SimulationBroker(
            initial_cash=config.initial_cash,
            asset_registry=registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
            execution_delay=False,  # Disable for VectorBT comparison (VectorBT fills same-bar)
        )

        # Create strategy and data feed
        strategy = SignalStrategy(entries, exits, precision_mgr)
        data_feed = SimpleDataFeed(ohlcv)

        # Run backtest
        engine = BacktestEngine(
            data_feed=data_feed,
            strategy=strategy,
            broker=broker,
            initial_capital=config.initial_cash,
        )

        results = engine.run()

        # Get round-trip trades from trade_tracker, not broker.get_trades()
        # broker.get_trades() returns individual orders/fills
        # trade_tracker.get_trades_df() returns entry/exit paired trades
        trades = engine.broker.trade_tracker.get_trades_df()

        # Also get open positions as trades (for comparison with VectorBT which includes open positions)
        if engine.broker.trade_tracker.get_open_position_count() > 0:
            # Get final timestamp and price
            final_timestamp = ohlcv.index[-1]
            final_price = ohlcv['close'].iloc[-1]

            # Convert open positions to trade records
            open_position_trades = engine.broker.trade_tracker.get_open_positions_as_trades(
                current_timestamp=final_timestamp,
                current_price=final_price
            )

            # Convert to DataFrame and append
            if open_position_trades:
                import polars as pl
                # Convert to pandas directly to avoid Polars schema issues
                open_trades_dicts = [t.to_dict() for t in open_position_trades]
                open_trades_pd = pd.DataFrame(open_trades_dicts)

                # Concatenate with existing trades
                if len(trades) > 0:
                    # Convert both to pandas for concatenation
                    if hasattr(trades, 'to_pandas'):
                        trades_pd = trades.to_pandas()
                    else:
                        trades_pd = trades
                    trades = pd.concat([trades_pd, open_trades_pd], ignore_index=True)
                else:
                    trades = open_trades_pd

        if hasattr(trades, 'to_pandas'):
            # Convert Polars DataFrame to pandas
            trades_df = trades.to_pandas()
        else:
            trades_df = trades

        # Standardize column names to match expected format
        # qengine uses: entry_dt, exit_dt, entry_quantity, exit_quantity
        # expected: entry_time, exit_time, size, pnl, entry_price, exit_price
        if len(trades_df) > 0:
            trades_df = trades_df.rename(columns={
                'entry_dt': 'entry_time',
                'exit_dt': 'exit_time',
                'entry_quantity': 'size',
            })

        # Get final values from results
        final_value = results['final_value']
        final_cash = engine.portfolio.cash
        final_position = engine.broker.position_tracker.get_position("BTC")
        final_price = ohlcv['close'].iloc[-1]

        return BacktestResult(
            trades=trades_df,
            final_value=final_value,
            final_cash=final_cash,
            final_position=final_position,
            total_pnl=results['total_return'] / 100 * config.initial_cash,  # Convert percentage to dollar PnL
            num_trades=len(trades_df),
            engine_name='qengine',
        )


class VectorBTWrapper(EngineWrapper):
    """Wrapper for VectorBT Pro."""

    def run_backtest(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: Optional[pd.Series] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Run VectorBT backtest."""
        try:
            import vectorbtpro as vbt
        except ImportError:
            raise ImportError("VectorBT Pro not installed")

        if config is None:
            config = BacktestConfig()

        # Align signals with OHLCV index
        # entries and exits come with integer index, need to map to datetime index
        entries_aligned = pd.Series(entries.values, index=ohlcv.index)

        if exits is not None:
            exits_aligned = pd.Series(exits.values, index=ohlcv.index)
        else:
            # No exits - let VectorBT handle (will exit on next entry)
            exits_aligned = None

        # Run portfolio simulation
        size = np.inf if config.size is None else config.size

        # Handle combined fees (percentage + fixed) or simple fees
        if isinstance(config.fees, dict):
            # Combined fees: VectorBT uses separate parameters
            percentage_fees = config.fees.get('percentage', 0.0)
            fixed_fees = config.fees.get('fixed', 0.0)
        else:
            # Simple percentage fees
            percentage_fees = config.fees
            fixed_fees = 0.0

        pf = vbt.Portfolio.from_signals(
            close=ohlcv['close'],
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            entries=entries_aligned,
            exits=exits_aligned,
            init_cash=config.initial_cash,
            size=size,
            fees=percentage_fees,
            fixed_fees=fixed_fees,
            slippage=config.slippage,
            freq='1min',
        )

        # Extract trades
        trades = pf.trades.records_readable

        # Standardize trade format
        trades_df = pd.DataFrame({
            'entry_time': trades['Entry Index'],
            'entry_price': trades['Avg Entry Price'],
            'exit_time': trades['Exit Index'],
            'exit_price': trades['Avg Exit Price'],
            'pnl': trades['PnL'],
            'size': trades['Size'],
        })

        return BacktestResult(
            trades=trades_df,
            final_value=pf.value.iloc[-1],
            final_cash=pf.cash.iloc[-1],
            final_position=pf.assets.iloc[-1],
            total_pnl=pf.total_profit,
            num_trades=len(trades_df),
            engine_name='VectorBT',
        )


class ZiplineWrapper(EngineWrapper):
    """Wrapper for Zipline (placeholder)."""

    def run_backtest(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: Optional[pd.Series] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Run Zipline backtest."""
        raise NotImplementedError("Zipline wrapper not yet implemented")


class BacktraderWrapper(EngineWrapper):
    """Wrapper for Backtrader (placeholder)."""

    def run_backtest(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: Optional[pd.Series] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Run Backtrader backtest."""
        raise NotImplementedError("Backtrader wrapper not yet implemented")
