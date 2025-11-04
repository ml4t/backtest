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
    fees: float = 0.0
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

        # Create strategy that trades on signals
        class SignalStrategy(Strategy):
            def __init__(self, entries_ser, exits_ser):
                super().__init__()
                self.entries = entries_ser
                self.exits = exits_ser if exits_ser is not None else pd.Series([False] * len(entries_ser))
                self.bar_idx = 0

            def on_market_event(self, event: MarketEvent):
                # Get signal for this bar
                entry_signal = self.entries.iloc[self.bar_idx] if self.bar_idx < len(self.entries) else False
                exit_signal = self.exits.iloc[self.bar_idx] if self.bar_idx < len(self.exits) else False

                # Check current position
                position = self.broker.get_position(event.asset_id)

                # Exit logic
                if exit_signal and position != 0:
                    # Exit entire position
                    self.broker.submit_market_order(
                        asset_id=event.asset_id,
                        quantity=abs(position),
                        side='sell' if position > 0 else 'buy',
                    )

                # Entry logic
                if entry_signal and position == 0:
                    # Calculate position size (use all cash if size not specified)
                    if config.size is None:
                        # Use all available cash
                        cash = self.broker.get_cash()
                        size = cash / event.close  # Buy as much as possible
                    else:
                        size = config.size

                    self.broker.submit_market_order(
                        asset_id=event.asset_id,
                        quantity=size,
                        side='buy',
                    )

                self.bar_idx += 1

        # Create data feed
        class SimpleDataFeed(DataFeed):
            def __init__(self, ohlcv_df):
                self.ohlcv = ohlcv_df.reset_index()
                self.idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.idx >= len(self.ohlcv):
                    raise StopIteration

                row = self.ohlcv.iloc[self.idx]
                event = MarketEvent(
                    timestamp=row['timestamp'],
                    asset_id="BTC",
                    price=row['close'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                )
                self.idx += 1
                return event

        # Create commission and slippage models
        commission_model = PercentageCommission(rate=config.fees) if config.fees > 0 else None
        slippage_model = PercentageSlippage(slippage=config.slippage) if config.slippage > 0 else None

        # Run backtest
        engine = BacktestEngine(
            initial_capital=config.initial_cash,
            asset_registry=registry,
            commission_model=commission_model,
            slippage_model=slippage_model,
        )

        strategy = SignalStrategy(entries, exits)
        data_feed = SimpleDataFeed(ohlcv)

        engine.run(strategy, data_feed)

        # Extract results
        trades_df = engine.broker.trade_tracker.get_trades_dataframe()

        # Calculate final values
        final_cash = engine.broker.get_cash()
        final_position = engine.broker.get_position("BTC")
        final_price = ohlcv['close'].iloc[-1]
        final_value = final_cash + final_position * final_price

        return BacktestResult(
            trades=trades_df,
            final_value=final_value,
            final_cash=final_cash,
            final_position=final_position,
            total_pnl=final_value - config.initial_cash,
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

        pf = vbt.Portfolio.from_signals(
            close=ohlcv['close'],
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            entries=entries_aligned,
            exits=exits_aligned,
            init_cash=config.initial_cash,
            size=size,
            fees=config.fees,
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
