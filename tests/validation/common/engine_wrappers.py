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
    limit_offset: float = 0.02  # Offset from market price for limit orders (default 2%)
    size: Optional[float] = None  # None = use all cash (size=np.inf in VBT)
    close_final_position: bool = False  # Auto-close open positions at backtest end (default: False)
    execution_delay_days: int = 0  # Execute signals N days later (0=same-day, 1=next-day like Zipline)


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
    returns: pd.Series | None = None  # Daily returns (percentage)
    equity_curve: pd.Series | None = None  # Portfolio value over time


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


class BacktestWrapper(EngineWrapper):
    """Wrapper for ml4t.backtest."""

    def run_backtest(
        self,
        ohlcv: pd.DataFrame,
        entries: pd.Series,
        exits: Optional[pd.Series] = None,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Run ml4t.backtest backtest."""
        # Import from new modular API
        from ml4t.backtest import (
            Engine,
            DataFeed,
            Strategy,
            PercentageCommission,
            NoCommission,
            NoSlippage,
            Order,
            OrderType,
            OrderSide,
            ExecutionMode,
        )
        import polars as pl

        if config is None:
            config = BacktestConfig()

        # Convert OHLCV to Polars DataFrame for DataFeed
        # Add asset column (single asset "BTC")
        # Ensure timestamps are datetime objects, not pandas Timestamps
        timestamps = [pd.Timestamp(ts).to_pydatetime() for ts in ohlcv.index]

        prices_df = pl.DataFrame({
            'timestamp': timestamps,
            'asset': ['BTC'] * len(ohlcv),
            'open': ohlcv['open'].values.astype(float),
            'high': ohlcv['high'].values.astype(float),
            'low': ohlcv['low'].values.astype(float),
            'close': ohlcv['close'].values.astype(float),
            'volume': ohlcv['volume'].values.astype(float),
        })

        # Convert signals to Polars DataFrame
        signals_df = pl.DataFrame({
            'timestamp': timestamps,
            'asset': ['BTC'] * len(entries),
            'entry': entries.values,
            'exit': exits.values if exits is not None else [False] * len(entries),
        })

        # Create strategy that trades on signals
        class SignalStrategy(Strategy):
            def __init__(self):
                super().__init__()

            def on_data(self, timestamp, data, context, broker):
                # data is dict[asset, dict] with keys: open, high, low, close, volume, signals
                asset_data = data.get('BTC', {})
                signals = asset_data.get('signals', {})

                entry_signal = signals.get('entry', False)
                exit_signal = signals.get('exit', False)

                # Get current position
                position = broker.get_position('BTC')
                current_qty = position.quantity if position else 0.0

                # Exit logic (explicit exit signal OR new entry while holding position)
                should_exit = (exit_signal and current_qty != 0) or (entry_signal and current_qty != 0)

                if should_exit:
                    # Close position
                    broker.submit_order(
                        asset='BTC',
                        side=OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                        quantity=abs(current_qty),
                        order_type=OrderType.MARKET,
                    )

                # Entry logic
                if entry_signal:
                    # Calculate position size
                    if config.size is None:
                        # Use all available cash
                        cash = broker.cash
                        price = asset_data.get('close')

                        # Account for slippage and commission in sizing
                        slippage_mult = 1.0 + config.slippage
                        effective_price = price * slippage_mult

                        if isinstance(config.fees, dict):
                            pct = config.fees.get('percentage', 0.0)
                            fixed = config.fees.get('fixed', 0.0)
                            size = (cash * 0.9999 - fixed) / (effective_price * (1 + pct))
                        elif config.fees > 0:
                            size = (cash * 0.9999) / (effective_price * (1 + config.fees))
                        else:
                            size = (cash * 0.9999) / effective_price
                    else:
                        size = config.size

                    if size > 0:
                        broker.submit_order(
                            asset='BTC',
                            side=OrderSide.BUY,
                            quantity=size,
                            order_type=OrderType.MARKET,
                        )

        # Create commission and slippage models
        if isinstance(config.fees, dict):
            # Combined fee model not supported in new API - use percentage only
            from ml4t.backtest import PerShareCommission
            commission_model = PercentageCommission(rate=config.fees.get('percentage', 0.0))
        elif config.fees > 0:
            commission_model = PercentageCommission(rate=config.fees)
        else:
            commission_model = NoCommission()

        from ml4t.backtest import PercentageSlippage
        slippage_model = PercentageSlippage(rate=config.slippage) if config.slippage > 0 else NoSlippage()

        # Create DataFeed with prices and signals
        feed = DataFeed(prices_df=prices_df, signals_df=signals_df)

        # Create strategy
        strategy = SignalStrategy()

        # Determine execution mode
        execution_mode = ExecutionMode.NEXT_BAR if config.execution_delay_days > 0 else ExecutionMode.SAME_BAR

        # Run backtest
        engine = Engine(
            feed=feed,
            strategy=strategy,
            initial_cash=config.initial_cash,
            commission_model=commission_model,
            slippage_model=slippage_model,
            execution_mode=execution_mode,
            account_type='cash',  # Use cash account for validation
        )

        results = engine.run()

        # Extract trades
        trades = engine.broker.trades

        # Calculate commission per trade based on config
        def calc_commission(price, quantity, fees):
            if isinstance(fees, dict):
                # Combined fees: percentage + fixed
                pct = fees.get('percentage', 0.0)
                fixed = fees.get('fixed', 0.0)
                return (price * quantity * pct) + fixed
            elif fees > 0:
                # Simple percentage fees
                return price * quantity * fees
            else:
                return 0.0

        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'entry_price': t.entry_price,
                'exit_time': t.exit_time,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'size': t.quantity,
                # Commission details - calculate from notional values
                'entry_commission': calc_commission(t.entry_price, t.quantity, config.fees),
                'exit_commission': calc_commission(t.exit_price, t.quantity, config.fees),
                'exit_quantity': t.quantity,  # Exit quantity same as entry quantity for round trips
            }
            for t in trades if t.exit_time is not None
        ])

        # Get final values from results
        final_value = results['final_value']
        final_cash = engine.broker.cash
        position = engine.broker.get_position('BTC')
        final_position = position.quantity if position else 0.0

        return BacktestResult(
            trades=trades_df,
            final_value=final_value,
            final_cash=final_cash,
            final_position=final_position,
            total_pnl=results['total_return'] * config.initial_cash,  # Convert ratio to dollars
            num_trades=len(trades_df),
            engine_name='ml4t.backtest',
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
