"""Moving Average Crossover signal generator."""
from datetime import datetime

import polars as pl

from .base import Signal, SignalGenerator


class MACrossoverSignals(SignalGenerator):
    """Simple moving average crossover strategy.

    Generates BUY when fast MA crosses above slow MA.
    Generates SELL when fast MA crosses below slow MA.

    Args:
        fast_period: Fast MA period (default: 10)
        slow_period: Slow MA period (default: 30)
        quantity: Fixed quantity per trade (default: 100)
        stop_loss_pct: Optional stop loss percentage (default: None)
        take_profit_pct: Optional take profit percentage (default: None)
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        quantity: float = 100,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        name: str = "MA_Crossover"
    ):
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate MA crossover signals."""
        self.validate_data(data)

        # Calculate moving averages
        df = data.sort('timestamp').with_columns([
            pl.col('close').rolling_mean(self.fast_period).alias('fast_ma'),
            pl.col('close').rolling_mean(self.slow_period).alias('slow_ma'),
        ])

        # Remove rows with null MAs (warm-up period)
        df = df.filter(pl.col('fast_ma').is_not_null() & pl.col('slow_ma').is_not_null())

        # Detect crossovers
        df = df.with_columns([
            (pl.col('fast_ma') > pl.col('slow_ma')).alias('fast_above_slow'),
        ])

        # Shift to detect changes
        df = df.with_columns([
            pl.col('fast_above_slow').shift(1).alias('prev_fast_above_slow'),
        ])

        # Filter for actual crossovers (skip first row due to shift)
        df = df.filter(pl.col('prev_fast_above_slow').is_not_null())

        # Generate signals
        signals = []

        for row in df.iter_rows(named=True):
            current_above = row['fast_above_slow']
            prev_above = row['prev_fast_above_slow']

            # Bullish crossover: fast crosses above slow
            if current_above and not prev_above:
                stop_loss = None
                take_profit = None

                if self.stop_loss_pct:
                    stop_loss = row['close'] * (1 - self.stop_loss_pct)
                if self.take_profit_pct:
                    take_profit = row['close'] * (1 + self.take_profit_pct)

                signals.append(Signal(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    action='BUY',
                    quantity=self.quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                ))

            # Bearish crossover: fast crosses below slow
            elif not current_above and prev_above:
                # Use CLOSE instead of SELL to exit long positions
                signals.append(Signal(
                    timestamp=row['timestamp'],
                    symbol=row['symbol'],
                    action='CLOSE',
                    quantity=self.quantity,
                ))

        return signals

    def __repr__(self) -> str:
        return (f"MACrossoverSignals(fast={self.fast_period}, slow={self.slow_period}, "
                f"qty={self.quantity}, sl={self.stop_loss_pct}, tp={self.take_profit_pct})")
