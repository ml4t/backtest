"""Mean reversion signal generator using RSI."""
import polars as pl

from .base import Signal, SignalGenerator


class MeanReversionSignals(SignalGenerator):
    """RSI-based mean reversion strategy.

    Generates BUY when RSI < oversold threshold.
    Generates SELL when RSI > overbought threshold.

    Args:
        rsi_period: RSI calculation period (default: 14)
        oversold: RSI oversold threshold (default: 30)
        overbought: RSI overbought threshold (default: 70)
        quantity: Fixed quantity per trade (default: 100)
        stop_loss_pct: Optional stop loss percentage
        take_profit_pct: Optional take profit percentage
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        quantity: float = 100,
        stop_loss_pct: float | None = 0.05,  # 5% default stop
        take_profit_pct: float | None = 0.10,  # 10% default target
        name: str = "Mean_Reversion"
    ):
        super().__init__(name)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.quantity = quantity
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def calculate_rsi(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate RSI indicator."""
        # Calculate price changes
        df = df.with_columns([
            (pl.col('close') - pl.col('close').shift(1)).alias('price_change')
        ])

        # Separate gains and losses
        df = df.with_columns([
            pl.when(pl.col('price_change') > 0)
              .then(pl.col('price_change'))
              .otherwise(0)
              .alias('gain'),
            pl.when(pl.col('price_change') < 0)
              .then(-pl.col('price_change'))
              .otherwise(0)
              .alias('loss'),
        ])

        # Calculate average gain/loss using EMA
        df = df.with_columns([
            pl.col('gain').ewm_mean(span=self.rsi_period, adjust=False).alias('avg_gain'),
            pl.col('loss').ewm_mean(span=self.rsi_period, adjust=False).alias('avg_loss'),
        ])

        # Calculate RS and RSI
        df = df.with_columns([
            (pl.col('avg_gain') / pl.col('avg_loss')).alias('rs')
        ])

        df = df.with_columns([
            (100 - (100 / (1 + pl.col('rs')))).alias('rsi')
        ])

        return df

    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate mean reversion signals."""
        self.validate_data(data)

        # Calculate RSI
        df = data.sort('timestamp')
        df = self.calculate_rsi(df)

        # Remove warm-up period
        df = df.filter(pl.col('rsi').is_not_null())

        # Track position state
        signals = []
        in_position = False

        for row in df.iter_rows(named=True):
            rsi = row['rsi']
            price = row['close']

            # Entry signals
            if not in_position:
                if rsi < self.oversold:
                    # Oversold: Buy
                    stop_loss = None
                    take_profit = None

                    if self.stop_loss_pct:
                        stop_loss = price * (1 - self.stop_loss_pct)
                    if self.take_profit_pct:
                        take_profit = price * (1 + self.take_profit_pct)

                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        action='BUY',
                        quantity=self.quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    ))
                    in_position = True

                elif rsi > self.overbought:
                    # Overbought: Sell short
                    stop_loss = None
                    take_profit = None

                    if self.stop_loss_pct:
                        stop_loss = price * (1 + self.stop_loss_pct)
                    if self.take_profit_pct:
                        take_profit = price * (1 - self.take_profit_pct)

                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        action='SELL',
                        quantity=self.quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                    ))
                    in_position = True

            # Exit signals (mean reversion: exit when RSI normalizes)
            else:
                if 45 < rsi < 55:  # RSI near 50 = exit
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        symbol=row['symbol'],
                        action='CLOSE',
                        quantity=None,  # Close all
                    ))
                    in_position = False

        return signals

    def __repr__(self) -> str:
        return (f"MeanReversionSignals(rsi={self.rsi_period}, "
                f"oversold={self.oversold}, overbought={self.overbought})")
