"""Random signal generator for stress testing."""
import random
from datetime import datetime

import polars as pl

from .base import Signal, SignalGenerator


class RandomSignals(SignalGenerator):
    """Generates random trading signals for stress testing.

    Useful for testing that platforms handle the same signals identically,
    even when signals don't follow any logical pattern.

    Args:
        signal_frequency: Probability of generating signal on each bar (0-1)
        quantity: Fixed quantity per trade
        seed: Random seed for reproducibility
        allow_stop_loss: Include random stop losses
        allow_take_profit: Include random take profits
        allow_trailing_stop: Include random trailing stops
    """

    def __init__(
        self,
        signal_frequency: float = 0.1,
        quantity: float = 100,
        seed: int = 42,
        allow_stop_loss: bool = True,
        allow_take_profit: bool = True,
        allow_trailing_stop: bool = False,
        name: str = "Random"
    ):
        super().__init__(name)
        self.signal_frequency = signal_frequency
        self.quantity = quantity
        self.seed = seed
        self.allow_stop_loss = allow_stop_loss
        self.allow_take_profit = allow_take_profit
        self.allow_trailing_stop = allow_trailing_stop
        self.rng = random.Random(seed)

    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate random signals."""
        self.validate_data(data)

        # Reset random state for reproducibility
        self.rng.seed(self.seed)

        df = data.sort('timestamp')
        signals = []
        position = 0  # Track position: 0=flat, 1=long, -1=short

        for row in df.iter_rows(named=True):
            # Randomly decide to signal
            if self.rng.random() > self.signal_frequency:
                continue

            # Randomly choose action based on current position
            if position == 0:
                # Flat: can only buy or sell
                action = self.rng.choice(['BUY', 'SELL'])
                position = 1 if action == 'BUY' else -1
            else:
                # In position: can close or do nothing
                if self.rng.random() < 0.7:  # 70% chance to close
                    action = 'CLOSE'
                    position = 0
                else:
                    continue  # Skip this signal

            # Random stop loss / take profit
            stop_loss = None
            take_profit = None
            trailing_stop_pct = None

            if action in ['BUY', 'SELL']:
                price = row['close']

                if self.allow_stop_loss and self.rng.random() < 0.3:
                    # 30% chance of stop loss (2-5% away)
                    sl_pct = self.rng.uniform(0.02, 0.05)
                    stop_loss = price * (1 - sl_pct) if action == 'BUY' else price * (1 + sl_pct)

                if self.allow_take_profit and self.rng.random() < 0.3:
                    # 30% chance of take profit (3-8% away)
                    tp_pct = self.rng.uniform(0.03, 0.08)
                    take_profit = price * (1 + tp_pct) if action == 'BUY' else price * (1 - tp_pct)

                if self.allow_trailing_stop and self.rng.random() < 0.2:
                    # 20% chance of trailing stop (1-3%)
                    trailing_stop_pct = self.rng.uniform(0.01, 0.03)

            signals.append(Signal(
                timestamp=row['timestamp'],
                symbol=row['symbol'],
                action=action,
                quantity=self.quantity if action != 'CLOSE' else None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop_pct=trailing_stop_pct,
            ))

        return signals

    def __repr__(self) -> str:
        return f"RandomSignals(freq={self.signal_frequency}, seed={self.seed})"
