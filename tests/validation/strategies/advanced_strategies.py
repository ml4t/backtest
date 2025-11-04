"""
Advanced Trading Strategies for Cross-Framework Validation

These strategies generate more realistic trade counts (50-200+ trades per year)
for comprehensive framework validation.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """Trading signal with metadata."""

    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL', 'SHORT', 'COVER'
    strength: float  # Signal strength 0-1
    reason: str  # Signal reason for debugging


class BollingerBandMeanReversion:
    """
    Mean reversion strategy using Bollinger Bands.

    Generates 100-200 trades per year on daily data.
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    - Use RSI for confirmation
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        min_holding_period: int = 2,  # Minimum bars to hold
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_holding_period = min_holding_period
        self.name = "BollingerBandMeanReversion"

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        df = data.copy()

        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * self.bb_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std * self.bb_std)

        # Calculate RSI
        df["rsi"] = self.calculate_rsi(df["close"])

        # Generate signals
        df["signal"] = 0
        position = 0
        holding_counter = 0

        entries = []
        exits = []

        for i in range(len(df)):
            if i < max(self.bb_period, self.rsi_period):
                continue

            price = df["close"].iloc[i]
            upper = df["bb_upper"].iloc[i]
            lower = df["bb_lower"].iloc[i]
            rsi = df["rsi"].iloc[i]

            # Decrement holding counter
            if holding_counter > 0:
                holding_counter -= 1

            # Entry conditions
            if position == 0 and holding_counter == 0:
                # Buy signal: price near lower band and RSI oversold
                if price <= lower * 1.01 and rsi < self.rsi_oversold:
                    entries.append(i)
                    position = 1
                    holding_counter = self.min_holding_period

            # Exit conditions
            elif position == 1 and holding_counter == 0:
                # Sell signal: price near upper band or RSI overbought
                if price >= upper * 0.99 or rsi > self.rsi_overbought:
                    exits.append(i)
                    position = 0
                    holding_counter = self.min_holding_period

        # Create boolean masks
        df["entries"] = False
        df["exits"] = False

        if entries:
            df.loc[df.index[entries], "entries"] = True
        if exits:
            df.loc[df.index[exits], "exits"] = True

        return df[["entries", "exits", "bb_upper", "bb_middle", "bb_lower", "rsi"]]

    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "min_holding_period": self.min_holding_period,
        }


class VolumeBreakoutStrategy:
    """
    Breakout strategy with volume confirmation.

    Generates 50-100 trades per year on daily data.
    - Buy on price breakout with volume surge
    - Use trailing stop for exits
    - ATR-based position sizing
    """

    def __init__(
        self,
        lookback_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        breakout_threshold: float = 0.02,  # 2% above resistance
    ):
        self.lookback_period = lookback_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.breakout_threshold = breakout_threshold
        self.name = "VolumeBreakoutStrategy"

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        df = data.copy()

        # Calculate indicators
        df["resistance"] = df["high"].rolling(window=self.lookback_period).max()
        df["support"] = df["low"].rolling(window=self.lookback_period).min()
        df["avg_volume"] = df["volume"].rolling(window=self.lookback_period).mean()
        df["atr"] = self.calculate_atr(df)

        # Initialize signal columns
        entries = []
        exits = []
        position = 0
        stop_price = 0

        for i in range(self.lookback_period, len(df)):
            price = df["close"].iloc[i]
            df["high"].iloc[i]
            volume = df["volume"].iloc[i]
            resistance = df["resistance"].iloc[i - 1]  # Previous bar's resistance
            avg_vol = df["avg_volume"].iloc[i]
            atr = df["atr"].iloc[i]

            if position == 0:
                # Check for breakout with volume
                if (
                    price > resistance * (1 + self.breakout_threshold)
                    and volume > avg_vol * self.volume_multiplier
                ):
                    entries.append(i)
                    position = 1
                    stop_price = price - (atr * self.atr_stop_multiplier)

            else:  # position == 1
                # Update trailing stop
                new_stop = price - (atr * self.atr_stop_multiplier)
                stop_price = max(stop_price, new_stop)

                # Check exit conditions
                if price <= stop_price or i == len(df) - 1:
                    exits.append(i)
                    position = 0
                    stop_price = 0

        # Create boolean masks
        df["entries"] = False
        df["exits"] = False

        if entries:
            df.loc[df.index[entries], "entries"] = True
        if exits:
            df.loc[df.index[exits], "exits"] = True

        return df[["entries", "exits", "resistance", "support", "atr"]]

    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "lookback_period": self.lookback_period,
            "volume_multiplier": self.volume_multiplier,
            "atr_period": self.atr_period,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "breakout_threshold": self.breakout_threshold,
        }


class ShortTermMomentumStrategy:
    """
    Short-term momentum strategy with frequent trading.

    Generates 150-300 trades per year on daily data.
    - Uses fast/slow EMA crossovers
    - ADX for trend strength confirmation
    - Quick exits on momentum loss
    """

    def __init__(
        self,
        fast_ema: int = 5,
        slow_ema: int = 15,
        adx_period: int = 14,
        adx_threshold: float = 25,
        profit_target: float = 0.03,  # 3% profit target
        stop_loss: float = 0.015,  # 1.5% stop loss
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.name = "ShortTermMomentumStrategy"

    def calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()

        return adx

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        df = data.copy()

        # Calculate indicators
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_ema, adjust=False).mean()
        df["adx"] = self.calculate_adx(df)

        # Generate signals
        entries = []
        exits = []
        position = 0
        entry_price = 0

        for i in range(max(self.slow_ema, self.adx_period), len(df)):
            price = df["close"].iloc[i]
            fast = df["ema_fast"].iloc[i]
            slow = df["ema_slow"].iloc[i]
            prev_fast = df["ema_fast"].iloc[i - 1]
            prev_slow = df["ema_slow"].iloc[i - 1]
            adx = df["adx"].iloc[i]

            if position == 0:
                # Entry: Fast EMA crosses above slow EMA with strong trend
                if prev_fast <= prev_slow and fast > slow and adx > self.adx_threshold:
                    entries.append(i)
                    position = 1
                    entry_price = price

            else:  # position == 1
                # Exit conditions
                profit = (price - entry_price) / entry_price

                # Exit on: profit target, stop loss, or momentum reversal
                if profit >= self.profit_target or profit <= -self.stop_loss or fast < slow:
                    exits.append(i)
                    position = 0
                    entry_price = 0

        # Create boolean masks
        df["entries"] = False
        df["exits"] = False

        if entries:
            df.loc[df.index[entries], "entries"] = True
        if exits:
            df.loc[df.index[exits], "exits"] = True

        return df[["entries", "exits", "ema_fast", "ema_slow", "adx"]]

    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "adx_period": self.adx_period,
            "adx_threshold": self.adx_threshold,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
        }
