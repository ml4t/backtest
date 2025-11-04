"""
Base Strategy Definition for Cross-Framework Validation

Defines strategies as pure signal generators that output entry/exit points.
Each framework adapter then executes these identical signals.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies should be pure signal generators that:
    1. Take OHLCV data
    2. Calculate indicators
    3. Return entry/exit signals

    The signals should be deterministic - same data = same signals.
    """

    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate entry and exit signals from OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  and DatetimeIndex

        Returns:
            DataFrame with at minimum:
            - 'entry': bool - True when entering position
            - 'exit': bool - True when exiting position
            - Any indicator columns for debugging
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters for reproducibility."""


class RSIMeanReversionStrategy(BaseStrategy):
    """
    Simple RSI mean reversion strategy with clear rules.

    Rules:
    - ENTRY: RSI < 30 (oversold)
    - EXIT: RSI > 70 (overbought) OR holding > max_holding_days

    This should generate 50-100 trades per year on daily data.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_level: float = 30,
        overbought_level: float = 70,
        max_holding_days: int = 20,
    ):
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.max_holding_days = max_holding_days

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI levels."""
        df = data.copy()

        # Calculate RSI
        df["rsi"] = self.calculate_rsi(df["close"])

        # Initialize signal columns
        df["entry"] = False
        df["exit"] = False

        # Track position state
        position = False
        holding_days = 0

        for i in range(self.rsi_period, len(df)):
            rsi = df["rsi"].iloc[i]

            if pd.isna(rsi):
                continue

            if not position:
                # Entry condition: RSI < oversold
                if rsi < self.oversold_level:
                    df.iloc[i, df.columns.get_loc("entry")] = True
                    position = True
                    holding_days = 0
            else:
                holding_days += 1
                # Exit conditions: RSI > overbought OR max holding reached
                if rsi > self.overbought_level or holding_days >= self.max_holding_days:
                    df.iloc[i, df.columns.get_loc("exit")] = True
                    position = False
                    holding_days = 0

        # Ensure last position is closed
        if position and len(df) > 0:
            df.iloc[-1, df.columns.get_loc("exit")] = True

        return df

    def get_name(self) -> str:
        return "RSIMeanReversion"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "rsi_period": self.rsi_period,
            "oversold_level": self.oversold_level,
            "overbought_level": self.overbought_level,
            "max_holding_days": self.max_holding_days,
        }


class DualMovingAverageCrossover(BaseStrategy):
    """
    Classic dual moving average crossover strategy.

    Rules:
    - ENTRY: Fast MA crosses above Slow MA
    - EXIT: Fast MA crosses below Slow MA

    This should generate 10-20 trades per year on daily data.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover."""
        df = data.copy()

        # Calculate moving averages
        df["ma_fast"] = df["close"].rolling(window=self.fast_period).mean()
        df["ma_slow"] = df["close"].rolling(window=self.slow_period).mean()

        # Calculate crossover points
        df["ma_diff"] = df["ma_fast"] - df["ma_slow"]
        df["ma_diff_prev"] = df["ma_diff"].shift(1)

        # Entry: fast crosses above slow
        df["entry"] = (df["ma_diff"] > 0) & (df["ma_diff_prev"] <= 0)

        # Exit: fast crosses below slow
        df["exit"] = (df["ma_diff"] <= 0) & (df["ma_diff_prev"] > 0)

        # Remove signals before we have enough data
        min_period = max(self.fast_period, self.slow_period)
        df.loc[: df.index[min_period - 1], ["entry", "exit"]] = False

        return df

    def get_name(self) -> str:
        return "DualMovingAverageCrossover"

    def get_parameters(self) -> dict[str, Any]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}


class BollingerBandBreakout(BaseStrategy):
    """
    Bollinger Band breakout strategy with volume confirmation.

    Rules:
    - ENTRY: Close > Upper BB AND Volume > 1.5x average volume
    - EXIT: Close < Middle BB OR holding > max_holding_days

    This should generate 30-60 trades per year on daily data.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_multiplier: float = 1.5,
        max_holding_days: int = 10,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_multiplier = volume_multiplier
        self.max_holding_days = max_holding_days

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on BB breakout."""
        df = data.copy()

        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std_dev = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std_dev * self.bb_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std_dev * self.bb_std)

        # Calculate average volume
        df["avg_volume"] = df["volume"].rolling(window=self.bb_period).mean()

        # Initialize signals
        df["entry"] = False
        df["exit"] = False

        # Track position
        position = False
        holding_days = 0

        for i in range(self.bb_period, len(df)):
            close = df["close"].iloc[i]
            volume = df["volume"].iloc[i]
            upper = df["bb_upper"].iloc[i]
            middle = df["bb_middle"].iloc[i]
            avg_vol = df["avg_volume"].iloc[i]

            if pd.isna(upper) or pd.isna(middle):
                continue

            if not position:
                # Entry: breakout above upper band with volume
                if close > upper and volume > avg_vol * self.volume_multiplier:
                    df.iloc[i, df.columns.get_loc("entry")] = True
                    position = True
                    holding_days = 0
            else:
                holding_days += 1
                # Exit: price returns to middle OR max holding
                if close < middle or holding_days >= self.max_holding_days:
                    df.iloc[i, df.columns.get_loc("exit")] = True
                    position = False
                    holding_days = 0

        # Close any open position
        if position and len(df) > 0:
            df.iloc[-1, df.columns.get_loc("exit")] = True

        return df

    def get_name(self) -> str:
        return "BollingerBandBreakout"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "volume_multiplier": self.volume_multiplier,
            "max_holding_days": self.max_holding_days,
        }
