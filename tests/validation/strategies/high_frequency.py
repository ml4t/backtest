"""
High-Frequency Trading Strategies for Cross-Framework Validation

These strategies generate 100+ trades per year on daily data using aggressive
signal generation and short holding periods.
"""

from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy


class ScalpingStrategy(BaseStrategy):
    """
    Aggressive scalping strategy using short-term momentum.

    Rules:
    - ENTRY: Fast EMA crosses above slow EMA AND momentum positive
    - EXIT: Fast EMA crosses below slow EMA OR profit target OR stop loss

    Targets 100-200 trades per year.
    """

    def __init__(
        self,
        fast_ema: int = 3,
        slow_ema: int = 8,
        momentum_period: int = 5,
        profit_target: float = 0.005,  # 0.5% profit target
        stop_loss: float = 0.003,  # 0.3% stop loss
        max_holding_days: int = 3,  # Exit after 3 days max
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.momentum_period = momentum_period
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate high-frequency trading signals."""
        df = data.copy()

        # Calculate EMAs
        df["ema_fast"] = df["close"].ewm(span=self.fast_ema, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_ema, adjust=False).mean()

        # Calculate momentum
        df["momentum"] = df["close"].pct_change(self.momentum_period)

        # Initialize signals
        df["entry"] = False
        df["exit"] = False

        # Track position
        position = False
        entry_price = 0
        holding_days = 0

        min_period = max(self.slow_ema, self.momentum_period)

        for i in range(min_period, len(df)):
            if position:
                holding_days += 1
                current_price = df["close"].iloc[i]
                pnl = (current_price - entry_price) / entry_price

                # Exit conditions
                exit_signal = False

                # Stop loss or take profit
                if (
                    pnl >= self.profit_target
                    or pnl <= -self.stop_loss
                    or holding_days >= self.max_holding_days
                    or df["ema_fast"].iloc[i] < df["ema_slow"].iloc[i]
                ):
                    exit_signal = True

                if exit_signal:
                    df.iloc[i, df.columns.get_loc("exit")] = True
                    position = False
                    holding_days = 0

            else:
                # Entry conditions
                if (
                    df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i]
                    and df["ema_fast"].iloc[i - 1] <= df["ema_slow"].iloc[i - 1]
                    and df["momentum"].iloc[i] > 0
                ):
                    df.iloc[i, df.columns.get_loc("entry")] = True
                    position = True
                    entry_price = df["close"].iloc[i]
                    holding_days = 0

        # Close any remaining position
        if position and len(df) > 0:
            df.iloc[-1, df.columns.get_loc("exit")] = True

        return df

    def get_name(self) -> str:
        return "ScalpingStrategy"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "momentum_period": self.momentum_period,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "max_holding_days": self.max_holding_days,
        }


class MicroReversalStrategy(BaseStrategy):
    """
    Micro reversal strategy catching small price corrections.

    Rules:
    - ENTRY: Price drops X% below 5-day MA AND RSI oversold
    - EXIT: Price returns to MA OR small profit target

    Very aggressive to generate many trades.
    """

    def __init__(
        self,
        ma_period: int = 5,
        entry_threshold: float = 0.01,  # Enter when 1% below MA
        rsi_period: int = 5,  # Very short RSI
        rsi_oversold: float = 35,  # Less extreme for more signals
        rsi_overbought: float = 65,  # Less extreme for more signals
        profit_target: float = 0.003,  # 0.3% quick profit
        max_holding_days: int = 2,  # Very short holding
    ):
        self.ma_period = ma_period
        self.entry_threshold = entry_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.profit_target = profit_target
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
        """Generate micro reversal signals."""
        df = data.copy()

        # Calculate indicators
        df["ma"] = df["close"].rolling(window=self.ma_period).mean()
        df["rsi"] = self.calculate_rsi(df["close"])
        df["price_to_ma"] = (df["close"] - df["ma"]) / df["ma"]

        # Initialize signals
        df["entry"] = False
        df["exit"] = False

        # Track position
        position = False
        entry_price = 0
        holding_days = 0

        min_period = max(self.ma_period, self.rsi_period)

        for i in range(min_period, len(df)):
            if pd.isna(df["ma"].iloc[i]) or pd.isna(df["rsi"].iloc[i]):
                continue

            if position:
                holding_days += 1
                current_price = df["close"].iloc[i]
                pnl = (current_price - entry_price) / entry_price

                # Exit conditions
                exit_signal = False

                # Quick profit target
                if (
                    pnl >= self.profit_target
                    or df["price_to_ma"].iloc[i] >= 0
                    or df["rsi"].iloc[i] > self.rsi_overbought
                    or holding_days >= self.max_holding_days
                ):
                    exit_signal = True

                if exit_signal:
                    df.iloc[i, df.columns.get_loc("exit")] = True
                    position = False
                    holding_days = 0

            else:
                # Entry: Price below MA and RSI oversold
                if (
                    df["price_to_ma"].iloc[i] < -self.entry_threshold
                    and df["rsi"].iloc[i] < self.rsi_oversold
                ):
                    df.iloc[i, df.columns.get_loc("entry")] = True
                    position = True
                    entry_price = df["close"].iloc[i]
                    holding_days = 0

        # Close any remaining position
        if position and len(df) > 0:
            df.iloc[-1, df.columns.get_loc("exit")] = True

        return df

    def get_name(self) -> str:
        return "MicroReversalStrategy"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "ma_period": self.ma_period,
            "entry_threshold": self.entry_threshold,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "profit_target": self.profit_target,
            "max_holding_days": self.max_holding_days,
        }


class VolatilityBreakoutScalper(BaseStrategy):
    """
    Scalp volatility breakouts with very tight stops.

    Rules:
    - ENTRY: Price breaks above/below Bollinger Band with volume spike
    - EXIT: Return to middle band OR tight stop/target

    Generates many quick trades.
    """

    def __init__(
        self,
        bb_period: int = 10,
        bb_std: float = 1.5,  # Tighter bands for more signals
        volume_threshold: float = 1.2,  # Lower threshold for more signals
        profit_target: float = 0.004,  # 0.4% quick profit
        stop_loss: float = 0.002,  # 0.2% tight stop
        max_holding_days: int = 2,  # Very short holding
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_threshold = volume_threshold
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days

    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals."""
        df = data.copy()

        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        bb_std_dev = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std_dev * self.bb_std)
        df["bb_lower"] = df["bb_middle"] - (bb_std_dev * self.bb_std)

        # Volume analysis
        df["volume_ma"] = df["volume"].rolling(window=self.bb_period).mean()
        df["volume_spike"] = df["volume"] / df["volume_ma"]

        # Initialize signals
        df["entry"] = False
        df["exit"] = False

        # Track position
        position = False
        position_type = None  # 'long' or 'short'
        entry_price = 0
        holding_days = 0

        for i in range(self.bb_period, len(df)):
            if pd.isna(df["bb_upper"].iloc[i]) or pd.isna(df["volume_spike"].iloc[i]):
                continue

            close = df["close"].iloc[i]

            if position:
                holding_days += 1
                pnl = (close - entry_price) / entry_price

                if position_type == "short":
                    pnl = -pnl

                # Exit conditions
                exit_signal = False

                # Stop loss or profit target
                if (
                    pnl >= self.profit_target
                    or pnl <= -self.stop_loss
                    or (position_type == "long" and close <= df["bb_middle"].iloc[i])
                    or (position_type == "short" and close >= df["bb_middle"].iloc[i])
                    or holding_days >= self.max_holding_days
                ):
                    exit_signal = True

                if exit_signal:
                    df.iloc[i, df.columns.get_loc("exit")] = True
                    position = False
                    position_type = None
                    holding_days = 0

            else:
                # Entry conditions with volume confirmation
                if df["volume_spike"].iloc[i] >= self.volume_threshold:
                    # Breakout above upper band - go long
                    if close > df["bb_upper"].iloc[i]:
                        df.iloc[i, df.columns.get_loc("entry")] = True
                        position = True
                        position_type = "long"
                        entry_price = close
                        holding_days = 0

                    # Breakout below lower band - treat as reversal entry
                    elif close < df["bb_lower"].iloc[i]:
                        df.iloc[i, df.columns.get_loc("entry")] = True
                        position = True
                        position_type = "long"  # Buy the dip
                        entry_price = close
                        holding_days = 0

        # Close any remaining position
        if position and len(df) > 0:
            df.iloc[-1, df.columns.get_loc("exit")] = True

        return df

    def get_name(self) -> str:
        return "VolatilityBreakoutScalper"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "volume_threshold": self.volume_threshold,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "max_holding_days": self.max_holding_days,
        }
