"""
High-Frequency Trading Strategies for Cross-Framework Validation

These strategies generate 100+ trades per year for more convincing validation.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""

    name: str
    description: str
    expected_trades_per_year: int

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70

    # Bollinger Band parameters
    bb_period: int = 20
    bb_std: float = 2.0

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Mean reversion parameters
    mean_reversion_period: int = 5
    mean_reversion_threshold: float = 0.02  # 2% deviation

    # Scalping parameters
    scalp_profit_target: float = 0.003  # 0.3% profit target
    scalp_stop_loss: float = 0.002  # 0.2% stop loss
    scalp_holding_period: int = 3  # Max days to hold


class HighFrequencyStrategies:
    """Collection of high-frequency trading strategies."""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def rsi_mean_reversion_signals(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """
        RSI Mean Reversion Strategy
        Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)
        Expected: 50-100 trades per year
        """
        signals = data.copy()

        # Calculate RSI
        signals["rsi"] = HighFrequencyStrategies.calculate_rsi(data["close"], config.rsi_period)

        # Generate signals
        signals["entry"] = signals["rsi"] < config.rsi_oversold
        signals["exit"] = signals["rsi"] > config.rsi_overbought

        # Clean up signals (can't enter if already in position)
        position = False
        clean_entries = []
        clean_exits = []

        for i in range(len(signals)):
            if signals["entry"].iloc[i] and not position:
                clean_entries.append(True)
                clean_exits.append(False)
                position = True
            elif signals["exit"].iloc[i] and position:
                clean_entries.append(False)
                clean_exits.append(True)
                position = False
            else:
                clean_entries.append(False)
                clean_exits.append(False)

        signals["entry"] = clean_entries
        signals["exit"] = clean_exits

        return signals

    @staticmethod
    def bollinger_squeeze_signals(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """
        Bollinger Band Squeeze Strategy
        Buy when price breaks above upper band, Sell when crosses below middle band
        Expected: 40-80 trades per year
        """
        signals = data.copy()

        # Calculate Bollinger Bands
        upper, middle, lower = HighFrequencyStrategies.calculate_bollinger_bands(
            data["close"],
            config.bb_period,
            config.bb_std,
        )

        signals["bb_upper"] = upper
        signals["bb_middle"] = middle
        signals["bb_lower"] = lower

        # Generate signals
        signals["price_above_upper"] = data["close"] > upper
        signals["price_below_middle"] = data["close"] < middle

        # Detect breakouts
        signals["entry"] = (signals["price_above_upper"]) & (~signals["price_above_upper"].shift(1))
        signals["exit"] = (signals["price_below_middle"]) & (
            ~signals["price_below_middle"].shift(1)
        )

        # Clean up signals
        position = False
        clean_entries = []
        clean_exits = []

        for i in range(len(signals)):
            if signals["entry"].iloc[i] and not position:
                clean_entries.append(True)
                clean_exits.append(False)
                position = True
            elif signals["exit"].iloc[i] and position:
                clean_entries.append(False)
                clean_exits.append(True)
                position = False
            else:
                clean_entries.append(False)
                clean_exits.append(False)

        signals["entry"] = clean_entries
        signals["exit"] = clean_exits

        return signals

    @staticmethod
    def macd_momentum_signals(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """
        MACD Momentum Strategy
        Buy when MACD crosses above signal, Sell when crosses below
        Expected: 30-60 trades per year
        """
        signals = data.copy()

        # Calculate MACD
        macd, signal_line = HighFrequencyStrategies.calculate_macd(
            data["close"],
            config.macd_fast,
            config.macd_slow,
            config.macd_signal,
        )

        signals["macd"] = macd
        signals["signal_line"] = signal_line

        # Generate crossover signals
        signals["macd_above"] = macd > signal_line
        signals["entry"] = (signals["macd_above"]) & (~signals["macd_above"].shift(1))
        signals["exit"] = (~signals["macd_above"]) & (signals["macd_above"].shift(1))

        # Clean up signals
        position = False
        clean_entries = []
        clean_exits = []

        for i in range(len(signals)):
            if signals["entry"].iloc[i] and not position:
                clean_entries.append(True)
                clean_exits.append(False)
                position = True
            elif signals["exit"].iloc[i] and position:
                clean_entries.append(False)
                clean_exits.append(True)
                position = False
            else:
                clean_entries.append(False)
                clean_exits.append(False)

        signals["entry"] = clean_entries
        signals["exit"] = clean_exits

        return signals

    @staticmethod
    def intraday_momentum_signals(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """
        Intraday Momentum Strategy (simulated with daily data)
        Buy on 5-day high breakout, Sell on 5-day low breakdown
        Expected: 80-150 trades per year
        """
        signals = data.copy()

        # Calculate rolling highs and lows
        period = 5
        signals["high_5d"] = data["high"].rolling(window=period).max()
        signals["low_5d"] = data["low"].rolling(window=period).min()

        # Shift to avoid look-ahead bias
        signals["prev_high_5d"] = signals["high_5d"].shift(1)
        signals["prev_low_5d"] = signals["low_5d"].shift(1)

        # Generate signals
        signals["entry"] = data["close"] > signals["prev_high_5d"]
        signals["exit"] = data["close"] < signals["prev_low_5d"]

        # Clean up signals
        position = False
        clean_entries = []
        clean_exits = []

        for i in range(len(signals)):
            if i < period:  # Skip warmup period
                clean_entries.append(False)
                clean_exits.append(False)
                continue

            if signals["entry"].iloc[i] and not position:
                clean_entries.append(True)
                clean_exits.append(False)
                position = True
            elif signals["exit"].iloc[i] and position:
                clean_entries.append(False)
                clean_exits.append(True)
                position = False
            else:
                clean_entries.append(False)
                clean_exits.append(False)

        signals["entry"] = clean_entries
        signals["exit"] = clean_exits

        return signals

    @staticmethod
    def mean_reversion_scalping_signals(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """
        Mean Reversion Scalping Strategy
        Buy when price deviates -2% from 5-day mean, Exit at +0.3% profit or -0.2% stop
        Expected: 100-200 trades per year
        """
        signals = data.copy()

        # Calculate short-term mean
        signals["sma_5"] = data["close"].rolling(window=config.mean_reversion_period).mean()

        # Calculate deviation from mean
        signals["deviation"] = (data["close"] - signals["sma_5"]) / signals["sma_5"]

        # Initialize tracking variables
        position = False
        entry_price = 0
        entry_bar = 0
        clean_entries = []
        clean_exits = []

        for i in range(len(signals)):
            if i < config.mean_reversion_period:  # Skip warmup
                clean_entries.append(False)
                clean_exits.append(False)
                continue

            current_price = data["close"].iloc[i]

            if not position:
                # Entry signal: price below mean by threshold
                if signals["deviation"].iloc[i] < -config.mean_reversion_threshold:
                    clean_entries.append(True)
                    clean_exits.append(False)
                    position = True
                    entry_price = current_price
                    entry_bar = i
                else:
                    clean_entries.append(False)
                    clean_exits.append(False)
            else:
                # Exit conditions
                profit = (current_price - entry_price) / entry_price
                bars_held = i - entry_bar

                # Exit if: profit target hit, stop loss hit, or max holding period reached
                if (
                    profit >= config.scalp_profit_target
                    or profit <= -config.scalp_stop_loss
                    or bars_held >= config.scalp_holding_period
                ):
                    clean_entries.append(False)
                    clean_exits.append(True)
                    position = False
                else:
                    clean_entries.append(False)
                    clean_exits.append(False)

        signals["entry"] = clean_entries
        signals["exit"] = clean_exits

        return signals

    @staticmethod
    def combined_high_frequency_signals(data: pd.DataFrame) -> pd.DataFrame:
        """
        Combined High-Frequency Strategy using multiple indicators
        This combines RSI, BB, and short-term momentum for maximum trades
        Expected: 150-300 trades per year
        """
        signals = data.copy()

        # Calculate all indicators
        signals["rsi"] = HighFrequencyStrategies.calculate_rsi(
            data["close"],
            period=7,
        )  # Faster RSI
        upper, middle, lower = HighFrequencyStrategies.calculate_bollinger_bands(
            data["close"],
            period=10,
            std_dev=1.5,  # Tighter bands
        )
        signals["bb_upper"] = upper
        signals["bb_lower"] = lower

        # Short-term momentum
        signals["mom_3"] = data["close"] / data["close"].shift(3) - 1

        # Generate entry signals (any condition)
        signals["rsi_oversold"] = signals["rsi"] < 25
        signals["bb_touch_lower"] = data["close"] <= lower
        signals["mom_reversal"] = (signals["mom_3"] < -0.02) & (
            signals["mom_3"] > signals["mom_3"].shift(1)
        )

        # Generate exit signals
        signals["rsi_overbought"] = signals["rsi"] > 75
        signals["bb_touch_upper"] = data["close"] >= upper
        signals["mom_exhausted"] = (signals["mom_3"] > 0.02) & (
            signals["mom_3"] < signals["mom_3"].shift(1)
        )

        # Combine signals
        position = False
        entry_bar = 0
        clean_entries = []
        clean_exits = []

        for i in range(len(signals)):
            if i < 10:  # Skip warmup
                clean_entries.append(False)
                clean_exits.append(False)
                continue

            # Entry logic
            if not position:
                if (
                    signals["rsi_oversold"].iloc[i]
                    or signals["bb_touch_lower"].iloc[i]
                    or signals["mom_reversal"].iloc[i]
                ):
                    clean_entries.append(True)
                    clean_exits.append(False)
                    position = True
                    entry_bar = i
                else:
                    clean_entries.append(False)
                    clean_exits.append(False)
            else:
                # Exit logic (must hold at least 1 bar)
                if i > entry_bar:
                    if (
                        signals["rsi_overbought"].iloc[i]
                        or signals["bb_touch_upper"].iloc[i]
                        or signals["mom_exhausted"].iloc[i]
                        or (i - entry_bar) >= 5
                    ):  # Max 5-day holding
                        clean_entries.append(False)
                        clean_exits.append(True)
                        position = False
                    else:
                        clean_entries.append(False)
                        clean_exits.append(False)
                else:
                    clean_entries.append(False)
                    clean_exits.append(False)

        signals["entry"] = clean_entries
        signals["exit"] = clean_exits

        return signals


# Strategy configurations
STRATEGY_CONFIGS = {
    "rsi_mean_reversion": StrategyConfig(
        name="RSI Mean Reversion",
        description="Buy oversold (RSI<30), Sell overbought (RSI>70)",
        expected_trades_per_year=75,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
    ),
    "bollinger_squeeze": StrategyConfig(
        name="Bollinger Band Squeeze",
        description="Trade breakouts from Bollinger Band squeeze",
        expected_trades_per_year=60,
        bb_period=20,
        bb_std=2.0,
    ),
    "macd_momentum": StrategyConfig(
        name="MACD Momentum",
        description="Trade MACD signal line crossovers",
        expected_trades_per_year=45,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
    ),
    "intraday_momentum": StrategyConfig(
        name="Intraday Momentum",
        description="Trade 5-day high/low breakouts",
        expected_trades_per_year=120,
    ),
    "mean_reversion_scalping": StrategyConfig(
        name="Mean Reversion Scalping",
        description="Scalp mean reversion with tight stops",
        expected_trades_per_year=150,
        mean_reversion_period=5,
        mean_reversion_threshold=0.02,
        scalp_profit_target=0.003,
        scalp_stop_loss=0.002,
        scalp_holding_period=3,
    ),
    "combined_high_frequency": StrategyConfig(
        name="Combined High-Frequency",
        description="Multiple indicators for maximum trade frequency",
        expected_trades_per_year=200,
    ),
}


def get_strategy_signals(data: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Get signals for a specific strategy."""
    if strategy_name not in STRATEGY_CONFIGS:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    config = STRATEGY_CONFIGS[strategy_name]

    if strategy_name == "rsi_mean_reversion":
        return HighFrequencyStrategies.rsi_mean_reversion_signals(data, config)
    if strategy_name == "bollinger_squeeze":
        return HighFrequencyStrategies.bollinger_squeeze_signals(data, config)
    if strategy_name == "macd_momentum":
        return HighFrequencyStrategies.macd_momentum_signals(data, config)
    if strategy_name == "intraday_momentum":
        return HighFrequencyStrategies.intraday_momentum_signals(data, config)
    if strategy_name == "mean_reversion_scalping":
        return HighFrequencyStrategies.mean_reversion_scalping_signals(data, config)
    if strategy_name == "combined_high_frequency":
        return HighFrequencyStrategies.combined_high_frequency_signals(data)
    raise ValueError(f"Strategy {strategy_name} not implemented")


def count_trades(signals: pd.DataFrame) -> int:
    """Count the number of trades in a signal DataFrame."""
    return signals["entry"].sum() + signals["exit"].sum()


if __name__ == "__main__":
    # Test with sample data
    print("High-Frequency Trading Strategies")
    print("=" * 50)

    for _name, config in STRATEGY_CONFIGS.items():
        print(f"\n{config.name}:")
        print(f"  Description: {config.description}")
        print(f"  Expected trades/year: {config.expected_trades_per_year}")
