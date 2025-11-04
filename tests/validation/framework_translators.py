"""
Framework Translators

These translate generic StrategySpec into framework-specific implementations.
Each framework has its own way of:
- Calculating indicators
- Detecting signals
- Managing positions
- Executing trades

The translators handle these differences while ensuring the SAME trading logic.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from strategy_specifications import SignalType, StrategySpec


class BaseTranslator(ABC):
    """Base class for framework-specific strategy translators."""

    @abstractmethod
    def translate_and_execute(
        self,
        spec: StrategySpec,
        data: pd.DataFrame,
        initial_capital: float,
    ) -> dict[str, Any]:
        """
        Translate strategy spec and execute backtest.

        Returns:
            Dictionary with standardized results:
            - final_value: float
            - total_return: float (%)
            - num_trades: int
            - trades: List[Dict]
            - execution_time: float
        """


class QEngineTranslator(BaseTranslator):
    """Translate strategy spec to QEngine implementation."""

    def translate_and_execute(
        self,
        spec: StrategySpec,
        data: pd.DataFrame,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Execute strategy using QEngine's manual approach."""
        import time

        start_time = time.time()

        # Calculate all indicators
        indicators = self._calculate_indicators(spec, data)

        # Initialize portfolio
        cash = initial_capital
        shares = 0.0
        position = 0
        trades = []
        entry_price = 0
        entry_bar = 0

        # Process each bar
        for i, (date, row) in enumerate(data.iterrows()):
            # Skip if indicators not ready
            if any(
                pd.isna(indicators.get(ind, [np.nan])[i])
                for ind in self._get_required_indicators(spec)
            ):
                continue

            # Current values
            price = row["close"]

            # Check position management (stop loss, take profit, max holding)
            if position == 1:
                # Check risk management rules
                if self._should_exit_risk_management(spec, i, entry_bar, price, entry_price):
                    cash = shares * price
                    trades.append(
                        {"date": date, "action": "SELL", "price": price, "shares": shares},
                    )
                    shares = 0.0
                    position = 0
                    continue

            # Check trading rules
            signals = self._check_rules(spec, indicators, i, position)

            if signals.get("long_entry") and position == 0 and cash > 0:
                # Enter long position
                if spec.position_sizing == "all_in":
                    shares = cash / price
                    cash = 0.0
                else:
                    # Fixed position sizing
                    position_value = min(cash, initial_capital * 0.1)
                    shares = position_value / price
                    cash -= position_value

                trades.append({"date": date, "action": "BUY", "price": price, "shares": shares})
                position = 1
                entry_price = price
                entry_bar = i

            elif signals.get("long_exit") and position == 1:
                # Exit long position
                cash += shares * price
                trades.append({"date": date, "action": "SELL", "price": price, "shares": shares})
                shares = 0.0
                position = 0

        # Close any remaining position
        if shares > 0:
            final_price = data["close"].iloc[-1]
            cash += shares * final_price

        # Calculate results
        final_value = cash
        total_return = (final_value / initial_capital - 1) * 100

        return {
            "framework": "QEngine",
            "final_value": final_value,
            "total_return": total_return,
            "num_trades": len(trades),
            "trades": trades,
            "execution_time": time.time() - start_time,
        }

    def _calculate_indicators(
        self,
        spec: StrategySpec,
        data: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Calculate all indicators specified in the strategy."""
        indicators = {}

        for ind_spec in spec.indicators:
            if ind_spec.name == "rsi":
                indicators["rsi"] = self._calculate_rsi(data["close"], ind_spec.params["period"])
            elif ind_spec.name == "sma":
                label = ind_spec.params.get("label", "sma")
                indicators[f"sma_{label}"] = (
                    data["close"].rolling(ind_spec.params["period"]).mean().values
                )
            elif ind_spec.name == "ema":
                label = ind_spec.params.get("label", "ema")
                indicators[f"ema_{label}"] = (
                    data["close"].ewm(span=ind_spec.params["period"]).mean().values
                )
            elif ind_spec.name == "bbands":
                period = ind_spec.params["period"]
                std = ind_spec.params["std"]
                sma = data["close"].rolling(period).mean()
                std_dev = data["close"].rolling(period).std()
                indicators["bb_upper"] = (sma + std * std_dev).values
                indicators["bb_middle"] = sma.values
                indicators["bb_lower"] = (sma - std * std_dev).values
            elif ind_spec.name == "volume_sma":
                indicators["volume_sma"] = (
                    data["volume"].rolling(ind_spec.params["period"]).mean().values
                )
            elif ind_spec.name == "macd":
                fast = ind_spec.params["fast"]
                slow = ind_spec.params["slow"]
                signal = ind_spec.params["signal"]
                ema_fast = data["close"].ewm(span=fast).mean()
                ema_slow = data["close"].ewm(span=slow).mean()
                macd_line = ema_fast - ema_slow
                macd_signal = macd_line.ewm(span=signal).mean()
                indicators["macd"] = macd_line.values
                indicators["macd_signal"] = macd_signal.values
            elif ind_spec.name == "atr":
                indicators["atr"] = self._calculate_atr(data, ind_spec.params["period"])
            elif ind_spec.name == "adx":
                indicators["adx"] = self._calculate_adx(data, ind_spec.params["period"])

        # Add price/volume for rules that need them
        indicators["close"] = data["close"].values
        indicators["volume"] = data["volume"].values

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int) -> np.ndarray:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate ATR."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.values

    def _calculate_adx(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate ADX."""
        high = data["high"]
        low = data["low"]
        data["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr = self._calculate_atr(data, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx.values

    def _check_rules(
        self,
        spec: StrategySpec,
        indicators: dict,
        bar_idx: int,
        position: int,
    ) -> dict[str, bool]:
        """Check trading rules and return signals."""
        signals = {}

        for rule in spec.rules:
            # Parse and evaluate condition
            if rule.signal_type == SignalType.LONG_ENTRY and position == 0:
                signals["long_entry"] = self._evaluate_condition(
                    rule.condition,
                    indicators,
                    bar_idx,
                )
            elif rule.signal_type == SignalType.LONG_EXIT and position == 1:
                signals["long_exit"] = self._evaluate_condition(rule.condition, indicators, bar_idx)

        return signals

    def _evaluate_condition(self, condition: str, indicators: dict, bar_idx: int) -> bool:
        """Evaluate a trading condition."""
        # Simple condition parser - in production would use proper expression parser

        # Handle crossover conditions
        if "crosses_above" in condition:
            parts = condition.split(" crosses_above ")
            fast_name = parts[0].strip()
            slow_name = parts[1].strip()

            if bar_idx > 0:
                fast_curr = indicators.get(fast_name, [0])[bar_idx]
                fast_prev = indicators.get(fast_name, [0])[bar_idx - 1]
                slow_curr = indicators.get(slow_name, [0])[bar_idx]
                slow_prev = indicators.get(slow_name, [0])[bar_idx - 1]

                return fast_prev <= slow_prev and fast_curr > slow_curr
            return False

        if "crosses_below" in condition:
            parts = condition.split(" crosses_below ")
            fast_name = parts[0].strip()
            slow_name = parts[1].strip()

            if bar_idx > 0:
                fast_curr = indicators.get(fast_name, [0])[bar_idx]
                fast_prev = indicators.get(fast_name, [0])[bar_idx - 1]
                slow_curr = indicators.get(slow_name, [0])[bar_idx]
                slow_prev = indicators.get(slow_name, [0])[bar_idx - 1]

                return fast_prev > slow_prev and fast_curr <= slow_curr
            return False

        # Handle simple comparisons
        if "<" in condition:
            parts = condition.split(" < ")
            left = parts[0].strip()
            right = (
                float(parts[1].strip())
                if parts[1].strip().replace(".", "").isdigit()
                else indicators.get(parts[1].strip(), [0])[bar_idx]
            )
            left_val = indicators.get(left, [0])[bar_idx]
            return left_val < right

        if ">" in condition:
            parts = condition.split(" > ")
            left = parts[0].strip()
            right = (
                float(parts[1].strip())
                if parts[1].strip().replace(".", "").isdigit()
                else indicators.get(parts[1].strip(), [0])[bar_idx]
            )
            left_val = indicators.get(left, [0])[bar_idx]
            return left_val > right

        # Handle AND conditions
        if " AND " in condition:
            parts = condition.split(" AND ")
            return all(self._evaluate_condition(p.strip("()"), indicators, bar_idx) for p in parts)

        # Handle OR conditions
        if " OR " in condition:
            parts = condition.split(" OR ")
            return any(self._evaluate_condition(p.strip("()"), indicators, bar_idx) for p in parts)

        return False

    def _should_exit_risk_management(
        self,
        spec: StrategySpec,
        current_bar: int,
        entry_bar: int,
        current_price: float,
        entry_price: float,
    ) -> bool:
        """Check if position should be exited based on risk management rules."""
        rm = spec.risk_management

        # Max holding period
        if rm.get("max_holding_period") and (current_bar - entry_bar) >= rm["max_holding_period"]:
            return True

        # Stop loss
        if rm.get("stop_loss"):
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct <= -rm["stop_loss"]:
                return True

        # Take profit
        if rm.get("take_profit"):
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= rm["take_profit"]:
                return True

        return False

    def _get_required_indicators(self, spec: StrategySpec) -> list[str]:
        """Get list of all required indicators."""
        required = set()
        for rule in spec.rules:
            required.update(rule.required_indicators)
        return list(required)
