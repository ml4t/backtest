"""
VectorBT Pro Framework Adapter for Cross-Framework Validation

Supports both open-source VectorBT and VectorBT Pro versions.
"""

import time
import tracemalloc
from typing import Any

import pandas as pd

from .base import BaseFrameworkAdapter, TradeRecord, ValidationResult


class VectorBTProAdapter(BaseFrameworkAdapter):
    """Adapter for VectorBT Pro backtesting framework."""

    def __init__(self):
        super().__init__("VectorBTPro")
        self.is_pro = False
        self._check_version()

    def _check_version(self):
        """Check if VectorBT Pro is available."""
        try:
            import vectorbtpro as vbt

            self.is_pro = True
            self.vbt = vbt
            print(f"✓ Using VectorBT Pro version {getattr(vbt, '__version__', 'unknown')}")
        except ImportError:
            try:
                import vectorbt as vbt

                self.is_pro = False
                self.vbt = vbt
                print(
                    f"⚠ VectorBT Pro not available, using open-source version {getattr(vbt, '__version__', 'unknown')}",
                )
            except ImportError:
                raise ImportError("Neither VectorBT Pro nor open-source VectorBT available")

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """Run backtest using VectorBT Pro (or fallback to open-source)."""

        result = ValidationResult(
            framework=self.framework_name if self.is_pro else "VectorBT",
            strategy=strategy_params.get("name", "Unknown"),
            initial_capital=initial_capital,
        )

        try:
            vbt = self.vbt

            tracemalloc.start()
            start_time = time.time()

            strategy_name = strategy_params.get("name", "Unknown")

            # Handle different strategy types
            if strategy_name == "MovingAverageCrossover":
                entries, exits = self._run_ma_crossover(data, strategy_params)
            elif strategy_name == "BollingerBandMeanReversion":
                entries, exits = self._run_bollinger_bands(data, strategy_params)
            elif strategy_name == "VolumeBreakoutStrategy":
                entries, exits = self._run_volume_breakout(data, strategy_params)
            elif strategy_name == "ShortTermMomentumStrategy":
                entries, exits = self._run_momentum(data, strategy_params)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            print(
                f"VectorBT{'Pro' if self.is_pro else ''} signals: {entries.sum()} entries, {exits.sum()} exits",
            )

            # Create portfolio
            close_prices = data["close"]

            # Create portfolio (same API for both Pro and open-source in recent versions)
            pf = vbt.Portfolio.from_signals(
                close_prices,
                entries=entries,
                exits=exits,
                init_cash=initial_capital,
                fees=0.0,
                slippage=0.0,
                freq="D",
            )

            # Extract results
            result.final_value = (
                float(pf.final_value()) if callable(pf.final_value) else float(pf.final_value)
            )
            result.total_return = (
                float(pf.total_return()) * 100
                if callable(pf.total_return)
                else float(pf.total_return) * 100
            )

            # Get trade count
            if hasattr(pf, "orders") and hasattr(pf.orders, "records"):
                result.num_trades = len(pf.orders.records)
            elif hasattr(pf, "orders"):
                result.num_trades = len(pf.orders)
            else:
                result.num_trades = 0

            # Performance metrics
            try:
                result.sharpe_ratio = (
                    float(pf.sharpe_ratio())
                    if callable(pf.sharpe_ratio)
                    else float(pf.sharpe_ratio)
                )
            except:
                result.sharpe_ratio = 0.0

            try:
                result.max_drawdown = (
                    float(pf.max_drawdown()) * 100
                    if callable(pf.max_drawdown)
                    else float(pf.max_drawdown) * 100
                )
            except:
                result.max_drawdown = 0.0

            # Get detailed metrics if Pro version
            if self.is_pro and hasattr(pf, "metrics"):
                try:
                    metrics = pf.metrics
                    if hasattr(metrics, "win_rate"):
                        result.win_rate = float(metrics.win_rate)
                except:
                    pass

            # Extract trades
            result.trades = []
            try:
                if hasattr(pf, "trades"):
                    if hasattr(pf.trades, "records_readable"):
                        trades_df = pf.trades.records_readable
                    elif hasattr(pf.trades, "records"):
                        trades_df = pf.trades.records
                    else:
                        trades_df = None

                    if trades_df is not None and len(trades_df) > 0:
                        for _, trade in (
                            trades_df.iterrows()
                            if hasattr(trades_df, "iterrows")
                            else enumerate(trades_df)
                        ):
                            trade_record = TradeRecord(
                                timestamp=trade.get(
                                    "Entry Timestamp",
                                    trade.get("entry_timestamp", pd.NaT),
                                ),
                                action="BUY"
                                if trade.get("Size", trade.get("size", 0)) > 0
                                else "SELL",
                                quantity=abs(float(trade.get("Size", trade.get("size", 0)))),
                                price=float(
                                    trade.get("Avg Entry Price", trade.get("avg_entry_price", 0)),
                                ),
                                value=abs(
                                    float(trade.get("Size", trade.get("size", 0)))
                                    * float(
                                        trade.get(
                                            "Avg Entry Price",
                                            trade.get("avg_entry_price", 0),
                                        ),
                                    ),
                                ),
                            )
                            result.trades.append(trade_record)
            except Exception as e:
                print(f"Warning: Could not extract trade details: {e}")

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print(f"✓ VectorBT{'Pro' if self.is_pro else ''} backtest completed")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")
            print(f"  Execution Time: {result.execution_time:.3f}s")

        except Exception as e:
            error_msg = f"VectorBT{'Pro' if self.is_pro else ''} backtest failed: {e}"
            print(f"✗ {error_msg}")
            result.errors.append(error_msg)

        return result

    def run_with_signals(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        config: "FrameworkConfig | None" = None,
    ) -> ValidationResult:
        """Run backtest with pre-computed signals."""
        from .base import FrameworkConfig

        # Validate inputs
        self.validate_data(data)
        self.validate_signals(signals, data)

        # Use default config if none provided
        if config is None:
            config = FrameworkConfig.realistic()

        result = ValidationResult(
            framework=self.framework_name if self.is_pro else "VectorBT",
            strategy="PrecomputedSignals",
            initial_capital=config.initial_capital,
        )

        try:
            vbt = self.vbt
            tracemalloc.start()
            start_time = time.time()

            # Extract signals
            entries = signals["entry"]
            exits = signals["exit"]

            print(f"VectorBT{'Pro' if self.is_pro else ''} signals: {entries.sum()} entries, {exits.sum()} exits")

            # Create portfolio
            close_prices = data["close"]

            # Map config to VectorBT parameters
            fees = config.commission_pct if config.commission_pct > 0 else 0.0
            slippage = config.slippage_pct if config.slippage_pct > 0 else 0.0

            pf = vbt.Portfolio.from_signals(
                close_prices,
                entries=entries,
                exits=exits,
                init_cash=config.initial_capital,
                fees=fees,
                slippage=slippage,
                freq="D",
            )

            # Extract results (same as run_backtest)
            result.final_value = (
                float(pf.final_value()) if callable(pf.final_value) else float(pf.final_value)
            )
            result.total_return = (
                float(pf.total_return()) * 100
                if callable(pf.total_return)
                else float(pf.total_return) * 100
            )

            # Get trade count
            if hasattr(pf, "orders") and hasattr(pf.orders, "records"):
                result.num_trades = len(pf.orders.records)
            elif hasattr(pf, "orders"):
                result.num_trades = len(pf.orders)
            else:
                result.num_trades = 0

            # Performance metrics
            try:
                result.sharpe_ratio = (
                    float(pf.sharpe_ratio())
                    if callable(pf.sharpe_ratio)
                    else float(pf.sharpe_ratio)
                )
            except:
                result.sharpe_ratio = 0.0

            try:
                result.max_drawdown = (
                    float(pf.max_drawdown()) * 100
                    if callable(pf.max_drawdown)
                    else float(pf.max_drawdown) * 100
                )
            except:
                result.max_drawdown = 0.0

            # Performance tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.execution_time = time.time() - start_time
            result.memory_usage = peak / 1024 / 1024

            print(f"✓ VectorBT{'Pro' if self.is_pro else ''} backtest completed")
            print(f"  Final Value: ${result.final_value:,.2f}")
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Total Trades: {result.num_trades}")

        except Exception as e:
            error_msg = f"VectorBT{'Pro' if self.is_pro else ''} backtest with signals failed: {e}"
            print(f"✗ {error_msg}")
            result.errors.append(error_msg)

        return result

    def _run_ma_crossover(self, data: pd.DataFrame, params: dict) -> tuple:
        """Run MA crossover strategy."""
        vbt = self.vbt

        close_prices = data["close"]
        short_window = params.get("short_window", 20)
        long_window = params.get("long_window", 50)

        # Calculate MAs
        ma_short = vbt.MA.run(close_prices, window=short_window).ma
        ma_long = vbt.MA.run(close_prices, window=long_window).ma

        # Generate signals
        entries = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        exits = (ma_short <= ma_long) & (ma_short.shift(1) > ma_long.shift(1))

        # Remove NaN values
        valid_mask = ~(ma_short.isna() | ma_long.isna())
        entries = entries & valid_mask
        exits = exits & valid_mask

        return entries, exits

    def _run_bollinger_bands(self, data: pd.DataFrame, params: dict) -> tuple:
        """Run Bollinger Bands mean reversion strategy."""
        vbt = self.vbt

        close_prices = data["close"]
        bb_period = params.get("bb_period", 20)
        bb_std = params.get("bb_std", 2.0)
        rsi_period = params.get("rsi_period", 14)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        # Calculate Bollinger Bands (use manual calculation for compatibility)
        ma = close_prices.rolling(window=bb_period).mean()
        std = close_prices.rolling(window=bb_period).std()
        upper = ma + (std * bb_std)
        lower = ma - (std * bb_std)

        # Calculate RSI
        rsi = vbt.RSI.run(close_prices, window=rsi_period).rsi

        # Generate signals
        entries = (close_prices <= lower * 1.01) & (rsi < rsi_oversold)
        exits = (close_prices >= upper * 0.99) | (rsi > rsi_overbought)

        # Remove NaN values
        valid_mask = ~(upper.isna() | lower.isna() | rsi.isna())
        entries = entries & valid_mask
        exits = exits & valid_mask

        return entries, exits

    def _run_volume_breakout(self, data: pd.DataFrame, params: dict) -> tuple:
        """Run volume breakout strategy."""
        close_prices = data["close"]
        high_prices = data["high"]
        volume = data["volume"]

        lookback = params.get("lookback_period", 20)
        vol_mult = params.get("volume_multiplier", 1.5)
        breakout_thresh = params.get("breakout_threshold", 0.02)

        # Calculate resistance and volume average
        resistance = high_prices.rolling(window=lookback).max()
        avg_volume = volume.rolling(window=lookback).mean()

        # Generate breakout signals
        entries = (close_prices > resistance.shift(1) * (1 + breakout_thresh)) & (
            volume > avg_volume * vol_mult
        )

        # Simple exit after fixed period or trailing stop
        # For simplicity, exit after 10 bars or 5% profit
        exits = pd.Series(False, index=data.index)

        entry_indices = entries[entries].index
        for entry_idx in entry_indices:
            entry_loc = data.index.get_loc(entry_idx)
            # Exit after 10 bars or at end
            exit_loc = min(entry_loc + 10, len(data) - 1)
            if exit_loc < len(data):
                exits.iloc[exit_loc] = True

        return entries, exits

    def _run_momentum(self, data: pd.DataFrame, params: dict) -> tuple:
        """Run short-term momentum strategy."""
        vbt = self.vbt

        close_prices = data["close"]
        fast_ema = params.get("fast_ema", 5)
        slow_ema = params.get("slow_ema", 15)

        # Calculate EMAs
        if hasattr(vbt, "EMA"):
            ema_fast = vbt.EMA.run(close_prices, window=fast_ema).ema
            ema_slow = vbt.EMA.run(close_prices, window=slow_ema).ema
        else:
            # Fallback to pandas
            ema_fast = close_prices.ewm(span=fast_ema, adjust=False).mean()
            ema_slow = close_prices.ewm(span=slow_ema, adjust=False).mean()

        # Generate signals
        entries = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        exits = (ema_fast <= ema_slow) & (ema_fast.shift(1) > ema_slow.shift(1))

        # Add profit target and stop loss exits
        params.get("profit_target", 0.03)
        params.get("stop_loss", 0.015)

        # This is simplified - in production would track actual entry prices
        # For now, use momentum reversal as primary exit

        # Remove NaN values
        valid_mask = ~(ema_fast.isna() | ema_slow.isna())
        entries = entries & valid_mask
        exits = exits & valid_mask

        return entries, exits
