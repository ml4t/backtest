"""Backtesting engine orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from .broker import Broker
from .datafeed import DataFeed
from .models import CommissionModel, PercentageCommission, PercentageSlippage, SlippageModel
from .strategy import Strategy
from .types import ExecutionMode

if TYPE_CHECKING:
    from .config import BacktestConfig


class Engine:
    """Backtesting engine."""

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        account_type: str = "cash",
        initial_margin: float = 0.5,
        maintenance_margin: float = 0.25,
        config: BacktestConfig | None = None,
    ):
        self.feed = feed
        self.strategy = strategy
        self.execution_mode = execution_mode
        self.config = config  # Store config for strategy access
        self.broker = Broker(
            initial_cash=initial_cash,
            commission_model=commission_model,
            slippage_model=slippage_model,
            execution_mode=execution_mode,
            account_type=account_type,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
        )
        self.equity_curve: list[tuple[datetime, float]] = []

    def run(self) -> dict:
        """Run backtest and return results."""
        self.strategy.on_start(self.broker)

        for timestamp, assets_data, context in self.feed:
            prices = {a: d["close"] for a, d in assets_data.items() if d.get("close")}
            opens = {a: d.get("open", d.get("close")) for a, d in assets_data.items()}
            volumes = {a: d.get("volume", 0) for a, d in assets_data.items()}
            signals = {a: d.get("signals", {}) for a, d in assets_data.items()}

            self.broker._update_time(timestamp, prices, opens, volumes, signals)

            if self.execution_mode == ExecutionMode.NEXT_BAR:
                # Next-bar mode: process pending orders at open price
                self.broker._process_orders(use_open=True)
                # Strategy generates new orders
                self.strategy.on_data(timestamp, assets_data, context, self.broker)
                # New orders will be processed next bar
            else:
                # Same-bar mode: process before and after strategy
                self.broker._process_orders()
                self.strategy.on_data(timestamp, assets_data, context, self.broker)
                self.broker._process_orders()

            self.equity_curve.append((timestamp, self.broker.get_account_value()))

        self.strategy.on_end(self.broker)
        return self._generate_results()

    def _generate_results(self) -> dict:
        """Generate backtest results."""
        if not self.equity_curve:
            return {}

        initial = self.broker.initial_cash
        final = self.equity_curve[-1][1]
        total_return = (final - initial) / initial

        peak = initial
        max_dd = 0.0
        for _, value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        winning = [t for t in self.broker.trades if t.pnl > 0]
        losing = [t for t in self.broker.trades if t.pnl <= 0]

        return {
            "initial_cash": initial,
            "final_value": final,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd * 100,
            "num_trades": len(self.broker.trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.broker.trades) if self.broker.trades else 0,
            "total_commission": sum(f.commission for f in self.broker.fills),
            "total_slippage": sum(f.slippage for f in self.broker.fills),
            "trades": self.broker.trades,
            "equity_curve": self.equity_curve,
            "fills": self.broker.fills,
        }

    @classmethod
    def from_config(
        cls,
        feed: DataFeed,
        strategy: Strategy,
        config: BacktestConfig,
    ) -> Engine:
        """
        Create an Engine instance from a BacktestConfig.

        This is the recommended way to create an engine when you want
        to replicate specific framework behavior (Backtrader, VectorBT, etc.).

        Example:
            from ml4t.backtest import Engine, BacktestConfig, DataFeed, Strategy

            # Use Backtrader-compatible settings
            config = BacktestConfig.from_preset("backtrader")
            engine = Engine.from_config(feed, strategy, config)
            results = engine.run()

        Args:
            feed: DataFeed with price data
            strategy: Strategy to execute
            config: BacktestConfig with all behavioral settings

        Returns:
            Configured Engine instance
        """
        from .config import CommissionModel as CommModelEnum
        from .config import FillTiming
        from .config import SlippageModel as SlipModelEnum

        # Map config fill timing to ExecutionMode
        if config.fill_timing == FillTiming.SAME_BAR:
            execution_mode = ExecutionMode.SAME_BAR
        else:
            # NEXT_BAR_OPEN or NEXT_BAR_CLOSE both use NEXT_BAR mode
            execution_mode = ExecutionMode.NEXT_BAR

        # Build commission model from config
        commission_model: CommissionModel | None = None
        if config.commission_model == CommModelEnum.PERCENTAGE:
            commission_model = PercentageCommission(
                rate=config.commission_rate,
            )
        elif config.commission_model == CommModelEnum.PER_SHARE:
            from .models import PerShareCommission

            commission_model = PerShareCommission(
                per_share=config.commission_per_share,
                minimum=config.commission_minimum,
            )
        elif config.commission_model == CommModelEnum.PER_TRADE:
            from .models import NoCommission

            # For per-trade, we'd need a new model, use NoCommission for now
            commission_model = NoCommission()
        # NONE or unrecognized -> None (will use NoCommission in Broker)

        # Build slippage model from config
        slippage_model: SlippageModel | None = None
        if config.slippage_model == SlipModelEnum.PERCENTAGE:
            slippage_model = PercentageSlippage(rate=config.slippage_rate)
        elif config.slippage_model == SlipModelEnum.FIXED:
            from .models import FixedSlippage

            slippage_model = FixedSlippage(amount=config.slippage_fixed)
        # NONE, VOLUME_BASED, or unrecognized -> None (will use NoSlippage)

        return cls(
            feed=feed,
            strategy=strategy,
            initial_cash=config.initial_cash,
            commission_model=commission_model,
            slippage_model=slippage_model,
            execution_mode=execution_mode,
            account_type=config.account_type,
            initial_margin=config.margin_requirement,
            maintenance_margin=config.margin_requirement * 0.5,  # Standard ratio
            config=config,  # Store config for strategy access
        )


# === Convenience Function ===


def run_backtest(
    prices: pl.DataFrame | str,
    strategy: Strategy,
    signals: pl.DataFrame | str | None = None,
    context: pl.DataFrame | str | None = None,
    config: BacktestConfig | str | None = None,
    # Legacy parameters (used if config is None)
    initial_cash: float = 100000.0,
    commission_model: CommissionModel | None = None,
    slippage_model: SlippageModel | None = None,
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
) -> dict:
    """
    Run a backtest with minimal setup.

    Args:
        prices: Price DataFrame or path to parquet file
        strategy: Strategy instance to execute
        signals: Optional signals DataFrame or path
        context: Optional context DataFrame or path
        config: BacktestConfig instance, preset name (str), or None for legacy params
        initial_cash: Starting cash (legacy, ignored if config provided)
        commission_model: Commission model (legacy, ignored if config provided)
        slippage_model: Slippage model (legacy, ignored if config provided)
        execution_mode: Execution mode (legacy, ignored if config provided)

    Returns:
        Results dictionary with metrics, trades, equity curve

    Example:
        # Using config preset
        results = run_backtest(prices_df, strategy, config="backtrader")

        # Using custom config
        config = BacktestConfig.from_preset("backtrader")
        config.commission_rate = 0.002  # Higher commission
        results = run_backtest(prices_df, strategy, config=config)
    """
    feed = DataFeed(
        prices_path=prices if isinstance(prices, str) else None,
        signals_path=signals if isinstance(signals, str) else None,
        context_path=context if isinstance(context, str) else None,
        prices_df=prices if isinstance(prices, pl.DataFrame) else None,
        signals_df=signals if isinstance(signals, pl.DataFrame) else None,
        context_df=context if isinstance(context, pl.DataFrame) else None,
    )

    # Handle config parameter
    if config is not None:
        from .config import BacktestConfig as ConfigCls

        if isinstance(config, str):
            config = ConfigCls.from_preset(config)
        return Engine.from_config(feed, strategy, config).run()

    # Legacy path: use individual parameters
    engine = Engine(
        feed,
        strategy,
        initial_cash,
        commission_model=commission_model,
        slippage_model=slippage_model,
        execution_mode=execution_mode,
    )
    return engine.run()


# Backward compatibility: BacktestEngine was renamed to Engine in v0.2.0
BacktestEngine = Engine
