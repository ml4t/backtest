"""Backtesting engine orchestration."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from .analytics import EquityCurve, TradeAnalyzer
from .analytics.metrics import calmar_ratio, sharpe_ratio, sortino_ratio
from .broker import Broker
from .config import InitialHwmSource, Mode, TrailStopTiming, WaterMarkSource
from .datafeed import DataFeed
from .models import CommissionModel, SlippageModel
from .strategy import Strategy
from .types import ContractSpec, ExecutionMode, StopFillMode, StopLevelBasis

if TYPE_CHECKING:
    from .config import BacktestConfig
    from .result import BacktestResult


class Engine:
    """Event-driven backtesting engine.

    The Engine orchestrates the backtest by iterating through market data,
    managing the broker, and calling the strategy on each bar.

    Execution Flow:
        1. Initialize strategy (on_start)
        2. For each bar:
           a. Update broker with current prices
           b. Process pending exits (NEXT_BAR_OPEN mode)
           c. Evaluate position rules (stops, trails)
           d. Process pending orders
           e. Call strategy.on_data()
           f. Process new orders (SAME_BAR mode)
           g. Update water marks
           h. Record equity
        3. Close open positions
        4. Finalize strategy (on_end)

    Attributes:
        feed: DataFeed providing price and signal data
        strategy: Strategy implementing trading logic
        broker: Broker handling order execution and positions
        execution_mode: Order execution timing (SAME_BAR or NEXT_BAR)
        equity_curve: List of (timestamp, equity) tuples

    Example:
        >>> from ml4t.backtest import Engine, DataFeed, Strategy
        >>>
        >>> class MyStrategy(Strategy):
        ...     def on_data(self, timestamp, data, context, broker):
        ...         for asset, bar in data.items():
        ...             if bar.get('signal', 0) > 0.5:
        ...                 broker.submit_order(asset, 100)
        >>>
        >>> feed = DataFeed(prices_df=df)
        >>> engine = Engine(feed=feed, strategy=MyStrategy())
        >>> result = engine.run()
        >>> print(result['total_return'])
    """

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        initial_cash: float = 100000.0,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        stop_slippage_rate: float = 0.0,
        execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
        stop_fill_mode: StopFillMode = StopFillMode.STOP_PRICE,
        stop_level_basis: StopLevelBasis = StopLevelBasis.FILL_PRICE,
        trail_hwm_source: WaterMarkSource = WaterMarkSource.CLOSE,
        initial_hwm_source: InitialHwmSource = InitialHwmSource.FILL_PRICE,
        trail_stop_timing: TrailStopTiming = TrailStopTiming.LAGGED,
        allow_short_selling: bool = False,
        allow_leverage: bool = False,
        initial_margin: float = 0.5,
        long_maintenance_margin: float = 0.25,
        short_maintenance_margin: float = 0.30,
        fixed_margin_schedule: dict[str, tuple[float, float]] | None = None,
        config: BacktestConfig | None = None,
        execution_limits=None,
        market_impact_model=None,
        contract_specs: dict[str, ContractSpec] | None = None,
    ):
        self.feed = feed
        self.strategy = strategy
        self.execution_mode = execution_mode
        self.stop_fill_mode = stop_fill_mode
        self.stop_level_basis = stop_level_basis
        self.config = config  # Store config for strategy access
        self.broker = Broker(
            initial_cash=initial_cash,
            commission_model=commission_model,
            slippage_model=slippage_model,
            stop_slippage_rate=stop_slippage_rate,
            execution_mode=execution_mode,
            stop_fill_mode=stop_fill_mode,
            stop_level_basis=stop_level_basis,
            trail_hwm_source=trail_hwm_source,
            initial_hwm_source=initial_hwm_source,
            trail_stop_timing=trail_stop_timing,
            allow_short_selling=allow_short_selling,
            allow_leverage=allow_leverage,
            initial_margin=initial_margin,
            long_maintenance_margin=long_maintenance_margin,
            short_maintenance_margin=short_maintenance_margin,
            fixed_margin_schedule=fixed_margin_schedule,
            execution_limits=execution_limits,
            market_impact_model=market_impact_model,
            contract_specs=contract_specs,
        )
        self.equity_curve: list[tuple[datetime, float]] = []

        # Calendar session enforcement (lazy initialized in run())
        self._calendar = None
        self._skipped_bars = 0

    def run(self) -> BacktestResult:
        """Run backtest and return structured results.

        Returns:
            BacktestResult with trades, equity curve, metrics, and export methods.
            Call .to_dict() for backward-compatible dictionary output.
        """
        # Lazy calendar initialization (zero cost if unused)
        is_trading_day_fn = None
        if self.config and self.config.calendar:
            from .calendar import get_calendar, is_trading_day

            self._calendar = get_calendar(self.config.calendar)
            is_trading_day_fn = is_trading_day

        self.strategy.on_start(self.broker)

        # Date-level cache for trading day checks (significant speedup for intraday data)
        trading_day_cache: dict[date, bool] = {}

        for timestamp, assets_data, context in self.feed:
            # Calendar session enforcement
            calendar_id = self.config.calendar if self.config else None
            if (
                self._calendar
                and calendar_id
                and self.config
                and self.config.enforce_sessions
                and is_trading_day_fn
            ):
                # For daily data, check trading day; for intraday, check market hours
                if self.config.data_frequency.value == "daily":
                    if not is_trading_day_fn(calendar_id, timestamp.date()):
                        self._skipped_bars += 1
                        continue
                else:
                    # Intraday: use cached trading day check (avoid expensive calendar.valid_days per bar)
                    bar_date = timestamp.date()
                    if bar_date not in trading_day_cache:
                        trading_day_cache[bar_date] = is_trading_day_fn(calendar_id, bar_date)
                    if not trading_day_cache[bar_date]:
                        self._skipped_bars += 1
                        continue

            prices = {a: d["close"] for a, d in assets_data.items() if d.get("close")}
            opens = {a: d.get("open", d.get("close")) for a, d in assets_data.items()}
            highs = {a: d.get("high", d.get("close")) for a, d in assets_data.items()}
            lows = {a: d.get("low", d.get("close")) for a, d in assets_data.items()}
            volumes = {a: d.get("volume", 0) for a, d in assets_data.items()}
            signals = {a: d.get("signals", {}) for a, d in assets_data.items()}

            self.broker._update_time(timestamp, prices, opens, highs, lows, volumes, signals)

            # Process pending exits from NEXT_BAR_OPEN mode (fills at open)
            # This must happen BEFORE evaluate_position_rules() to clear deferred exits
            self.broker._process_pending_exits()

            # Evaluate position rules (stops, trails, etc.) - generates exit orders
            self.broker.evaluate_position_rules()

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

            # Update water marks at END of bar, AFTER all orders processed
            # This ensures new positions get their HWM updated from entry bar's high
            # VBT Pro behavior: HWM updated at bar end, used in NEXT bar's trail evaluation
            self.broker._update_water_marks()

            self.equity_curve.append((timestamp, self.broker.get_account_value()))

        self.strategy.on_end(self.broker)
        return self._generate_results()

    def run_dict(self) -> dict[str, Any]:
        """Run backtest and return dictionary (backward compatible).

        This is equivalent to run().to_dict() but more explicit for code
        that requires dictionary output.

        Returns:
            Dictionary with metrics, trades, and equity curve.
        """
        return self.run().to_dict()

    def _generate_results(self) -> BacktestResult:
        """Generate backtest results with full analytics."""
        from .result import BacktestResult
        from .types import Trade

        if not self.equity_curve:
            # Return empty result for no-data case
            return BacktestResult(
                trades=[],
                equity_curve=[],
                fills=[],
                metrics={"skipped_bars": self._skipped_bars},
                config=self.config,
            )

        # Build EquityCurve from raw data
        equity = EquityCurve()
        for ts, value in self.equity_curve:
            equity.append(ts, value)

        # Collect all trades (closed + open)
        all_trades = list(self.broker.trades)  # Closed trades

        # Add open positions as trades with status="open" (mark-to-market)
        if self.equity_curve:
            last_timestamp = self.equity_curve[-1][0]
            for asset, pos in self.broker.positions.items():
                # Get last known price for this asset
                last_price = self.broker._current_prices.get(asset, pos.entry_price)

                # Calculate mark-to-market PnL
                pnl = (last_price - pos.entry_price) * pos.quantity - pos.entry_commission
                pnl_pct = (last_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0

                open_trade = Trade(
                    symbol=asset,  # Asset identifier (Position.asset -> Trade.symbol)
                    entry_time=pos.entry_time,
                    exit_time=last_timestamp,  # Mark-to-market time
                    entry_price=pos.entry_price,
                    exit_price=last_price,  # Mark-to-market price
                    quantity=pos.quantity,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    bars_held=pos.bars_held,
                    fees=pos.entry_commission,  # Only entry fees so far
                    slippage=0.0,  # No exit slippage yet
                    exit_reason="end_of_backtest",
                    status="open",
                    mfe=pos.max_favorable_excursion,
                    mae=pos.max_adverse_excursion,
                )
                all_trades.append(open_trade)

        # Build TradeAnalyzer (only on closed trades for accurate stats)
        closed_trades = [t for t in all_trades if t.status == "closed"]
        trade_analyzer = TradeAnalyzer(closed_trades)

        # Build metrics dictionary (backward compatible)
        metrics = {
            # Core metrics (backward compatible)
            "initial_cash": equity.initial_value,
            "final_value": equity.final_value,
            "total_return": equity.total_return,
            "total_return_pct": equity.total_return * 100,
            "max_drawdown": abs(equity.max_dd),  # Keep as positive for backward compat
            "max_drawdown_pct": abs(equity.max_dd) * 100,
            "num_trades": trade_analyzer.num_trades,
            "winning_trades": trade_analyzer.num_winners,
            "losing_trades": trade_analyzer.num_losers,
            "win_rate": trade_analyzer.win_rate,
            # Commission/slippage from fills (includes open positions)
            "total_commission": sum(f.commission for f in self.broker.fills),
            "total_slippage": sum(f.slippage for f in self.broker.fills),
            # Additional metrics
            "sharpe": sharpe_ratio(equity.returns),
            "sortino": sortino_ratio(equity.returns),
            "calmar": calmar_ratio(equity.cagr, equity.max_dd),
            "cagr": equity.cagr,
            "volatility": equity.volatility,
            "profit_factor": trade_analyzer.profit_factor,
            "expectancy": trade_analyzer.expectancy,
            "avg_trade": trade_analyzer.avg_trade,
            "avg_win": trade_analyzer.avg_win,
            "avg_loss": trade_analyzer.avg_loss,
            "largest_win": trade_analyzer.largest_win,
            "largest_loss": trade_analyzer.largest_loss,
            # Calendar enforcement
            "skipped_bars": self._skipped_bars,
        }

        return BacktestResult(
            trades=all_trades,  # Includes both closed and open trades
            equity_curve=self.equity_curve,
            fills=self.broker.fills,
            metrics=metrics,
            config=self.config,
            equity=equity,
            trade_analyzer=trade_analyzer,
        )

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

        All configuration is handled by BacktestConfig and Broker.from_config(),
        ensuring consistent behavior across the system.

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
        # Create broker from config (handles all commission/slippage/account setup)
        broker = Broker.from_config(config)

        # Create engine with pre-configured broker
        engine = cls.__new__(cls)
        engine.feed = feed
        engine.strategy = strategy
        engine.execution_mode = config.execution_mode
        engine.stop_fill_mode = broker.stop_fill_mode
        engine.stop_level_basis = broker.stop_level_basis
        engine.config = config
        engine.broker = broker
        engine.equity_curve = []
        engine._calendar = None
        engine._skipped_bars = 0

        return engine

    @classmethod
    def from_mode(
        cls,
        feed: DataFeed,
        strategy: Strategy,
        mode: Mode,
    ) -> Engine:
        """
        Create an Engine instance from a Mode enum.

        This is the simplest way to create an engine when you want sensible
        defaults without configuring every detail.

        Example:
            from ml4t.backtest import Engine, Mode, DataFeed

            # Quick prototype with no transaction costs
            engine = Engine.from_mode(feed, strategy, Mode.FAST)

            # Production-ready conservative settings
            engine = Engine.from_mode(feed, strategy, Mode.REALISTIC)

            # Match VectorBT behavior exactly
            engine = Engine.from_mode(feed, strategy, Mode.VECTORBT)

        Args:
            feed: DataFeed with price data
            strategy: Strategy to execute
            mode: Mode enum specifying desired behavior

        Returns:
            Configured Engine instance
        """
        config = mode.to_config()
        return cls.from_config(feed, strategy, config)


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
) -> BacktestResult:
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
        BacktestResult with metrics, trades, equity curve, and export methods.
        Call .to_dict() for backward-compatible dictionary output.

    Example:
        # Using config preset
        result = run_backtest(prices_df, strategy, config="backtrader")
        print(result.metrics["sharpe"])

        # Export to Parquet
        result.to_parquet("./results")

        # Using custom config
        config = BacktestConfig.from_preset("backtrader")
        config.commission_rate = 0.002  # Higher commission
        result = run_backtest(prices_df, strategy, config=config)
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
