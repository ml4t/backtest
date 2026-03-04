"""Backtesting engine orchestration."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from .analytics import EquityCurve, TradeAnalyzer
from .analytics.metrics import calmar_ratio
from .broker import Broker
from .datafeed import DataFeed
from .strategy import Strategy
from .types import ExecutionMode

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
        config: BacktestConfig with all behavioral settings
        equity_curve: List of (timestamp, equity) tuples

    Example:
        >>> from ml4t.backtest import Engine, DataFeed, Strategy, BacktestConfig
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
        config: BacktestConfig | None = None,
        *,
        contract_specs: dict[str, Any] | None = None,
        market_impact_model: Any | None = None,
        execution_limits: Any | None = None,
    ):
        from .config import BacktestConfig as ConfigCls

        if config is None:
            config = ConfigCls()

        self.feed = feed
        self.strategy = strategy
        self.config = config
        self.execution_mode = config.execution_mode
        self.broker = Broker.from_config(
            config,
            contract_specs=contract_specs,
            market_impact_model=market_impact_model,
            execution_limits=execution_limits,
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

            prices = getattr(assets_data, "_prices", None)
            opens = getattr(assets_data, "_opens", None)
            highs = getattr(assets_data, "_highs", None)
            lows = getattr(assets_data, "_lows", None)
            volumes = getattr(assets_data, "_volumes", None)
            signals = getattr(assets_data, "_signals", None)

            if (
                prices is None
                or opens is None
                or highs is None
                or lows is None
                or volumes is None
                or signals is None
            ):
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

                # Calculate mark-to-market PnL (include multiplier for futures)
                pnl = (
                    last_price - pos.entry_price
                ) * pos.quantity * pos.multiplier - pos.entry_commission
                raw_pct = (
                    (last_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
                )
                pnl_pct = raw_pct if pos.quantity > 0 else -raw_pct

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
                    entry_slippage=pos.entry_slippage,
                    multiplier=pos.multiplier,
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
            "sharpe": equity.sharpe,
            "sortino": equity.sortino,
            "calmar": calmar_ratio(equity.cagr, equity.max_dd),
            "cagr": equity.cagr,
            "volatility": equity.volatility,
            "profit_factor": trade_analyzer.profit_factor,
            # Per-trade return metrics (percentage-based, direction-aware)
            "expectancy": trade_analyzer.expectancy,
            "avg_trade": trade_analyzer.avg_trade,
            "avg_win": trade_analyzer.avg_win,
            "avg_loss": trade_analyzer.avg_loss,
            "largest_win": trade_analyzer.largest_win,
            "largest_loss": trade_analyzer.largest_loss,
            "payoff_ratio": trade_analyzer.payoff_ratio,
            # Cost decomposition
            "total_gross_pnl": trade_analyzer.total_gross_pnl,
            "total_costs": trade_analyzer.total_costs,
            "avg_cost_drag": trade_analyzer.avg_cost_drag,
            "gross_profit_factor": trade_analyzer.gross_profit_factor,
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
        *,
        contract_specs: dict[str, Any] | None = None,
        market_impact_model: Any | None = None,
        execution_limits: Any | None = None,
    ) -> Engine:
        """Create an Engine instance from a BacktestConfig.

        Equivalent to ``Engine(feed, strategy, config)``. Kept as a convenience
        for code that reads more clearly with a named constructor.

        Args:
            feed: DataFeed with price data
            strategy: Strategy to execute
            config: BacktestConfig with all behavioral settings
            contract_specs: Per-asset contract specifications (futures multipliers, etc.)
            market_impact_model: Market impact model for fill simulation
            execution_limits: Execution limits (max order size, etc.)

        Returns:
            Configured Engine instance
        """
        return cls(
            feed,
            strategy,
            config,
            contract_specs=contract_specs,
            market_impact_model=market_impact_model,
            execution_limits=execution_limits,
        )


# === Convenience Function ===


def run_backtest(
    prices: pl.DataFrame | str,
    strategy: Strategy,
    signals: pl.DataFrame | str | None = None,
    context: pl.DataFrame | str | None = None,
    config: BacktestConfig | str | None = None,
    *,
    contract_specs: dict[str, Any] | None = None,
    market_impact_model: Any | None = None,
    execution_limits: Any | None = None,
) -> BacktestResult:
    """Run a backtest with minimal setup.

    Args:
        prices: Price DataFrame or path to parquet file
        strategy: Strategy instance to execute
        signals: Optional signals DataFrame or path
        context: Optional context DataFrame or path
        config: BacktestConfig instance, preset name (str), or None for defaults
        contract_specs: Per-asset contract specifications (futures multipliers, etc.)
        market_impact_model: Market impact model for fill simulation
        execution_limits: Execution limits (max order size, etc.)

    Returns:
        BacktestResult with metrics, trades, equity curve, and export methods.

    Example:
        # Using config preset
        result = run_backtest(prices_df, strategy, config="backtrader")
        print(result.metrics["sharpe"])

        # Using custom config
        config = BacktestConfig.from_preset("backtrader")
        config.commission_rate = 0.002
        result = run_backtest(prices_df, strategy, config=config)

        # Futures with contract specs
        from ml4t.backtest import ContractSpec, AssetClass
        specs = {"ES": ContractSpec(symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50.0)}
        result = run_backtest(prices_df, strategy, config=config, contract_specs=specs)
    """
    feed = DataFeed(
        prices_path=prices if isinstance(prices, str) else None,
        signals_path=signals if isinstance(signals, str) else None,
        context_path=context if isinstance(context, str) else None,
        prices_df=prices if isinstance(prices, pl.DataFrame) else None,
        signals_df=signals if isinstance(signals, pl.DataFrame) else None,
        context_df=context if isinstance(context, pl.DataFrame) else None,
    )

    if isinstance(config, str):
        from .config import BacktestConfig as ConfigCls

        config = ConfigCls.from_preset(config)

    return Engine(
        feed,
        strategy,
        config,
        contract_specs=contract_specs,
        market_impact_model=market_impact_model,
        execution_limits=execution_limits,
    ).run()
