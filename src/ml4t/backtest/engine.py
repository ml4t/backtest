"""Backtesting engine orchestration."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from .analytics import EquityCurve, TradeAnalyzer
from .analytics.metrics import calmar_ratio
from .broker import Broker
from .config import DataFrequency
from .datafeed import DataFeed
from .strategy import Strategy
from .types import ExecutionMode, OrderSide, OrderType

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
        self.config = config.merge_feed_spec(getattr(feed, "feed_spec", None))
        self.execution_mode = self.config.execution_mode
        self.broker = Broker.from_config(
            self.config,
            contract_specs=contract_specs,
            market_impact_model=market_impact_model,
            execution_limits=execution_limits,
        )
        self.equity_curve: list[tuple[datetime, float]] = []
        self.portfolio_state: list[tuple[datetime, float, float, float, float, int]] = []

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
        if self.config and self.config.resolved_calendar:
            from .calendar import get_calendar, is_trading_day

            self._calendar = get_calendar(self.config.resolved_calendar)
            is_trading_day_fn = is_trading_day

        self.strategy.on_prepare(self.broker, self.feed.timestamps, self.config)
        self.strategy.on_start(self.broker)

        # Date-level cache for trading day checks (significant speedup for intraday data)
        trading_day_cache: dict[date, bool] = {}

        for timestamp, assets_data, context in self.feed:
            # Calendar session enforcement
            calendar_id = self.config.resolved_calendar if self.config else None
            if (
                self._calendar
                and calendar_id
                and self.config
                and self.config.enforce_sessions
                and is_trading_day_fn
            ):
                # For daily data, check trading day; for intraday, check market hours
                if self.config.resolved_data_frequency == DataFrequency.DAILY:
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
            closes = getattr(assets_data, "_closes", None)
            volumes = getattr(assets_data, "_volumes", None)
            bids = getattr(assets_data, "_bids", None)
            asks = getattr(assets_data, "_asks", None)
            mids = getattr(assets_data, "_mids", None)
            bid_sizes = getattr(assets_data, "_bid_sizes", None)
            ask_sizes = getattr(assets_data, "_ask_sizes", None)
            signals = getattr(assets_data, "_signals", None)

            if (
                prices is None
                or opens is None
                or highs is None
                or lows is None
                or closes is None
                or volumes is None
                or bids is None
                or asks is None
                or mids is None
                or bid_sizes is None
                or ask_sizes is None
                or signals is None
            ):
                prices = {
                    a: price
                    for a, d in assets_data.items()
                    if (price := d.get("price", d.get("close"))) is not None
                }
                opens = {}
                highs = {}
                lows = {}
                closes = {
                    a: close
                    for a, d in assets_data.items()
                    if (close := d.get("close", d.get("price"))) is not None
                }
                for asset, data in assets_data.items():
                    base_price = data.get("close")
                    if base_price is None:
                        base_price = data.get("price")
                    opens[asset] = data.get("open") if data.get("open") is not None else base_price
                    highs[asset] = data.get("high") if data.get("high") is not None else base_price
                    lows[asset] = data.get("low") if data.get("low") is not None else base_price
                volumes = {a: d.get("volume", 0) for a, d in assets_data.items()}
                bids = {a: d["bid"] for a, d in assets_data.items() if d.get("bid") is not None}
                asks = {a: d["ask"] for a, d in assets_data.items() if d.get("ask") is not None}
                mids = {a: d["mid"] for a, d in assets_data.items() if d.get("mid") is not None}
                bid_sizes = {
                    a: d["bid_size"]
                    for a, d in assets_data.items()
                    if d.get("bid_size") is not None
                }
                ask_sizes = {
                    a: d["ask_size"]
                    for a, d in assets_data.items()
                    if d.get("ask_size") is not None
                }
                signals = {a: d.get("signals", {}) for a, d in assets_data.items()}

            self.broker._update_time(
                timestamp,
                prices,
                opens,
                highs,
                lows,
                closes,
                volumes,
                bids,
                asks,
                mids,
                bid_sizes,
                ask_sizes,
                signals,
            )

            # Process pending exits from NEXT_BAR_OPEN mode (fills at open)
            # This must happen BEFORE evaluate_position_rules() to clear deferred exits
            self.broker._process_pending_exits()

            # Optional strategy phase for opening orders that must receive risk
            # protection during the current bar. Existing strategies inherit a no-op.
            self.strategy.on_before_risk(timestamp, assets_data, context, self.broker)

            # Evaluate position rules (stops, trails, etc.) - generates exit orders
            self.broker.evaluate_position_rules()

            if self.execution_mode == ExecutionMode.NEXT_BAR:
                # Next-bar mode: process pending orders at open price
                self.broker._process_orders(use_open=True)
                # Strategy generates new orders
                self.strategy.on_data(timestamp, assets_data, context, self.broker)
                # MOC orders are the one next-bar exception: they execute on the
                # current session close after strategy logic runs.
                self.broker._process_orders(
                    order_types={OrderType.MOC},
                    include_orders_this_bar=True,
                )
            else:
                # Same-bar mode: process before and after strategy
                self.broker._process_orders()
                self.strategy.on_data(timestamp, assets_data, context, self.broker)
                self.broker._process_orders()

            # Update water marks at END of bar, AFTER all orders processed
            # This ensures new positions get their HWM updated from entry bar's high
            # VBT Pro behavior: HWM updated at bar end, used in NEXT bar's trail evaluation
            self.broker._update_water_marks()

            self._record_portfolio_state(timestamp)

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

    def _record_portfolio_state(self, timestamp: datetime) -> None:
        """Capture per-bar portfolio state for reporting."""
        cash = self.broker.cash
        gross_exposure = 0.0
        net_exposure = 0.0

        for asset, pos in self.broker.positions.items():
            price = self.broker.get_mark_price(asset, quantity=pos.quantity)
            if price is None:
                price = self.broker._last_prices.get(asset, pos.current_price or pos.entry_price)
            position_value = pos.quantity * price * pos.multiplier
            gross_exposure += abs(position_value)
            net_exposure += position_value

        equity = cash + net_exposure
        self.equity_curve.append((timestamp, equity))
        self.portfolio_state.append(
            (timestamp, equity, cash, gross_exposure, net_exposure, len(self.broker.positions))
        )

    def _build_activity_metrics(self) -> dict[str, int | float]:
        """Compute fill and portfolio activity metrics."""
        fills = self.broker.fills
        if not fills:
            avg_open_positions = (
                sum(state[5] for state in self.portfolio_state) / len(self.portfolio_state)
                if self.portfolio_state
                else 0.0
            )
            max_open_positions = max((state[5] for state in self.portfolio_state), default=0)
            return {
                "num_fills": 0,
                "num_rebalance_events": 0,
                "unique_symbols_traded": 0,
                "total_filled_notional": 0.0,
                "avg_turnover": 0.0,
                "max_turnover": 0.0,
                "avg_open_positions": avg_open_positions,
                "max_open_positions": max_open_positions,
            }

        fill_notional_by_timestamp: dict[datetime, float] = {}
        total_filled_notional = 0.0
        traded_symbols: set[str] = set()
        rebalance_events: set[str | datetime] = set()

        for fill in fills:
            multiplier = self.broker.get_multiplier(fill.asset)
            notional = abs(fill.quantity) * fill.price * multiplier
            total_filled_notional += notional
            fill_notional_by_timestamp[fill.timestamp] = (
                fill_notional_by_timestamp.get(fill.timestamp, 0.0) + notional
            )
            traded_symbols.add(fill.asset)
            rebalance_events.add(fill.rebalance_id or fill.timestamp)

        turnovers = [
            fill_notional_by_timestamp.get(timestamp, 0.0) / equity if equity else 0.0
            for timestamp, equity, *_ in self.portfolio_state
        ]
        avg_open_positions = (
            sum(state[5] for state in self.portfolio_state) / len(self.portfolio_state)
            if self.portfolio_state
            else 0.0
        )
        max_open_positions = max((state[5] for state in self.portfolio_state), default=0)

        return {
            "num_fills": len(fills),
            "num_rebalance_events": len(rebalance_events),
            "unique_symbols_traded": len(traded_symbols),
            "total_filled_notional": total_filled_notional,
            "avg_turnover": sum(turnovers) / len(turnovers) if turnovers else 0.0,
            "max_turnover": max(turnovers, default=0.0),
            "avg_open_positions": avg_open_positions,
            "max_open_positions": max_open_positions,
        }

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
                predictions=self.feed.signals,
                portfolio_state=[],
                metrics={"skipped_bars": self._skipped_bars},
                config=self.config,
            )

        # Build EquityCurve from raw data
        equity = EquityCurve.from_config(self.config)
        for ts, value in self.equity_curve:
            equity.append(ts, value)

        # Collect all trades (closed + open)
        all_trades = list(self.broker.trades)  # Closed trades

        # Add open positions as trades with status="open" (mark-to-market)
        if self.equity_curve:
            last_timestamp = self.equity_curve[-1][0]
            for asset, pos in self.broker.positions.items():
                # Get last known price for this asset
                last_price = (
                    self.broker.get_mark_price(asset, quantity=pos.quantity) or pos.entry_price
                )
                entry_quote = pos.context.get("entry_quote_context", {})
                exit_quote = self.broker.get_quote_context(
                    asset,
                    OrderSide.BUY if pos.quantity < 0 else OrderSide.SELL,
                )

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
                    exit_slippage=0.0,  # No exit slippage yet
                    exit_reason="end_of_backtest",
                    status="open",
                    mfe=pos.max_favorable_excursion,
                    mae=pos.max_adverse_excursion,
                    entry_slippage=pos.entry_slippage,
                    multiplier=pos.multiplier,
                    entry_quote_mid_price=entry_quote.get("quote_mid_price"),
                    entry_bid_price=entry_quote.get("bid_price"),
                    entry_ask_price=entry_quote.get("ask_price"),
                    entry_spread=entry_quote.get("spread"),
                    entry_available_size=entry_quote.get("available_size"),
                    exit_quote_mid_price=exit_quote.get("quote_mid_price"),
                    exit_bid_price=exit_quote.get("bid_price"),
                    exit_ask_price=exit_quote.get("ask_price"),
                    exit_spread=exit_quote.get("spread"),
                    exit_available_size=exit_quote.get("available_size"),
                )
                all_trades.append(open_trade)

        # Build TradeAnalyzer (only on closed trades for accurate stats)
        closed_trades = [t for t in all_trades if t.status == "closed"]
        trade_analyzer = TradeAnalyzer(closed_trades)
        activity_metrics = self._build_activity_metrics()

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
            "total_slippage": sum(t.total_slippage_cost for t in all_trades),
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
            # Activity and exposure summaries
            **activity_metrics,
        }

        return BacktestResult(
            trades=all_trades,  # Includes both closed and open trades
            equity_curve=self.equity_curve,
            fills=self.broker.fills,
            predictions=self.feed.signals,
            portfolio_state=self.portfolio_state,
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
    feed_spec: Any | None = None,
    contract: Any | None = None,
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
        feed_spec: Optional shared dataset contract for schema and temporal metadata
        contract: Alias for feed_spec
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
        feed_spec=feed_spec,
        contract=contract,
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
