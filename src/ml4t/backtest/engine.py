"""Backtesting engine orchestration."""

from __future__ import annotations

import inspect
import warnings
from datetime import date, datetime
from math import ceil
from typing import TYPE_CHECKING, Any

import polars as pl
from ml4t.specs import (
    LIFECYCLE_V1,
    ExecutionPolicy,
    HistoricalStrategyCompatibilityError,
    LifecyclePhase,
    LifecycleVersion,
    negotiate_lifecycle_version,
)

from .analytics import EquityCurve, TradeAnalyzer
from .analytics.metrics import calmar_ratio
from .broker import Broker
from .config import DataFrequency
from .datafeed import DataFeed
from .lifecycle import LifecycleDispatcher
from .preopen import PreOpenTargetManager, default_execution_policy
from .strategy import Strategy
from .types import ExecutionMode, OrderSide, OrderType

if TYPE_CHECKING:
    from .config import BacktestConfig
    from .result import BacktestResult

# Compare at most 20 dates; require min(4, 10% of sample) bars and 25% relative recovery.
_TIMEZONE_DIAGNOSTIC_MAX_DATES = 20
_TIMEZONE_DIAGNOSTIC_MAX_RETENTION = 0.5
_TIMEZONE_DIAGNOSTIC_MAX_REQUIRED_GAIN = 4
_TIMEZONE_DIAGNOSTIC_MIN_GAIN_FRACTION = 0.1
_TIMEZONE_DIAGNOSTIC_MIN_RELATIVE_GAIN = 1.25
_TIMEZONE_DIAGNOSTIC_NEAR_TOTAL_LOSS = 0.1


class Engine:
    """Event-driven backtesting engine.

    The Engine orchestrates the backtest by iterating through market data,
    managing the broker, and calling the strategy on each bar.
    Engine instances are single-use; create a new instance for each run.

    Execution Flow:
        1. Call on_prepare with the feed timestamps and resolved config.
        2. Call on_start before any market bar is registered.
        3. For each accepted session bar, register data, process eligible
           deferred orders and risk, call the per-bar strategy callbacks, process
           configured current-bar orders, and record marked portfolio state.
        4. Call on_end after the final timestamp.
        5. Return closed trades plus marked open positions. The engine does not
           submit automatic end-of-data liquidation orders.

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
        lifecycle_version: LifecycleVersion | str = LifecycleVersion.V1,
        execution_policy: ExecutionPolicy | None = None,
        target_intent_state: dict[str, Any] | None = None,
    ):
        from .config import BacktestConfig as ConfigCls

        negotiated_version = negotiate_lifecycle_version(lifecycle_version)
        self._validate_strategy_lifecycle(strategy)
        if config is None:
            config = ConfigCls()

        self.feed = feed
        self.strategy = strategy
        self.config = config.merge_feed_spec(getattr(feed, "feed_spec", None))
        self.execution_mode = self.config.execution_mode
        self.lifecycle_version = negotiated_version
        self.execution_policy = execution_policy or default_execution_policy(self.config)
        if execution_limits is None and self.execution_policy.liquidity_fraction < 1.0:
            from .execution import VolumeParticipationLimit

            execution_limits = VolumeParticipationLimit(
                max_participation=self.execution_policy.liquidity_fraction
            )
        self.broker = Broker.from_config(
            self.config,
            contract_specs=contract_specs,
            market_impact_model=market_impact_model,
            execution_limits=execution_limits,
        )
        self.equity_curve: list[tuple[datetime, float]] = []
        self.portfolio_state: list[tuple[datetime, float, float, float, float, int]] = []
        self.lifecycle_dispatcher = LifecycleDispatcher(strategy, LIFECYCLE_V1)
        self.preopen_target_manager = PreOpenTargetManager(
            self.broker,
            self.execution_policy,
            self.lifecycle_version,
            calendar=self.config.resolved_calendar,
        )
        self.broker._preopen_target_manager = self.preopen_target_manager
        if target_intent_state is not None:
            self.preopen_target_manager.restore_state(target_intent_state)
        self._strategy_finalized = False
        self._market_event_count = 0

        # Calendar session enforcement (lazy initialized in run())
        self._calendar = None
        self._skipped_bars = 0
        self._has_run = False

    def run(self) -> BacktestResult:
        """Run backtest and return structured results.

        Returns:
            BacktestResult with trades, equity curve, metrics, and export methods.
            Call .to_dict() for backward-compatible dictionary output.

        Raises:
            RuntimeError: If a run was already started on this Engine instance.
        """
        if self._has_run:
            raise RuntimeError("Engine.run() was already started; create a new Engine for each run")
        self._has_run = True

        # Lazy calendar initialization (zero cost if unused)
        is_trading_day_fn = None
        valid_intraday_bar_mask: bytearray | None = None
        timestamps = self.feed.timestamps
        if self.config and self.config.resolved_calendar:
            from .calendar import filter_to_trading_sessions, get_calendar, is_trading_day

            self._calendar = get_calendar(self.config.resolved_calendar)
            is_trading_day_fn = is_trading_day

            if (
                self.config.enforce_sessions
                and self.config.resolved_data_frequency != DataFrequency.DAILY
            ):
                prices_frame = self.feed.prices
                timestamp_dtype = (
                    prices_frame.schema[self.feed.feed_spec.timestamp_col]
                    if prices_frame is not None
                    else None
                )
                naive_timestamps = (
                    isinstance(timestamp_dtype, pl.Datetime) and timestamp_dtype.time_zone is None
                )
                timestamp_frame = pl.DataFrame(
                    {
                        "timestamp": timestamps,
                        "__feed_bar_index": range(len(timestamps)),
                    }
                )
                filtered = filter_to_trading_sessions(
                    timestamp_frame,
                    self.config.resolved_calendar,
                    naive_tz=self.config.resolved_timezone,
                )
                retained_bars = len(filtered)
                total_bars = len(timestamp_frame)
                calendar_id = self.config.resolved_calendar
                retention = retained_bars / total_bars if total_bars else 1.0
                calendar_timezone = str(self._calendar.tz)
                compare_calendar_timezone = (
                    naive_timestamps
                    and self.config.resolved_timezone != calendar_timezone
                    and retention <= _TIMEZONE_DIAGNOSTIC_MAX_RETENTION
                )
                configured_sample_retained = retained_bars
                alternative_sample_retained = retained_bars
                sample_bars = total_bars
                if compare_calendar_timezone:
                    sample_dates: set[date] = set()
                    sample_bars = 0
                    for sample_bars, timestamp in enumerate(timestamps, start=1):
                        sample_date = timestamp.date()
                        if (
                            sample_date not in sample_dates
                            and len(sample_dates) == _TIMEZONE_DIAGNOSTIC_MAX_DATES
                        ):
                            sample_bars -= 1
                            break
                        sample_dates.add(sample_date)
                    diagnostic_frame = timestamp_frame.head(sample_bars)
                    if sample_bars != total_bars:
                        configured_sample_retained = len(
                            filter_to_trading_sessions(
                                diagnostic_frame,
                                calendar_id,
                                naive_tz=self.config.resolved_timezone,
                            )
                        )
                    alternative_sample_retained = len(
                        filter_to_trading_sessions(
                            diagnostic_frame,
                            calendar_id,
                            naive_tz=calendar_timezone,
                        )
                    )
                if configured_sample_retained:
                    relative_retention_gain = (
                        alternative_sample_retained / configured_sample_retained
                    )
                elif alternative_sample_retained:
                    relative_retention_gain = float("inf")
                else:
                    relative_retention_gain = 1.0
                minimum_absolute_gain = max(
                    1,
                    min(
                        _TIMEZONE_DIAGNOSTIC_MAX_REQUIRED_GAIN,
                        ceil(sample_bars * _TIMEZONE_DIAGNOSTIC_MIN_GAIN_FRACTION),
                    ),
                )
                alternative_timezone_explains_loss = (
                    compare_calendar_timezone
                    and alternative_sample_retained - configured_sample_retained
                    >= minimum_absolute_gain
                    and relative_retention_gain >= _TIMEZONE_DIAGNOSTIC_MIN_RELATIVE_GAIN
                )
                possible_session_misconfiguration = (
                    retention <= _TIMEZONE_DIAGNOSTIC_NEAR_TOTAL_LOSS
                    or alternative_timezone_explains_loss
                )
                should_warn = possible_session_misconfiguration and any(
                    is_trading_day_fn(calendar_id, feed_date)
                    for feed_date in {ts.date() for ts in timestamps}
                )
                if should_warn:
                    timezone_note = ""
                    if alternative_timezone_explains_loss:
                        timezone_note = (
                            f" Interpreting naive timestamps as {calendar_timezone!r} would "
                            f"retain {alternative_sample_retained} bars instead of "
                            f"{configured_sample_retained} in a {sample_bars}-bar sample."
                        )
                    elif naive_timestamps:
                        timezone_note = (
                            f" Naive timestamps were interpreted as "
                            f"{self.config.resolved_timezone!r}."
                        )
                    warnings.warn(
                        f"Session filtering for {self.config.resolved_calendar!r} retained "
                        f"{retained_bars} of {total_bars} intraday bars.{timezone_note} "
                        "Verify the configured calendar, data timezone, and session coverage.",
                        UserWarning,
                        stacklevel=2,
                    )
                valid_intraday_bar_mask = bytearray(len(timestamps))
                for index in filtered["__feed_bar_index"]:
                    valid_intraday_bar_mask[index] = 1

        self.lifecycle_dispatcher.dispatch(
            LifecyclePhase.RUN_START,
            self.broker,
            self.broker,
        )
        self.lifecycle_dispatcher.dispatch(
            LifecyclePhase.CAUSAL_INITIALIZATION,
            self.broker,
            self.broker,
            self.config,
        )

        for feed_bar_index, (timestamp, assets_data, context) in enumerate(self.feed):
            # Calendar session enforcement
            calendar_id = self.config.resolved_calendar if self.config else None
            if (
                self._calendar
                and calendar_id
                and self.config
                and self.config.enforce_sessions
                and is_trading_day_fn
            ):
                # Daily bars use valid dates. Intraday bars use precomputed session intervals.
                if self.config.resolved_data_frequency == DataFrequency.DAILY:
                    if not is_trading_day_fn(calendar_id, timestamp.date()):
                        self._skipped_bars += 1
                        continue
                elif valid_intraday_bar_mask is None or not valid_intraday_bar_mask[feed_bar_index]:
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

            self.preopen_target_manager.process_opening(timestamp)

            # Process pending exits from NEXT_BAR_OPEN mode (fills at open)
            # This must happen BEFORE evaluate_position_rules() to clear deferred exits
            self.broker._process_pending_exits()

            # Evaluate position rules (stops, trails, etc.) - generates exit orders
            self.broker.evaluate_position_rules()

            if self.execution_mode == ExecutionMode.NEXT_BAR:
                # Process same-cycle risk exits before ordinary strategy decisions.
                self.broker._process_orders(use_open=True)
                # Strategy generates new orders
                self._dispatch_market_event(timestamp, assets_data, context)
                # MOC orders are the one next-bar exception: they execute on the
                # current session close after strategy logic runs.
                self.broker._process_orders(
                    order_types={OrderType.MOC},
                    include_orders_this_bar=True,
                )
            else:
                # Same-bar mode: process before and after strategy
                self.broker._process_orders()
                self._dispatch_market_event(timestamp, assets_data, context)
                self.broker._process_orders()

            self.preopen_target_manager.reconcile(timestamp)

            # Update water marks at END of bar, AFTER all orders processed
            # This ensures new positions get their HWM updated from entry bar's high
            # VBT Pro behavior: HWM updated at bar end, used in NEXT bar's trail evaluation
            self.broker._update_water_marks()

            self._record_portfolio_state(timestamp)

        self._finalize_strategy()
        self.lifecycle_dispatcher.validate_completed_run(self._market_event_count)
        return self._generate_results()

    @staticmethod
    def _validate_strategy_lifecycle(strategy: Strategy) -> None:
        strategy_type = type(strategy)
        if getattr(strategy_type, "on_before_risk", None) is not None:
            raise HistoricalStrategyCompatibilityError(
                strategy_type.__name__,
                "on_before_risk",
                LifecyclePhase.PRE_OPEN,
            )
        parameters = tuple(inspect.signature(strategy_type.on_prepare).parameters.values())
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if any(parameter.name == "timestamps" for parameter in parameters) or len(positional) > 3:
            raise HistoricalStrategyCompatibilityError(
                strategy_type.__name__,
                "on_prepare(timestamps)",
                LifecyclePhase.CAUSAL_INITIALIZATION,
            )

    def _dispatch_market_event(
        self,
        timestamp: datetime,
        assets_data: Any,
        context: dict[str, Any],
    ) -> None:
        try:
            self.lifecycle_dispatcher.dispatch(
                LifecyclePhase.MARKET_EVENT,
                self.broker,
                timestamp,
                assets_data,
                context,
                self.broker,
                event_time=timestamp,
            )
            self._market_event_count += 1
        except BaseException as failure:
            try:
                self._finalize_strategy()
            except BaseException as finalization_failure:
                failure.add_note(
                    "on_end also failed during cleanup: "
                    f"{type(finalization_failure).__name__}: {finalization_failure}"
                )
            raise

    def _finalize_strategy(self) -> None:
        if self._strategy_finalized:
            return
        self.lifecycle_dispatcher.dispatch(
            LifecyclePhase.RUN_END,
            self.broker,
            self.broker,
        )
        self._strategy_finalized = True

    def run_dict(self) -> dict[str, Any]:
        """Run backtest and return dictionary (backward compatible).

        This is equivalent to run().to_dict() but more explicit for code
        that requires dictionary output.

        Returns:
            Dictionary with metrics, trades, and equity curve.

        Raises:
            RuntimeError: If a run was already started on this Engine instance.
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
                price = self.broker.get_last_price(asset) or pos.current_price or pos.entry_price
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
        if not self.broker.fills:
            avg_open_positions = (
                sum(state[5] for state in self.portfolio_state) / len(self.portfolio_state)
                if self.portfolio_state
                else 0.0
            )
            max_open_positions = max((state[5] for state in self.portfolio_state), default=0)
            return {
                "num_orders": len(self.broker.orders),
                "num_rejected_orders": len(self.broker.get_rejected_orders()),
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

        for fill in self.broker.fills:
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
            "num_orders": len(self.broker.orders),
            "num_rejected_orders": len(self.broker.get_rejected_orders()),
            "num_fills": len(self.broker.fills),
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

        contract_evidence = {
            "lifecycle_version": self.lifecycle_version.value,
            "execution_policy": self.execution_policy.to_dict(),
            "target_intents": [intent.to_dict() for intent in self.preopen_target_manager.targets],
            "child_order_intents": [
                child.to_dict() for child in self.preopen_target_manager.children
            ],
            "intent_reconciliations": [
                record.to_dict() for record in self.preopen_target_manager.reconciliations
            ],
        }

        if not self.equity_curve:
            # Return empty result for no-data case
            return BacktestResult(
                trades=[],
                equity_curve=[],
                fills=[],
                rejected_orders=self.broker.get_rejected_orders(),
                predictions=self.feed.signals,
                portfolio_state=[],
                metrics={
                    "skipped_bars": self._skipped_bars,
                    "num_orders": len(self.broker.orders),
                    "num_rejected_orders": len(self.broker.get_rejected_orders()),
                    **contract_evidence,
                },
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

        # Realized-P&L metrics include partial reductions. TradeAnalyzer limits
        # lifecycle metrics such as holding period and excursions to full closes.
        realized_trades = [t for t in all_trades if t.status in {"closed", "partial"}]
        trade_analyzer = TradeAnalyzer(realized_trades)
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
            **contract_evidence,
        }

        return BacktestResult(
            trades=all_trades,  # Includes both closed and open trades
            equity_curve=list(self.equity_curve),
            fills=list(self.broker.fills),
            rejected_orders=self.broker.get_rejected_orders(),
            predictions=self.feed.signals,
            portfolio_state=list(self.portfolio_state),
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
        lifecycle_version: LifecycleVersion | str = LifecycleVersion.V1,
        execution_policy: ExecutionPolicy | None = None,
        target_intent_state: dict[str, Any] | None = None,
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
            lifecycle_version=lifecycle_version,
            execution_policy=execution_policy,
            target_intent_state=target_intent_state,
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
    lifecycle_version: LifecycleVersion | str = LifecycleVersion.V1,
    execution_policy: ExecutionPolicy | None = None,
    target_intent_state: dict[str, Any] | None = None,
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
        lifecycle_version=lifecycle_version,
        execution_policy=execution_policy,
        target_intent_state=target_intent_state,
    ).run()
