"""Zipline framework driver for validation scenarios.

Consolidates 15 run_zipline() functions into a single parameterized driver.

Zipline uses next-bar execution with open fills. Requires:
- exchange_calendars for NYSE trading calendar dates
- Custom bundle registration for test data
- Custom slippage model for open-price fills
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.types import FrameworkResult, ScenarioConfig


def run(
    scenario: ScenarioConfig,
    prices_df: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None = None,
) -> FrameworkResult:
    """Run a validation scenario using Zipline.

    Args:
        scenario: Scenario configuration.
        prices_df: OHLCV DataFrame with DatetimeIndex.
        entries: Boolean entry signal array.
        exits: Boolean exit signal array (None for risk-rule-only exits).

    Returns:
        FrameworkResult with trade data.
    """
    try:
        from zipline import run_algorithm
        from zipline.api import order, order_target, set_slippage, symbol
        from zipline.finance import slippage as zipline_slippage
    except ImportError:
        raise ImportError("Zipline not installed. Run in .venv-zipline environment.")

    constants = scenario.constants

    # Store signals for the algo to access
    signal_data = {
        "entries": entries,
        "exits": exits,
        "dates": prices_df.index,
    }
    shares = scenario.shares
    risk_rules = scenario.risk_rules
    is_short = scenario.strategy_type == "short_only"

    def initialize(context):
        context.asset = symbol("TEST")
        context.signal_data = signal_data
        context.bar_count = 0
        context.in_position = False
        context.entry_price = None
        context.high_water_mark = None
        # Use custom slippage to fill at open price
        set_slippage(_create_open_price_slippage())

    def handle_data(context, data):
        idx = context.bar_count
        if idx >= len(context.signal_data["entries"]):
            return

        entry = context.signal_data["entries"][idx]
        exit_sig = context.signal_data["exits"][idx] if context.signal_data["exits"] is not None else False

        current_pos = context.portfolio.positions[context.asset].amount
        current_price = data.current(context.asset, "close")

        # Risk rule evaluation for manual stop/take-profit (Zipline has no built-in rules)
        if current_pos != 0 and context.entry_price is not None:
            should_exit = False

            for rule in risk_rules:
                if rule["type"] == "StopLoss":
                    if current_pos > 0:
                        loss_pct = (current_price - context.entry_price) / context.entry_price
                        if loss_pct <= -rule["pct"]:
                            should_exit = True
                    elif current_pos < 0:
                        loss_pct = (context.entry_price - current_price) / context.entry_price
                        if loss_pct <= -rule["pct"]:
                            should_exit = True

                elif rule["type"] == "TakeProfit":
                    if current_pos > 0:
                        gain_pct = (current_price - context.entry_price) / context.entry_price
                        if gain_pct >= rule["pct"]:
                            should_exit = True
                    elif current_pos < 0:
                        gain_pct = (context.entry_price - current_price) / context.entry_price
                        if gain_pct >= rule["pct"]:
                            should_exit = True

                elif rule["type"] == "TrailingStop":
                    if context.high_water_mark is None:
                        context.high_water_mark = current_price
                    if current_pos > 0:
                        context.high_water_mark = max(context.high_water_mark, current_price)
                        drawdown = (context.high_water_mark - current_price) / context.high_water_mark
                        if drawdown >= rule["pct"]:
                            should_exit = True
                    elif current_pos < 0:
                        context.high_water_mark = min(context.high_water_mark, current_price)
                        drawup = (current_price - context.high_water_mark) / context.high_water_mark
                        if drawup >= rule["pct"]:
                            should_exit = True

            if should_exit:
                order_target(context.asset, 0)
                context.in_position = False
                context.entry_price = None
                context.high_water_mark = None
                context.bar_count += 1
                return

        # Signal-based exits
        if exit_sig and current_pos > 0:
            order_target(context.asset, 0)
            context.in_position = False
            context.entry_price = None
        elif exit_sig and current_pos < 0:
            order_target(context.asset, 0)
            context.in_position = False
            context.entry_price = None
        # Entries
        elif entry and current_pos == 0:
            if is_short:
                order(context.asset, -shares)
            else:
                order(context.asset, shares)
            context.in_position = True
            context.entry_price = current_price
            context.high_water_mark = current_price

        context.bar_count += 1

    def analyze(context, perf):
        pass

    # Setup bundle
    bundle_name = _setup_bundle(prices_df)

    # Run
    start = prices_df.index[0]
    end = prices_df.index[-1]
    if start.tz is not None:
        start = start.tz_convert(None)
        end = end.tz_convert(None)

    results = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=scenario.initial_cash,
        bundle=bundle_name,
        data_frequency="daily",
    )

    # Extract results
    final_value = results["portfolio_value"].iloc[-1]

    # Count trades from transactions
    num_trades = 0
    for txn_list in results["transactions"]:
        if txn_list:
            num_trades += len(txn_list)
    num_trades = num_trades // 2  # Entry + exit = 1 round trip

    return FrameworkResult(
        framework="Zipline",
        final_value=final_value,
        total_pnl=final_value - scenario.initial_cash,
        num_trades=num_trades,
    )


def _create_open_price_slippage():
    """Create a custom slippage model that fills at open price."""
    from zipline.finance.slippage import SlippageModel

    class OpenPriceSlippage(SlippageModel):
        @staticmethod
        def process_order(data, order):
            return (data.current(order.asset, "open"), order.amount)

    return OpenPriceSlippage()


def _setup_bundle(prices_df: pd.DataFrame, bundle_name: str = "test_validation") -> str:
    """Register and ingest a custom bundle with test data."""
    from zipline.data.bundles import ingest, register

    def make_ingest_func(df):
        def ingest_func(
            environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
            adjustment_writer, calendar, start_session, end_session,
            cache, show_progress, output_dir,
        ):
            sessions = calendar.sessions_in_range(start_session, end_session)

            df_naive = df.copy()
            if df_naive.index.tz is not None:
                df_naive.index = df_naive.index.tz_convert(None)

            valid_mask = df_naive.index.isin(sessions)
            trading_df = df_naive[valid_mask].copy()

            if len(trading_df) == 0:
                raise ValueError(
                    f"No trading days found. Data: {df_naive.index[:5].tolist()}, "
                    f"Calendar: {sessions[:5].tolist()}"
                )

            asset_db_writer.write(
                equities=pd.DataFrame({
                    "symbol": ["TEST"],
                    "asset_name": ["Test Asset"],
                    "exchange": ["NYSE"],
                })
            )

            daily_bar_writer.write(
                [(0, trading_df[["open", "high", "low", "close", "volume"]])],
                show_progress=show_progress,
            )
            adjustment_writer.write()

        return ingest_func

    start_session = prices_df.index[0]
    end_session = prices_df.index[-1]
    if start_session.tz is not None:
        start_session = start_session.tz_convert(None)
        end_session = end_session.tz_convert(None)

    register(
        bundle_name,
        make_ingest_func(prices_df),
        calendar_name="XNYS",
        start_session=start_session,
        end_session=end_session,
    )

    ingest(bundle_name, show_progress=False)
    return bundle_name
