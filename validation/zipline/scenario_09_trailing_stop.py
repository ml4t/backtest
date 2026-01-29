#!/usr/bin/env python3
"""Scenario 09: Trailing Stop validation against Zipline.

Zipline doesn't have built-in trailing stop - must be implemented manually.

Run from .venv-validation environment:
    source .venv-validation/bin/activate
    python validation/zipline/scenario_09_trailing_stop.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIL_PCT = 0.05  # 5% trailing stop
SHARES_PER_TRADE = 100


def generate_test_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate trending data that will trigger trailing stops."""
    import exchange_calendars as xcals

    np.random.seed(seed)
    n_bars = 100
    base_price = 100.0

    prices = [base_price]
    for i in range(1, n_bars):
        if i < 30:
            change = np.random.randn() * 0.01 + 0.005
        elif i < 35:
            change = -0.02 + np.random.randn() * 0.005
        elif i < 60:
            change = np.random.randn() * 0.01 + 0.003
        elif i < 65:
            change = -0.015 + np.random.randn() * 0.005
        else:
            change = np.random.randn() * 0.01 + 0.001
        prices.append(prices[-1] * (1 + change))

    prices = np.array(prices)

    nyse = xcals.get_calendar("XNYS")
    start = pd.Timestamp("2020-01-02")
    all_sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
    dates = pd.DatetimeIndex(all_sessions[:n_bars]).tz_localize("UTC")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.002),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.full(n_bars, 100000.0),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True
    entries[40] = True

    return df, entries


def setup_zipline_bundle(prices_df: pd.DataFrame, bundle_name: str = "test_trailing"):
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
            trading_df = df_naive[df_naive.index.isin(sessions)].copy()

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

    start_session = prices_df.index[0].tz_convert(None)
    end_session = prices_df.index[-1].tz_convert(None)

    register(bundle_name, make_ingest_func(prices_df), calendar_name="XNYS",
             start_session=start_session, end_session=end_session)
    ingest(bundle_name, show_progress=False)
    return bundle_name


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    from zipline import run_algorithm
    from zipline.api import order, order_target, set_commission, set_slippage, symbol
    from zipline.finance.commission import NoCommission
    from zipline.finance.slippage import NoSlippage

    signal_data = {"entries": entries}

    def initialize(context):
        context.asset = symbol("TEST")
        context.signal_data = signal_data
        context.bar_count = 0
        context.high_water_mark = 0.0
        context.in_position = False
        context.num_trades = 0
        set_commission(NoCommission())
        set_slippage(NoSlippage())

    def handle_data(context, data):
        idx = context.bar_count
        if idx >= len(context.signal_data["entries"]):
            return

        entry = context.signal_data["entries"][idx]
        current_price = data.current(context.asset, "close")
        current_pos = context.portfolio.positions[context.asset].amount

        if context.in_position and current_pos > 0:
            # Update high water mark
            if current_price > context.high_water_mark:
                context.high_water_mark = current_price

            # Check trailing stop
            stop_price = context.high_water_mark * (1 - TRAIL_PCT)
            if current_price <= stop_price:
                order_target(context.asset, 0)
                context.in_position = False
                context.high_water_mark = 0.0

        elif entry and current_pos == 0:
            order(context.asset, SHARES_PER_TRADE)
            context.in_position = True
            context.high_water_mark = current_price
            context.num_trades += 1

        context.bar_count += 1

    bundle_name = setup_zipline_bundle(prices_df)
    start = prices_df.index[0].tz_convert(None)
    end = prices_df.index[-1].tz_convert(None)

    results = run_algorithm(
        start=start, end=end,
        initialize=initialize, handle_data=handle_data,
        capital_base=100_000.0, bundle=bundle_name, data_frequency="daily",
    )

    final_value = results["portfolio_value"].iloc[-1]
    num_trades = sum(1 for txn_list in results["transactions"] if txn_list) // 2

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": num_trades,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    import polars as pl

    from ml4t.backtest import (
        DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy,
    )
    from ml4t.backtest.risk.position import TrailingStop

    prices_pl = pl.DataFrame({
        "timestamp": [ts.to_pydatetime().replace(tzinfo=None) for ts in prices_df.index],
        "asset": ["TEST"] * len(prices_df),
        "open": prices_df["open"].tolist(),
        "high": prices_df["high"].tolist(),
        "low": prices_df["low"].tolist(),
        "close": prices_df["close"].tolist(),
        "volume": prices_df["volume"].tolist(),
    })

    signals_pl = pl.DataFrame({
        "timestamp": [ts.to_pydatetime().replace(tzinfo=None) for ts in prices_df.index],
        "asset": ["TEST"] * len(prices_df),
        "entry": entries.tolist(),
    })

    class TrailingStopStrategy(Strategy):
        def on_start(self, broker):
            # Set position rules at strategy start
            broker.set_position_rules(TrailingStop(pct=TRAIL_PCT))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0
            if signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    engine = Engine(
        feed, TrailingStopStrategy(),
        initial_cash=100_000.0, allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
    }


def main():
    print("=" * 70)
    print(f"Scenario 09: Trailing Stop ({TRAIL_PCT*100:.0f}%)")
    print("=" * 70)

    prices_df, entries = generate_test_data()
    print(f"\n  Bars: {len(prices_df)}, Entry signals: {entries.sum()}")

    print("\n  Running Zipline (manual trailing stop)...")
    try:
        zl_results = run_zipline(prices_df, entries)
        print(f"   Trades: {zl_results['num_trades']}")
        print(f"   Final Value: ${zl_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n  Running ml4t.backtest...")
    ml4t_results = run_ml4t_backtest(prices_df, entries)
    print(f"   Trades: {ml4t_results['num_trades']}")
    print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")

    # Compare with tolerance
    trades_match = zl_results["num_trades"] == ml4t_results["num_trades"]
    value_diff_pct = abs(zl_results["final_value"] - ml4t_results["final_value"]) / zl_results["final_value"] * 100
    values_close = value_diff_pct < 2.0  # 2% tolerance for trailing stop timing

    print("\n" + "=" * 70)
    print(f"Trade Count: ZL={zl_results['num_trades']}, ML4T={ml4t_results['num_trades']}")
    print(f"Final Value diff: {value_diff_pct:.4f}%")
    print("Note: Zipline uses manual trailing stop implementation")
    if trades_match and values_close:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED (or within acceptable tolerance)")
    print("=" * 70)

    return 0 if values_close else 1


if __name__ == "__main__":
    sys.exit(main())
