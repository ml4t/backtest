#!/usr/bin/env python3
"""Scenario 03: Stop-Loss validation against Zipline.

This script validates that ml4t.backtest stop-loss behavior matches Zipline
when using a custom bundle with stop-loss logic implemented in the strategy.

Run from .venv-validation environment:
    .venv-validation/bin/python3 validation/zipline/scenario_03_stop_loss.py

Zipline doesn't have built-in sl_stop like VectorBT. Stop-loss is implemented
in handle_data() by checking price against entry price.

Success criteria:
- Trade count: Exact match
- Exit trigger timing: Same bar
- Exit price: Within tolerance (Zipline fills at open on next bar)
- Final P&L: Within tolerance
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_stop_loss_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate test data where price drops to trigger stop-loss.

    Uses NYSE trading calendar for Zipline compatibility.
    """
    import exchange_calendars as xcals

    np.random.seed(seed)

    # Price path: 100 -> gradual decline to trigger 5% stop
    n_bars = 20
    prices = np.array(
        [
            100.0,  # Bar 0: Entry
            99.0,  # Bar 1: -1%
            98.0,  # Bar 2: -2%
            97.0,  # Bar 3: -3%
            96.0,  # Bar 4: -4%
            94.5,  # Bar 5: -5.5% -> STOP TRIGGERED
            93.0,  # Bar 6: -7%
            92.0,  # Bar 7: -8%
            91.0,  # Bar 8: -9%
            90.0,  # Bar 9: -10%
            89.0,  # Bar 10
            88.0,  # Bar 11
            87.0,  # Bar 12
            86.0,  # Bar 13
            85.0,  # Bar 14
            84.0,  # Bar 15
            83.0,  # Bar 16
            82.0,  # Bar 17
            81.0,  # Bar 18
            80.0,  # Bar 19
        ]
    )

    # Use NYSE calendar
    nyse = xcals.get_calendar("XNYS")
    start = pd.Timestamp("2020-01-02")
    all_sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
    dates = pd.DatetimeIndex(all_sessions[:n_bars]).tz_localize("UTC")

    # Generate OHLCV with open = close for simplicity
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.full(n_bars, 100000.0),
        },
        index=dates,
    )

    # Entry on bar 0 only
    entries = np.zeros(n_bars, dtype=bool)
    entries[0] = True

    return df, entries


def setup_zipline_bundle(prices_df: pd.DataFrame, bundle_name: str = "test_sl"):
    """Register and ingest a custom bundle with test data."""
    from zipline.data.bundles import ingest, register

    def make_ingest_func(df):
        def ingest_func(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            sessions = calendar.sessions_in_range(start_session, end_session)

            df_naive = df.copy()
            if df_naive.index.tz is not None:
                df_naive.index = df_naive.index.tz_convert(None)

            valid_mask = df_naive.index.isin(sessions)
            trading_df = df_naive[valid_mask].copy()

            if len(trading_df) == 0:
                raise ValueError("No trading days found")

            asset_db_writer.write(
                equities=pd.DataFrame(
                    {
                        "symbol": ["TEST"],
                        "asset_name": ["Test Asset"],
                        "exchange": ["NYSE"],
                    }
                )
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


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray, sl_pct: float) -> dict:
    """Run backtest using Zipline with stop-loss logic in strategy."""
    try:
        from zipline import run_algorithm
        from zipline.api import order, order_target, set_commission, set_slippage, symbol
        from zipline.finance.commission import NoCommission
        from zipline.finance.slippage import SlippageModel
    except ImportError:
        raise ImportError("Zipline not installed. Run in .venv-validation environment.")

    signal_data = {
        "entries": entries,
        "dates": prices_df.index,
        "sl_pct": sl_pct,
    }

    # Custom slippage to fill at open price (matching ml4t.backtest NEXT_BAR)
    class OpenPriceSlippage(SlippageModel):
        @staticmethod
        def process_order(data, order):
            return (data.current(order.asset, "open"), order.amount)

    def initialize(context):
        context.asset = symbol("TEST")
        context.signal_data = signal_data
        context.bar_count = 0
        context.entry_price = None
        context.sl_pct = sl_pct
        set_slippage(OpenPriceSlippage())
        set_commission(NoCommission())

    def handle_data(context, data):
        idx = context.bar_count
        if idx >= len(context.signal_data["entries"]):
            return

        entry = context.signal_data["entries"][idx]
        current_pos = context.portfolio.positions[context.asset].amount
        current_price = data.current(context.asset, "close")

        # Check stop-loss
        if current_pos > 0 and context.entry_price is not None:
            loss_pct = (current_price - context.entry_price) / context.entry_price
            if loss_pct <= -context.sl_pct:
                order_target(context.asset, 0)
                context.entry_price = None
        # Check entry
        elif entry and current_pos == 0:
            order(context.asset, 100)
            # Entry price is the NEXT bar's open (when order fills)
            # We'll capture it when we check position next bar
            context.entry_price = current_price  # Use signal price for stop calc

        context.bar_count += 1

    def analyze(context, perf):
        pass

    bundle_name = setup_zipline_bundle(prices_df)

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
        capital_base=100_000.0,
        bundle=bundle_name,
        data_frequency="daily",
    )

    final_value = results["portfolio_value"].iloc[-1]
    transactions = results["transactions"]
    num_trades = sum(1 for txn_list in transactions if txn_list) // 2

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": num_trades,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, sl_pct: float) -> dict:
    """Run backtest using ml4t.backtest with stop-loss."""
    import polars as pl

    from ml4t.backtest import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        StopFillMode,
        StopLevelBasis,
        Strategy,
    )
    from ml4t.backtest.risk import StopLoss

    prices_pl = pl.DataFrame(
        {
            "timestamp": [ts.to_pydatetime().replace(tzinfo=None) for ts in prices_df.index],
            "asset": ["TEST"] * len(prices_df),
            "open": prices_df["open"].tolist(),
            "high": prices_df["high"].tolist(),
            "low": prices_df["low"].tolist(),
            "close": prices_df["close"].tolist(),
            "volume": prices_df["volume"].tolist(),
        }
    )

    signals_pl = pl.DataFrame(
        {
            "timestamp": [ts.to_pydatetime().replace(tzinfo=None) for ts in prices_df.index],
            "asset": ["TEST"] * len(prices_df),
            "entry": entries.tolist(),
        }
    )

    class StopLossStrategy(Strategy):
        def __init__(self, sl_pct: float):
            self.sl_pct = sl_pct

        def on_start(self, broker):
            broker.set_position_rules(StopLoss(pct=self.sl_pct))

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return

            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            if signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", 100)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = StopLossStrategy(sl_pct=sl_pct)

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Zipline
        stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,  # Match Zipline: exit at next bar's open
        stop_level_basis=StopLevelBasis.SIGNAL_PRICE,  # Zipline uses signal close
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": results["trades"],
    }


def compare_results(zipline_results: dict, ml4t_results: dict, sl_pct: float) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: Zipline vs ml4t.backtest (Stop-Loss={sl_pct:.0%})")
    print("=" * 70)

    all_match = True

    # Trade count
    zl_trades = zipline_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = zl_trades == ml4t_trades
    print(
        f"\nTrade Count: ZL={zl_trades}, ML4T={ml4t_trades} {'EXACT MATCH' if trades_match else 'FAIL'}"
    )
    all_match &= trades_match

    # Final value - require EXACT MATCH (using NEXT_BAR_OPEN mode)
    zl_value = zipline_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(zl_value - ml4t_value)
    values_match = value_diff < 0.01  # Within 1 cent (floating point tolerance)
    print(
        f"Final Value: ZL=${zl_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff=${value_diff:.2f}) {'EXACT MATCH' if values_match else 'FAIL'}"
    )
    all_match &= values_match

    # Total P&L - require EXACT MATCH
    zl_pnl = zipline_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(zl_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 0.01  # Within 1 cent (floating point tolerance)
    print(
        f"Total P&L: ZL=${zl_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f}) {'EXACT MATCH' if pnl_match else 'FAIL'}"
    )
    all_match &= pnl_match

    print("\n" + "=" * 70)
    print("Mode: StopFillMode.NEXT_BAR_OPEN (Zipline-compatible)")
    print("Both frameworks now fill exits at next bar's open price")
    print("=" * 70)

    if all_match:
        print("\n✅ VALIDATION PASSED: EXACT MATCH")
    else:
        print("\n❌ VALIDATION FAILED: Results do not match")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print("Scenario 03: Stop-Loss Validation (Zipline)")
    print("=" * 70)

    sl_pct = 0.05

    print(f"\n📊 Generating test data for {sl_pct:.0%} stop-loss...")
    prices_df, entries = generate_stop_loss_data()
    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    print(f"   Price at entry: ${prices_df['close'].iloc[0]:.2f}")
    print(f"   Stop level: ${prices_df['close'].iloc[0] * (1 - sl_pct):.2f}")

    print("\n🔷 Running Zipline...")
    try:
        zipline_results = run_zipline(prices_df, entries, sl_pct)
        print(f"   Trades: {zipline_results['num_trades']}")
        print(f"   Final Value: ${zipline_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"   ❌ {e}")
        return 1
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n🔶 Running ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, sl_pct)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
        if ml4t_results["trades"]:
            print(f"   Exit price: ${ml4t_results['trades'][0].exit_price:.2f}")
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    success = compare_results(zipline_results, ml4t_results, sl_pct)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
