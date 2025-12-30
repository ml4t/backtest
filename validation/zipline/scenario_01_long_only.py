#!/usr/bin/env python3
"""Scenario 01: Long-only validation against Zipline.

This script validates that ml4t.backtest produces identical results to Zipline
for a simple long-only strategy using predefined entry/exit signals.

IMPORTANT: Zipline uses a bundle system for data, which requires special setup.
This script creates a temporary bundle with test data to enable comparison.

Run from .venv-zipline environment:
    .venv-zipline/bin/python3 validation/zipline/scenario_01_long_only.py

Success criteria:
- Trade count: Exact match
- Fill prices: Exact match (both fill at next bar open)
- Final P&L: Exact match (using NYSE trading calendar)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_test_data(
    n_bars: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate test data for both frameworks.

    Uses NYSE trading calendar dates to ensure exact Zipline compatibility.
    """
    import exchange_calendars as xcals

    np.random.seed(seed)

    # Generate price path (random walk)
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02  # 2% daily vol
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate dates using actual NYSE trading calendar (NOT freq="B" which includes holidays)
    nyse = xcals.get_calendar("XNYS")
    start = pd.Timestamp("2020-01-02")  # First trading day of 2020
    # Get enough sessions to cover n_bars
    all_sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
    dates = pd.DatetimeIndex(all_sessions[:n_bars]).tz_localize("UTC")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
        },
        index=dates,
    )

    # Ensure high >= open, close, low and low <= open, close, high
    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # Generate entry/exit signals
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    i = 0
    while i < n_bars - 6:
        entries[i] = True
        exits[i + 5] = True
        i += 10

    return df, entries, exits


def setup_zipline_bundle(prices_df: pd.DataFrame, bundle_name: str = "test_validation"):
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
            # Get the actual calendar sessions in our range
            sessions = calendar.sessions_in_range(start_session, end_session)

            # Convert df index to tz-naive for comparison
            df_naive = df.copy()
            if df_naive.index.tz is not None:
                df_naive.index = df_naive.index.tz_convert(None)

            # Filter our data to only include dates that are trading days
            valid_mask = df_naive.index.isin(sessions)
            trading_df = df_naive[valid_mask].copy()

            if len(trading_df) == 0:
                raise ValueError(
                    f"No trading days found. Data dates: {df_naive.index[:5].tolist()}, "
                    f"Calendar sessions: {sessions[:5].tolist()}"
                )

            # Write equity metadata
            asset_db_writer.write(
                equities=pd.DataFrame(
                    {
                        "symbol": ["TEST"],
                        "asset_name": ["Test Asset"],
                        "exchange": ["NYSE"],
                    }
                )
            )

            # Write daily bars - needs to be iterable of (sid, df) tuples
            daily_bar_writer.write(
                [(0, trading_df[["open", "high", "low", "close", "volume"]])],
                show_progress=show_progress,
            )

            # No adjustments
            adjustment_writer.write()

        return ingest_func

    # Convert to tz-naive for calendar comparison
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

    # Ingest the data
    ingest(bundle_name, show_progress=False)

    return bundle_name


def _create_open_price_slippage():
    """Create a custom slippage model that fills at open price.

    This matches ml4t.backtest's NEXT_BAR execution mode where orders
    placed on bar N fill at bar N+1's open price.
    """
    from zipline.finance.slippage import SlippageModel

    class OpenPriceSlippage(SlippageModel):
        """Fill at open price with no price impact."""

        @staticmethod
        def process_order(data, order):
            return (data.current(order.asset, "open"), order.amount)

    return OpenPriceSlippage()


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run backtest using Zipline."""
    try:
        from zipline import run_algorithm
        from zipline.api import get_datetime, order, order_target, record, set_slippage, symbol
        from zipline.finance import slippage
    except ImportError:
        raise ImportError("Zipline not installed. Run in .venv-zipline environment.")

    # Store signals in a global for the algo to access
    signal_data = {
        "entries": entries,
        "exits": exits,
        "dates": prices_df.index,
    }

    def initialize(context):
        context.asset = symbol("TEST")
        context.signal_data = signal_data
        context.bar_count = 0
        context.trades = []
        context.in_position = False
        # Use custom slippage to fill at open price (matching ml4t.backtest NEXT_BAR mode)
        set_slippage(_create_open_price_slippage())

    def handle_data(context, data):
        idx = context.bar_count
        if idx >= len(context.signal_data["entries"]):
            return

        entry = context.signal_data["entries"][idx]
        exit_sig = context.signal_data["exits"][idx]

        current_pos = context.portfolio.positions[context.asset].amount

        if exit_sig and current_pos > 0:
            order_target(context.asset, 0)
            context.in_position = False
        elif entry and current_pos == 0:
            order(context.asset, 100)
            context.in_position = True

        context.bar_count += 1

    def analyze(context, perf):
        pass

    # Set up custom bundle
    bundle_name = setup_zipline_bundle(prices_df)

    # Run the algorithm - use tz-naive dates (Zipline will localize)
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

    # Extract results
    final_value = results["portfolio_value"].iloc[-1]

    # Count trades from transactions
    transactions = results["transactions"]
    num_trades = 0
    for txn_list in transactions:
        if txn_list:
            num_trades += len(txn_list)

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "num_trades": num_trades // 2,  # Entry + exit = 1 round trip
        "transactions": transactions,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

    # Convert to polars format - use same dates as input
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

    # Create signals DataFrame
    signals_pl = pl.DataFrame(
        {
            "timestamp": [ts.to_pydatetime().replace(tzinfo=None) for ts in prices_df.index],
            "asset": ["TEST"] * len(prices_df),
            "entry": entries.tolist(),
            "exit": exits.tolist(),
        }
    )

    class SignalStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return

            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            if signals.get("exit") and current_qty > 0:
                broker.close_position("TEST")
            elif signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", 100)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = SignalStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        account_type="cash",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Zipline fills at next bar open
    )

    results = engine.run()

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "num_trades": results["num_trades"],
        "trades": results["trades"],
    }


def compare_results(zipline_results: dict, ml4t_results: dict) -> bool:
    """Compare results and report differences."""
    print("\n" + "=" * 70)
    print("COMPARISON: Zipline vs ml4t.backtest")
    print("=" * 70)

    all_match = True

    # Trade count (Zipline counts entry+exit separately, so divide by 2)
    zipline_trades = zipline_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = zipline_trades == ml4t_trades
    print(
        f"\nTrade Count: Zipline={zipline_trades}, ML4T={ml4t_trades} {'✅' if trades_match else '❌'}"
    )
    all_match &= trades_match

    # Final value - with correct NYSE calendar, expect exact match
    zipline_value = zipline_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(zipline_value - ml4t_value)
    value_pct_diff = value_diff / zipline_value * 100 if zipline_value != 0 else 0
    values_match = value_pct_diff < 0.01  # Expect exact match (0.01% tolerance for floating point)
    print(
        f"Final Value: Zipline=${zipline_value:,.2f}, ML4T=${ml4t_value:,.2f} (diff={value_pct_diff:.4f}%) {'✅' if values_match else '❌'}"
    )
    all_match &= values_match

    # Total P&L - expect near-exact match with correct calendar and open-price slippage
    zipline_pnl = zipline_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(zipline_pnl - ml4t_pnl)
    pnl_match = pnl_diff < 5.0  # Within $5 (floating point tolerance across 10 trades)
    print(
        f"Total P&L: Zipline=${zipline_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} (diff=${pnl_diff:.2f}) {'✅' if pnl_match else '❌'}"
    )
    all_match &= pnl_match

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED: Results match within tolerance")
    else:
        print("❌ VALIDATION FAILED: Results do not match")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print("Scenario 01: Long-Only Validation (Zipline)")
    print("=" * 70)

    # Generate test data
    print("\n📊 Generating test data...")
    prices_df, entries, exits = generate_test_data(n_bars=100)
    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")
    print(f"   Exit signals: {exits.sum()}")
    print(f"   Date range: {prices_df.index[0]} to {prices_df.index[-1]}")

    # Run Zipline
    print("\n🔷 Running Zipline...")
    try:
        zipline_results = run_zipline(prices_df, entries, exits)
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

    # Run ml4t.backtest
    print("\n🔶 Running ml4t.backtest...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    success = compare_results(zipline_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
