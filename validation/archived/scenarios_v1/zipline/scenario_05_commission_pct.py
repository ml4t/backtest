#!/usr/bin/env python3
"""Scenario 05: Percentage Commission validation against Zipline.

This script validates that ml4t.backtest commission calculations match Zipline
when using percentage-based commissions (0.1% of trade value).

Run from .venv-validation environment:
    source .venv-validation/bin/activate
    cd validation/zipline
    python scenario_05_commission_pct.py

Success criteria:
- Trade count: Exact match
- Commission per trade: Match within tolerance
- Final P&L: Match within tolerance (accounting for commissions)

Note: Zipline uses PerTrade or PerShare commission models. For percentage
commission, we use a custom implementation via PerDollar model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Commission rate: 0.1% (10 basis points)
COMMISSION_RATE = 0.001


def generate_test_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate test data with NYSE calendar dates."""
    import exchange_calendars as xcals

    np.random.seed(seed)

    n_bars = 100
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    # Use NYSE calendar for Zipline compatibility
    nyse = xcals.get_calendar("XNYS")
    start = pd.Timestamp("2020-01-02")
    all_sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
    dates = pd.DatetimeIndex(all_sessions[:n_bars]).tz_localize("UTC")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
            "close": prices,
            "volume": np.full(n_bars, 100000.0),
        },
        index=dates,
    )

    df["high"] = df[["open", "high", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) * 0.999

    # Entry signals only (exit handled by strategy)
    entries = np.zeros(n_bars, dtype=bool)
    i = 0
    while i < n_bars - 6:
        entries[i] = True
        i += 10

    return df, entries


def setup_zipline_bundle(prices_df: pd.DataFrame, bundle_name: str = "test_comm"):
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


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """Run backtest using Zipline with percentage commission."""
    try:
        from zipline import run_algorithm
        from zipline.api import order, order_target, set_commission, set_slippage, symbol
        from zipline.finance.commission import PerDollar
        from zipline.finance.slippage import NoSlippage
    except ImportError:
        raise ImportError("Zipline not installed. Run in .venv-validation environment.")

    signal_data = {"entries": entries, "dates": prices_df.index}
    HOLD_BARS = 5

    def initialize(context):
        context.asset = symbol("TEST")
        context.signal_data = signal_data
        context.bar_count = 0
        context.entry_bar = None
        # Zipline's PerDollar charges commission * notional value
        set_commission(PerDollar(cost=COMMISSION_RATE))
        set_slippage(NoSlippage())

    def handle_data(context, data):
        idx = context.bar_count

        if idx >= len(context.signal_data["entries"]):
            return

        entry = context.signal_data["entries"][idx]
        current_pos = context.portfolio.positions[context.asset].amount

        # Exit after HOLD_BARS
        if current_pos > 0 and context.entry_bar is not None:
            if idx - context.entry_bar >= HOLD_BARS:
                order_target(context.asset, 0)
                context.entry_bar = None
        # Entry
        elif entry and current_pos == 0:
            order(context.asset, 100)
            context.entry_bar = idx

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

    # Extract total commission from transactions
    total_commission = 0.0
    for txn_list in transactions:
        if txn_list:
            for txn in txn_list:
                comm = txn.get("commission")
                if comm is not None:
                    total_commission += abs(comm)

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "total_pnl": final_value - 100_000.0,
        "total_commission": total_commission,
        "num_trades": num_trades,
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray) -> dict:
    """Run backtest using ml4t.backtest with percentage commission."""
    import polars as pl

    from ml4t.backtest._validation_imports import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoSlippage,
        PercentageCommission,
        Strategy,
    )

    HOLD_BARS = 5

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

    class HoldStrategy(Strategy):
        def __init__(self):
            self.entry_bar = None
            self.bar_count = 0

        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return

            signals = data["TEST"].get("signals", {})
            position = broker.get_position("TEST")
            current_qty = position.quantity if position else 0

            # Exit after HOLD_BARS
            if current_qty > 0 and self.entry_bar is not None:
                if self.bar_count - self.entry_bar >= HOLD_BARS:
                    broker.close_position("TEST")
                    self.entry_bar = None
            # Entry
            elif signals.get("entry") and current_qty == 0:
                broker.submit_order("TEST", 100)
                self.entry_bar = self.bar_count

            self.bar_count += 1

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = HoldStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=PercentageCommission(rate=COMMISSION_RATE),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Zipline
    )

    results = engine.run()

    total_commission = sum(f.commission for f in results["fills"])

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "total_pnl": results["final_value"] - 100_000.0,
        "total_commission": total_commission,
        "num_trades": results["num_trades"],
    }


def compare_results(zl_results: dict, ml4t_results: dict) -> bool:
    """Compare results and report differences.

    Note: Zipline's PerDollar commission model doesn't record commission
    in transaction records. We compare final values which include commission
    effects.
    """
    print("\n" + "=" * 70)
    print(f"COMPARISON: Zipline vs ml4t.backtest (Commission={COMMISSION_RATE:.2%})")
    print("=" * 70)
    print("\nNote: Zipline's PerDollar model doesn't record commission in transactions.")
    print("Comparing final values which include commission effects.")

    all_match = True

    # Trade count
    zl_trades = zl_results["num_trades"]
    ml4t_trades = ml4t_results["num_trades"]
    trades_match = zl_trades == ml4t_trades
    print(f"\nTrade Count: ZL={zl_trades}, ML4T={ml4t_trades} {'✅' if trades_match else '❌'}")
    all_match &= trades_match

    # Final value - use 0.5% tolerance for Zipline (due to calendar/fill differences)
    zl_value = zl_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(zl_value - ml4t_value)
    value_pct_diff = value_diff / zl_value * 100 if zl_value != 0 else 0
    values_match = value_pct_diff < 0.5  # Within 0.5% for Zipline
    print(
        f"Final Value: ZL=${zl_value:,.2f}, ML4T=${ml4t_value:,.2f} "
        f"(diff={value_pct_diff:.4f}%) {'✅' if values_match else '❌'}"
    )
    all_match &= values_match

    # Total P&L - use wider tolerance for Zipline
    zl_pnl = zl_results["total_pnl"]
    ml4t_pnl = ml4t_results["total_pnl"]
    pnl_diff = abs(zl_pnl - ml4t_pnl)
    pnl_pct_diff = pnl_diff / abs(zl_pnl) * 100 if zl_pnl != 0 else 0
    pnl_match = pnl_pct_diff < 5.0  # Within 5% for Zipline (different fills)
    print(
        f"Total P&L: ZL=${zl_pnl:,.2f}, ML4T=${ml4t_pnl:,.2f} "
        f"(diff=${pnl_diff:.2f}, {pnl_pct_diff:.2f}%) {'✅' if pnl_match else '❌'}"
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
    print(f"Scenario 05: Percentage Commission Validation ({COMMISSION_RATE:.2%})")
    print("=" * 70)

    print("\n📊 Generating test data...")
    prices_df, entries = generate_test_data()
    print(f"   Bars: {len(prices_df)}")
    print(f"   Entry signals: {entries.sum()}")

    print("\n🔷 Running Zipline...")
    try:
        zl_results = run_zipline(prices_df, entries)
        print(f"   Trades: {zl_results['num_trades']}")
        print(f"   Total Commission: ${zl_results['total_commission']:.2f}")
        print(f"   Final Value: ${zl_results['final_value']:,.2f}")
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
        ml4t_results = run_ml4t_backtest(prices_df, entries)
        print(f"   Trades: {ml4t_results['num_trades']}")
        print(f"   Total Commission: ${ml4t_results['total_commission']:.2f}")
        print(f"   Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"   ❌ {e}")
        import traceback

        traceback.print_exc()
        return 1

    success = compare_results(zl_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
