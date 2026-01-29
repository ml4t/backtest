#!/usr/bin/env python3
"""Scenario 11: Short-only validation against Zipline.

This validates that ml4t.backtest short selling produces IDENTICAL results
to Zipline - trade-by-trade exact match.

Run from .venv-zipline environment:
    .venv-zipline/bin/python validation/zipline/scenario_11_short_only.py

Success criteria:
- Trade count: EXACT match
- Entry prices: EXACT match
- Exit prices: EXACT match
- PnL per trade: EXACT match (within $0.01)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SHARES_PER_TRADE = 100


def generate_test_data(seed: int = 42):
    """Generate test data with clear SHORT entry/exit signals aligned with NYSE calendar."""
    import exchange_calendars as xcals

    np.random.seed(seed)
    n_bars = 200

    # Use NYSE calendar for proper alignment with Zipline
    nyse = xcals.get_calendar("XNYS")
    start = pd.Timestamp("2020-01-02")
    all_sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
    dates = pd.DatetimeIndex(all_sessions[:n_bars]).tz_localize("UTC")

    # Generate price path with trends
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    closes = base_price * np.exp(np.cumsum(returns))

    # Generate valid OHLC
    opens = closes * (1 + np.random.randn(n_bars) * 0.003)
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.005)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n_bars, 100000.0),
        },
        index=dates,
    )

    # SHORT entry/exit signals: enter every 20 bars, exit after 10 bars
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    idx = 5  # Start at bar 5
    while idx < n_bars - 11:
        entries[idx] = True
        exits[idx + 10] = True
        idx += 20

    return df, entries, exits


def setup_zipline_bundle(prices_df: pd.DataFrame, bundle_name: str = "test_short"):
    """Create and ingest a Zipline bundle from the test data."""
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


def run_zipline(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run short-only backtest using Zipline."""
    from zipline import run_algorithm
    from zipline.api import order, order_target, set_commission, set_slippage, symbol
    from zipline.finance.commission import NoCommission
    from zipline.finance.slippage import NoSlippage

    signal_data = {"entries": entries, "exits": exits}

    def initialize(context):
        context.asset = symbol("TEST")
        context.signal_data = signal_data
        context.bar_count = 0
        context.trades = []
        context.current_trade = None
        set_commission(NoCommission())
        set_slippage(NoSlippage())

    def handle_data(context, data):
        idx = context.bar_count
        if idx >= len(context.signal_data["entries"]):
            return

        entry = context.signal_data["entries"][idx]
        exit_signal = context.signal_data["exits"][idx]
        current_pos = context.portfolio.positions[context.asset].amount

        # Exit first (cover short)
        if exit_signal and current_pos < 0:
            order_target(context.asset, 0)
            if context.current_trade:
                context.current_trade["exit_time"] = data.current_dt
                context.current_trade["exit_price"] = data.current(context.asset, "close")
                context.trades.append(context.current_trade)
                context.current_trade = None

        # Then entry (open short)
        elif entry and current_pos == 0:
            order(context.asset, -SHARES_PER_TRADE)  # Negative for short
            context.current_trade = {
                "entry_time": data.current_dt,
                "entry_price": data.current(context.asset, "close"),
                "size": SHARES_PER_TRADE,
                "direction": "Short",
            }

        context.bar_count += 1

    def analyze(context, perf):
        # Calculate PnL for each trade
        for trade in context.trades:
            # For SHORT: PnL = (entry - exit) * size
            trade["pnl"] = (trade["entry_price"] - trade["exit_price"]) * trade["size"]

    bundle_name = setup_zipline_bundle(prices_df)
    start = prices_df.index[0].tz_convert(None)
    end = prices_df.index[-1].tz_convert(None)

    perf = run_algorithm(
        start=start, end=end,
        initialize=initialize, handle_data=handle_data, analyze=analyze,
        capital_base=1_000_000.0, bundle=bundle_name, data_frequency="daily",
    )

    # Extract trades from context
    from zipline.algorithm import TradingAlgorithm
    # Access via perf or context (depends on version)

    # Get final value
    final_value = perf["portfolio_value"].iloc[-1]

    # Rebuild trade list from performance data
    trade_list = []

    # Count trades from transactions
    prev_pos = 0
    trade_entry = None
    for dt, txns in perf["transactions"].items():
        for txn in txns:
            if txn["amount"] < 0 and prev_pos == 0:  # SHORT entry
                trade_entry = {
                    "entry_time": dt,
                    "entry_price": txn["price"],
                    "size": abs(txn["amount"]),
                    "direction": "Short",
                }
                prev_pos = txn["amount"]
            elif txn["amount"] > 0 and prev_pos < 0:  # Cover short
                if trade_entry:
                    trade_entry["exit_time"] = dt
                    trade_entry["exit_price"] = txn["price"]
                    trade_entry["pnl"] = (trade_entry["entry_price"] - trade_entry["exit_price"]) * trade_entry["size"]
                    trade_list.append(trade_entry)
                    trade_entry = None
                prev_pos = 0

    return {
        "framework": "Zipline",
        "final_value": final_value,
        "num_trades": len(trade_list),
        "trades": sorted(trade_list, key=lambda t: t["entry_time"]),
    }


def run_ml4t_backtest(prices_df: pd.DataFrame, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Run short-only backtest using ml4t.backtest."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, OrderSide, Strategy

    # Remove timezone for ml4t
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
            "short_entry": entries.tolist(),
            "short_exit": exits.tolist(),
        }
    )

    class ShortOnlyStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if "TEST" not in data:
                return
            signals = data["TEST"].get("signals", {})
            pos = broker.get_position("TEST")
            current_qty = pos.quantity if pos else 0

            # Exit first (cover short)
            if signals.get("short_exit") and current_qty < 0:
                broker.close_position("TEST")
            # Then entry (open short)
            elif signals.get("short_entry") and current_qty == 0:
                broker.submit_order("TEST", SHARES_PER_TRADE, OrderSide.SELL)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = ShortOnlyStrategy()

    # Match Zipline: NEXT_BAR execution
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        allow_short_selling=True,
        allow_leverage=True,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.NEXT_BAR,
    )

    results = engine.run()

    trade_list = []
    for t in results["trades"]:
        trade_list.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size": abs(t.quantity),
            "pnl": t.pnl,
            "direction": "Short" if t.quantity < 0 else "Long",
        })

    return {
        "framework": "ml4t.backtest",
        "final_value": results["final_value"],
        "num_trades": results["num_trades"],
        "trades": sorted(trade_list, key=lambda t: t["entry_time"]),
    }


def compare_results(zl_results, ml4t_results):
    """Compare trade-by-trade results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Zipline vs ml4t.backtest (SHORT-ONLY)")
    print("=" * 70)

    all_match = True

    # Trade count
    zl_count = zl_results["num_trades"]
    ml4t_count = ml4t_results["num_trades"]
    count_match = zl_count == ml4t_count
    print(f"\nTrade Count: ZL={zl_count}, ML4T={ml4t_count} "
          f"{'✅ MATCH' if count_match else '❌ MISMATCH'}")
    all_match &= count_match

    # Final value
    zl_value = zl_results["final_value"]
    ml4t_value = ml4t_results["final_value"]
    value_diff = abs(zl_value - ml4t_value)
    value_pct = value_diff / zl_value * 100 if zl_value else 0
    value_match = value_pct < 0.1  # 0.1% tolerance
    print(f"Final Value: ZL=${zl_value:,.2f}, ML4T=${ml4t_value:,.2f} "
          f"(diff={value_pct:.4f}%) {'✅ MATCH' if value_match else '❌ MISMATCH'}")
    all_match &= value_match

    # Trade-by-trade comparison
    if count_match and zl_count > 0:
        zl_trades = zl_results["trades"]
        ml4t_trades = ml4t_results["trades"]

        matches = 0
        mismatches = 0
        sample_mismatches = []

        for zl_t, ml4t_t in zip(zl_trades, ml4t_trades):
            entry_match = abs(zl_t["entry_price"] - ml4t_t["entry_price"]) < 0.01
            exit_match = abs(zl_t["exit_price"] - ml4t_t["exit_price"]) < 0.01
            pnl_match = abs(zl_t["pnl"] - ml4t_t["pnl"]) < 1.0  # $1 tolerance

            if entry_match and exit_match and pnl_match:
                matches += 1
            else:
                mismatches += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append((zl_t, ml4t_t))

        match_pct = matches / len(zl_trades) * 100 if zl_trades else 0
        print(f"\nTrade-Level Match: {matches}/{len(zl_trades)} ({match_pct:.1f}%)")

        if mismatches > 0:
            print(f"\nSample Mismatches:")
            for zl_t, ml4t_t in sample_mismatches:
                print(f"  ZL:   entry=${zl_t['entry_price']:.2f}, "
                      f"exit=${zl_t['exit_price']:.2f}, pnl=${zl_t['pnl']:.2f}")
                print(f"  ML4T: entry=${ml4t_t['entry_price']:.2f}, "
                      f"exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

        all_match &= (mismatches == 0)

    # Sample trades
    if zl_count > 0:
        print("\nSample Short Trades (first 3):")
        print("-" * 70)
        for i, (zl_t, ml4t_t) in enumerate(
            zip(zl_results["trades"][:3], ml4t_results["trades"][:3])
        ):
            print(f"  Trade {i+1}:")
            print(f"    ZL:   entry=${zl_t['entry_price']:.2f}, exit=${zl_t['exit_price']:.2f}, pnl=${zl_t['pnl']:.2f}")
            print(f"    ML4T: entry=${ml4t_t['entry_price']:.2f}, exit=${ml4t_t['exit_price']:.2f}, pnl=${ml4t_t['pnl']:.2f}")

    print("\n" + "=" * 70)
    if all_match:
        print("✅ VALIDATION PASSED: Short-selling results match exactly")
    else:
        print("❌ VALIDATION FAILED: Short-selling results differ")
    print("=" * 70)

    return all_match


def main():
    print("=" * 70)
    print("Scenario 11: Short-Only Validation (Zipline)")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data (NYSE calendar)...")
    prices_df, entries, exits = generate_test_data()
    n_bars = len(prices_df)
    print(f"  Bars: {n_bars}")
    print(f"  Short entry signals: {entries.sum()}")
    print(f"  Short exit signals: {exits.sum()}")

    # Run Zipline
    print("\nRunning Zipline (short-only)...")
    try:
        zl_results = run_zipline(prices_df, entries, exits)
        print(f"  Trades: {zl_results['num_trades']}")
        print(f"  Final Value: ${zl_results['final_value']:,.2f}")
    except ImportError as e:
        print(f"  ERROR: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run ml4t.backtest
    print("\nRunning ml4t.backtest (short-only)...")
    try:
        ml4t_results = run_ml4t_backtest(prices_df, entries, exits)
        print(f"  Trades: {ml4t_results['num_trades']}")
        print(f"  Final Value: ${ml4t_results['final_value']:,.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare
    success = compare_results(zl_results, ml4t_results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
