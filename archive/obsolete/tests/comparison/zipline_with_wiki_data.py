"""
Zipline Implementation with Wiki Prices Data

This creates a custom Zipline bundle using the Wiki Prices data from projects/daily_us_equities/
to enable proper comparison with ml4t.backtest and Backtrader.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

# Project paths
backtest_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(backtest_src))

# Try importing Zipline
try:
    from exchange_calendars import get_calendar
    from zipline import run_algorithm
    from zipline.api import (
        get_datetime,
        order_target_percent,
        record,
        set_commission,
        set_slippage,
        symbol,
    )
    from zipline.data.bundles import ingest, register
    from zipline.data.bundles.core import load
    from zipline.finance import commission, slippage

    ZIPLINE_AVAILABLE = True
except ImportError as e:
    print(f"Zipline not available: {e}")
    ZIPLINE_AVAILABLE = False


def create_wiki_bundle():
    """Create a custom Zipline bundle from Wiki Prices data."""
    if not ZIPLINE_AVAILABLE:
        return None

    wiki_path = projects_dir / "daily_us_equities" / "wiki_prices.parquet"
    if not wiki_path.exists():
        print(f"Wiki prices not found at {wiki_path}")
        return None

    # Load Wiki prices data
    df = pd.read_parquet(wiki_path)
    print(f"Loaded Wiki data: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Select a subset for testing (SPY and a few others)
    test_tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
    df_subset = df[df["ticker"].isin(test_tickers)].copy()
    print(f"Using {len(df_subset)} rows for tickers: {test_tickers}")

    def wiki_bundle_ingest(
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
        """Ingest function for Wiki bundle."""

        # Prepare data for Zipline format
        data = df_subset.copy()
        data["date"] = pd.to_datetime(data["date"]).dt.tz_localize("UTC")
        data = data.set_index(["ticker", "date"]).sort_index()

        # Create assets metadata
        assets = []
        for ticker in test_tickers:
            ticker_data = df_subset[df_subset["ticker"] == ticker]
            if len(ticker_data) > 0:
                start_date = pd.to_datetime(ticker_data["date"].min()).tz_localize("UTC")
                end_date = pd.to_datetime(ticker_data["date"].max()).tz_localize("UTC")
                assets.append(
                    {
                        "sid": len(assets),
                        "symbol": ticker,
                        "asset_name": ticker,
                        "start_date": start_date,
                        "end_date": end_date,
                        "exchange": "NYSE",
                        "auto_close_date": end_date + pd.Timedelta(days=1),
                    },
                )

        # Write assets metadata
        asset_db_writer.write(assets)

        # Write daily bar data
        def get_data_for_sid(sid):
            ticker = assets[sid]["symbol"]
            ticker_data = df_subset[df_subset["ticker"] == ticker].copy()
            ticker_data["date"] = pd.to_datetime(ticker_data["date"]).dt.tz_localize("UTC")
            ticker_data = ticker_data.set_index("date")

            # Rename columns to match Zipline expectations
            ticker_data = ticker_data.rename(
                columns={
                    "adj_close": "close",  # Use adjusted close as close
                    "adj_volume": "volume",
                },
            )

            # Ensure all required columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in ticker_data.columns:
                    if col == "volume":
                        ticker_data[col] = ticker_data.get("volume", 1000000)
                    else:
                        ticker_data[col] = ticker_data.get(
                            "close",
                            ticker_data.get("adj_close", 100),
                        )

            return ticker_data[required_cols].dropna()

        # Write daily data for each asset
        for asset in assets:
            sid = asset["sid"]
            ticker_data = get_data_for_sid(sid)
            if len(ticker_data) > 0:
                daily_bar_writer.write(
                    sid,
                    ticker_data.itertuples(),
                )

        # Write empty adjustments (no splits/dividends for simplicity)
        adjustment_writer.write()

    # Register the bundle
    register(
        "wiki",
        wiki_bundle_ingest,
        calendar_name="NYSE",
        start_session=pd.Timestamp("2010-01-01", tz="UTC"),
        end_session=pd.Timestamp("2018-12-31", tz="UTC"),
    )

    return "wiki"


def run_zipline_momentum_strategy():
    """Run the momentum strategy using Zipline with Wiki data."""
    if not ZIPLINE_AVAILABLE:
        return {
            "framework": "Zipline",
            "error": "Zipline not available",
            "final_value": 10000,
            "total_return": 0,
            "trades": [],
        }

    # Create and ingest bundle
    bundle_name = create_wiki_bundle()
    if not bundle_name:
        return {
            "framework": "Zipline",
            "error": "Failed to create bundle",
            "final_value": 10000,
            "total_return": 0,
            "trades": [],
        }

    try:
        # Ingest the bundle
        print("Ingesting Wiki bundle...")
        ingest(bundle_name, os.environ, pd.Timestamp.utcnow())
        print("Bundle ingestion complete")

    except Exception as e:
        print(f"Bundle ingestion failed: {e}")
        return {
            "framework": "Zipline",
            "error": f"Bundle ingestion failed: {e}",
            "final_value": 10000,
            "total_return": 0,
            "trades": [],
        }

    # Define the strategy
    def initialize(context):
        """Initialize strategy."""
        context.asset = symbol("SPY")
        context.short_window = 20
        context.long_window = 50
        context.position = 0
        context.trades = []

        # Set commission and slippage to zero for comparison
        set_commission(commission.PerShare(cost=0))
        set_slippage(slippage.FixedSlippage(spread=0))

    def handle_data(context, data):
        """Handle each data point."""
        # Get historical data
        try:
            history = data.history(context.asset, "price", context.long_window + 1, "1d")
        except:
            return

        if len(history) < context.long_window:
            return

        # Calculate MAs
        short_ma = history[-context.short_window :].mean()
        long_ma = history.mean()

        # Get current price
        try:
            current_price = data.current(context.asset, "price")
        except:
            return

        # Generate signals
        if short_ma > long_ma and context.position <= 0:
            # Buy signal
            order_target_percent(context.asset, 1.0)
            context.position = 1
            context.trades.append(
                {
                    "timestamp": get_datetime(),
                    "action": "BUY",
                    "price": current_price,
                },
            )

        elif short_ma < long_ma and context.position >= 0:
            # Sell signal
            order_target_percent(context.asset, 0.0)
            context.position = 0
            context.trades.append(
                {
                    "timestamp": get_datetime(),
                    "action": "SELL",
                    "price": current_price,
                },
            )

        record(price=current_price, short_ma=short_ma, long_ma=long_ma, position=context.position)

    # Run the algorithm
    try:
        print("Running Zipline algorithm...")
        result = run_algorithm(
            start=pd.Timestamp("2015-01-01", tz="UTC"),
            end=pd.Timestamp("2016-01-01", tz="UTC"),
            initialize=initialize,
            handle_data=handle_data,
            capital_base=10000,
            data_frequency="daily",
            bundle="wiki",
        )

        # Extract results
        final_value = result.portfolio_value.iloc[-1]
        total_return = (final_value / 10000 - 1) * 100

        # Extract trades from recorded data
        trades = []
        prev_position = 0
        for idx, row in result.iterrows():
            current_position = row.get("position", 0)
            if current_position != prev_position:
                action = "BUY" if current_position > prev_position else "SELL"
                trades.append(
                    {
                        "timestamp": idx,
                        "action": action,
                        "price": row.get("price", 0),
                    },
                )
                prev_position = current_position

        print("✓ Zipline backtest completed")
        print(f"  Final value: ${final_value:,.2f}")
        print(f"  Return: {total_return:.2f}%")
        print(f"  Trades: {len(trades)}")

        return {
            "framework": "Zipline",
            "final_value": final_value,
            "total_return": total_return,
            "trades": trades,
            "result": result,
        }

    except Exception as e:
        print(f"✗ Zipline algorithm failed: {e}")
        return {
            "framework": "Zipline",
            "error": str(e),
            "final_value": 10000,
            "total_return": 0,
            "trades": [],
        }


def test_zipline_setup():
    """Test the Zipline setup with Wiki data."""
    print("\n" + "=" * 60)
    print("TESTING ZIPLINE WITH WIKI DATA")
    print("=" * 60)

    result = run_zipline_momentum_strategy()

    print("\nZipline Test Results:")
    print(f"  Framework: {result['framework']}")
    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Final Value: ${result['final_value']:,.2f}")
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"  Number of Trades: {len(result['trades'])}")

        if result["trades"]:
            print("\n  First few trades:")
            for i, trade in enumerate(result["trades"][:5]):
                print(
                    f"    {i + 1}. {trade['timestamp']} {trade['action']} @ ${trade['price']:.2f}"
                )

    return result


if __name__ == "__main__":
    test_zipline_setup()
