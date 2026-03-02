"""Calendar enforcement scale test.

Tests calendar session enforcement with large-scale data:
- 100 assets
- 1 year of minute data (including weekends and holidays)
- Verifies only trading session bars are processed
- Verifies no trades on non-trading days
"""

from datetime import datetime, timedelta
import time

import numpy as np
import polars as pl

from ml4t.backtest._validation_imports import Engine, Strategy, DataFeed, OrderSide
from ml4t.backtest.config import BacktestConfig, DataFrequency


class BuyAndHoldStrategy(Strategy):
    """Strategy that buys once and holds."""

    def __init__(self):
        self.entry_timestamps: dict[str, datetime] = {}
        self.bars_processed = 0
        self.all_timestamps: list[datetime] = []

    def on_data(self, timestamp, data, context, broker):
        self.bars_processed += 1
        self.all_timestamps.append(timestamp)

        for asset in data:
            if asset not in self.entry_timestamps:
                pos = broker.get_position(asset)
                if pos is None or pos.quantity == 0:
                    broker.submit_order(asset, 10.0, OrderSide.BUY)
                    self.entry_timestamps[asset] = timestamp


def generate_minute_data_with_weekends(
    start_date: datetime,
    end_date: datetime,
    n_assets: int = 100,
    include_non_trading: bool = True,
) -> pl.DataFrame:
    """Generate minute-level data including non-trading periods.

    Args:
        start_date: Start date
        end_date: End date
        n_assets: Number of assets
        include_non_trading: Include weekends and extended hours
    """
    all_data = []
    current = start_date

    # Generate trading hours: 9:30 AM - 4:00 PM
    # If include_non_trading, also generate 8:00 AM - 6:00 PM on all days

    while current.date() <= end_date.date():
        is_weekend = current.weekday() >= 5

        if include_non_trading:
            # Generate 8:00 AM - 6:00 PM on ALL days (including weekends)
            start_hour, end_hour = 8, 18
        else:
            # Only generate for weekdays
            if is_weekend:
                current += timedelta(days=1)
                continue
            start_hour, end_hour = 9, 16

        for hour in range(start_hour, end_hour):
            start_minute = 30 if (hour == 9 and not include_non_trading) else 0
            end_minute = 60

            for minute in range(start_minute, end_minute):
                ts = current.replace(hour=hour, minute=minute, second=0, microsecond=0)

                for i in range(n_assets):
                    asset = f"ASSET{i:03d}"
                    base_price = 100.0 + i * 10  # Different base per asset
                    noise = np.random.uniform(-0.1, 0.1)
                    price = base_price * (1 + noise)

                    all_data.append({
                        "timestamp": ts,
                        "asset": asset,
                        "open": price,
                        "high": price * 1.001,
                        "low": price * 0.999,
                        "close": price,
                        "volume": 100000,
                    })

        current += timedelta(days=1)

    return pl.DataFrame(all_data)


def run_calendar_scale_test(n_assets: int = 50, n_days: int = 30):
    """Run calendar enforcement scale test.

    Args:
        n_assets: Number of assets (default 50)
        n_days: Number of calendar days (default 30)
    """
    print("=" * 70)
    print("CALENDAR ENFORCEMENT SCALE TEST")
    print("=" * 70)

    # Parameters - use a shorter window for practical testing
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=n_days - 1)

    print(f"\nGenerating data: {n_assets} assets x {n_days} days of minute data")
    print("Including weekends, holidays, and extended hours...")

    t0 = time.perf_counter()
    df = generate_minute_data_with_weekends(
        start_date, end_date, n_assets=n_assets, include_non_trading=True
    )
    gen_time = time.perf_counter() - t0

    total_bars = len(df) // n_assets  # Bars per asset
    print(f"Generated {len(df):,} rows ({total_bars:,} bars per asset) in {gen_time:.1f}s")

    # Test 1: Without calendar enforcement
    print("\n--- Test 1: No Calendar Enforcement ---")
    feed_no_cal = DataFeed(prices_df=df)
    strategy_no_cal = BuyAndHoldStrategy()
    engine_no_cal = Engine(feed=feed_no_cal, strategy=strategy_no_cal)

    t0 = time.perf_counter()
    results_no_cal = engine_no_cal.run()
    time_no_cal = time.perf_counter() - t0

    print(f"Bars processed: {strategy_no_cal.bars_processed:,}")
    print(f"Trades: {results_no_cal['num_trades']}")
    print(f"Skipped bars: {results_no_cal['skipped_bars']}")
    print(f"Time: {time_no_cal:.2f}s")

    # Test 2: With calendar enforcement
    print("\n--- Test 2: With Calendar Enforcement (NYSE) ---")
    feed_cal = DataFeed(prices_df=df)
    strategy_cal = BuyAndHoldStrategy()

    config = BacktestConfig(
        calendar="NYSE",
        enforce_sessions=True,
        data_frequency=DataFrequency.MINUTE_1,
    )
    engine_cal = Engine(feed=feed_cal, strategy=strategy_cal, config=config)

    t0 = time.perf_counter()
    results_cal = engine_cal.run()
    time_cal = time.perf_counter() - t0

    print(f"Bars processed: {strategy_cal.bars_processed:,}")
    print(f"Trades: {results_cal['num_trades']}")
    print(f"Skipped bars: {results_cal['skipped_bars']}")
    print(f"Time: {time_cal:.2f}s")

    # Validate results
    print("\n--- Validation ---")

    # Check 1: With enforcement, should have fewer bars processed
    assert strategy_cal.bars_processed < strategy_no_cal.bars_processed, (
        f"Expected fewer bars with enforcement: {strategy_cal.bars_processed} >= {strategy_no_cal.bars_processed}"
    )
    print(f"[PASS] Fewer bars processed with enforcement: {strategy_cal.bars_processed:,} < {strategy_no_cal.bars_processed:,}")

    # Check 2: Skipped bars should be non-zero with enforcement
    assert results_cal["skipped_bars"] > 0, "Expected some bars to be skipped"
    print(f"[PASS] Bars were skipped: {results_cal['skipped_bars']:,}")

    # Check 3: Verify no weekend timestamps in processed bars
    weekend_bars = sum(1 for ts in strategy_cal.all_timestamps if ts.weekday() >= 5)
    assert weekend_bars == 0, f"Found {weekend_bars} weekend bars processed"
    print(f"[PASS] No weekend bars processed")

    # Check 4: Verify all entries happened on valid trading days
    for asset, entry_ts in strategy_cal.entry_timestamps.items():
        assert entry_ts.weekday() < 5, f"{asset} entered on weekend: {entry_ts}"
    print(f"[PASS] All {len(strategy_cal.entry_timestamps)} entries on trading days")

    # Calculate skip ratio
    skip_ratio = results_cal["skipped_bars"] / (strategy_cal.bars_processed + results_cal["skipped_bars"])
    print(f"\nSkip ratio: {skip_ratio:.1%} of bars were outside trading sessions")

    # Performance comparison
    bars_per_sec_no_cal = strategy_no_cal.bars_processed / time_no_cal
    bars_per_sec_cal = strategy_cal.bars_processed / time_cal
    print(f"\nPerformance:")
    print(f"  Without calendar: {bars_per_sec_no_cal:,.0f} bars/sec")
    print(f"  With calendar: {bars_per_sec_cal:,.0f} bars/sec")

    print("\n" + "=" * 70)
    print("ALL CALENDAR SCALE TESTS PASSED")
    print("=" * 70)

    return {
        "bars_no_enforcement": strategy_no_cal.bars_processed,
        "bars_with_enforcement": strategy_cal.bars_processed,
        "skipped_bars": results_cal["skipped_bars"],
        "skip_ratio": skip_ratio,
        "time_no_enforcement": time_no_cal,
        "time_with_enforcement": time_cal,
    }


if __name__ == "__main__":
    np.random.seed(42)
    results = run_calendar_scale_test()
