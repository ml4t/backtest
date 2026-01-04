#!/usr/bin/env python3
"""NautilusTrader Quickstart Test.

Tests basic functionality with EUR/USD sample data.
"""

import os
import urllib.request
from pathlib import Path

# NautilusTrader imports
from nautilus_trader.backtest.node import BacktestDataConfig
from nautilus_trader.backtest.node import BacktestEngineConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.node import BacktestRunConfig
from nautilus_trader.backtest.node import BacktestVenueConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import Quantity
from nautilus_trader.model import QuoteTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import CSVTickDataLoader
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.trading.strategy import StrategyConfig


# =============================================================================
# Simple EMA Crossover Strategy
# =============================================================================

class SimpleEMAConfig(StrategyConfig):
    """Configuration for simple EMA crossover strategy."""
    instrument_id: InstrumentId
    fast_period: int = 10
    slow_period: int = 20
    trade_size: int = 100_000


class SimpleEMAStrategy(Strategy):
    """Simple EMA crossover strategy - minimal implementation."""

    def __init__(self, config: SimpleEMAConfig):
        super().__init__(config=config)
        self.fast_ema = ExponentialMovingAverage(config.fast_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_period)
        self.trade_size = Quantity.from_int(config.trade_size)
        self.in_position = False

    def on_start(self):
        """Subscribe to market data on start."""
        self.subscribe_quote_ticks(instrument_id=self.config.instrument_id)
        self._log.info("Strategy started - subscribed to quote ticks")

    def on_quote_tick(self, tick: QuoteTick):
        """Process incoming quote ticks."""
        # Update indicators with mid price
        mid = (float(tick.bid_price) + float(tick.ask_price)) / 2
        self.fast_ema.update_raw(mid)
        self.slow_ema.update_raw(mid)

        # Wait for indicators to initialize
        if not self.slow_ema.initialized:
            return

        fast_val = self.fast_ema.value
        slow_val = self.slow_ema.value

        # Simple crossover logic
        if fast_val > slow_val and not self.in_position:
            # Golden cross - go long
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.trade_size,
            )
            self.submit_order(order)
            self.in_position = True
            self._log.info(f"BUY @ {tick.ask_price}")

        elif fast_val < slow_val and self.in_position:
            # Death cross - close position
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.trade_size,
            )
            self.submit_order(order)
            self.in_position = False
            self._log.info(f"SELL @ {tick.bid_price}")

    def on_stop(self):
        """Clean up on stop."""
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)


# =============================================================================
# Data Setup
# =============================================================================

def setup_sample_data():
    """Download EUR/USD sample data and write to Parquet catalog."""
    catalog_path = Path("catalog")
    catalog_path.mkdir(exist_ok=True)

    # Check if data already exists
    catalog = ParquetDataCatalog(str(catalog_path))
    instruments = catalog.instruments()
    if instruments:
        print(f"Using existing data: {len(instruments)} instruments")
        return catalog_path

    print("Downloading EUR/USD sample data...")
    url = "https://raw.githubusercontent.com/nautechsystems/nautilus_data/main/raw_data/fx_hist_data/DAT_ASCII_EURUSD_T_202001.csv.gz"
    filename = "EURUSD_202001.csv.gz"

    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete")
    except Exception as e:
        print(f"Download failed: {e}")
        raise

    # Create instrument
    print("Creating EUR/USD instrument...")
    instrument = TestInstrumentProvider.default_fx_ccy("EUR/USD")

    # Load and process tick data
    print("Loading tick data...")
    df = CSVTickDataLoader.load(
        filename,
        index_col=0,
        datetime_format="%Y%m%d %H%M%S%f",
    )
    df.columns = ["bid_price", "ask_price", "size"]
    print(f"Loaded {len(df):,} ticks")

    # Wrangle to NautilusTrader format
    wrangler = QuoteTickDataWrangler(instrument)
    ticks = wrangler.process(df)
    print(f"Processed {len(ticks):,} quote ticks")

    # Write to catalog
    print("Writing to Parquet catalog...")
    catalog.write_data([instrument])
    catalog.write_data(ticks)
    print("Data setup complete!")

    # Cleanup
    os.unlink(filename)

    return catalog_path


# =============================================================================
# Run Backtest
# =============================================================================

def run_backtest():
    """Run the EMA crossover backtest."""
    print("\n" + "=" * 60)
    print("NautilusTrader Quickstart Test")
    print("=" * 60)

    # Setup data
    catalog_path = setup_sample_data()
    catalog = ParquetDataCatalog(str(catalog_path))
    instruments = catalog.instruments()

    if not instruments:
        print("ERROR: No instruments found in catalog")
        return 1

    instrument = instruments[0]
    print(f"\nInstrument: {instrument.id}")

    # Configure venue (simulated FX broker)
    venue = BacktestVenueConfig(
        name="SIM",
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency="USD",
        starting_balances=["1_000_000 USD"],
    )

    # Configure data source
    data = BacktestDataConfig(
        catalog_path=str(catalog.path),
        data_cls=QuoteTick,
        instrument_id=instrument.id,
        end_time="2020-01-05",  # Use 5 days of data for quick test
    )

    # Configure engine with strategy
    engine_config = BacktestEngineConfig(
        strategies=[
            ImportableStrategyConfig(
                strategy_path="__main__:SimpleEMAStrategy",
                config_path="__main__:SimpleEMAConfig",
                config={
                    "instrument_id": instrument.id,
                    "fast_period": 10,
                    "slow_period": 20,
                    "trade_size": 100_000,
                },
            )
        ],
        logging=LoggingConfig(log_level="ERROR"),
    )

    # Create run configuration
    run_config = BacktestRunConfig(
        engine=engine_config,
        venues=[venue],
        data=[data],
    )

    # Run backtest
    print("\nRunning backtest...")
    node = BacktestNode(configs=[run_config])

    import time
    start_time = time.perf_counter()
    results = node.run()
    elapsed = time.perf_counter() - start_time

    print(f"\nBacktest completed in {elapsed:.2f} seconds")

    # Get results
    engine = node.get_engine(run_config.id)

    # Generate reports
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    # Account report
    from nautilus_trader.model import Venue
    account_report = engine.trader.generate_account_report(Venue("SIM"))
    print(f"\nAccount Report:\n{account_report}")

    # Positions report
    positions = engine.trader.generate_positions_report()
    print(f"\nPositions: {len(positions)}")

    if len(positions) > 0:
        print(positions.to_string())

    # Order fills
    fills = engine.trader.generate_order_fills_report()
    print(f"\nOrder Fills: {len(fills)}")

    print("\n" + "=" * 60)
    print("TEST PASSED - NautilusTrader working correctly!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_backtest())
