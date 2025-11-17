"""
Bundle-free Zipline example using DataFrameLoader.

This demonstrates the pattern from zipline/tests/pipeline/test_pipeline_algo.py
showing how to inject custom OHLCV data without requiring a bundle.

Key insight: Use DataFrameLoader + get_pipeline_loader parameter to TradingAlgorithm.
"""

import pandas as pd
import numpy as np

from zipline.algorithm import TradingAlgorithm
from zipline.api import order_target_percent, set_commission, set_slippage
from zipline.finance import commission, slippage
from zipline.finance.trading import SimulationParameters
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.testing import make_simple_equity_info, make_test_asset_finder_and_asset_info
from zipline.testing.fixtures import WithDataPortal, WithTradingSessions
from zipline.utils.calendar_utils import get_calendar


def run_example():
    # Create simple OHLCV data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    dates = dates[dates.day_name().isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]  # Trading days
    dates = dates.tz_localize('UTC')

    sid = 1
    ohlcv_data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 105,
        'low': np.random.randn(len(dates)).cumsum() + 95,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.rand(len(dates)) * 1000000,
    }, index=dates)

    # Reshape for DataFrameLoader (columns must be sids)
    ohlcv_pivoted = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        ohlcv_pivoted[col] = pd.DataFrame({sid: ohlcv_data[col].values}, index=dates)

    # Create loaders
    loaders = {
        USEquityPricing.open: DataFrameLoader(USEquityPricing.open, ohlcv_pivoted['open']),
        USEquityPricing.high: DataFrameLoader(USEquityPricing.high, ohlcv_pivoted['high']),
        USEquityPricing.low: DataFrameLoader(USEquityPricing.low, ohlcv_pivoted['low']),
        USEquityPricing.close: DataFrameLoader(USEquityPricing.close, ohlcv_pivoted['close']),
        USEquityPricing.volume: DataFrameLoader(USEquityPricing.volume, ohlcv_pivoted['volume']),
    }

    def get_loader(column):
        return loaders.get(column)

    # Strategy
    def initialize(context):
        context.sid = sid
        set_commission(commission.PerShare(cost=0.0))
        set_slippage(slippage.FixedSlippage(spread=0.0))

    def handle_data(context, data):
        price = data.current(context.sid, 'close')
        # Simple strategy: buy if we don't have position
        if context.portfolio.positions[context.sid].amount == 0:
            order_target_percent(context.sid, 1.0)

    # This is the key part - using WithDataPortal fixture pattern
    # See zipline/testing/fixtures.py for implementation

    print("This example shows the pattern needed for bundle-free Zipline")
    print("Full implementation requires creating asset_finder and data_portal")
    print("See zipline/tests/pipeline/test_pipeline_algo.py for complete example")


if __name__ == '__main__':
    run_example()
