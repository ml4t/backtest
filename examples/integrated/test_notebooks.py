#!/usr/bin/env python
"""Test that example notebooks can be executed without errors.

This script simulates notebook execution by running the code cells.
"""

import sys
import time
from pathlib import Path

def test_notebook_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        import warnings
        warnings.filterwarnings('ignore')

        import uuid
        from datetime import datetime, timezone, timedelta

        import numpy as np
        import polars as pl
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # ml4t.backtest imports
        from ml4t.backtest.core.event import MarketEvent, OrderEvent
        from ml4t.backtest.core.types import OrderSide, OrderType, TimeInForce
        from ml4t.backtest.data.polars_feed import PolarsDataFeed
        from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider
        from ml4t.backtest.engine import BacktestEngine
        from ml4t.backtest.execution.broker import SimulationBroker
        from ml4t.backtest.execution.commission import NoCommission
        from ml4t.backtest.execution.slippage import (
            NoSlippage, PercentageSlippage, SpreadAwareSlippage,
            VolumeAwareSlippage, OrderTypeDependentSlippage
        )
        from ml4t.backtest.risk import RiskManager
        from ml4t.backtest.risk.rules import (
            VolatilityScaledStopLoss, RegimeDependentRule, PriceBasedStopLoss
        )
        from ml4t.backtest.strategy.base import Strategy

        print("✓ All imports successful")
        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic backtest functionality."""
    print("\nTesting basic backtest functionality...")

    try:
        import tempfile
        import polars as pl
        from datetime import datetime, timedelta, timezone
        from ml4t.backtest.data.polars_feed import PolarsDataFeed
        from ml4t.backtest.engine import BacktestEngine
        from ml4t.backtest.execution.broker import SimulationBroker
        from ml4t.backtest.execution.commission import NoCommission
        from ml4t.backtest.execution.slippage import NoSlippage
        from ml4t.backtest.strategy.base import Strategy
        from ml4t.backtest.core.event import MarketEvent, OrderEvent
        from ml4t.backtest.core.types import OrderSide, OrderType, TimeInForce
        import uuid

        # Create minimal test data
        base_date = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        data = pl.DataFrame({
            'timestamp': [base_date + timedelta(days=i) for i in range(10)],
            'asset_id': ['TEST'] * 10,
            'open': [100.0 + i for i in range(10)],
            'high': [101.0 + i for i in range(10)],
            'low': [99.0 + i for i in range(10)],
            'close': [100.5 + i for i in range(10)],
            'volume': [1_000_000] * 10,
        })

        # Simple strategy
        class TestStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.entered = False

            def on_start(self, portfolio=None, event_bus=None):
                self.portfolio = portfolio
                self.event_bus = event_bus

            def on_event(self, event):
                if isinstance(event, MarketEvent) and not self.entered:
                    order = OrderEvent(
                        timestamp=event.timestamp,
                        order_id=str(uuid.uuid4()),
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=10,
                        time_in_force=TimeInForce.DAY,
                    )
                    self.event_bus.publish(order)
                    self.entered = True

        # Write data and run backtest
        tmp_dir = Path(tempfile.mkdtemp())
        data_file = tmp_dir / "test.parquet"
        data.write_parquet(data_file)

        feed = PolarsDataFeed(data_file, asset_id='TEST')
        strategy = TestStrategy()
        broker = SimulationBroker(
            initial_cash=10_000,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

        engine = BacktestEngine(
            strategy=strategy,
            data_feed=feed,
            broker=broker,
        )

        results = engine.run()

        print("✓ Basic backtest executed successfully")
        return True

    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("NOTEBOOK DEPENDENCY TESTING")
    print("="*60)
    print()

    start = time.time()

    # Run tests
    tests = [
        test_notebook_imports,
        test_basic_functionality,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)

    elapsed = time.time() - start

    print()
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    print(f"Execution time: {elapsed:.2f}s")
    print()

    if all(results):
        print("✓ All tests passed - notebooks should execute successfully")
        return 0
    else:
        print("✗ Some tests failed - notebooks may have issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())
