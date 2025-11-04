"""
Scenario 005: Multi-Asset Trading

Purpose: Validate multi-asset execution and position tracking across platforms.

Test: Trading AAPL and MSFT simultaneously over 2017.
- 4 complete trades (2 per asset)
- Interleaved signals testing concurrent positions
- Asset isolation verification
- Market orders only (simplest case for multi-asset)
- Fixed commission only

What we're testing:
1. Can all platforms trade multiple assets simultaneously?
2. Are positions tracked independently per asset?
3. Do assets interfere with each other?
4. Does portfolio tracking work correctly with multiple concurrent positions?
5. Are trades extracted correctly for each asset?

Multi-Asset Strategy:
- Trade AAPL and MSFT with interleaved signals
- Open AAPL position first
- Open MSFT position while AAPL still open (test concurrent positions)
- Close both positions separately
- Repeat for second round-trip on each asset

Data: Real AAPL and MSFT prices from Quandl Wiki dataset (2017)
"""

from dataclasses import dataclass
from datetime import datetime
import polars as pl
import pytz
import sys
from pathlib import Path

# Add validation directory to path for fixtures
sys.path.insert(0, str(Path(__file__).parents[1]))
from fixtures.market_data import get_ticker_data


@dataclass
class Signal:
    """Platform-independent signal specification."""

    timestamp: datetime
    asset: str
    action: str  # 'BUY', 'SELL', 'CLOSE'
    quantity: float
    order_type: str = 'MARKET'
    limit_price: float | None = None
    stop_price: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    trailing_stop_pct: float | None = None


class Scenario005:
    """Multi-asset trading - validate concurrent position tracking."""

    name = "005_multi_asset"
    description = "Two assets traded simultaneously (AAPL + MSFT)"

    # ===================================================================
    # DATA SPECIFICATION
    # ===================================================================

    @staticmethod
    def get_data() -> pl.DataFrame:
        """
        Return OHLCV data for this scenario.

        Returns concatenated DataFrame with both AAPL and MSFT.
        Uses real data from Quandl Wiki prices (2017).
        """
        aapl_data = get_ticker_data(
            ticker='AAPL',
            start_date='2017-01-01',
            end_date='2017-12-31',
            use_adjusted=False
        )
        msft_data = get_ticker_data(
            ticker='MSFT',
            start_date='2017-01-01',
            end_date='2017-12-31',
            use_adjusted=False
        )

        # Concatenate both datasets
        combined = pl.concat([aapl_data, msft_data], how='vertical')

        # Sort by timestamp to maintain chronological order
        combined = combined.sort('timestamp')

        return combined

    # ===================================================================
    # SIGNALS (HARDCODED)
    # ===================================================================

    # CRITICAL: All signals must be timezone-aware (UTC) to match market data timestamps
    #
    # MULTI-ASSET STRATEGY:
    # Trade 1 (AAPL): Entry 2017-01-09 → Exit 2017-03-01
    #   Entry: 2017-01-09 close=$118.99, expect entry ~$119 at next open
    #   Exit:  2017-03-01 close=$139.79, expect exit ~$139.79 at next open
    #
    # Trade 2 (MSFT): Entry 2017-01-27 → Exit 2017-04-03 (overlaps with AAPL Trade 1)
    #   Entry: 2017-01-27 close=$65.78, expect entry ~$65.69 at next open
    #   Exit:  2017-04-03 close=$65.86, expect exit ~$65.71 at next open
    #
    # Trade 3 (AAPL): Entry 2017-06-02 → Exit 2017-08-01
    #   Entry: 2017-06-02 close=$155.45, expect entry ~$154.34 at next open
    #   Exit:  2017-08-01 close=$146.39, expect exit ~$149.10 at next open
    #
    # Trade 4 (MSFT): Entry 2017-06-05 → Exit 2017-09-01 (overlaps with AAPL Trade 3)
    #   Entry: 2017-06-05 close=$72.28, expect entry ~$72.30 at next open
    #   Exit:  2017-09-01 close=$74.77, expect exit ~$75.23 at next open

    signals = [
            # ===================================================================
            # AAPL Trade 1: Entry
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 1, 9, 0, 0, 0, tzinfo=pytz.UTC),
                asset='AAPL',
                action='BUY',
                quantity=100,
                order_type='MARKET'
            ),
            # Price on 2017-01-09: Close=$118.99
            # Expected entry: 2017-01-10 open=$119.11 (100 shares)

            # ===================================================================
            # MSFT Trade 1: Entry (while AAPL position is still open)
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 1, 27, 0, 0, 0, tzinfo=pytz.UTC),
                asset='MSFT',
                action='BUY',
                quantity=200,
                order_type='MARKET'
            ),
            # Price on 2017-01-27: Close=$65.78
            # Expected entry: 2017-01-30 open=$65.69 (200 shares)
            # Now holding: AAPL 100 shares + MSFT 200 shares (concurrent positions)

            # ===================================================================
            # AAPL Trade 1: Exit (while MSFT position is still open)
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 3, 1, 0, 0, 0, tzinfo=pytz.UTC),
                asset='AAPL',
                action='SELL',
                quantity=100,
                order_type='MARKET'
            ),
            # Price on 2017-03-01: Close=$139.79
            # Expected exit: 2017-03-02 open=$140.00 (100 shares)
            # Now holding: MSFT 200 shares only

            # ===================================================================
            # MSFT Trade 1: Exit
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 4, 3, 0, 0, 0, tzinfo=pytz.UTC),
                asset='MSFT',
                action='SELL',
                quantity=200,
                order_type='MARKET'
            ),
            # Price on 2017-04-03: Close=$65.86
            # Expected exit: 2017-04-04 open=$65.71 (200 shares)
            # Now flat (no positions)

            # ===================================================================
            # AAPL Trade 2: Entry
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 6, 2, 0, 0, 0, tzinfo=pytz.UTC),
                asset='AAPL',
                action='BUY',
                quantity=150,
                order_type='MARKET'
            ),
            # Price on 2017-06-02: Close=$155.45
            # Expected entry: 2017-06-05 open=$154.34 (150 shares)

            # ===================================================================
            # MSFT Trade 2: Entry (while AAPL position is still open)
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 6, 5, 0, 0, 0, tzinfo=pytz.UTC),
                asset='MSFT',
                action='BUY',
                quantity=250,
                order_type='MARKET'
            ),
            # Price on 2017-06-05: Close=$72.28
            # Expected entry: 2017-06-06 open=$72.30 (250 shares)
            # Now holding: AAPL 150 shares + MSFT 250 shares (concurrent positions)

            # ===================================================================
            # AAPL Trade 2: Exit (while MSFT position is still open)
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 8, 1, 0, 0, 0, tzinfo=pytz.UTC),
                asset='AAPL',
                action='SELL',
                quantity=150,
                order_type='MARKET'
            ),
            # Price on 2017-08-01: Close=$146.39
            # Expected exit: 2017-08-02 open=$149.10 (150 shares)
            # Now holding: MSFT 250 shares only

            # ===================================================================
            # MSFT Trade 2: Exit
            # ===================================================================
            Signal(
                timestamp=datetime(2017, 9, 1, 0, 0, 0, tzinfo=pytz.UTC),
                asset='MSFT',
                action='SELL',
                quantity=250,
                order_type='MARKET'
            ),
            # Price on 2017-09-01: Close=$74.77
            # Expected exit: 2017-09-05 open=$75.23 (250 shares)
            # Now flat (no positions)
        ]

    # ===================================================================
    # METADATA
    # ===================================================================

    config = {
        'initial_capital': 100_000.0,
        'commission': 0.001,  # 0.1% per trade
        'slippage': 0.0,      # No slippage for simplicity
    }

    # ===================================================================
    # VALIDATION EXPECTATIONS
    # ===================================================================

    @staticmethod
    def get_expected_trades() -> dict:
        """
        Document expected trade results for validation.

        This helps identify platform-specific differences in multi-asset execution.
        """
        return {
            'total_trades': 4,  # 2 AAPL + 2 MSFT
            'aapl_trades': 2,
            'msft_trades': 2,
            'concurrent_positions': True,  # Both assets held simultaneously
            'max_concurrent_assets': 2,
            'expected_differences': {
                'execution_timing': 'Platforms may execute at different times (same-bar vs next-bar)',
                'fill_prices': 'Slight variations due to different execution models',
                'position_tracking': 'All platforms should maintain separate positions per asset',
            },
            'critical_validations': [
                'Both assets traded successfully',
                'Positions tracked independently',
                'No cross-asset interference',
                'Correct portfolio value with concurrent positions',
            ]
        }


# ===================================================================
# HELPER FUNCTION
# ===================================================================

def get_scenario() -> Scenario005:
    """
    Factory function to get scenario instance.

    Used by runner.py to load scenarios dynamically.
    """
    return Scenario005()


# ===================================================================
# MAIN (for testing)
# ===================================================================

if __name__ == "__main__":
    # Test that scenario loads correctly
    scenario = get_scenario()
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")

    # Load data
    print("\nLoading data...")
    data = scenario.get_data()
    print(f"  Total bars: {len(data)}")
    print(f"  AAPL bars: {len(data.filter(pl.col('symbol') == 'AAPL'))}")
    print(f"  MSFT bars: {len(data.filter(pl.col('symbol') == 'MSFT'))}")

    # Load signals
    print("\nLoading signals...")
    signals = scenario.signals
    print(f"  Total signals: {len(signals)}")

    # Group signals by asset
    aapl_signals = [s for s in signals if s.asset == 'AAPL']
    msft_signals = [s for s in signals if s.asset == 'MSFT']
    print(f"  AAPL signals: {len(aapl_signals)}")
    print(f"  MSFT signals: {len(msft_signals)}")

    # Show signal timeline
    print("\nSignal timeline:")
    for i, signal in enumerate(signals, 1):
        print(f"  {i}. {signal.timestamp.date()} - {signal.asset} {signal.action} {signal.quantity}")

    # Show expected trades
    print("\nExpected trades:")
    expected = scenario.get_expected_trades()
    print(f"  Total: {expected['total_trades']}")
    print(f"  AAPL: {expected['aapl_trades']}")
    print(f"  MSFT: {expected['msft_trades']}")
    print(f"  Concurrent positions: {expected['concurrent_positions']}")

    print("\n✅ Scenario loaded successfully!")
