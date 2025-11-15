"""
Scenario 001: Simple Market Orders

Purpose: Validate basic market order execution across platforms.

Test: 4 market orders on daily AAPL data over 2017.
- 2 complete trades (BUY→SELL, BUY→SELL)
- No stop loss or take profit
- Fixed commission only

What we're testing:
1. Do all platforms execute all 4 signals?
2. What price do they use (open vs close)?
3. When do they execute (same bar vs next bar)?
4. Do commissions calculate correctly?

Data: Real AAPL prices from Quandl Wiki dataset (2017)
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


class Scenario001:
    """Simple market orders - baseline validation."""

    name = "001_simple_market_orders"
    description = "Three market orders (BUY, SELL, BUY) with no exits"

    # ===================================================================
    # DATA SPECIFICATION
    # ===================================================================

    @staticmethod
    def get_data() -> pl.DataFrame:
        """
        Return OHLCV data for this scenario.

        Uses real AAPL data from Quandl Wiki prices (2017).
        """
        return get_ticker_data(
            ticker='AAPL',
            start_date='2017-01-01',
            end_date='2017-12-31',
            use_adjusted=False  # Use unadjusted prices for consistency
        )

    # ===================================================================
    # SIGNALS (HARDCODED)
    # ===================================================================

    # CRITICAL: Signals must be timezone-aware (UTC) to match market data timestamps
    signals = [
        # Trade 1: BUY early in year (Feb)
        Signal(
            timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),
        # Trade 1: SELL in spring (Apr)
        Signal(
            timestamp=datetime(2017, 4, 17, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='MARKET',
        ),
        # Trade 2: BUY in summer (Jul)
        Signal(
            timestamp=datetime(2017, 7, 17, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),
        # Trade 2: SELL late in year (Dec)
        Signal(
            timestamp=datetime(2017, 12, 18, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='MARKET',
        ),
    ]

    # ===================================================================
    # BACKTEST CONFIGURATION
    # ===================================================================

    config = {
        'initial_capital': 100_000.0,
        'commission': 0.001,  # 0.1% per trade
        'slippage': 0.0,      # No slippage for simplicity
    }

    # ===================================================================
    # EXPECTED RESULTS
    # ===================================================================

    expected = {
        # How many complete trades?
        # (BUY→SELL counts as 1, so we expect 2)
        'trade_count': 2,

        # When should orders execute?
        # Options: 'same_bar' (at signal bar close) or 'next_bar' (at next bar open)
        'execution_timing': 'next_bar',  # Most realistic

        # What price should be used?
        # Options: 'close' (signal bar close) or 'open' (next bar open)
        'price_used': 'open',  # For next_bar execution

        # Open positions at end?
        'final_position': 0,  # Flat (all closed)
    }

    # ===================================================================
    # COMPARISON RULES
    # ===================================================================

    comparison = {
        # How close do entry prices need to be?
        'price_tolerance_pct': 0.1,  # 0.1% = $0.70 on $700

        # How close does PnL need to be?
        'pnl_tolerance': 10.0,  # $10 total PnL difference acceptable

        # Must timestamps match exactly?
        'timestamp_exact': True,

        # Must trade counts match?
        'trade_count_exact': True,
    }

    # ===================================================================
    # ANALYSIS HELPERS
    # ===================================================================

    @staticmethod
    def analyze_data_at_signals():
        """
        Print what the data looks like at each signal timestamp.

        This helps understand what price each platform should use.
        """
        df = Scenario001.get_data()

        print("Data at signal timestamps:")
        print("=" * 80)

        for i, signal in enumerate(Scenario001.signals):
            print(f"\nSignal {i+1}: {signal.action} at {signal.timestamp.date()}")

            # Get signal bar
            signal_bar = df.filter(pl.col('timestamp') == signal.timestamp)
            if len(signal_bar) > 0:
                row = signal_bar[0]
                print(f"  Signal bar: O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")

            # Get next bar
            next_bar = df.filter(pl.col('timestamp') > signal.timestamp).head(1)
            if len(next_bar) > 0:
                row = next_bar[0]
                print(f"  Next bar:   O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")
            else:
                print("  Next bar:   (none)")

    @staticmethod
    def expected_pnl():
        """
        Calculate expected PnL if using next bar open prices.

        This is what we expect ml4t.backtest to produce.
        """
        df = Scenario001.get_data()

        trades = []
        position = None

        for signal in Scenario001.signals:
            # Find next bar after signal
            next_bar = df.filter(pl.col('timestamp') > signal.timestamp).head(1)
            if len(next_bar) == 0:
                print(f"Warning: No next bar for signal at {signal.timestamp}")
                continue

            price = next_bar[0]['open'][0]

            if signal.action == 'BUY':
                position = {
                    'entry_price': price,
                    'entry_timestamp': signal.timestamp,
                    'quantity': signal.quantity,
                }
                print(f"Open position: BUY {signal.quantity} @ ${price:.2f}")

            elif signal.action == 'SELL' and position:
                exit_price = price
                pnl = (exit_price - position['entry_price']) * position['quantity']

                # Commission: 0.1% on entry + 0.1% on exit
                entry_commission = position['entry_price'] * position['quantity'] * 0.001
                exit_commission = exit_price * position['quantity'] * 0.001
                total_commission = entry_commission + exit_commission

                net_pnl = pnl - total_commission

                trades.append({
                    'entry': position['entry_timestamp'],
                    'exit': signal.timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'gross_pnl': pnl,
                    'commission': total_commission,
                    'net_pnl': net_pnl,
                })

                print(f"Close position: SELL {position['quantity']} @ ${exit_price:.2f}")
                print(f"  Gross PnL: ${pnl:.2f}")
                print(f"  Commission: ${total_commission:.2f}")
                print(f"  Net PnL: ${net_pnl:.2f}")

                position = None

        print(f"\nTotal trades: {len(trades)}")
        print(f"Total Net PnL: ${sum(t['net_pnl'] for t in trades):.2f}")

        return trades


# ===================================================================
# USAGE
# ===================================================================

if __name__ == "__main__":
    """
    Run this to see what the scenario looks like.
    """
    print("Scenario 001: Simple Market Orders")
    print("=" * 80)
    print(f"Description: {Scenario001.description}")
    print(f"Signals: {len(Scenario001.signals)}")
    print(f"Expected trades: {Scenario001.expected['trade_count']}")
    print()

    # Show data at signal points
    Scenario001.analyze_data_at_signals()

    print("\n" + "=" * 80)
    print("Expected Results (next bar open execution):")
    print("=" * 80)
    Scenario001.expected_pnl()
