"""
Scenario 004: Position Re-Entry and Accumulation

Purpose: Validate position accumulation and re-entry patterns across platforms.

Test: 8 market orders on daily AAPL data over 2017.
- Trade sequence 1: Position accumulation (BUY → BUY more → SELL all)
- Trade sequence 2: Re-entry after partial exit (BUY → SELL partial → BUY more → SELL all)
- Market orders only for cross-platform simplicity
- Fixed commission only

What we're testing:
1. Do all platforms handle position accumulation correctly?
2. Can platforms track cumulative positions (100 → 200 → 0)?
3. Do re-entry patterns work (exit then re-enter same symbol)?
4. Are position sizes tracked correctly throughout?

Position Accumulation Logic:
- BUY 100 shares → position = 100
- BUY 100 more shares → position = 200
- SELL 200 shares → position = 0 (flat)

Re-Entry Logic:
- BUY 100 shares → position = 100
- SELL 50 shares → position = 50
- BUY 100 more shares → position = 150
- SELL 150 shares → position = 0 (flat)

Strategy:
- Use real AAPL price data from March-May 2017
- Test position accumulation in uptrend (March)
- Test re-entry during consolidation (May)
- All market orders execute at next bar open

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


class Scenario004:
    """Position re-entry - validate accumulation and re-entry patterns."""

    name = "004_position_reentry"
    description = "Position accumulation and re-entry pattern testing"

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
    #
    # POSITION ACCUMULATION STRATEGY (Trade Sequence 1):
    # Signal 1: BUY 100 on 2017-03-01 (entry at ~$137.89-140.15 range)
    #           Price: O=$137.89, H=$140.15, L=$137.59, C=$139.79
    #           Expected: Execute next bar (2017-03-02) at open ~$140.15
    #
    # Signal 2: BUY 100 MORE on 2017-03-06 (accumulate to 200 shares)
    #           Price: O=$139.37, H=$139.77, L=$138.60, C=$139.34
    #           Position before: 100 shares
    #           Expected: Execute next bar (2017-03-07) at open ~$139.00
    #           Position after: 200 shares
    #
    # Signal 3: SELL 200 on 2017-03-10 (exit all accumulated position)
    #           Price: O=$139.25, H=$139.36, L=$138.64, C=$139.14
    #           Position before: 200 shares
    #           Expected: Execute next bar (2017-03-13) at open ~$139.84
    #           Position after: 0 shares (flat)
    #
    # RE-ENTRY PATTERN (Trade Sequence 2):
    # Signal 4: BUY 100 on 2017-05-01 (new position)
    #           Price: O=$145.10, H=$147.20, L=$144.96, C=$146.60
    #           Expected: Execute next bar (2017-05-02) at open ~$147.51
    #           Position after: 100 shares
    #
    # Signal 5: SELL 50 on 2017-05-08 (partial exit)
    #           Price: O=$149.03, H=$153.70, L=$149.03, C=$153.00
    #           Position before: 100 shares
    #           Expected: Execute next bar (2017-05-09) at open ~$153.74
    #           Position after: 50 shares
    #
    # Signal 6: BUY 100 on 2017-05-15 (re-enter and accumulate)
    #           Price: O=$156.01, H=$156.65, L=$155.05, C=$155.70
    #           Position before: 50 shares
    #           Expected: Execute next bar (2017-05-16) at open ~$155.47
    #           Position after: 150 shares
    #
    # Signal 7: SELL 150 on 2017-05-22 (exit all)
    #           Price: O=$154.00, H=$154.58, L=$152.91, C=$153.99
    #           Position before: 150 shares
    #           Expected: Execute next bar (2017-05-23) at open ~$154.45
    #           Position after: 0 shares (flat)

    signals = [
        # ===================================================================
        # TRADE SEQUENCE 1: Position Accumulation
        # ===================================================================

        # Signal 1: Initial BUY (100 shares)
        Signal(
            timestamp=datetime(2017, 3, 1, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),

        # Signal 2: Accumulate position (BUY 100 more → total 200)
        Signal(
            timestamp=datetime(2017, 3, 6, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),

        # Signal 3: Exit all accumulated position (SELL 200)
        Signal(
            timestamp=datetime(2017, 3, 10, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=200,
            order_type='MARKET',
        ),

        # ===================================================================
        # TRADE SEQUENCE 2: Re-Entry Pattern
        # ===================================================================

        # Signal 4: New position (BUY 100)
        Signal(
            timestamp=datetime(2017, 5, 1, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),

        # Signal 5: Partial exit (SELL 50 → position = 50)
        Signal(
            timestamp=datetime(2017, 5, 8, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=50,
            order_type='MARKET',
        ),

        # Signal 6: Re-enter and accumulate (BUY 100 → position = 150)
        Signal(
            timestamp=datetime(2017, 5, 15, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),

        # Signal 7: Exit all (SELL 150 → position = 0)
        Signal(
            timestamp=datetime(2017, 5, 22, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=150,
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
        # Sequence 1: BUY 100 + BUY 100 → SELL 200 (1 complete trade)
        # Sequence 2: BUY 100 → SELL 50 + BUY 100 → SELL 150 (2 complete trades)
        # Total: 3 complete trades
        'trade_count': 3,

        # Market orders execute at next bar open
        'execution_timing': 'next_bar_open',

        # What price should be used?
        'price_used': 'next_bar_open',

        # Open positions at end?
        'final_position': 0,  # Flat (all closed)

        # Position tracking
        'position_tracking': {
            'after_signal_1': 100,   # BUY 100
            'after_signal_2': 200,   # BUY 100 more
            'after_signal_3': 0,     # SELL 200 (flat)
            'after_signal_4': 100,   # BUY 100 (new position)
            'after_signal_5': 50,    # SELL 50
            'after_signal_6': 150,   # BUY 100 more
            'after_signal_7': 0,     # SELL 150 (flat)
        },
    }

    # ===================================================================
    # COMPARISON RULES
    # ===================================================================

    comparison = {
        # Position tracking must be correct (critical!)
        'position_tracking': True,

        # Market orders use next bar open
        'price_tolerance_pct': 0.5,  # 0.5% = ~$0.70 on $140

        # PnL tolerance
        'pnl_tolerance': 100.0,  # $100 total PnL difference acceptable

        # Timestamps may differ slightly (same-bar vs next-bar)
        'timestamp_exact': False,
        'timestamp_tolerance_days': 1,  # Allow 1-day difference

        # Must trade counts match?
        # Note: Different platforms may group trades differently
        # ml4t.backtest/backtrader/zipline: 3 trades (accumulation counts as 1 trade)
        # VectorBT: May show 4-5 trades (each entry/exit separate)
        'trade_count_exact': False,  # Allow some variation
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
        df = Scenario004.get_data()

        print("Data at signal timestamps:")
        print("=" * 80)

        position = 0  # Track expected position

        for i, signal in enumerate(Scenario004.signals):
            print(f"\nSignal {i+1}: {signal.action} {signal.quantity} @ MARKET")
            print(f"  Signal date: {signal.timestamp.date()}")

            # Get signal bar
            signal_bar = df.filter(pl.col('timestamp') == signal.timestamp)
            if len(signal_bar) > 0:
                row = signal_bar[0]
                print(f"  Signal bar: O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")

            # Get next bar (execution bar)
            next_bar = df.filter(pl.col('timestamp') > signal.timestamp).head(1)
            if len(next_bar) > 0:
                row = next_bar[0]
                exec_date = row['timestamp'][0].date()
                exec_price = row['open'][0]
                print(f"  Next bar:   O={exec_price:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")
                print(f"  Expected execution: {exec_date} at ${exec_price:.2f}")
            else:
                print("  Next bar:   (none)")

            # Update position tracking
            if signal.action == 'BUY':
                position += signal.quantity
            elif signal.action == 'SELL':
                position -= signal.quantity

            print(f"  Position after: {position:.0f} shares")

    @staticmethod
    def expected_pnl():
        """
        Calculate expected PnL assuming next bar open execution.
        """
        df = Scenario004.get_data()

        trades = []
        position = 0
        position_entries = []  # Track entry prices for FIFO

        print("Expected Trade Execution:")
        print("=" * 80)

        for i, signal in enumerate(Scenario004.signals):
            # Find next bar after signal (execution bar)
            next_bar = df.filter(pl.col('timestamp') > signal.timestamp).head(1)
            if len(next_bar) == 0:
                print(f"Warning: No next bar for signal at {signal.timestamp}")
                continue

            exec_price = next_bar[0]['open'][0]
            exec_date = next_bar[0]['timestamp'][0].date()

            print(f"\nSignal {i+1}: {signal.action} {signal.quantity:.0f} @ {signal.timestamp.date()}")
            print(f"  Execution: {exec_date} at ${exec_price:.2f}")
            print(f"  Position before: {position:.0f}")

            if signal.action == 'BUY':
                # Accumulate position
                for _ in range(int(signal.quantity / 100)):
                    position_entries.append(exec_price)
                position += signal.quantity

            elif signal.action == 'SELL':
                # Exit position (FIFO)
                quantity_to_sell = signal.quantity
                gross_pnl = 0

                while quantity_to_sell > 0 and position_entries:
                    entry_price = position_entries.pop(0)
                    shares = min(100, quantity_to_sell)
                    pnl = (exec_price - entry_price) * shares
                    gross_pnl += pnl
                    quantity_to_sell -= shares

                position -= signal.quantity

                # Commission: 0.1% on entry + 0.1% on exit
                # For accumulated positions, commission on each leg
                entry_commission = sum(p * 100 * 0.001 for p in position_entries[-int(signal.quantity/100):])
                exit_commission = exec_price * signal.quantity * 0.001
                total_commission = entry_commission + exit_commission

                net_pnl = gross_pnl - total_commission

                trades.append({
                    'exit_date': exec_date,
                    'exit_price': exec_price,
                    'quantity': signal.quantity,
                    'gross_pnl': gross_pnl,
                    'commission': total_commission,
                    'net_pnl': net_pnl,
                })

                print(f"  Gross PnL: ${gross_pnl:.2f}")
                print(f"  Commission: ${total_commission:.2f}")
                print(f"  Net PnL: ${net_pnl:.2f}")

            print(f"  Position after: {position:.0f}")

        print(f"\nTotal complete trades: {len(trades)}")
        if trades:
            total_net_pnl = sum(t['net_pnl'] for t in trades)
            print(f"Total Net PnL: ${total_net_pnl:.2f}")

        return trades


# ===================================================================
# USAGE
# ===================================================================

if __name__ == "__main__":
    """
    Run this to see what the scenario looks like.
    """
    print("Scenario 004: Position Re-Entry")
    print("=" * 80)
    print(f"Description: {Scenario004.description}")
    print(f"Signals: {len(Scenario004.signals)}")
    print(f"Expected complete trades: {Scenario004.expected['trade_count']}")
    print()

    # Show data at signal points
    Scenario004.analyze_data_at_signals()

    print("\n" + "=" * 80)
    print("Expected Results (position accumulation):")
    print("=" * 80)
    Scenario004.expected_pnl()
