"""
Scenario 003: Stop-Loss Protection (Manual Exits)

Purpose: Validate stop-loss protection behavior across platforms using manual exit signals.

Test: 4 signals on daily AAPL data over 2017.
- 2 complete trades demonstrating stop-loss protection
- Manual SELL signals triggered when price drops below stop level
- Fixed commission only

What we're testing:
1. Do all platforms execute manual stop-loss exits correctly?
2. Are exits triggered at appropriate prices when stop level is breached?
3. Does the exit timing match expected behavior (next bar after breach)?
4. Does stop-loss protection work as expected (limit losses)?

Stop-Loss Simulation:
- Enter long positions with market orders
- Monitor price movements
- Exit with market SELL when price breaches stop level
- Tests *result* of stop-loss protection without platform-specific stop order types

Strategy:
- Enter long positions
- Place manual SELL orders when stop level is breached
- Simulates stop-loss protection behavior

Note: This scenario uses manual exit signals to simulate stop-loss behavior,
ensuring compatibility across all 4 platforms without requiring platform-specific
stop order implementations.

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


class Scenario003:
    """Stop-loss protection - validate manual exits when stop level breached."""

    name = "003_stop_orders"
    description = "Two trades demonstrating stop-loss protection with manual exits"

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
    # STOP-LOSS PROTECTION STRATEGY (Manual Exits):
    # Trade 1: BUY at market → Manual SELL when stop level breached
    #          Entry 2017-02-22: Close=$137.11
    #          Stop level: $136.00 (0.8% below entry)
    #          Price drops on 2017-02-24: Low=$135.28 (breaches stop!)
    #          Exit 2017-02-24: Manual SELL to protect against further loss
    #
    # Trade 2: BUY at market → Manual SELL when stop level breached
    #          Entry 2017-07-26: Close=$153.46
    #          Stop level: $151.00 (1.6% below entry)
    #          Price drops on 2017-07-27: Low=$147.30 (breaches stop!)
    #          Exit 2017-07-27: Manual SELL to protect against dramatic loss

    signals = [
        # Trade 1: BUY entry
        Signal(
            timestamp=datetime(2017, 2, 22, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),
        # Trade 1: Manual SELL exit (simulating stop-loss trigger on 2017-02-24)
        Signal(
            timestamp=datetime(2017, 2, 24, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='MARKET',
        ),

        # Trade 2: BUY entry
        Signal(
            timestamp=datetime(2017, 7, 26, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='MARKET',
        ),
        # Trade 2: Manual SELL exit (simulating stop-loss trigger on 2017-07-27)
        Signal(
            timestamp=datetime(2017, 7, 27, tzinfo=pytz.UTC),
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
        # (BUY with stop-loss exit counts as 1)
        'trade_count': 2,

        # Stop-loss should trigger automatically
        'execution_timing': 'next_bar',  # When stop is checked

        # Stop-loss exits at stop price or worse
        'price_used': 'stop_or_worse',

        # Open positions at end?
        'final_position': 0,  # Flat (stop-loss closed both positions)
    }

    # ===================================================================
    # COMPARISON RULES
    # ===================================================================

    comparison = {
        # Stop-loss must trigger correctly (critical!)
        'stop_loss_enforcement': True,

        # Exit prices should be near stop price
        'price_tolerance_pct': 1.0,  # 1% tolerance (stop may execute worse than stop price)

        # PnL should show losses (stopped out)
        'pnl_tolerance': 100.0,  # $100 tolerance (both trades stopped out for losses)

        # Timing may vary (when stop is checked)
        'timestamp_exact': False,
        'timestamp_tolerance_days': 1,  # Stop may trigger same day or next

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

        For stop-loss orders, show when price drops below stop level.
        """
        df = Scenario003.get_data()

        print("Data at signal timestamps and stop-loss triggers:")
        print("=" * 80)

        for i, signal in enumerate(Scenario003.signals):
            print(f"\nSignal {i+1}: {signal.action} with STOP LOSS @ ${signal.stop_loss:.2f}")
            print(f"  Signal date: {signal.timestamp.date()}")

            # Get signal bar (entry)
            signal_bar = df.filter(pl.col('timestamp') == signal.timestamp)
            if len(signal_bar) > 0:
                row = signal_bar[0]
                entry_price = row['close'][0]  # Assume entry at close
                print(f"  Entry bar:  O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={entry_price:.2f}")
                print(f"  Stop-loss: ${signal.stop_loss:.2f} ({((signal.stop_loss - entry_price) / entry_price * 100):.1f}% below entry)")

            # Find when stop-loss would trigger
            future_bars = df.filter(pl.col('timestamp') > signal.timestamp)
            for j in range(min(20, len(future_bars))):  # Check next 20 days
                row = future_bars[j]
                if row['low'][0] <= signal.stop_loss:
                    print(f"\n  🔴 STOP TRIGGERED on {row['timestamp'][0].strftime('%Y-%m-%d')}:")
                    print(f"     O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                          f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")
                    print(f"     Low ${row['low'][0]:.2f} <= Stop ${signal.stop_loss:.2f}")
                    break

    @staticmethod
    def expected_pnl():
        """
        Calculate expected PnL assuming stop-loss triggers.

        Both trades should result in losses (stopped out).
        """
        df = Scenario003.get_data()

        trades = []

        for signal in Scenario003.signals:
            # Entry at signal bar close
            signal_bar = df.filter(pl.col('timestamp') == signal.timestamp)
            if len(signal_bar) == 0:
                print(f"Warning: No data for signal at {signal.timestamp}")
                continue

            entry_price = signal_bar[0]['close'][0]

            # Find stop-loss trigger
            future_bars = df.filter(pl.col('timestamp') > signal.timestamp)
            exit_price = None
            exit_timestamp = None

            for j in range(len(future_bars)):
                row = future_bars[j]
                if row['low'][0] <= signal.stop_loss:
                    # Stop triggered - exit at stop price (or worse if gap down)
                    # Conservative: assume exit at stop price
                    exit_price = signal.stop_loss
                    exit_timestamp = row['timestamp'][0]
                    break

            if exit_price is None:
                print(f"Warning: Stop-loss never triggered for signal at {signal.timestamp}")
                continue

            # Calculate P&L
            pnl = (exit_price - entry_price) * signal.quantity

            # Commission
            entry_commission = entry_price * signal.quantity * 0.001
            exit_commission = exit_price * signal.quantity * 0.001
            total_commission = entry_commission + exit_commission

            net_pnl = pnl - total_commission

            trades.append({
                'entry': signal.timestamp,
                'exit': exit_timestamp,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_price': signal.stop_loss,
                'quantity': signal.quantity,
                'gross_pnl': pnl,
                'commission': total_commission,
                'net_pnl': net_pnl,
            })

            print(f"Position: BUY {signal.quantity} @ ${entry_price:.2f}")
            print(f"Stop-loss triggered: SELL @ ${exit_price:.2f} (stop ${signal.stop_loss:.2f})")
            print(f"  Gross PnL: ${pnl:.2f}")
            print(f"  Commission: ${total_commission:.2f}")
            print(f"  Net PnL: ${net_pnl:.2f}")
            print()

        print(f"Total trades: {len(trades)}")
        if trades:
            print(f"Total Net PnL: ${sum(t['net_pnl'] for t in trades):.2f}")

        return trades


# ===================================================================
# USAGE
# ===================================================================

if __name__ == "__main__":
    """
    Run this to see what the scenario looks like.
    """
    print("Scenario 003: Stop Orders")
    print("=" * 80)
    print(f"Description: {Scenario003.description}")
    print(f"Signals: {len(Scenario003.signals)}")
    print(f"Expected trades: {Scenario003.expected['trade_count']}")
    print()

    # Show data at signal points and stop triggers
    Scenario003.analyze_data_at_signals()

    print("\n" + "=" * 80)
    print("Expected Results (stop-loss execution):")
    print("=" * 80)
    Scenario003.expected_pnl()
