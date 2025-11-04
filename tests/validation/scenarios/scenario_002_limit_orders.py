"""
Scenario 002: Limit Orders

Purpose: Validate limit order execution across platforms.

Test: 4 limit orders on daily AAPL data over 2017.
- 2 complete trades (BUY LIMIT → SELL LIMIT, BUY LIMIT → SELL LIMIT)
- Testing "marketable" limit orders (execute immediately or very quickly)
- Fixed commission only

What we're testing:
1. Do all platforms execute limit orders correctly?
2. Are limit prices respected (at or better than limit)?
3. How do platforms handle limit order fills (same bar vs next bar)?
4. Do limit order commissions calculate correctly?

Limit Order Logic:
- BUY LIMIT: Execute when market price <= limit price (buy at or below limit)
- SELL LIMIT: Execute when market price >= limit price (sell at or above limit)

Strategy:
- Use "marketable" limits that execute quickly for reliable testing
- BUY limit set above current market (executes at better price immediately)
- SELL limit set below current market (executes at better price immediately)

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


class Scenario002:
    """Limit orders - validate limit price execution."""

    name = "002_limit_orders"
    description = "Two limit order trades testing immediate execution"

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
    # LIMIT ORDER STRATEGY:
    # Trade 1: BUY at $132 limit (above market at $130.29 close)
    #          Price on 2017-02-06: Close=$130.29
    #          Next bar 2017-02-07: Open=$130.54 (below limit, should execute)
    #          Expected entry: ~$130.54 (better than limit of $132)
    #
    # Trade 1: SELL at $149 limit (below market at ~$149.04 close on 2017-07-14)
    #          Price on 2017-07-14: Close=$149.04
    #          Next bar 2017-07-17: Open=$148.82 (close to limit, should execute)
    #          Expected exit: ~$148.82 (very close to limit of $149)
    #
    # Trade 2: BUY at $151 limit (above market at ~$150.34 close on 2017-07-20)
    #          Price on 2017-07-20: Close=$150.34
    #          Next bar 2017-07-21: Open=$149.99 (below limit, should execute)
    #          Expected entry: ~$149.99 (better than limit of $151)
    #
    # Trade 2: SELL at $147 limit (below market at ~$148.13 low on 2017-07-31)
    #          Price on 2017-07-31: Close=$148.85
    #          Next bar 2017-08-01: Open=$149.10 (above limit, should execute)
    #          Expected exit: ~$149.10 (better than limit of $147)

    signals = [
        # Trade 1: BUY LIMIT early in year (Feb)
        Signal(
            timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='LIMIT',
            limit_price=132.00,  # Above market - will execute at better price
        ),
        # Trade 1: SELL LIMIT in summer (Jul)
        Signal(
            timestamp=datetime(2017, 7, 14, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='LIMIT',
            limit_price=149.00,  # At/near market - will execute close to limit
        ),
        # Trade 2: BUY LIMIT later in summer (Jul)
        Signal(
            timestamp=datetime(2017, 7, 20, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
            order_type='LIMIT',
            limit_price=151.00,  # Above market - will execute at better price
        ),
        # Trade 2: SELL LIMIT end of summer (Aug)
        Signal(
            timestamp=datetime(2017, 7, 31, tzinfo=pytz.UTC),
            asset='AAPL',
            action='SELL',
            quantity=100,
            order_type='LIMIT',
            limit_price=147.00,  # Below market - will execute at better price
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
        # Limit orders execute when price reaches limit
        'execution_timing': 'next_bar',  # Most realistic for next-bar systems

        # What price should be used?
        # For limit orders: at or better than limit price
        'price_used': 'limit_or_better',

        # Open positions at end?
        'final_position': 0,  # Flat (all closed)
    }

    # ===================================================================
    # COMPARISON RULES
    # ===================================================================

    comparison = {
        # Limit orders must respect limit prices (critical!)
        'limit_price_enforcement': True,

        # How close do entry prices need to be?
        # Wider tolerance since limit execution can vary
        'price_tolerance_pct': 0.5,  # 0.5% = $0.70 on $140

        # How close does PnL need to be?
        'pnl_tolerance': 50.0,  # $50 total PnL difference acceptable

        # Timestamps may differ more due to limit conditions
        'timestamp_exact': False,
        'timestamp_tolerance_days': 2,  # Allow 2-day difference

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
        df = Scenario002.get_data()

        print("Data at signal timestamps:")
        print("=" * 80)

        for i, signal in enumerate(Scenario002.signals):
            print(f"\nSignal {i+1}: {signal.action} {signal.order_type} @ ${signal.limit_price:.2f}")
            print(f"  Signal date: {signal.timestamp.date()}")

            # Get signal bar
            signal_bar = df.filter(pl.col('timestamp') == signal.timestamp)
            if len(signal_bar) > 0:
                row = signal_bar[0]
                print(f"  Signal bar: O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")

                # Check if limit is marketable
                if signal.action == 'BUY' and signal.limit_price >= row['high'][0]:
                    print(f"  ✅ BUY limit ${signal.limit_price:.2f} >= high ${row['high'][0]:.2f} (marketable)")
                elif signal.action == 'SELL' and signal.limit_price <= row['low'][0]:
                    print(f"  ✅ SELL limit ${signal.limit_price:.2f} <= low ${row['low'][0]:.2f} (marketable)")

            # Get next bar
            next_bar = df.filter(pl.col('timestamp') > signal.timestamp).head(1)
            if len(next_bar) > 0:
                row = next_bar[0]
                print(f"  Next bar:   O={row['open'][0]:.2f} H={row['high'][0]:.2f} "
                      f"L={row['low'][0]:.2f} C={row['close'][0]:.2f}")

                # Check if limit would execute on next bar
                if signal.action == 'BUY' and row['low'][0] <= signal.limit_price:
                    print(f"  ✅ BUY limit ${signal.limit_price:.2f} >= low ${row['low'][0]:.2f} (will execute)")
                elif signal.action == 'SELL' and row['high'][0] >= signal.limit_price:
                    print(f"  ✅ SELL limit ${signal.limit_price:.2f} <= high ${row['high'][0]:.2f} (will execute)")
            else:
                print("  Next bar:   (none)")

    @staticmethod
    def expected_pnl():
        """
        Calculate expected PnL assuming next bar execution at limit-or-better prices.
        """
        df = Scenario002.get_data()

        trades = []
        position = None

        for signal in Scenario002.signals:
            # Find next bar after signal
            next_bar = df.filter(pl.col('timestamp') > signal.timestamp).head(1)
            if len(next_bar) == 0:
                print(f"Warning: No next bar for signal at {signal.timestamp}")
                continue

            # For limit orders, use the better of (open, limit)
            open_price = next_bar[0]['open'][0]

            if signal.action == 'BUY':
                # BUY limit: execute if open <= limit, at open price (better than limit)
                if open_price <= signal.limit_price:
                    exec_price = open_price
                else:
                    # Wait for price to drop to limit (simplified: use limit)
                    exec_price = signal.limit_price

                position = {
                    'entry_price': exec_price,
                    'entry_timestamp': signal.timestamp,
                    'quantity': signal.quantity,
                }
                print(f"Open position: BUY {signal.quantity} @ ${exec_price:.2f} (limit ${signal.limit_price:.2f})")

            elif signal.action == 'SELL' and position:
                # SELL limit: execute if open >= limit, at open price (better than limit)
                if open_price >= signal.limit_price:
                    exec_price = open_price
                else:
                    # Wait for price to rise to limit (simplified: use limit)
                    exec_price = signal.limit_price

                exit_price = exec_price
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

                print(f"Close position: SELL {position['quantity']} @ ${exit_price:.2f} (limit ${signal.limit_price:.2f})")
                print(f"  Gross PnL: ${pnl:.2f}")
                print(f"  Commission: ${total_commission:.2f}")
                print(f"  Net PnL: ${net_pnl:.2f}")

                position = None

        print(f"\nTotal trades: {len(trades)}")
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
    print("Scenario 002: Limit Orders")
    print("=" * 80)
    print(f"Description: {Scenario002.description}")
    print(f"Signals: {len(Scenario002.signals)}")
    print(f"Expected trades: {Scenario002.expected['trade_count']}")
    print()

    # Show data at signal points
    Scenario002.analyze_data_at_signals()

    print("\n" + "=" * 80)
    print("Expected Results (limit order execution):")
    print("=" * 80)
    Scenario002.expected_pnl()
