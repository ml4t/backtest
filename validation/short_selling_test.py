"""Short selling basic sanity tests.

NOTE: For actual validation, use:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/scenario_11_short_only.py

That script validates 100% trade-level exact match against VectorBT Pro.

These tests are just basic sanity checks:
1. Short-only strategy profits in falling market
2. Short-only strategy loses in rising market
3. Long-short hedged portfolio has both directions
4. Short position PnL formula is correct

Run with:
    source .venv/bin/activate
    python validation/short_selling_test.py
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml4t.backtest import (
    DataFeed,
    Engine,
    NoCommission,
    NoSlippage,
    OrderSide,
    Strategy,
)


def generate_trending_data(
    n_assets: int = 50,
    n_bars: int = 500,
    trend_direction: int = 1,  # 1=up, -1=down
    seed: int = 42,
) -> pl.DataFrame:
    """Generate price data with a trend direction."""
    np.random.seed(seed)

    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_bars)]
    all_data = []

    for i in range(n_assets):
        asset = f"ASSET{i:04d}"
        base_price = 100.0

        # Generate trend + noise
        trend = trend_direction * 0.0002 * np.arange(n_bars)  # Slight trend
        noise = np.random.randn(n_bars) * 0.015
        log_returns = trend + noise
        closes = base_price * np.exp(np.cumsum(log_returns))

        # Generate valid OHLC
        opens = closes * (1 + np.random.randn(n_bars) * 0.003)
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.004)
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.004)

        for j in range(n_bars):
            all_data.append({
                "timestamp": dates[j],
                "asset": asset,
                "open": opens[j],
                "high": highs[j],
                "low": lows[j],
                "close": closes[j],
                "volume": float(np.random.randint(100000, 1000000)),
            })

    return pl.DataFrame(all_data)


class ShortOnlyStrategy(Strategy):
    """Strategy that goes short on every bar."""

    def __init__(self, n_assets: int, hold_period: int = 10):
        self.n_assets = n_assets
        self.hold_period = hold_period
        self.bar_count = 0
        self.positions_opened = {}

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1

        for i in range(self.n_assets):
            asset = f"ASSET{i:04d}"
            if asset not in data:
                continue

            pos = broker.get_position(asset)
            current_qty = pos.quantity if pos else 0

            # Close positions after hold period
            if asset in self.positions_opened:
                if self.bar_count - self.positions_opened[asset] >= self.hold_period:
                    if current_qty < 0:  # Have a short position
                        broker.close_position(asset)
                        del self.positions_opened[asset]

            # Open new short positions periodically
            if self.bar_count % 20 == (i % 20) and current_qty == 0:
                broker.submit_order(asset, 100, OrderSide.SELL)
                self.positions_opened[asset] = self.bar_count


class LongShortStrategy(Strategy):
    """Long-short hedged strategy (market neutral)."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.bar_count = 0
        self.rebalance_freq = 20
        self.positions_opened = {}

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1

        # Close positions after hold period
        for asset in list(self.positions_opened.keys()):
            if self.bar_count - self.positions_opened[asset] >= 15:
                pos = broker.get_position(asset)
                if pos and pos.quantity != 0:
                    broker.close_position(asset)
                    del self.positions_opened[asset]

        # Open new positions periodically
        if self.bar_count % self.rebalance_freq != 1:
            return

        # Long top half of assets, short bottom half
        n_long = self.n_assets // 2

        for i in range(self.n_assets):
            asset = f"ASSET{i:04d}"
            if asset not in data:
                continue

            pos = broker.get_position(asset)
            current_qty = pos.quantity if pos else 0

            if current_qty != 0:
                continue  # Already have a position

            if i < n_long:
                # Long position
                broker.submit_order(asset, 50, OrderSide.BUY)
                self.positions_opened[asset] = self.bar_count
            else:
                # Short position
                broker.submit_order(asset, 50, OrderSide.SELL)
                self.positions_opened[asset] = self.bar_count


def test_short_only_falling_market():
    """Test short-only strategy in falling market (should profit)."""
    print("\n" + "=" * 70)
    print("TEST 1: Short-Only Strategy in Falling Market")
    print("=" * 70)

    n_assets = 50
    n_bars = 500

    # Generate falling market data
    df = generate_trending_data(n_assets=n_assets, n_bars=n_bars, trend_direction=-1)

    feed = DataFeed(prices_df=df)
    strategy = ShortOnlyStrategy(n_assets=n_assets)

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        account_type="margin",  # Need margin for shorts
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )

    t0 = time.perf_counter()
    results = engine.run()
    run_time = time.perf_counter() - t0

    print(f"\nResults:")
    print(f"  Time: {run_time:.2f}s")
    print(f"  Final value: ${results['final_value']:,.2f}")
    print(f"  Return: {results['total_return_pct']:.2f}%")
    print(f"  Trades: {results['num_trades']:,}")
    print(f"  Win rate: {results['win_rate']*100:.1f}%")

    # Validate: Should profit in falling market
    success = True

    # Should have positive returns
    if results['total_return_pct'] < 0:
        print(f"\n[FAIL] Short strategy should profit in falling market")
        success = False
    else:
        print(f"\n[PASS] Short strategy profited in falling market")

    # Should have positive win rate
    if results['win_rate'] < 0.4:
        print(f"[FAIL] Win rate too low for falling market")
        success = False
    else:
        print(f"[PASS] Win rate acceptable ({results['win_rate']*100:.1f}%)")

    # Check some trade PnLs
    if results['trades']:
        sample_trades = results['trades'][:5]
        print(f"\nSample trades:")
        for t in sample_trades:
            direction = "SHORT" if t.quantity < 0 else "LONG"
            print(f"  {t.asset}: {direction} entry=${t.entry_price:.2f}, "
                  f"exit=${t.exit_price:.2f}, pnl=${t.pnl:.2f}")

    return success


def test_short_only_rising_market():
    """Test short-only strategy in rising market (should lose)."""
    print("\n" + "=" * 70)
    print("TEST 2: Short-Only Strategy in Rising Market")
    print("=" * 70)

    n_assets = 50
    n_bars = 500

    # Generate rising market data
    df = generate_trending_data(n_assets=n_assets, n_bars=n_bars, trend_direction=1)

    feed = DataFeed(prices_df=df)
    strategy = ShortOnlyStrategy(n_assets=n_assets)

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        account_type="margin",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )

    results = engine.run()

    print(f"\nResults:")
    print(f"  Final value: ${results['final_value']:,.2f}")
    print(f"  Return: {results['total_return_pct']:.2f}%")
    print(f"  Trades: {results['num_trades']:,}")

    # Validate: Should lose in rising market
    if results['total_return_pct'] > 0:
        print(f"\n[FAIL] Short strategy shouldn't profit in rising market")
        return False
    else:
        print(f"\n[PASS] Short strategy correctly lost in rising market")
        return True


def test_long_short_hedged():
    """Test long-short hedged strategy (market neutral)."""
    print("\n" + "=" * 70)
    print("TEST 3: Long-Short Hedged Strategy")
    print("=" * 70)

    n_assets = 50
    n_bars = 500

    # Use flat market (no trend)
    df = generate_trending_data(n_assets=n_assets, n_bars=n_bars, trend_direction=0, seed=123)

    feed = DataFeed(prices_df=df)
    strategy = LongShortStrategy(n_assets=n_assets)

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
        account_type="margin",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )

    results = engine.run()

    print(f"\nResults:")
    print(f"  Final value: ${results['final_value']:,.2f}")
    print(f"  Return: {results['total_return_pct']:.2f}%")
    print(f"  Trades: {results['num_trades']:,}")

    # Count long vs short positions
    long_trades = sum(1 for t in results['trades'] if t.quantity > 0)
    short_trades = sum(1 for t in results['trades'] if t.quantity < 0)
    print(f"  Long trades: {long_trades}")
    print(f"  Short trades: {short_trades}")

    success = True

    # Should have both long and short trades
    if long_trades == 0 or short_trades == 0:
        print(f"\n[FAIL] Should have both long and short trades")
        success = False
    else:
        print(f"\n[PASS] Has both long and short trades")

    # Returns should be relatively small (market neutral)
    if abs(results['total_return_pct']) > 50:
        print(f"[WARN] Returns high for market-neutral strategy")
    else:
        print(f"[PASS] Returns reasonable for market-neutral strategy")

    return success


def test_short_pnl_correctness():
    """Test that short position PnL is calculated correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Short Position PnL Correctness")
    print("=" * 70)

    # Create simple controlled data
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    data = []

    # Asset that falls from 100 to 90 over 10 bars
    for i, d in enumerate(dates):
        price = 100.0 - i * 0.5  # Falls $0.50 per bar
        data.append({
            "timestamp": d,
            "asset": "FALLING",
            "open": price + 0.1,
            "high": price + 0.2,
            "low": price - 0.1,
            "close": price,
            "volume": 100000.0,
        })

    df = pl.DataFrame(data)

    class SimpleShortStrategy(Strategy):
        def __init__(self):
            self.bar_count = 0
            self.entry_price = None

        def on_data(self, timestamp, data, context, broker):
            self.bar_count += 1

            if "FALLING" not in data:
                return

            pos = broker.get_position("FALLING")
            qty = pos.quantity if pos else 0

            if self.bar_count == 1 and qty == 0:
                broker.submit_order("FALLING", 100, OrderSide.SELL)
            elif self.bar_count == 10 and qty < 0:
                broker.close_position("FALLING")

    feed = DataFeed(prices_df=df)
    strategy = SimpleShortStrategy()

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=100_000.0,
        account_type="margin",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )

    results = engine.run()

    if results['trades']:
        trade = results['trades'][0]
        print(f"\nTrade details:")
        print(f"  Entry price: ${trade.entry_price:.2f}")
        print(f"  Exit price: ${trade.exit_price:.2f}")
        print(f"  Quantity: {trade.quantity}")
        print(f"  PnL: ${trade.pnl:.2f}")

        # Calculate expected PnL
        # Short 100 shares at ~$100, cover at ~$95 = $500 profit
        # (entry - exit) * qty for short
        expected_pnl = (trade.entry_price - trade.exit_price) * abs(trade.quantity)

        print(f"\nExpected PnL calculation:")
        print(f"  (entry - exit) * qty = ({trade.entry_price:.2f} - {trade.exit_price:.2f}) * {abs(trade.quantity)}")
        print(f"  = ${expected_pnl:.2f}")

        if abs(trade.pnl - expected_pnl) < 0.01:
            print(f"\n[PASS] Short PnL calculated correctly")
            return True
        else:
            print(f"\n[FAIL] PnL mismatch: expected ${expected_pnl:.2f}, got ${trade.pnl:.2f}")
            return False
    else:
        print(f"\n[FAIL] No trades executed")
        return False


def main():
    print("=" * 70)
    print("SHORT SELLING VALIDATION")
    print("=" * 70)

    all_pass = True

    all_pass &= test_short_only_falling_market()
    all_pass &= test_short_only_rising_market()
    all_pass &= test_long_short_hedged()
    all_pass &= test_short_pnl_correctness()

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL SHORT SELLING TESTS PASSED")
    else:
        print("SOME SHORT SELLING TESTS FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
