"""Portfolio rebalancing scale validation.

Tests portfolio rebalancing at scale:
- 100 assets, 5 years, monthly rebalancing
- Equal-weight and risk-parity strategies
- Validates weight accuracy, cash balance, performance

Run with:
    source .venv/bin/activate
    python validation/rebalancing_scale_test.py
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
    RebalanceConfig,
    Strategy,
    TargetWeightExecutor,
)


def generate_multi_asset_data(
    n_assets: int = 100,
    n_bars: int = 1260,  # 5 years daily
    seed: int = 42,
) -> pl.DataFrame:
    """Generate multi-asset price data."""
    np.random.seed(seed)

    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_bars)]
    all_data = []

    for i in range(n_assets):
        asset = f"ASSET{i:04d}"
        base_price = 50.0 + i * 0.5
        returns = np.random.randn(n_bars) * 0.015  # 1.5% daily vol
        closes = base_price * np.exp(np.cumsum(returns))

        # Generate valid OHLC
        opens = closes * (1 + np.random.randn(n_bars) * 0.002)
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_bars)) * 0.003)
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_bars)) * 0.003)

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


class EqualWeightRebalanceStrategy(Strategy):
    """Monthly equal-weight rebalancing strategy."""

    def __init__(self, n_assets: int, rebalance_freq: int = 21):
        self.n_assets = n_assets
        self.rebalance_freq = rebalance_freq
        self.bar_count = 0
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=100.0,
                allow_fractional=True,
                min_weight_change=0.001,
            )
        )
        self.rebalance_count = 0
        self.weight_errors = []
        self.total_weights = []

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1

        # Rebalance monthly
        if self.bar_count % self.rebalance_freq == 1 or self.bar_count == 1:
            # Equal weights for all available assets
            target_weight = 1.0 / self.n_assets
            target_weights = {
                f"ASSET{i:04d}": target_weight for i in range(self.n_assets)
            }

            orders = self.executor.execute(target_weights, data, broker)
            for order in orders:
                broker.submit_order(order.asset, order.quantity, order.side)

            self.rebalance_count += 1

        # Track weight accuracy after rebalance settles
        if self.bar_count % self.rebalance_freq == 5 and self.bar_count > 5:
            account_value = broker.get_account_value()
            total_weight = 0.0
            max_weight_error = 0.0

            for i in range(self.n_assets):
                asset = f"ASSET{i:04d}"
                pos = broker.get_position(asset)
                if pos and pos.quantity > 0 and asset in data:
                    price = data[asset].get("close", 0)
                    pos_value = pos.quantity * price
                    actual_weight = pos_value / account_value if account_value > 0 else 0
                    target_weight = 1.0 / self.n_assets
                    weight_error = abs(actual_weight - target_weight)
                    max_weight_error = max(max_weight_error, weight_error)
                    total_weight += actual_weight

            self.weight_errors.append(max_weight_error)
            self.total_weights.append(total_weight)


class RiskParityStrategy(Strategy):
    """Quarterly inverse-volatility (risk parity proxy) strategy."""

    def __init__(self, n_assets: int, lookback: int = 63, rebalance_freq: int = 63):
        self.n_assets = n_assets
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.bar_count = 0
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=100.0,
                allow_fractional=True,
            )
        )
        self.rebalance_count = 0
        self.price_history = {f"ASSET{i:04d}": [] for i in range(n_assets)}

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1

        # Update price history
        for asset in data:
            if asset in self.price_history:
                self.price_history[asset].append(data[asset].get("close", 0))
                # Keep only lookback period
                if len(self.price_history[asset]) > self.lookback:
                    self.price_history[asset] = self.price_history[asset][-self.lookback:]

        # Rebalance quarterly
        if self.bar_count % self.rebalance_freq == 1 and self.bar_count > self.lookback:
            # Calculate inverse volatility weights
            vols = []
            assets_with_data = []

            for i in range(self.n_assets):
                asset = f"ASSET{i:04d}"
                if len(self.price_history[asset]) >= self.lookback:
                    prices = np.array(self.price_history[asset])
                    returns = np.diff(prices) / prices[:-1]
                    vol = np.std(returns) * np.sqrt(252)  # Annualized
                    if vol > 0:
                        vols.append(vol)
                        assets_with_data.append(asset)

            if not vols:
                return

            # Inverse volatility weights
            inv_vols = [1.0 / v for v in vols]
            total_inv_vol = sum(inv_vols)
            target_weights = {
                asset: inv_vol / total_inv_vol
                for asset, inv_vol in zip(assets_with_data, inv_vols)
            }

            orders = self.executor.execute(target_weights, data, broker)
            for order in orders:
                broker.submit_order(order.asset, order.quantity, order.side)

            self.rebalance_count += 1


def run_equal_weight_test(n_assets: int = 100, n_bars: int = 1260):
    """Run equal-weight rebalancing test."""
    print("\n" + "=" * 70)
    print("TEST 1: Equal-Weight Rebalancing")
    print("=" * 70)
    print(f"Assets: {n_assets}, Bars: {n_bars} (5 years), Monthly rebalancing")

    # Generate data
    t0 = time.perf_counter()
    df = generate_multi_asset_data(n_assets=n_assets, n_bars=n_bars)
    gen_time = time.perf_counter() - t0
    print(f"\nData generated: {len(df):,} rows in {gen_time:.2f}s")

    # Run backtest
    feed = DataFeed(prices_df=df)
    strategy = EqualWeightRebalanceStrategy(n_assets=n_assets)

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
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
    print(f"  Rebalances: {strategy.rebalance_count}")

    # Validate
    success = True

    # Check weight accuracy
    if strategy.weight_errors:
        avg_weight_error = np.mean(strategy.weight_errors)
        max_weight_error = np.max(strategy.weight_errors)
        print(f"\nWeight Accuracy:")
        print(f"  Avg max weight error: {avg_weight_error:.4f} ({avg_weight_error*100:.2f}%)")
        print(f"  Max weight error: {max_weight_error:.4f} ({max_weight_error*100:.2f}%)")

        if avg_weight_error > 0.02:  # 2% average error threshold
            print(f"  [FAIL] Average weight error too high (>{2}%)")
            success = False
        else:
            print(f"  [PASS] Weight accuracy within tolerance")

    # Check total weight (should be ~100%)
    if strategy.total_weights:
        avg_total = np.mean(strategy.total_weights)
        print(f"\nTotal Weight (should be ~1.0):")
        print(f"  Average: {avg_total:.4f}")

        # Allow 10% cash buffer - with 100 assets and $1M, some cash is expected
        if avg_total < 0.8:
            print(f"  [FAIL] Total weight too low (<80%)")
            success = False
        else:
            print(f"  [PASS] Total weight within tolerance (>80%)")

    # Check cash balance is non-negative
    cash = engine.broker.cash
    print(f"\nCash Balance: ${cash:,.2f}")
    if cash < -0.01:
        print(f"  [FAIL] Cash balance is negative")
        success = False
    else:
        print(f"  [PASS] Cash balance non-negative")

    return success, results


def run_risk_parity_test(n_assets: int = 50, n_bars: int = 1260):
    """Run risk parity rebalancing test."""
    print("\n" + "=" * 70)
    print("TEST 2: Risk Parity (Inverse Volatility) Rebalancing")
    print("=" * 70)
    print(f"Assets: {n_assets}, Bars: {n_bars} (5 years), Quarterly rebalancing")

    # Generate data
    df = generate_multi_asset_data(n_assets=n_assets, n_bars=n_bars)

    # Run backtest
    feed = DataFeed(prices_df=df)
    strategy = RiskParityStrategy(n_assets=n_assets)

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=1_000_000.0,
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
    print(f"  Rebalances: {strategy.rebalance_count}")

    # Basic validation
    success = results['final_value'] > 0 and results['num_trades'] > 0
    if success:
        print(f"\n[PASS] Risk parity rebalancing completed successfully")
    else:
        print(f"\n[FAIL] Risk parity rebalancing failed")

    return success, results


def run_large_scale_test(n_assets: int = 200, n_bars: int = 2520):
    """Run large scale performance test."""
    print("\n" + "=" * 70)
    print("TEST 3: Large Scale Performance")
    print("=" * 70)
    print(f"Assets: {n_assets}, Bars: {n_bars} (10 years)")

    # Generate data
    t0 = time.perf_counter()
    df = generate_multi_asset_data(n_assets=n_assets, n_bars=n_bars)
    gen_time = time.perf_counter() - t0
    print(f"\nData generated: {len(df):,} rows in {gen_time:.2f}s")

    # Run backtest with simple monthly rebalancing
    feed = DataFeed(prices_df=df)
    strategy = EqualWeightRebalanceStrategy(n_assets=n_assets)

    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=10_000_000.0,  # Larger capital
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
    )

    t0 = time.perf_counter()
    results = engine.run()
    run_time = time.perf_counter() - t0

    print(f"\nResults:")
    print(f"  Time: {run_time:.2f}s")
    print(f"  Bars/second: {n_bars / run_time:,.0f}")
    print(f"  Final value: ${results['final_value']:,.2f}")
    print(f"  Trades: {results['num_trades']:,}")

    # Performance threshold: should complete in reasonable time
    if run_time > 60:
        print(f"\n[WARNING] Performance slower than expected (>60s)")
        success = False
    else:
        print(f"\n[PASS] Performance acceptable ({run_time:.1f}s)")
        success = True

    return success, results


def main():
    print("=" * 70)
    print("PORTFOLIO REBALANCING SCALE VALIDATION")
    print("=" * 70)

    all_pass = True

    # Test 1: Equal-weight (100 assets, 5 years)
    success1, _ = run_equal_weight_test(n_assets=100, n_bars=1260)
    all_pass &= success1

    # Test 2: Risk parity (50 assets, 5 years)
    success2, _ = run_risk_parity_test(n_assets=50, n_bars=1260)
    all_pass &= success2

    # Test 3: Large scale (200 assets, 10 years)
    success3, _ = run_large_scale_test(n_assets=200, n_bars=2520)
    all_pass &= success3

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL REBALANCING SCALE TESTS PASSED")
    else:
        print("SOME REBALANCING TESTS FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
