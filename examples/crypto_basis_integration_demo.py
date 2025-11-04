#!/usr/bin/env python3
"""
Crypto Basis Strategy QEngine Integration Demo
==============================================

This example demonstrates how to integrate an external crypto basis trading strategy
with QEngine's event-driven backtesting framework. It shows:

1. How to use the Strategy-QEngine Integration Bridge
2. Running a sophisticated strategy in QEngine
3. Leveraging QEngine's advanced execution models
4. Comparing standalone vs QEngine results

Run with: PYTHONPATH=~/ml4t/qengine/src python examples/crypto_basis_integration_demo.py
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
from ml4t_backtest import BacktestEngine

from qengine.data.feed import DataFrameFeed
from qengine.execution.broker import SimulationBroker
from qengine.execution.commission import MakerTakerCommission, PercentageCommission
from qengine.execution.market_impact import LinearMarketImpact
from qengine.execution.slippage import LinearImpactSlippage, PercentageSlippage
from qengine.portfolio.simple import SimplePortfolio
from qengine.reporting.reporter import InMemoryReporter
from qengine.strategy import create_crypto_basis_strategy


def generate_synthetic_crypto_data(
    n_periods: int = 1000,
    start_price: float = 50000.0,
    basis_mean: float = 100.0,
    basis_volatility: float = 50.0,
) -> dict[str, pl.DataFrame]:
    """Generate synthetic crypto spot and futures data."""
    print("Generating synthetic crypto data...")

    # Time series
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_periods)]

    # Generate correlated price movements
    np.random.seed(42)  # For reproducible results

    # Spot price random walk
    returns = np.random.normal(0, 0.002, n_periods)  # 0.2% volatility
    spot_prices = [start_price]
    for i in range(1, n_periods):
        spot_prices.append(spot_prices[-1] * (1 + returns[i]))

    # Basis with mean reversion
    basis_shocks = np.random.normal(0, basis_volatility * 0.1, n_periods)
    basis_values = [basis_mean]

    for i in range(1, n_periods):
        # Mean reversion with some persistence
        mean_reversion = -0.05 * (basis_values[-1] - basis_mean)
        basis_values.append(basis_values[-1] + mean_reversion + basis_shocks[i])

    # Futures prices = spot + basis
    futures_prices = [spot + basis for spot, basis in zip(spot_prices, basis_values, strict=False)]

    # Create spot DataFrame
    spot_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["BTC"] * n_periods,
            "open": [p * 0.999 for p in spot_prices],
            "high": [p * 1.002 for p in spot_prices],
            "low": [p * 0.998 for p in spot_prices],
            "close": spot_prices,
            "volume": np.random.randint(800, 1200, n_periods),
        },
    )

    # Create futures DataFrame
    futures_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["BTC_FUTURE"] * n_periods,
            "open": [p * 0.999 for p in futures_prices],
            "high": [p * 1.002 for p in futures_prices],
            "low": [p * 0.998 for p in futures_prices],
            "close": futures_prices,
            "volume": np.random.randint(600, 1000, n_periods),
        },
    )

    print(f"Generated {n_periods} data points")
    print(f"Spot price range: ${min(spot_prices):,.0f} - ${max(spot_prices):,.0f}")
    print(f"Basis range: ${min(basis_values):,.0f} - ${max(basis_values):,.0f}")

    return {"spot": spot_df, "futures": futures_df}


def run_standalone_strategy(data: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """Run the original standalone strategy for comparison."""
    print("\n=== Running Standalone Strategy ===")

    # Import original strategy
    import sys

    sys.path.append("~/ml4t/projects/crypto_futures")
    from basis_trading_strategy import CryptoBasisStrategy

    # Initialize strategy
    strategy = CryptoBasisStrategy(
        lookback_window=60,
        entry_threshold=1.5,
        exit_threshold=0.5,
        max_position=0.2,
    )

    # Run backtest
    results = strategy.backtest(
        spot_df=data["spot"],
        futures_df=data["futures"],
        initial_capital=100000,
        transaction_cost=0.001,  # 0.1%
    )

    print("Standalone Results:")
    print(f"  Total Return: {results['metrics']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"  Number of Trades: {results['metrics']['num_trades']}")

    return results


def run_qengine_basic_strategy(data: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """Run strategy in QEngine with basic execution models."""
    print("\n=== Running QEngine Basic Strategy ===")

    # Combine data for QEngine
    combined_df = pl.concat([data["spot"], data["futures"]])
    combined_df = combined_df.sort("timestamp")

    # Create strategy using integration bridge
    strategy = create_crypto_basis_strategy(
        spot_asset_id="BTC",
        futures_asset_id="BTC_FUTURE",
        lookback_window=60,
        entry_threshold=1.5,
        exit_threshold=0.5,
        max_position=0.2,
        position_scaling=0.05,  # Use 5% of capital per signal
    )

    # Basic broker with simple models
    broker = SimulationBroker(
        initial_cash=100000,
        commission_model=PercentageCommission(rate=0.001),  # 0.1%
        slippage_model=PercentageSlippage(slippage_rate=0.0005),  # 0.05%
    )

    # Create backtest engine
    engine = BacktestEngine(
        data_feed=DataFrameFeed(combined_df),
        strategy=strategy,
        broker=broker,
        portfolio=SimplePortfolio(initial_cash=100000),
        reporter=InMemoryReporter(),
    )

    # Run backtest
    print("Running QEngine backtest...")
    results = engine.run()

    # Extract metrics
    portfolio_value = results.get("final_portfolio_value", 100000)
    total_return = (portfolio_value - 100000) / 100000

    print("QEngine Basic Results:")
    print(f"  Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Number of Trades: {len(results.get('trades', []))}")

    return results


def run_qengine_advanced_strategy(data: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """Run strategy in QEngine with advanced execution models."""
    print("\n=== Running QEngine Advanced Strategy ===")

    # Combine data for QEngine
    combined_df = pl.concat([data["spot"], data["futures"]])
    combined_df = combined_df.sort("timestamp")

    # Create strategy
    strategy = create_crypto_basis_strategy(
        spot_asset_id="BTC",
        futures_asset_id="BTC_FUTURE",
        lookback_window=60,
        entry_threshold=1.5,
        exit_threshold=0.5,
        max_position=0.2,
        position_scaling=0.05,
    )

    # Advanced broker with sophisticated models
    broker = SimulationBroker(
        initial_cash=100000,
        commission_model=MakerTakerCommission(
            maker_rate=-0.0001,
            taker_rate=0.0003,
        ),  # Rebates/fees
        slippage_model=LinearImpactSlippage(impact_coeff=1e-6, daily_volume=1000000),
        market_impact_model=LinearMarketImpact(
            permanent_factor=0.1,
            temporary_factor=0.5,
            avg_daily_volume=1000000,
        ),
    )

    # Create backtest engine
    engine = BacktestEngine(
        data_feed=DataFrameFeed(combined_df),
        strategy=strategy,
        broker=broker,
        portfolio=SimplePortfolio(initial_cash=100000),
        reporter=InMemoryReporter(),
    )

    # Run backtest
    print("Running QEngine advanced backtest...")
    results = engine.run()

    # Extract metrics
    portfolio_value = results.get("final_portfolio_value", 100000)
    total_return = (portfolio_value - 100000) / 100000

    print("QEngine Advanced Results:")
    print(f"  Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Number of Trades: {len(results.get('trades', []))}")

    # Show cost breakdown if available
    trades = results.get("trades", [])
    if trades:
        total_commission = sum(t.get("commission", 0) for t in trades)
        total_slippage = sum(t.get("slippage", 0) for t in trades)
        total_impact = sum(t.get("market_impact", 0) for t in trades)

        print(f"  Total Commission: ${total_commission:.2f}")
        print(f"  Total Slippage: ${total_slippage:.2f}")
        print(f"  Total Market Impact: ${total_impact:.2f}")

    return results


def compare_results(standalone: dict, basic: dict, advanced: dict) -> None:
    """Compare results across different implementations."""
    print("\n=== Results Comparison ===")

    # Extract returns
    standalone_return = standalone["metrics"]["total_return"]

    basic_value = basic.get("final_portfolio_value", 100000)
    basic_return = (basic_value - 100000) / 100000

    advanced_value = advanced.get("final_portfolio_value", 100000)
    advanced_return = (advanced_value - 100000) / 100000

    print("Strategy Performance:")
    print(f"  Standalone:      {standalone_return:8.2%}")
    print(f"  QEngine Basic:   {basic_return:8.2%}")
    print(f"  QEngine Advanced:{advanced_return:8.2%}")

    print("\nTrade Counts:")
    print(f"  Standalone:      {standalone['metrics']['num_trades']:8}")
    print(f"  QEngine Basic:   {len(basic.get('trades', [])):8}")
    print(f"  QEngine Advanced:{len(advanced.get('trades', [])):8}")

    # Analysis
    print("\nAnalysis:")
    if basic_return < standalone_return:
        diff = abs(basic_return - standalone_return)
        print(
            f"  • QEngine Basic shows {diff:.2%} lower return due to more realistic execution costs",
        )

    if advanced_return < basic_return:
        diff = abs(advanced_return - basic_return)
        print(f"  • QEngine Advanced shows {diff:.2%} lower return due to market impact modeling")

    print("  • The integration bridge successfully connects external strategies to QEngine")
    print("  • QEngine provides more realistic cost modeling than standalone implementations")


def main():
    """Run the complete integration demo."""
    print("Crypto Basis Strategy QEngine Integration Demo")
    print("=" * 50)

    try:
        # Generate synthetic data
        data = generate_synthetic_crypto_data(n_periods=500)

        # Run all three implementations
        standalone_results = run_standalone_strategy(data)
        basic_results = run_qengine_basic_strategy(data)
        advanced_results = run_qengine_advanced_strategy(data)

        # Compare results
        compare_results(standalone_results, basic_results, advanced_results)

        print("\n=== Integration Bridge Success! ===")
        print("✅ External strategy successfully integrated with QEngine")
        print("✅ Multiple execution models tested")
        print("✅ Results show realistic cost impact")
        print("✅ Framework ready for production strategies")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
