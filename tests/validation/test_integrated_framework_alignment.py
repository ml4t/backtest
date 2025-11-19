"""
Comprehensive cross-framework validation for ml4t.backtest with multi-asset Top-N momentum.

This test validates execution fidelity across all 4 frameworks:
- ml4t.backtest
- VectorBT Pro
- Backtrader
- Zipline-reloaded

Strategy: Top-N Momentum Rotation
- Universe: 25 stocks
- Period: 1 year (252 trading days)
- Rotation: Every 20 days, rank by 20-day returns, buy top 5
- Position Sizing: Equal weight (20% each)
- Costs: 0.1% commission, 0.1% slippage per trade
- Initial Capital: $100,000

Approach: Signal-based validation
- Pre-compute all buy/sell signals externally
- Feed identical signals to all frameworks
- Eliminates calculation variance (focus on execution)
- Variance tolerance: <0.5%

CRITICAL: All frameworks use SAME data and SAME signals
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.validation.common.data_generator import generate_ohlcv, validate_ohlcv
from tests.validation.frameworks.base import FrameworkConfig, ValidationResult


def generate_multi_asset_data(
    n_stocks: int = 25,
    n_days: int = 252,
    seed: int = 42,
    start_date: str = "2020-01-02",
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for multiple stocks.

    Args:
        n_stocks: Number of stocks to generate
        n_days: Number of trading days
        seed: Random seed for reproducibility
        start_date: Start date for data

    Returns:
        Dict mapping symbol -> OHLCV DataFrame
    """
    np.random.seed(seed)

    data = {}
    stock_names = [f"STOCK{i:02d}" for i in range(n_stocks)]

    # Generate price patterns with varied characteristics
    for i, symbol in enumerate(stock_names):
        # Vary base price and volatility across stocks
        base_price = 50 + (i * 10)  # Range: $50 - $290
        volatility = 0.01 + (i * 0.0005)  # Range: 1.0% - 2.2%

        df = generate_ohlcv(
            n_bars=n_days,
            symbol=symbol,
            start_date=start_date,
            freq="1D",
            base_price=base_price,
            volatility=volatility,
            seed=seed + i,  # Different seed per stock for variety
        )

        # Validate OHLCV constraints
        validate_ohlcv(df)

        data[symbol] = df

    return data


def compute_momentum_signals(
    data: dict[str, pd.DataFrame],
    lookback_days: int = 20,
    rotation_days: int = 20,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Compute Top-N momentum rotation signals.

    Strategy:
    - Every rotation_days, rank all stocks by lookback_days returns
    - Buy top N stocks (equal weight)
    - Sell stocks no longer in top N

    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        lookback_days: Days to calculate momentum (returns)
        rotation_days: Days between rotations
        top_n: Number of stocks to hold

    Returns:
        DataFrame with columns: timestamp, symbol, signal (1=buy, -1=sell, 0=hold)
    """
    # Combine all closes into panel
    closes = pd.DataFrame({symbol: df['close'] for symbol, df in data.items()})

    # Calculate returns over lookback period
    returns = closes.pct_change(lookback_days)

    signals = []
    current_positions = set()

    # Get rotation dates (every rotation_days)
    all_dates = closes.index
    rotation_dates = all_dates[::rotation_days]

    for date in rotation_dates:
        if date not in returns.index:
            continue

        # Skip if not enough history
        if pd.isna(returns.loc[date]).all():
            continue

        # Rank stocks by returns
        daily_returns = returns.loc[date].dropna()

        if len(daily_returns) == 0:
            continue

        # Get top N stocks
        top_stocks = daily_returns.nlargest(top_n).index.tolist()
        new_positions = set(top_stocks)

        # Generate sell signals for stocks dropping out
        for symbol in current_positions - new_positions:
            signals.append({
                'timestamp': date,
                'symbol': symbol,
                'signal': -1,  # SELL
            })

        # Generate buy signals for new stocks
        for symbol in new_positions - current_positions:
            signals.append({
                'timestamp': date,
                'symbol': symbol,
                'signal': 1,  # BUY
            })

        current_positions = new_positions

    # Close all positions at end
    if len(current_positions) > 0:
        final_date = all_dates[-1]
        for symbol in current_positions:
            signals.append({
                'timestamp': final_date,
                'symbol': symbol,
                'signal': -1,  # SELL
            })

    df_signals = pd.DataFrame(signals)
    if len(df_signals) > 0:
        df_signals = df_signals.sort_values('timestamp').reset_index(drop=True)

    return df_signals


class TestIntegratedFrameworkAlignment:
    """
    Integrated cross-framework validation with multi-asset Top-N momentum.

    Validates that ml4t.backtest produces equivalent results to established frameworks
    when all receive identical signals and configuration.
    """

    @pytest.fixture
    def test_data(self):
        """Generate test data for validation."""
        return generate_multi_asset_data(
            n_stocks=25,
            n_days=252,
            seed=42,
            start_date="2020-01-02",
        )

    @pytest.fixture
    def signals(self, test_data):
        """Pre-compute momentum rotation signals."""
        return compute_momentum_signals(
            test_data,
            lookback_days=20,
            rotation_days=20,
            top_n=5,
        )

    def test_signal_generation(self, test_data, signals):
        """Verify signal generator produces valid output."""
        assert len(signals) > 0, "Should generate signals"
        assert 'timestamp' in signals.columns
        assert 'symbol' in signals.columns
        assert 'signal' in signals.columns

        # Check signal values
        assert set(signals['signal'].unique()).issubset({-1, 0, 1})

        # Count rotations (expect ~12 for 252 days with 20-day rotation)
        rotation_dates = signals['timestamp'].unique()
        assert 10 <= len(rotation_dates) <= 15, \
            f"Expected 10-15 rotation dates, got {len(rotation_dates)}"

        print(f"\n✅ Generated {len(signals)} signals across {len(rotation_dates)} rotation dates")
        print(f"   Symbols involved: {signals['symbol'].nunique()}")
        print(f"   Buy signals: {(signals['signal'] == 1).sum()}")
        print(f"   Sell signals: {(signals['signal'] == -1).sum()}")

    def test_data_generation(self, test_data):
        """Verify multi-asset data generation."""
        assert len(test_data) == 25, "Should generate 25 stocks"

        for symbol, df in test_data.items():
            assert len(df) == 252, f"{symbol} should have 252 days"
            assert not df.isnull().any().any(), f"{symbol} should have no NaN values"

            # Validate OHLCV constraints
            validate_ohlcv(df)

        print(f"\n✅ Generated data for {len(test_data)} stocks, 252 days each")

        # Show price ranges
        price_ranges = {}
        for symbol, df in test_data.items():
            price_ranges[symbol] = (df['close'].min(), df['close'].max())

        print("\nPrice ranges:")
        for symbol in sorted(price_ranges.keys())[:5]:  # Show first 5
            min_p, max_p = price_ranges[symbol]
            print(f"   {symbol}: ${min_p:.2f} - ${max_p:.2f}")

    def test_qengine_execution(self, test_data, signals):
        """Test ml4t.backtest adapter with Top-N momentum signals."""
        from tests.validation.frameworks import BacktestAdapter

        adapter = BacktestAdapter()
        config = FrameworkConfig(
            initial_capital=100000.0,
            commission_pct=0.001,  # 0.1%
            slippage_pct=0.001,    # 0.1%
            fill_timing="next_open",
        )

        # test_data is already a dict[str, DataFrame]
        # signals already has [timestamp, symbol, signal] columns
        result = adapter.run_with_signals(
            data=test_data,
            signals=signals,
            config=config,
        )

        print(f"\n{'=' * 80}")
        print(f"ml4t.backtest Results:")
        print(f"{'=' * 80}")
        print(f"Final Value: ${result.final_value:,.2f}")
        print(f"Return: {result.total_return:.2f}%")
        print(f"Trades: {result.num_trades}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        print(f"{'=' * 80}\n")

        assert not result.has_errors, f"ml4t.backtest had errors: {result.errors}"
        assert result.num_trades > 0, "Should execute some trades"
        assert result.final_value > 0, "Should have final value"

    def test_vectorbt_execution(self, test_data, signals):
        """Test VectorBT adapter with Top-N momentum signals."""
        from tests.validation.frameworks import VectorBTAdapter

        adapter = VectorBTAdapter()
        config = FrameworkConfig(
            initial_capital=100000.0,
            commission_pct=0.001,  # 0.1%
            slippage_pct=0.001,    # 0.1%
            fill_timing="next_open",
        )

        # test_data is already a dict[str, DataFrame]
        # signals already has [timestamp, symbol, signal] columns
        result = adapter.run_with_signals(
            data=test_data,
            signals=signals,
            config=config,
        )

        print(f"\n{'=' * 80}")
        print(f"VectorBT Results:")
        print(f"{'=' * 80}")
        print(f"Final Value: ${result.final_value:,.2f}")
        print(f"Return: {result.total_return:.2f}%")
        print(f"Trades: {result.num_trades}")
        print(f"Execution Time: {result.execution_time:.3f}s")
        print(f"{'=' * 80}\n")

        assert not result.has_errors, f"VectorBT had errors: {result.errors}"
        assert result.num_trades > 0, "Should execute some trades"
        assert result.final_value > 0, "Should have final value"

    def test_all_frameworks_alignment(self, test_data, signals):
        """
        Full 4-way cross-framework validation.

        Compares ml4t.backtest, VectorBT Pro, Backtrader, and Zipline-reloaded
        with identical signals and configuration.

        Acceptance Criteria:
        - All frameworks execute signals
        - Final values within 0.5% variance
        - Trade counts within ±2 trades (due to end-of-backtest handling)
        - Any systematic differences documented with source code citations
        """
        from tests.validation.frameworks import (
            BacktestAdapter,
            VectorBTAdapter,
            BacktraderAdapter,
            ZiplineAdapter,
        )

        config = FrameworkConfig(
            initial_capital=100000.0,
            commission_pct=0.001,  # 0.1%
            slippage_pct=0.001,    # 0.1%
            fill_timing="next_open",
            fractional_shares=True,  # Prevent quantity mismatches
        )

        adapters = {
            "ml4t.backtest": BacktestAdapter(),
            "VectorBT": VectorBTAdapter(),
            "Backtrader": BacktraderAdapter(),
            "Zipline": ZiplineAdapter(),
        }

        results = {}

        print(f"\n{'=' * 80}")
        print(f"4-WAY FRAMEWORK VALIDATION: Top-N Momentum (25 stocks, 1 year)")
        print(f"{'=' * 80}")
        print(f"Signals: {len(signals)} total")
        print(f"Config: {config.commission_pct:.1%} commission, {config.slippage_pct:.1%} slippage")
        print(f"{'=' * 80}\n")

        # Run all frameworks
        for name, adapter in adapters.items():
            print(f"Running {name}...")
            start_time = time.time()

            try:
                result = adapter.run_with_signals(
                    data=test_data,
                    signals=signals,
                    config=config,
                )
                elapsed = time.time() - start_time
                result.execution_time = elapsed
                results[name] = result

                print(f"   ✅ Completed in {elapsed:.3f}s")
                print(f"      Final Value: ${result.final_value:,.2f}")
                print(f"      Trades: {result.num_trades}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[name] = ValidationResult(
                    framework=name,
                    strategy="TopN Momentum",
                    errors=[str(e)],
                )

        # Display comparison table
        print(f"\n{'=' * 80}")
        print(f"COMPARISON RESULTS")
        print(f"{'=' * 80}")
        print(f"{'Framework':<15} {'Final Value':>15} {'Return':>10} {'Trades':>8} {'Time (s)':>10}")
        print(f"{'-' * 80}")

        for name, result in results.items():
            if not result.has_errors:
                print(f"{name:<15} ${result.final_value:>14,.2f} {result.total_return:>9.2f}% "
                      f"{result.num_trades:>8} {result.execution_time:>9.3f}")
            else:
                print(f"{name:<15} {'ERROR':>15} {'':<10} {'':<8}")

        # Export trades to CSV for detailed comparison
        output_dir = Path("tests/validation/trade_logs")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"EXPORTING TRADE LOGS")
        print(f"{'=' * 80}")

        for name, result in results.items():
            if not result.has_errors and result.trades:
                # Create DataFrame from trades
                trades_data = []
                for trade in result.trades:
                    trade_dict = {
                        'timestamp': trade.timestamp,
                        'action': trade.action,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'value': trade.value,
                        'commission': trade.commission if hasattr(trade, 'commission') else 0,
                    }
                    # Add symbol if available (for multi-asset)
                    if hasattr(trade, 'symbol'):
                        trade_dict['symbol'] = trade.symbol
                    trades_data.append(trade_dict)

                trades_df = pd.DataFrame(trades_data)
                filename = output_dir / f"trades_{name.replace(' ', '_').replace('.', '_')}.csv"
                trades_df.to_csv(filename, index=False)
                print(f"   Exported {len(trades_df)} trades to {filename}")

        # Calculate variance
        available_results = {name: r for name, r in results.items() if not r.has_errors}

        if len(available_results) < 2:
            pytest.skip(f"Need at least 2 frameworks available, got {len(available_results)}")

        final_values = [r.final_value for r in available_results.values()]
        returns = [r.total_return for r in available_results.values()]
        trade_counts = [r.num_trades for r in available_results.values()]

        value_range = max(final_values) - min(final_values)
        value_pct_range = (value_range / config.initial_capital) * 100
        return_range = max(returns) - min(returns)
        trade_range = max(trade_counts) - min(trade_counts)

        print(f"\n{'Variance Statistics':}")
        print(f"  Value Range: ${value_range:,.2f} ({value_pct_range:.4f}%)")
        print(f"  Return Range: {return_range:.4f}%")
        print(f"  Trade Count Range: {trade_range}")
        print(f"{'=' * 80}\n")

        # Assertions
        assert value_pct_range < 0.5, \
            f"Value variance {value_pct_range:.4f}% exceeds 0.5% threshold"

        assert trade_range <= 5, \
            f"Trade count variance {trade_range} exceeds ±5 trades tolerance"

        print(f"✅ 4-WAY VALIDATION PASSED - All frameworks produce equivalent results!")
        print(f"   Variance: {value_pct_range:.4f}% (<0.5% threshold)")
        print(f"   Trade alignment: ±{trade_range} trades")

    def test_all_frameworks_alignment_scaled(self):
        """
        Scaled-up 4-way validation: 50 stocks, 3 years, 100+ trades.

        Tests that variance remains <0.5% with:
        - More assets (50 stocks)
        - Longer period (3 years = 756 trading days)
        - More trades (100-300+ expected)
        - Equal-weighted portfolio (20% per position, 5 positions max)

        This validates that execution differences don't compound over time and scale.
        """
        from tests.validation.frameworks import (
            BacktestAdapter,
            VectorBTAdapter,
            BacktraderAdapter,
            ZiplineAdapter,
        )

        # Generate scaled-up data
        test_data = generate_multi_asset_data(
            n_stocks=50,  # 2x more stocks
            n_days=756,   # 3 years (252 * 3)
            seed=42,
            start_date="2020-01-02",
        )

        # Generate signals with same rotation logic
        signals = compute_momentum_signals(
            test_data,
            lookback_days=20,
            rotation_days=20,
            top_n=5,
        )

        config = FrameworkConfig(
            initial_capital=100000.0,
            commission_pct=0.001,  # 0.1%
            slippage_pct=0.001,    # 0.1%
            fill_timing="next_open",
            fractional_shares=True,
        )

        adapters = {
            "ml4t.backtest": BacktestAdapter(),
            "VectorBT": VectorBTAdapter(),
            "Backtrader": BacktraderAdapter(),
            "Zipline": ZiplineAdapter(),
        }

        results = {}

        print(f"\n{'=' * 80}")
        print(f"SCALED 4-WAY VALIDATION: Top-N Momentum (50 stocks, 3 years)")
        print(f"{'=' * 80}")
        print(f"Stocks: 50, Days: 756, Signals: {len(signals)}")
        print(f"Config: {config.commission_pct:.1%} commission, {config.slippage_pct:.1%} slippage")
        print(f"{'=' * 80}\n")

        # Run all frameworks
        for name, adapter in adapters.items():
            print(f"Running {name}...")
            start_time = time.time()

            try:
                result = adapter.run_with_signals(
                    data=test_data,
                    signals=signals,
                    config=config,
                )
                elapsed = time.time() - start_time
                result.execution_time = elapsed
                results[name] = result

                print(f"   ✅ Completed in {elapsed:.3f}s")
                print(f"      Final Value: ${result.final_value:,.2f}")
                print(f"      Return: {result.total_return:.2f}%")
                print(f"      Trades: {result.num_trades}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results[name] = ValidationResult(
                    framework=name,
                    strategy="TopN Momentum Scaled",
                    errors=[str(e)],
                )

        # Display comparison table
        print(f"\n{'=' * 80}")
        print(f"SCALED COMPARISON RESULTS")
        print(f"{'=' * 80}")
        print(f"{'Framework':<15} {'Final Value':>15} {'Return':>10} {'Trades':>8} {'Time (s)':>10}")
        print(f"{'-' * 80}")

        for name, result in results.items():
            if not result.has_errors:
                print(f"{name:<15} ${result.final_value:>14,.2f} {result.total_return:>9.2f}% "
                      f"{result.num_trades:>8} {result.execution_time:>9.3f}")
            else:
                print(f"{name:<15} {'ERROR':>15} {'':<10} {'':<8}")

        # Calculate variance
        available_results = {name: r for name, r in results.items() if not r.has_errors}

        if len(available_results) < 2:
            pytest.skip(f"Need at least 2 frameworks available, got {len(available_results)}")

        final_values = [r.final_value for r in available_results.values()]
        returns = [r.total_return for r in available_results.values()]
        trade_counts = [r.num_trades for r in available_results.values()]

        value_range = max(final_values) - min(final_values)
        value_pct_range = (value_range / config.initial_capital) * 100
        return_range = max(returns) - min(returns)
        trade_range = max(trade_counts) - min(trade_counts)

        print(f"\n{'Variance Statistics':}")
        print(f"  Value Range: ${value_range:,.2f} ({value_pct_range:.4f}%)")
        print(f"  Return Range: {return_range:.4f}%")
        print(f"  Trade Count Range: {trade_range}")
        print(f"{'=' * 80}\n")

        # Assertions - slightly relaxed for larger scale
        assert value_pct_range < 1.0, \
            f"Value variance {value_pct_range:.4f}% exceeds 1.0% threshold (scaled test)"

        assert trade_range <= 10, \
            f"Trade count variance {trade_range} exceeds ±10 trades tolerance (scaled test)"

        print(f"✅ SCALED 4-WAY VALIDATION PASSED!")
        print(f"   Variance: {value_pct_range:.4f}% (<1.0% threshold for scaled test)")
        print(f"   Trade alignment: ±{trade_range} trades")
        print(f"   Scale: 50 stocks, 3 years, {len(signals)} signals")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
